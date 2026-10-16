"""High-level API: weights from the Hugging Face Hub or a local folder, references, boxes, optional upsampling.

    from flux3_mlx import Flux3Klein
    pipe = Flux3Klein.from_pretrained("8bit")
    image, record = pipe("a red bicycle leaning against a brick wall", seed=7)
"""

import gc
import hashlib
import json
import os
from pathlib import Path

import mlx.core as mx
from PIL import Image

from . import bbox
from .pipeline import MAX_REFERENCES, Pipeline, prepare_multi_reference, prepare_reference

HUB = "VisualInference"
MODELS = {"8bit": f"{HUB}/FLUX-3-Image-Klein-MLX-8bit", "6bit": f"{HUB}/FLUX-3-Image-Klein-MLX-6bit"}
UPSAMPLERS = {"4bit": f"{HUB}/FLUX-3-Image-Klein-pup-MLX-4bit", "8bit": f"{HUB}/FLUX-3-Image-Klein-pup-MLX-8bit"}


def system_memory_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError):
        return 0.0


def default_model() -> str:
    """8-bit (closest to BF16) when there is room for it, 6-bit otherwise."""
    return "8bit" if system_memory_gb() >= 24 else "6bit"


def resolve(name: str, table: dict) -> Path:
    """A key of ``table``, a local folder, or a Hub repo id -> local folder (downloaded on first use)."""
    if Path(name).expanduser().is_dir():
        return Path(name).expanduser()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(table.get(name, name)))


FORMAT_VERSION = 1


def check_format(root: Path) -> None:
    """Refuse model folders written for a newer runner (root config.json "format_version")."""
    config = root / "config.json"
    if config.exists():
        meta = json.loads(config.read_text())
        if meta.get("format") == "flux3-mlx" and meta.get("format_version", 1) > FORMAT_VERSION:
            raise SystemExit(f"{root} needs a newer flux3-mlx: pip install -U flux3-mlx")


class Flux3Klein:
    def __init__(self, pipeline: Pipeline, upsampler_path: Path | None = None, low_memory: bool = False):
        self.pipeline, self.upsampler_path, self.low_memory = pipeline, upsampler_path, low_memory
        self._upsampler = None

    @classmethod
    def from_pretrained(cls, model: str | None = None, upsampler: str | None = None,
                        low_memory: bool | None = None) -> "Flux3Klein":
        """model: "8bit", "6bit", a Hub repo id or a local folder (default: by installed memory).
        upsampler: "4bit", "8bit", a repo id or a folder to expand prompts with the FLUX 3 prompt upsampler.
        low_memory: never hold the text encoder, DiT and upsampler together (default: on below 24 GB)."""
        low_memory = system_memory_gb() < 24 if low_memory is None else low_memory
        root = resolve(model or default_model(), MODELS)
        check_format(root)
        pup = resolve(upsampler, UPSAMPLERS) if upsampler else None
        return cls(Pipeline(root, low_memory=low_memory), pup, low_memory)

    def upsample(self, prompt: str, images: list[Image.Image], boxes: list[dict] | None = None) -> str:
        from .pup import Upsampler

        if self._upsampler is None:
            self._upsampler = Upsampler(str(self.upsampler_path))
        try:
            return self._upsampler(prompt, images, boxes)
        finally:
            if self.low_memory or system_memory_gb() < 32:  # the 9B upsampler is the largest component
                self._upsampler = None
                gc.collect()
                mx.clear_cache()

    def __call__(self, prompt: str, images: list[Image.Image] | None = None, boxes: list[dict] | None = None,
                 width: int | None = None, height: int | None = None, steps: int | None = None,
                 guidance: float | None = None, seed: int | None = None, upsample: bool | None = None,
                 callback=None) -> tuple[Image.Image, dict]:
        """Prompt (+ up to 10 reference images, + boxes) -> (image, record). Mirrors flux3/generate.py:
        a single reference sets the output size and is padded to the 16-px grid, then cropped back."""
        images = list(images or [])
        if len(images) > MAX_REFERENCES:
            raise ValueError(f"at most {MAX_REFERENCES} references")
        upsample = self.upsampler_path is not None if upsample is None else upsample
        if upsample and self.upsampler_path is None:
            raise ValueError("no upsampler loaded: pass upsampler= to from_pretrained")
        finished = not boxes and bbox.is_finished(prompt)
        caption = None
        if upsample and not finished:
            try:
                caption = self.upsample(prompt, images, boxes)
            except ValueError as error:  # e.g. a reference beyond the upsampler's aspect limit
                print(f"upsampling skipped: {error}", flush=True)
                upsample = False
        if caption is None:
            caption = f"{prompt.rstrip()} {json.dumps(bbox.to_canvas(boxes), ensure_ascii=False)}" if boxes else prompt

        if len(images) > 1:
            references, crop = [prepare_multi_reference(i) for i in images], None
        else:
            prepared = [prepare_reference(i) for i in images]
            references = [i for i, _ in prepared]
            crop = prepared[0][1] if prepared and not (width or height) else None
        default_w, default_h = references[0].size if references else (1024, 1024)
        width, height = width or default_w, height or default_h
        seed = seed if seed is not None else int(hashlib.sha256(caption.encode()).hexdigest(), 16) % 2**32

        image = self.pipeline.generate(caption, references, width, height, steps, guidance, seed, callback=callback)
        if crop is not None:
            image = image.crop(crop)
        record = {"prompt": caption, "input_prompt": prompt, "boxes": boxes, "width": image.width,
                  "height": image.height, "steps": steps or (16 if self.pipeline.dit.flow_map else 50), "seed": seed, "upsampled": bool(upsample),
                  "references": len(images)}
        return image, record
