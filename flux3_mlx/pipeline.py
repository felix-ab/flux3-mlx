"""Prompt + reference images -> image, in MLX. Mirrors src/flux3/pipeline.py and sampling.py of the BFL reference.

Weights directory layout (both the BFL repo and our MLX repos):
  <root>/model.safetensors (BFL bf16)  or  <root>/transformer/{config.json, model.safetensors} (MLX, maybe quantized)
  <root>/ae.safetensors                or  <root>/vae/ae.safetensors
  text encoder: <root>/text_encoder/ (Qwen3-VL-4B-Instruct files, maybe MLX-quantized)
"""

import gc
import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image, ImageOps

from .autoencoder import DOWNSAMPLING_RATIO, load_autoencoder, latent_ids
from .dit import DiT, IN_CHANNELS, pack
from .text_encoder import TextEncoder
from .torch_randn import randn

FLOW_MAP_KEY = "ref_time_in.in_layer.weight"
IMAGE_ROPE_SCALE = 0.5
MATCH_MAX_PIXELS = 4 * 1024 * 1024
MIN_SIDE, MAX_SIDE = 64, 4096
MAX_REFERENCES = 10
MULTI_REFERENCE_MAX_PIXELS = 1024 * 1024
ALPHA = 6.93
DEFAULTS = {True: {"steps": 16, "guidance": None}, False: {"steps": 50, "guidance": 3.5}}  # keyed by flow_map

# Layers left in bf16 when quantizing: small, and they set every token's scale/shift/gate or embed the inputs.
KEEP_BF16 = ("modulation", "time_in", "emb_in", "txt_in", "final_layer")


def dit_quant_predicate(path: str, module: nn.Module) -> bool:
    return isinstance(module, nn.Linear) and not any(k in path for k in KEEP_BF16)


def schedule(steps: int, alpha: float = ALPHA) -> np.ndarray:
    t = np.linspace(1.0, 0.0, steps + 1, dtype=np.float32)
    return (alpha * t / (1.0 + (alpha - 1.0) * t)).astype(np.float32)


def _find(root: Path, *candidates: str) -> Path:
    for c in candidates:
        if (root / c).exists():
            return root / c
    raise FileNotFoundError(f"none of {candidates} under {root}")


def load_dit(root: str | Path) -> DiT:
    root = Path(root)
    file = _find(root, "transformer/model.safetensors", "model.safetensors")
    weights = mx.load(str(file))
    config = json.loads((file.parent / "config.json").read_text()) if (file.parent / "config.json").exists() else {}
    dit = DiT(flow_map=FLOW_MAP_KEY in weights)
    quant = config.get("quantization")
    if quant:
        nn.quantize(dit, group_size=quant["group_size"], bits=quant["bits"], class_predicate=dit_quant_predicate)
    else:
        weights = {k: v.astype(mx.bfloat16) for k, v in weights.items()}
    dit.load_weights(list(weights.items()), strict=True)
    mx.eval(dit.parameters())
    return dit


def load_text_encoder(root: str | Path) -> TextEncoder:
    return TextEncoder(str(_find(Path(root), "text_encoder")))


def load_vae(root: str | Path):
    return load_autoencoder(str(_find(Path(root), "vae/ae.safetensors", "ae.safetensors")))


class Pipeline:
    """DiT, text encoder and autoencoder from one weights root. ``low_memory`` drops the text encoder after
    encoding and loads the DiT only when sampling, so the peak is max(TE, DiT) instead of their sum."""

    def __init__(self, root: str | Path, text_encoder: str | Path | None = None, low_memory: bool = False):
        self.root, self.te_root, self.low_memory = Path(root), Path(text_encoder or root), low_memory
        self._te = self._dit = None
        self.vae = load_vae(self.root)
        if not low_memory:
            self._te, self._dit = load_text_encoder(self.te_root), load_dit(self.root)

    @property
    def dit(self) -> DiT:
        if self._dit is None:
            self._dit = load_dit(self.root)
        return self._dit

    def encode_prompts(self, prompts: list[str]) -> list:
        te = self._te or load_text_encoder(self.te_root)
        out = []
        for p in prompts:
            tokens, ids = te.encode(p)
            mx.eval(tokens)
            out.append((tokens, ids))
        if self.low_memory:
            del te
            gc.collect()
            mx.clear_cache()
        return out

    def encode_references(self, references: list[Image.Image]):
        if not references:
            return None
        tokens, ids = [], []
        for index, reference in enumerate(references):
            t, i = self.vae.encode(to_array(reference))
            i = scaled_ids(i)
            i[:, 3] = -(1 + index)  # the l axis tells references apart
            tokens.append(t.astype(mx.bfloat16))
            ids.append(i)
        cond = mx.concatenate(tokens)
        mx.eval(cond)
        return cond, np.concatenate(ids)

    def generate(self, prompt: str, references: list[Image.Image], width: int, height: int, steps: int | None = None,
                 guidance: float | None = None, seed: int = 0, noise: np.ndarray | None = None,
                 callback=None) -> Image.Image:
        if len(references) > MAX_REFERENCES:
            raise ValueError(f"at most {MAX_REFERENCES} references")
        if height % DOWNSAMPLING_RATIO or width % DOWNSAMPLING_RATIO:
            raise ValueError(f"width and height must be multiples of {DOWNSAMPLING_RATIO}")
        cond = self.encode_references(references)
        h, w = height // DOWNSAMPLING_RATIO, width // DOWNSAMPLING_RATIO
        if noise is None:
            noise = randn((h * w, IN_CHANNELS["image"]), seed)  # same noise as the torch reference for this seed
        x = mx.array(noise).astype(mx.bfloat16).astype(mx.float32)  # bf16 values, fp32 state across the steps
        ids = scaled_ids(latent_ids(h, w))

        dit = self.dit
        flow_map = dit.flow_map
        steps = steps or DEFAULTS[flow_map]["steps"]
        if flow_map and guidance not in (None, 1.0):
            raise ValueError("the distilled model samples without CFG; drop the guidance")
        guidance = DEFAULTS[flow_map]["guidance"] if guidance is None else guidance
        sched = schedule(steps)
        texts = self.encode_prompts([prompt] if flow_map else ["", prompt])
        if flow_map:
            packed = pack([[texts[0], cond, (x, ids)]])
        else:
            packed = pack([[texts[0], cond, (x, ids)], [texts[1], cond, (x, ids)]])
        n = h * w
        for step in range(steps):
            t, t_next = float(sched[step]), float(sched[step + 1])
            dt = float(np.float32(t_next) - np.float32(t))
            if flow_map:
                v = dit(packed, t, t_ref=t_next)
            else:
                out = dit(packed, t)
                uncond, condv = out[:n], out[n:]
                v = uncond + guidance * (condv - uncond)
            x = x + dt * v.astype(mx.float32)
            mx.eval(x)
            packed.data["image"] = x if flow_map else mx.concatenate([x, x])
            if callback:
                callback(step + 1, steps)
        image = self.vae.decode(x, h, w)
        mx.eval(image)
        return to_pil(image)


def scaled_ids(ids: np.ndarray) -> np.ndarray:
    ids = ids.astype(np.float32)
    ids[:, 1:3] *= IMAGE_ROPE_SCALE
    return ids


def _ceil16(v: int) -> int:
    return -(-v // DOWNSAMPLING_RATIO) * DOWNSAMPLING_RATIO


def reference_canvas(width: int, height: int):
    scale = min(1.0, math.sqrt(MATCH_MAX_PIXELS / (width * height)), MAX_SIDE / max(width, height))
    while True:
        fit = (max(1, round(width * scale)), max(1, round(height * scale)))
        canvas = tuple(min(MAX_SIDE, max(MIN_SIDE, _ceil16(v))) for v in fit)
        if canvas[0] * canvas[1] <= MATCH_MAX_PIXELS:
            return fit, canvas
        scale *= 0.995


def prepare_reference(image: Image.Image):
    """A single reference: fitted, reflect-padded to its 16-px canvas, centred. Returns (image, crop box or None)."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    fit, (w, h) = reference_canvas(image.width, image.height)
    if fit != image.size:
        image = image.resize(fit, Image.Resampling.LANCZOS)
    if (w, h) == image.size:
        return image, None
    left, top = (w - image.width) // 2, (h - image.height) // 2
    padding = ((top, h - image.height - top), (left, w - image.width - left), (0, 0))
    mode = "reflect" if min(image.size) > DOWNSAMPLING_RATIO else "symmetric"
    return Image.fromarray(np.pad(np.asarray(image), padding, mode=mode)), (left, top, left + image.width, top + image.height)


def prepare_multi_reference(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if width * height > MULTI_REFERENCE_MAX_PIXELS:
        scale = math.sqrt(MULTI_REFERENCE_MAX_PIXELS / (width * height))
        width, height = int(width * scale), int(height * scale)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    w, h = width // DOWNSAMPLING_RATIO * DOWNSAMPLING_RATIO, height // DOWNSAMPLING_RATIO * DOWNSAMPLING_RATIO
    left, top = (width - w) // 2, (height - h) // 2
    return image.crop((left, top, left + w, top + h))


def to_array(image: Image.Image) -> mx.array:
    return mx.array(np.asarray(image).astype(np.float32) / 127.5 - 1.0)


def to_pil(image: mx.array) -> Image.Image:
    # torch: ((x.clamp(-1, 1) + 1) * 127.5).to(uint8) truncates
    array = ((np.clip(np.array(image.astype(mx.float32)), -1, 1) + 1) * 127.5).astype(np.uint8)
    return Image.fromarray(array)
