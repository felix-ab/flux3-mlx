"""FLUX 3 Image Klein prompt upsampler (pup) on MLX via mlx-vlm (optional: pip install "flux3-mlx[pup]").

Follows black-forest-labs/flux-3-image-klein src/flux3/pup.py (Apache-2.0): the shipped system prompts, the edit
scaffold, the upsampler's own reference preprocessing, greedy decoding with presence penalty 1.5 for
text-to-image and 0 for edits, and locked bounding boxes kept verbatim."""

import io
import json
import math
from pathlib import Path

from PIL import Image, ImageOps

from . import bbox

MAX_TOKENS = 3584
PRESENCE_PENALTY = {"t2i": 1.5, "i2i": 0.0}
I2I_SCAFFOLD = {
    False: ("Reference images follow in order (the first image, the second image, ...).\n"
            "User request: {instruction}\n"
            "Output the caption."),
    True: ("Reference images follow in attachment order: ref_image_0, ref_image_1, ...\n"
           "User editing instruction: {instruction}\n"
           "Decide the edit, imagine the target image, and output the annotation record."),
}
REFERENCE_MULTIPLE = 32
MAX_ASPECT_RATIO = 200
MAX_REFERENCES = 10
MAX_PIXELS_BY_REFS = {1: 1024 * 1024, 2: 724 * 724, **{n: 512 * 512 for n in range(3, MAX_REFERENCES + 1)}}
JPEG_QUALITY = 92


def reference_size(width: int, height: int, n_refs: int) -> tuple[int, int]:
    if max(width, height) / min(width, height) > MAX_ASPECT_RATIO:
        raise ValueError(f"aspect ratio of {width}x{height} exceeds {MAX_ASPECT_RATIO}:1")
    floor = lambda v: max(REFERENCE_MULTIPLE, int(math.floor(v / REFERENCE_MULTIPLE)) * REFERENCE_MULTIPLE)
    cap = MAX_PIXELS_BY_REFS[n_refs]
    w = max(REFERENCE_MULTIPLE, round(width / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE)
    h = max(REFERENCE_MULTIPLE, round(height / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE)
    if w * h > cap:
        beta = math.sqrt(width * height / cap)
        w, h = floor(width / beta), floor(height / beta)
    if w * h > cap:
        if w == REFERENCE_MULTIPLE:
            h = floor(cap / w)
        else:
            w = floor(cap / h)
    return w, h


def policy_reference(image: Image.Image, n_refs: int) -> Image.Image:
    """The upsampler's reference preprocessing: RGB over white, LANCZOS to the size budget, JPEG q92."""
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    image = image.convert("RGB")
    size = reference_size(image.width, image.height, n_refs)
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


class Upsampler:
    def __init__(self, path: str):
        try:
            from mlx_vlm import load
        except ImportError as error:
            raise SystemExit('the prompt upsampler needs mlx-vlm: pip install "flux3-mlx[pup]"') from error
        self.model, self.processor = load(path)
        self.prompts = json.loads((Path(path) / "system_prompts.json").read_text())

    def __call__(self, prompt: str, images: list[Image.Image], boxes: list[dict] | None = None) -> str:
        from mlx_vlm import generate

        if len(images) > MAX_REFERENCES:
            raise ValueError(f"the upsampler takes at most {MAX_REFERENCES} references")
        task = "i2i" if images else "t2i"
        system_key, prompt, locked = bbox.plan(prompt, boxes, len(images))
        if images:
            references = [policy_reference(image, len(images)) for image in images]
            scaffold = I2I_SCAFFOLD[locked is not None].format(instruction=prompt)
            content = [{"type": "image"} for _ in references] + [{"type": "text", "text": scaffold}]
            messages = [{"role": "system", "content": self.prompts[system_key]}, {"role": "user", "content": content}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                      enable_thinking=False)
        else:
            references = None
            messages = [{"role": "system", "content": self.prompts[system_key]}, {"role": "user", "content": prompt}]
            text = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                                enable_thinking=False)
        penalty = PRESENCE_PENALTY[task]
        out = generate(self.model, self.processor, text, image=references, max_tokens=MAX_TOKENS, temperature=0.0,
                       presence_penalty=penalty or None, presence_context_size=MAX_TOKENS, verbose=False)
        caption = (out.text if hasattr(out, "text") else out).strip()
        if not caption:
            raise RuntimeError("empty generation")
        return bbox.reconcile(caption, locked) if locked else caption
