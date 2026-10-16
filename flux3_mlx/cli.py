"""FLUX 3 Image Klein on Apple Silicon.

  flux3-mlx "a red bicycle leaning against a brick wall" --out bike.png
  flux3-mlx "put a red bandana around the dog's neck" --image dog.png --upsample --out edit.png
  flux3-mlx "hot-air balloons over a valley" --boxes boxes.json --upsample --out layout.png
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from PIL import Image

from .api import MODELS, UPSAMPLERS, Flux3Klein, default_model
from .pipeline import MAX_REFERENCES


def main() -> None:
    parser = argparse.ArgumentParser(prog="flux3-mlx", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt")
    parser.add_argument("--model", default=None,
                        help=f"{' | '.join(MODELS)} | Hub repo id | local folder (default here: {default_model()})")
    parser.add_argument("--image", action="append", default=[], help=f"reference image; repeat (max {MAX_REFERENCES})")
    parser.add_argument("--boxes", help="JSON file: [{id, desc, bbox: [y_min, x_min, y_max, x_max] in 0-1}]")
    parser.add_argument("--upsample", nargs="?", const="4bit", default=None, metavar="UPSAMPLER",
                        help=f"expand the prompt with the FLUX 3 prompt upsampler first ({' | '.join(UPSAMPLERS)}, "
                             "repo id or folder; default 4bit). Recommended for short prompts.")
    parser.add_argument("--out", default="out.png")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--steps", type=int, help="default 16 (distilled) / 50 (base)")
    parser.add_argument("--guidance", type=float, help="CFG scale, base model only (default 3.5)")
    parser.add_argument("--seed", type=int)
    memory = parser.add_mutually_exclusive_group()
    memory.add_argument("--low-memory", dest="low_memory", action="store_true", default=None,
                        help="never hold the text encoder and the DiT together (default below 24 GB)")
    memory.add_argument("--no-low-memory", dest="low_memory", action="store_false")
    args = parser.parse_args()

    boxes = json.loads(Path(args.boxes).read_text()) if args.boxes else None
    images = [Image.open(p) for p in args.image]
    t0 = time.time()
    pipe = Flux3Klein.from_pretrained(args.model, upsampler=args.upsample, low_memory=args.low_memory)
    loaded = time.time()
    progress = lambda i, n: print(f"\rstep {i}/{n}  {(time.time() - started) / i:.2f} s/step", end="", flush=True)
    started = time.time()
    image, record = pipe(args.prompt, images, boxes, args.width, args.height, args.steps, args.guidance, args.seed,
                         callback=progress)
    print()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    image.save(args.out)
    record |= {"references": args.image, "model": args.model or default_model(), "load_s": round(loaded - t0, 1),
               "generate_s": round(time.time() - started, 1), "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2)}
    Path(args.out).with_suffix(".json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    if record["upsampled"]:
        print(f"prompt: {record['prompt']}")
    print(f"wrote {args.out}  ({record['generate_s']}s, peak {record['peak_memory_gb']} GB)")


if __name__ == "__main__":
    main()
