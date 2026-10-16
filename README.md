# flux3-mlx

**FLUX 3 Image Klein on Apple Silicon.** Text-to-image, editing with up to 10 reference images and bounding-box
layout, in [MLX](https://github.com/ml-explore/mlx), validated against Black Forest Labs' reference implementation.

![Images generated with flux3-mlx (8-bit) on an M4 Pro](docs/showcase.jpg)

```sh
pip install "flux3-mlx[pup] @ git+https://github.com/felix-ab/flux3-mlx"
flux3-mlx "a ceramicist's studio at golden hour, shelves of glazed bowls" --upsample --out studio.png
```

On first use, the weights download from the Hugging Face Hub, with no account or gating needed. By downloading
them you agree to the [FLUX Kommunity License](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md).

## Models

| Repo | DiT | Download | Peak memory (1024², low-memory) | Use |
|---|---|---|---|---|
| [FLUX-3-Image-Klein-MLX-8bit](https://huggingface.co/VisualInference/FLUX-3-Image-Klein-MLX-8bit) | 8-bit | 9.6 GB | 11.0 GB | default on 24 GB+ Macs; closest to BF16 |
| [FLUX-3-Image-Klein-MLX-6bit](https://huggingface.co/VisualInference/FLUX-3-Image-Klein-MLX-6bit) | 6-bit | 8.4 GB | 9.6 GB | 16 GB Macs |
| [FLUX-3-Image-Klein-pup-MLX-4bit](https://huggingface.co/VisualInference/FLUX-3-Image-Klein-pup-MLX-4bit) | prompt upsampler | 5.6 GB | 7.5 GB | `--upsample` (default) |
| [FLUX-3-Image-Klein-pup-MLX-8bit](https://huggingface.co/VisualInference/FLUX-3-Image-Klein-pup-MLX-8bit) | prompt upsampler | 9.8 GB | 12 GB | `--upsample 8bit` |

Each model repo bundles the transformer, the Qwen3-VL-4B text encoder (8-bit) and the VAE. **There is no 4-bit
model on purpose.** In our tests, 4-bit added 5–23% high-frequency grain and changed design details (see
Validation), so it does not meet the bar for this release.

## Usage

```sh
# text-to-image; long, fully described prompts work best, or let the upsampler write one
flux3-mlx "Full-color photograph, ..." --width 1216 --height 832 --seed 7 --out a.png
flux3-mlx "a red bicycle in the rain" --upsample --out b.png

# edit: the first reference sets the output size
flux3-mlx "put a red bandana around the dog's neck" --image dog.png --upsample --out edit.png

# combine up to 10 references
flux3-mlx "the woman from the first image wearing the jacket from the second" --image a.png --image b.png --upsample

# layout with bounding boxes ([y_min, x_min, y_max, x_max] in 0-1)
flux3-mlx "hot-air balloons over a green valley at sunrise" --boxes boxes.json --upsample
```

```python
from flux3_mlx import Flux3Klein

pipe = Flux3Klein.from_pretrained("8bit", upsampler="4bit")
image, record = pipe("a ceramicist's studio at golden hour", seed=3)
image.save("studio.png")
```

Every image gets a `.json` record next to it: final prompt, seed, size, steps, time and peak memory. Seeds are
compatible with the BFL reference: the same seed gives the same starting noise as `torch.randn` on the CPU, so
reference outputs can be reproduced on a Mac.

## Performance (M4 Pro, 20-core GPU, 48 GB)

| | 1024², 16 steps | Peak |
|---|---|---|
| 8-bit | ~115 s (7.3 s/step) | 14.3 GB (11.0 GB `--low-memory`) |
| 6-bit | ~114 s | 13.0 GB (9.6 GB `--low-memory`) |
| edit, 1 reference (~1 MP) | ~200–270 s | about the same |
| edit, 2 / 3 references | ~290 s / ~540 s | about the same |
| prompt upsampler 4-bit | ~8–10 s per prompt | 7.5 GB |

At 1 MP the model is compute-bound, so quantization buys memory, not speed. Klein's distilled checkpoint needs
16 steps. Do not lower the step count to save time: at 4 steps the output is visibly under-denoised and grainy, in
BFL's own code too.

## Validation

All comparisons are against BFL's reference code (`45e57c9`) and BFL's published example outputs.

- **Text encoder:** same chat template and token ids. Hidden-state cosine is ≥ 0.9999 (mean) at all 8 tapped layers.
- **VAE:** encode and decode match the reference (cos 1.000000).
- **Transformer:** per-step velocity cos 0.9999 (text-to-image and edit).
- **End to end, BFL `man_dog` (1024², seed 7):** BF16 port vs BFL's published B200 image is 35.9 dB, visually
  indistinguishable. 8-bit is 34.2 dB.
- **Edit and layout examples (6-bit):** `3_bandana` 45.5 dB, `balloons` 35.4 dB vs BFL's images.
- **Quantization study** (5 cases, texture = high-frequency energy relative to BF16, 1.00 = identical):

  | DiT | Texture | Verdict |
  |---|---|---|
  | 8-bit | 0.99 | shipped |
  | 6-bit | 1.00 | shipped |
  | 5-bit | 0.95 | blurs |
  | 4-bit (g32 / g64 / mixed) | 1.05–1.18 | grain, design changes |

Memory: the VAE runs its convolutions in row strips with a one-row halo, over global GroupNorm statistics, and its
attention in query chunks. The output is identical to the full-tensor computation, but a 1024² decode needs 3.9 GB
instead of 14 GB, and a 4 MP edit reference can be encoded in 8 GB.

## Licenses

- **This code:** Apache-2.0. It is a port of [black-forest-labs/flux-3-image-klein](https://github.com/black-forest-labs/flux-3-image-klein)
  (Apache-2.0); see `NOTICE`.
- **FLUX 3 Image Klein weights,** including these MLX conversions: [FLUX Kommunity License](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md).
  Non-commercial use; commercial use of outputs is limited to Qualifying Users (under US$5M revenue) and comes
  with content-filtering and AI-disclosure duties.
- **Qwen3-VL-4B text encoder and prompt upsampler:** Apache-2.0.

This is an independent community port. It is not affiliated with or endorsed by Black Forest Labs.
