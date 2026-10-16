"""torch.randn(n, generator=torch.Generator().manual_seed(seed)) on the CPU, reproduced in numpy, so an MLX
image uses the same starting noise as the BFL reference for the same seed (without depending on torch).

torch's CPU normal_ for float32 tensors of >= 16 elements fills blocks of 16 uniforms (24-bit, from MT19937), then
Box-Muller pairs element j with j + 8 inside each block. Matches torch to ~2e-6 (libm rounding), invisible after the
bf16 rounding the pipeline applies. Latents are 128 x N, so only sizes divisible by 16 are supported."""

import numpy as np


def _uniform24(rng: np.random.RandomState, n: int) -> np.ndarray:
    raw = rng.randint(0, 2**32, size=n, dtype=np.uint64).astype(np.uint32)
    return (raw & np.uint32((1 << 24) - 1)).astype(np.float32) * np.float32(1.0 / (1 << 24))


def _box_muller16(u: np.ndarray) -> np.ndarray:
    """u (..., 16) uniforms -> (..., 16) normals, torch's normal_fill_16 in float32."""
    u1 = np.float32(1.0) - u[..., :8]
    u2 = u[..., 8:]
    radius = np.sqrt(np.float32(-2.0) * np.log(u1))
    theta = np.float32(2.0 * np.pi) * u2
    return np.concatenate([radius * np.cos(theta), radius * np.sin(theta)], axis=-1).astype(np.float32)


def randn(shape: tuple[int, ...], seed: int) -> np.ndarray:
    n = int(np.prod(shape))
    if n % 16:
        raise ValueError("only sizes divisible by 16 are reproduced")
    rng = np.random.RandomState(seed)  # MT19937 seeded like std::mt19937(seed)
    return _box_muller16(_uniform24(rng, n).reshape(-1, 16)).reshape(shape)
