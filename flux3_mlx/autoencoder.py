"""FLUX 3 / FLUX.2 image autoencoder in MLX (NHWC). Mirrors src/flux3/autoencoder.py of the BFL reference:
8x VAE + 2x2 patchify = 16x; latents are 128-channel tokens with (t, h, w, l) ids. Runs in fp32.

Memory: every 3x3 convolution runs in horizontal strips with a one-row halo, and GroupNorm statistics are reduced
over the whole tensor strip by strip before any strip is normalised. The result equals the whole-tensor computation
(up to reduction order) without tile seams, while the peak stays near a few full-resolution activations instead of
the ~14 GB a 1024px fp32 decode otherwise needs."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

EMBED_DIM = 32
CH = 128
CH_MULT = (1, 2, 4, 4)
NUM_RES_BLOCKS = 2
DOWNSAMPLING_RATIO = 16
LATENT_CHANNELS = 128


STRIP_BYTES = 256 * 1024 * 1024  # budget for one strip's conv input (the im2col temporary is ~9x this)


def normalize(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, channels, eps=1e-6, affine=True, pytorch_compatible=True)


def _strip_rows(x: mx.array) -> int:
    _, h, w, c = x.shape
    return max(8, min(h, STRIP_BYTES // (w * c * 4)))


def group_stats(norm: nn.GroupNorm, x: mx.array) -> tuple[mx.array, mx.array]:
    """Two-pass GroupNorm mean and 1/std over the whole tensor, reduced strip by strip. Returns (B, groups)."""
    b, h, w, c = x.shape
    g, rows = norm.num_groups, _strip_rows(x)
    n = h * w * (c // g)
    total = mx.zeros((b, g), dtype=mx.float32)
    for r in range(0, h, rows):
        total = total + x[:, r:r + rows].astype(mx.float32).reshape(b, -1, g, c // g).sum(axis=(1, 3))
        mx.eval(total)
    mean = total / n
    sq = mx.zeros((b, g), dtype=mx.float32)
    for r in range(0, h, rows):
        d = x[:, r:r + rows].astype(mx.float32).reshape(b, -1, g, c // g) - mean[:, None, :, None]
        sq = sq + (d * d).sum(axis=(1, 3))
        mx.eval(sq)
    return mean, mx.rsqrt(sq / n + norm.eps)


def group_apply(norm: nn.GroupNorm, x: mx.array, stats) -> mx.array:
    mean, rstd = stats
    b, h, w, c = x.shape
    g = norm.num_groups
    y = (x.reshape(b, h, w, g, c // g) - mean[:, None, None, :, None]) * rstd[:, None, None, :, None]
    return y.reshape(b, h, w, c) * norm.weight + norm.bias


def conv3_strips(conv: nn.Conv2d, x: mx.array, pre=None, post=None) -> mx.array:
    """3x3, stride 1, padding 1 conv over row strips with a one-row halo: equal to conv(pre(x)) (+ post)."""
    _, h, _, _ = x.shape
    rows, outs = _strip_rows(x), []
    for r in range(0, h, rows):
        a, e = max(r - 1, 0), min(r + rows + 1, h)
        xs = x[:, a:e]
        y = conv(pre(xs) if pre else xs)[:, r - a: r - a + min(rows, h - r)]
        if post is not None:
            y = post(y, r, r + y.shape[1])
        mx.eval(y)
        outs.append(y)
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def __call__(self, x):
        """Nearest 2x then conv, strip by strip: input rows [a, e) cover output rows [2r, 2r + 2s) plus halo."""
        b, h, w, c = x.shape
        rows, outs = max(4, _strip_rows(x) // 4), []
        for r in range(0, h, rows):
            s = min(rows, h - r)
            a, e = max(r - 1, 0), min(r + s + 1, h)
            xs = x[:, a:e]
            up = mx.broadcast_to(xs[:, :, None, :, None, :], (b, e - a, 2, w, 2, c)).reshape(b, 2 * (e - a), 2 * w, c)
            y = self.conv(up)[:, 2 * (r - a): 2 * (r - a) + 2 * s]
            mx.eval(y)
            outs.append(y)
        return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def __call__(self, x):
        """Pad right/bottom by one, stride-2 conv; output rows [r, r + s) read input rows [2r, 2r + 2s + 1)."""
        b, h, w, c = x.shape
        ho = (h + 1 - 3) // 2 + 1
        rows, outs = max(4, _strip_rows(x) // 2), []
        for r in range(0, ho, rows):
            s = min(rows, ho - r)
            xs = x[:, 2 * r: min(2 * r + 2 * s + 1, h)]
            xs = mx.pad(xs, [(0, 0), (0, 2 * s + 1 - xs.shape[1]), (0, 1), (0, 0)])
            y = self.conv(xs)
            mx.eval(y)
            outs.append(y)
        return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def __call__(self, x):
        s1 = group_stats(self.norm1, x)
        h = conv3_strips(self.conv1, x, pre=lambda t: nn.silu(group_apply(self.norm1, t, s1)))
        s2 = group_stats(self.norm2, h)
        shortcut = self.nin_shortcut if self.in_channels != self.out_channels else (lambda t: t)
        return conv3_strips(self.conv2, h, pre=lambda t: nn.silu(group_apply(self.norm2, t, s2)),
                            post=lambda y, a, e: shortcut(x[:, a:e]) + y)


class AttnBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = normalize(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def __call__(self, x):
        b, height, width, c = x.shape
        h = self.norm(x)
        flat = lambda t: t.reshape(b, 1, height * width, c)
        q, k, v = flat(self.q(h)), flat(self.k(h)), flat(self.v(h))
        # One head of dim 512 materialises (HW)^2 scores; query chunks keep that to chunk x HW (exact: softmax is per row).
        chunk = max(1024, (1 << 30) // (4 * height * width))
        parts = []
        for i in range(0, height * width, chunk):
            parts.append(mx.fast.scaled_dot_product_attention(q[:, :, i:i + chunk], k, v, scale=c**-0.5))
            mx.eval(parts[-1])
        h = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=2)
        return x + self.proj_out(h.reshape(b, height, width, c))


class _Level(nn.Module):
    pass


class _Mid(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block_1 = ResnetBlock(channels, channels)
        self.attn_1 = AttnBlock(channels)
        self.block_2 = ResnetBlock(channels, channels)

    def __call__(self, x):
        return self.block_2(self.attn_1(self.block_1(x)))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant_conv = nn.Conv2d(2 * EMBED_DIM, 2 * EMBED_DIM, 1)
        self.conv_in = nn.Conv2d(3, CH, 3, padding=1)
        self.down = []
        block_in = CH
        for level, mult in enumerate(CH_MULT):
            down = _Level()
            down.block = []
            for _ in range(NUM_RES_BLOCKS):
                down.block.append(ResnetBlock(block_in, CH * mult))
                block_in = CH * mult
            if level != len(CH_MULT) - 1:
                down.downsample = Downsample(block_in)
            self.down.append(down)
        self.mid = _Mid(block_in)
        self.norm_out = normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, 2 * EMBED_DIM, 3, padding=1)

    def __call__(self, x):
        h = conv3_strips(self.conv_in, x)
        for level, down in enumerate(self.down):
            for block in down.block:
                h = block(h)
            if level != len(CH_MULT) - 1:
                h = down.downsample(h)
        h = self.mid(h)
        stats = group_stats(self.norm_out, h)
        return self.quant_conv(conv3_strips(self.conv_out, h, pre=lambda t: nn.silu(group_apply(self.norm_out, t, stats))))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.post_quant_conv = nn.Conv2d(EMBED_DIM, EMBED_DIM, 1)
        block_in = CH * CH_MULT[-1]
        self.conv_in = nn.Conv2d(EMBED_DIM, block_in, 3, padding=1)
        self.mid = _Mid(block_in)
        ups = []
        for level in reversed(range(len(CH_MULT))):
            up = _Level()
            up.block = []
            for _ in range(NUM_RES_BLOCKS + 1):
                up.block.append(ResnetBlock(block_in, CH * CH_MULT[level]))
                block_in = CH * CH_MULT[level]
            if level != 0:
                up.upsample = Upsample(block_in)
            ups.insert(0, up)
        self.up = ups
        self.norm_out = normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, 3, 3, padding=1)

    def __call__(self, z):
        h = self.mid(self.conv_in(self.post_quant_conv(z)))
        for level in reversed(range(len(CH_MULT))):
            for block in self.up[level].block:
                h = block(h)
            if level != 0:
                h = self.up[level].upsample(h)
        stats = group_stats(self.norm_out, h)
        return conv3_strips(self.conv_out, h, pre=lambda t: nn.silu(group_apply(self.norm_out, t, stats)))


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.bn_eps = 1e-4
        self.running_mean = mx.zeros((LATENT_CHANNELS,))
        self.running_var = mx.ones((LATENT_CHANNELS,))

    def encode(self, image: mx.array) -> tuple[mx.array, np.ndarray]:
        """image (H, W, 3) in [-1, 1] -> tokens (h*w, 128) fp32 and ids (h*w, 4) = (0, row, col, 0)."""
        mean = mx.split(self.encoder(image[None].astype(mx.float32)), 2, axis=-1)[0]  # (1, H/8, W/8, 32)
        _, hh, ww, _ = mean.shape
        h, w = hh // 2, ww // 2
        # torch: (1, 32, h, 2, w, 2) -> (1, 32, 2, 2, h, w) -> 128 channels ordered (c, dy, dx)
        z = mean.reshape(1, h, 2, w, 2, EMBED_DIM).transpose(0, 1, 3, 5, 2, 4).reshape(h * w, LATENT_CHANNELS)
        z = (z - self.running_mean) / mx.sqrt(self.running_var + self.bn_eps)
        return z, latent_ids(h, w)

    def decode(self, tokens: mx.array, h: int, w: int) -> mx.array:
        """tokens (h*w, 128) in id order -> image (H, W, 3) in [-1, 1]."""
        z = tokens.astype(mx.float32) * mx.sqrt(self.running_var + self.bn_eps) + self.running_mean
        z = z.reshape(1, h, w, EMBED_DIM, 2, 2).transpose(0, 1, 4, 2, 5, 3).reshape(1, 2 * h, 2 * w, EMBED_DIM)
        return self.decoder(z)[0]


def latent_ids(h: int, w: int) -> np.ndarray:
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    zeros = np.zeros_like(rows)
    return np.stack([zeros, rows, cols, zeros], axis=-1).reshape(h * w, 4)


def load_autoencoder(path: str) -> AutoEncoder:
    """Load the BFL-format ``ae.safetensors`` (torch layout) into the MLX module."""
    weights = mx.load(path)
    out = {}
    for k, v in weights.items():
        if k == "bn.num_batches_tracked":
            continue
        if k.startswith("bn."):
            out[k[3:]] = v.astype(mx.float32)
            continue
        if v.ndim == 4:  # conv OIHW -> OHWI
            v = v.transpose(0, 2, 3, 1)
        out[k] = v.astype(mx.float32)
    ae = AutoEncoder()
    ae.load_weights(list(out.items()), strict=True)
    mx.eval(ae.parameters())
    return ae
