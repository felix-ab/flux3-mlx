"""FLUX 3 Image Klein DiT in MLX. Mirrors black-forest-labs/flux-3-image-klein src/flux3/dit.py (Apache-2.0):
per-mode early blocks, then shared single-stream blocks with per-mode modulation. Parameter names match the
BFL safetensors export, so the checkpoint loads without renaming."""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

MODES = ("image", "image_cond", "text")  # the order tokens take inside a sample
IN_CHANNELS = {"image": 128, "image_cond": 128, "text": 20480}
HIDDEN = 3072
HEADS = 24
HEAD_DIM = HIDDEN // HEADS
DEPTH = 5
DEPTH_SINGLE = 28
AXES_DIM = (32, 32, 32, 32)
MLP_RATIO = 3.0
THETA = 10000
COMPUTE_DTYPE = mx.bfloat16  # matmul dtype; float32 only for numerical checks


def set_compute_dtype(dtype) -> None:
    global COMPUTE_DTYPE, _layernorm_modulation
    COMPUTE_DTYPE = dtype
    _layernorm_modulation = mx.compile(layernorm_modulation)  # the traced graph captured the old dtype


def timestep_embedding(t: mx.array, dim: int = 256, max_period: int = 10000) -> mx.array:
    """t (N,) -> (N, dim) fp32, [cos, sin] like the reference."""
    t = 1000.0 * t.astype(mx.float32)
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
    args = t[:, None] * freqs[None]
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


def rope_cos_sin(ids: np.ndarray) -> tuple[mx.array, mx.array]:
    """ids (L, 4) -> cos, sin of shape (L, HEAD_DIM // 2), fp32, angles computed in float64 like the reference."""
    parts = []
    for axis, dim in enumerate(AXES_DIM):
        scale = np.arange(0, dim, 2, dtype=np.float64) / dim
        parts.append(ids[:, axis].astype(np.float64)[:, None] * (1.0 / THETA**scale)[None])
    angles = np.concatenate(parts, axis=-1)
    return mx.array(np.cos(angles), dtype=mx.float32), mx.array(np.sin(angles), dtype=mx.float32)


def rmsnorm_rope(x: mx.array, scale: mx.array, cos: mx.array, sin: mx.array, eps: float = 1e-6) -> mx.array:
    """x (1, H, L, D): RMSNorm with a learned scale, then the interleaved rotation, in fp32; returns x's dtype."""
    dtype = x.dtype
    x = mx.fast.rms_norm(x.astype(mx.float32), scale.astype(mx.float32), eps)
    even, odd = x[..., 0::2], x[..., 1::2]
    out = mx.stack([cos * even - sin * odd, sin * even + cos * odd], axis=-1)
    return out.reshape(x.shape).astype(dtype)


def layernorm_modulation(x: mx.array, scale: mx.array, shift: mx.array) -> mx.array:
    """fp32 LayerNorm (autocast runs layer_norm in fp32), modulated; the result goes to the next matmul in bf16."""
    out = (1.0 + scale.astype(mx.float32)) * mx.fast.layer_norm(x.astype(mx.float32), None, None, 1e-6)
    return (out + shift.astype(mx.float32)).astype(COMPUTE_DTYPE)


def _swiglu(gate: mx.array, up: mx.array) -> mx.array:
    return nn.silu(gate) * up


def _gated_residual(x: mx.array, gate: mx.array, attn: mx.array, mlp: mx.array) -> mx.array:
    return x + gate * (attn + mlp)


# Fused elementwise kernels (traced once per shape; the arithmetic is unchanged).
_rmsnorm_rope = mx.compile(rmsnorm_rope)
_layernorm_modulation = mx.compile(layernorm_modulation)
_swiglu_c = mx.compile(_swiglu)
_gated_residual_c = mx.compile(_gated_residual)


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = mx.ones((dim,))


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=False)
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.out_layer(nn.silu(self.in_layer(x)))


class ModeModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, 3 * dim, bias=False)

    def __call__(self, vec: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        return tuple(mx.split(self.lin(nn.silu(vec)), 3, axis=-1))  # shift, scale, gate


class LastLayer(nn.Module):
    def __init__(self, hidden: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(hidden, out_channels, bias=False)
        self.adaLN_modulation = [nn.SiLU(), nn.Linear(hidden, 2 * hidden, bias=False)]

    def __call__(self, x: mx.array, vec: mx.array) -> mx.array:
        shift, scale = mx.split(self.adaLN_modulation[1](nn.silu(vec)), 2, axis=-1)
        return self.linear(layernorm_modulation(x, scale, shift))


class Block(nn.Module):
    def __init__(self, hidden: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.heads = heads
        mlp_hidden = int(hidden * mlp_ratio)
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp_in = nn.Linear(hidden, mlp_hidden * 2, bias=False)
        self.attn_out = nn.Linear(hidden, hidden, bias=False)
        self.mlp_out = nn.Linear(mlp_hidden, hidden, bias=False)
        self.norm = QKNorm(hidden // heads)

    def __call__(self, x, cos, sin, bounds, mod):
        """x (L, hidden). bounds: (start, end) per sample; samples never attend to each other."""
        shift, scale, gate = mod
        x_mod = _layernorm_modulation(x, scale, shift)
        split = lambda t: t.reshape(1, t.shape[0], self.heads, -1).transpose(0, 2, 1, 3)  # L (H D) -> 1 H L D
        q = _rmsnorm_rope(split(self.q_proj(x_mod)), self.norm.query_norm.scale, cos, sin)
        k = _rmsnorm_rope(split(self.k_proj(x_mod)), self.norm.key_norm.scale, cos, sin)
        v = split(self.v_proj(x_mod)).astype(q.dtype)
        outs = []
        for start, end in bounds:
            if start == end:
                continue
            o = mx.fast.scaled_dot_product_attention(q[:, :, start:end], k[:, :, start:end], v[:, :, start:end],
                                                     scale=HEAD_DIM**-0.5)
            outs.append(o[0].transpose(1, 0, 2).reshape(end - start, -1))
        attn = outs[0] if len(outs) == 1 else mx.concatenate(outs)
        gate_in, up = mx.split(self.mlp_in(x_mod), 2, axis=-1)
        return _gated_residual_c(x, gate, self.attn_out(attn), self.mlp_out(_swiglu_c(gate_in, up)))


class DiT(nn.Module):
    def __init__(self, flow_map: bool = False):
        super().__init__()
        self.flow_map = flow_map
        self.emb_in = {m: nn.Linear(IN_CHANNELS[m], HIDDEN, bias=False) for m in ("image", "image_cond")}
        self.txt_in = nn.Linear(IN_CHANNELS["text"], HIDDEN, bias=False)
        self.time_in = MLPEmbedder(256, HIDDEN)
        if flow_map:
            self.ref_time_in = MLPEmbedder(256, HIDDEN)
        self.early_stream_modulations = {k: ModeModulation(HIDDEN) for k in ("image", "image_cond", "txt")}
        self.single_stream_modulations = {k: ModeModulation(HIDDEN) for k in ("image", "image_cond", "txt")}
        self.content_mode_blocks = {m: [Block(HIDDEN, HEADS, MLP_RATIO) for _ in range(DEPTH)]
                                    for m in ("image", "image_cond")}
        self.txt_mode_blocks = [Block(HIDDEN, HEADS, MLP_RATIO) for _ in range(DEPTH)]
        self.single_blocks = [Block(HIDDEN, HEADS, MLP_RATIO) for _ in range(DEPTH_SINGLE)]
        self.final_layer = {"image": LastLayer(HIDDEN, IN_CHANNELS["image"])}

    def __call__(self, packed: "Packed", t: float, t_ref: float | None = None) -> mx.array:
        """Velocity for packed.data["image"]. Every sample shares the image timestep ``t``; text and reference
        tokens stay at time 0, as in the reference. t_ref: flow map only, the time the step jumps to."""
        if (t_ref is not None) != self.flow_map:
            raise ValueError("t_ref is required by, and only by, a flow-map DiT")
        modes = [m for m in MODES if packed.data[m].shape[0] > 0]  # empty modes are never attended
        mod_key = lambda m: "txt" if m == "text" else m
        # Timesteps live in bf16 in the reference (packed.timesteps), so round t like it does.
        bf16 = lambda v: mx.array([v], dtype=mx.bfloat16)
        emb = lambda v: timestep_embedding(bf16(v)).astype(COMPUTE_DTYPE)
        ref = self.ref_time_in(emb(t_ref)) if t_ref is not None else 0
        vec = {m: self.time_in(emb(t if m == "image" else 0.0)) + ref for m in modes}  # (1, HIDDEN), one row per mode
        early = {m: self.early_stream_modulations[mod_key(m)](vec[m]) for m in modes}
        single = {m: self.single_stream_modulations[mod_key(m)](vec[m]) for m in modes}
        x = {m: (self.txt_in if m == "text" else self.emb_in[m])(packed.data[m].astype(COMPUTE_DTYPE)) for m in modes}
        for d in range(DEPTH):
            for m in modes:
                block = self.txt_mode_blocks[d] if m == "text" else self.content_mode_blocks[m][d]
                x[m] = block(x[m], *packed.rope[m], packed.mode_indices[m], early[m])

        chunks, mods, bounds, placement, offset = [], ([], [], []), [], [], 0
        for s in range(packed.num_samples):
            start = offset
            for m in modes:
                a, b = packed.mode_indices[m][s]
                if b > a:
                    chunks.append(x[m][a:b])
                    for i in range(3):
                        mods[i].append(mx.broadcast_to(single[m][i], (b - a, HIDDEN)))
                    placement.append((m, a, b, offset))
                    offset += b - a
            bounds.append((start, offset))
        h = mx.concatenate(chunks)
        mods = tuple(mx.concatenate(c) for c in mods)
        cos, sin = packed.single_rope
        for block in self.single_blocks:
            h = block(h, cos, sin, bounds, mods)
        image = mx.concatenate([h[o : o + b - a] for m, a, b, o in placement if m == "image"])
        return self.final_layer["image"](image, vec["image"])


class Packed:
    """Mode-major token pack, as in the reference. data[m] (N_m, C_m); ids[m] (N_m, 4) numpy;
    mode_indices[m][s] = (start, end) of sample s in data[m]. RoPE tables are computed once per pack."""

    def __init__(self, data: dict, ids: dict, mode_indices: dict):
        self.data, self.ids, self.mode_indices = data, ids, mode_indices
        self.rope = {m: rope_cos_sin(ids[m]) for m in MODES if ids[m].shape[0] > 0}
        order = []
        for s in range(self.num_samples):
            for m in MODES:
                a, b = mode_indices[m][s]
                if b > a:
                    order.append(ids[m][a:b])
        self.single_rope = rope_cos_sin(np.concatenate(order))

    @property
    def num_samples(self) -> int:
        return len(self.mode_indices["image"])


def pack(samples: list[list]) -> Packed:
    """samples[s] = [text, image_cond or None, image]; each a (tokens mx.array, ids numpy (N, 4)) pair."""
    data = {m: [] for m in MODES}
    ids = {m: [] for m in MODES}
    mode_indices = {m: [] for m in MODES}
    counts = dict.fromkeys(MODES, 0)
    for text, cond, image in samples:
        for mode, item in (("text", text), ("image_cond", cond), ("image", image)):
            start = counts[mode]
            if item is not None:
                data[mode].append(item[0])
                ids[mode].append(item[1])
                counts[mode] += item[0].shape[0]
            mode_indices[mode].append((start, counts[mode]))
    data = {m: mx.concatenate(data[m]) if data[m] else mx.zeros((0, IN_CHANNELS[m])) for m in MODES}
    ids = {m: np.concatenate(ids[m]).astype(np.float32) if ids[m] else np.zeros((0, 4), np.float32) for m in MODES}
    return Packed(data, ids, mode_indices)
