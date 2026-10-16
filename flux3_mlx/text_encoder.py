"""Qwen3-VL-4B-Instruct text path as the FLUX 3 Image Klein text encoder, in MLX.

Mirrors src/flux3/text_encoder.py: chat-templated prompt, right-padded to a multiple of 80 (max 4096), pad positions
set to 1, hidden states of layers 4, 8, ..., 32 stacked into (L, 8 * 2560 = 20480). Text-only input gives equal
t/h/w MRoPE positions, which is ordinary RoPE, so no vision tower or MRoPE sectioning is needed."""

import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

OUTPUT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
MAX_LENGTH = 4096
PAD_MULTIPLE = 80
PAD_TOKEN = "<|endoftext|>"


class Attention(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        dim, self.n_heads, self.n_kv = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.head_dim = cfg.get("head_dim", dim // self.n_heads)
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=cfg["rms_norm_eps"])
        self.k_norm = nn.RMSNorm(self.head_dim, eps=cfg["rms_norm_eps"])

    def __call__(self, x, cos, sin, mask):
        b, l, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(b, l, self.n_heads, -1)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(b, l, self.n_kv, -1)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, l, self.n_kv, -1).transpose(0, 2, 1, 3)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5, mask=mask)
        return self.o_proj(o.transpose(0, 2, 1, 3).reshape(b, l, -1))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg["hidden_size"], cfg["intermediate_size"])
        self.input_layernorm = nn.RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.post_attention_layernorm = nn.RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])

    def __call__(self, x, cos, sin, mask):
        h = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
        return h + self.mlp(self.post_attention_layernorm(h))


class LanguageModel(nn.Module):
    def __init__(self, cfg: dict, num_layers: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = [Layer(cfg) for _ in range(num_layers)]


def apply_rope(x, cos, sin):
    """Non-interleaved (rotate_half) RoPE, as in HF Qwen3; cos/sin (L, head_dim) fp32."""
    half = x.shape[-1] // 2
    x32 = x.astype(mx.float32)
    rotated = mx.concatenate([-x32[..., half:], x32[..., :half]], axis=-1)
    return (x32 * cos + rotated * sin).astype(x.dtype)


class TextEncoder:
    def __init__(self, path: str, dtype=mx.bfloat16):
        from tokenizers import Tokenizer

        path = Path(path)
        cfg = json.loads((path / "config.json").read_text())
        self.cfg = cfg.get("text_config", cfg)
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.pad_id = self.tokenizer.token_to_id(PAD_TOKEN)
        # Only the layers whose outputs are read are needed: layers 33-36 and the final norm never run.
        self.model = LanguageModel(self.cfg, max(OUTPUT_LAYERS))
        quant = cfg.get("quantization")
        if quant:  # an MLX export of this module (see scripts/export.py)
            nn.quantize(self.model, group_size=quant["group_size"], bits=quant["bits"])
        self.model.load_weights(list(self._weights(path, dtype).items()), strict=True)
        mx.eval(self.model.parameters())
        head_dim = self.cfg.get("head_dim", self.cfg["hidden_size"] // self.cfg["num_attention_heads"])
        self.inv_freq = 1.0 / (self.cfg["rope_theta"] ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))

    def _weights(self, path: Path, dtype) -> dict:
        prefixes = ("model.language_model.", "language_model.model.", "model.")
        out = {}
        files = sorted(path.glob("*.safetensors"))
        for file in files:
            for k, v in mx.load(str(file)).items():
                for p in prefixes:
                    if k.startswith(p):
                        name = k[len(p):]
                        break
                else:
                    continue
                if name.startswith(("visual.", "lm_head", "norm.")):
                    continue
                if name.startswith("layers."):
                    if int(name.split(".")[1]) >= max(OUTPUT_LAYERS):
                        continue
                out[name] = v if v.dtype == mx.uint32 or name.endswith((".scales", ".biases")) else v.astype(dtype)
        return out

    @staticmethod
    def _template(prompt: str) -> str:
        # Qwen3-VL's chat template for one user turn with a generation prompt (checked equal to transformers).
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    def encode(self, prompt: str) -> tuple[mx.array, np.ndarray]:
        """prompt -> tokens (L, 20480) bf16 and ids (L, 4) = (0, 0, 0, position); L padded to a multiple of 80."""
        ids = self.tokenizer.encode(self._template(prompt)).ids[:MAX_LENGTH]  # right truncation, like the reference
        real = len(ids)
        length = min(math.ceil(real / PAD_MULTIPLE) * PAD_MULTIPLE, MAX_LENGTH)
        input_ids = np.array(ids + [self.pad_id] * (length - real))
        valid = np.arange(length) < real
        positions = np.where(valid, np.cumsum(valid) - 1, 1)
        angles = positions[:, None].astype(np.float64) * self.inv_freq[None]
        angles = np.concatenate([angles, angles], axis=-1)
        cos = mx.array(np.cos(angles), dtype=mx.float32)
        sin = mx.array(np.sin(angles), dtype=mx.float32)
        # causal + key padding mask; a padded query still sees every real token before it
        idx = np.arange(length)
        mask = mx.array((idx[None, :] <= idx[:, None]) & valid[None, :])
        h = self.model.embed_tokens(mx.array(input_ids)[None])
        states = []
        for i, layer in enumerate(self.model.layers, start=1):
            h = layer(h, cos, sin, mask)
            if i in OUTPUT_LAYERS:
                states.append(h)
        tokens = mx.stack(states, axis=2)[0].reshape(length, -1).astype(mx.bfloat16)  # (L, 8 * 2560)
        ids = np.zeros((length, 4), dtype=np.float32)
        ids[:, 3] = np.arange(length)
        return tokens, ids
