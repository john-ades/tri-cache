# src/llama/attn.py
from typing import Optional, Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.llama import ModelArgs

from src.sdpa import scaled_dot_product_attention
from src.cache import TriCache


def apply_rope(x: mx.array, positions: mx.array, freqs: mx.array) -> mx.array:
    dtype = x.dtype

    # Calculate angles strictly in float32
    theta = mx.outer(positions.astype(mx.float32), freqs.astype(mx.float32))
    cos = mx.cos(theta)[None, None, :, :]
    sin = mx.sin(theta)[None, None, :, :]

    # Cast back to original precision here
    cos = mx.concatenate([cos, cos], axis=-1).astype(dtype)
    sin = mx.concatenate([sin, sin], axis=-1).astype(dtype)

    M = x.shape[-1] // 2
    x1 = x[..., :M]
    x2 = x[..., M:]
    rotated = mx.concatenate([-x2, x1], axis=-1)

    out = (x * cos) + (rotated * sin)
    return out


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, cache: TriCache):
        super().__init__()

        dim = args.hidden_size
        self.cache = cache
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.head_dim or args.hidden_size // n_heads

        self.scale = head_dim ** -0.5
        if hasattr(args, "attention_bias"):
            attention_bias = args.attention_bias
        else:
            attention_bias = False

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attention_bias)

        # TODO: Make sure that we can specify the rope positions per token. Not continuous.
        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            args.rope_traditional,
            args.rope_scaling,
            args.max_position_embeddings,
        )

    def __call__(
            self,
            x: mx.array,
            mask: Optional[mx.array] = None,
            cache: Optional[Any] = None,
    ) -> mx.array:

        layer_id = getattr(self, 'layer_idx', 'Unknown')
        print(f"[DEBUG Layer {layer_id}] Attention __call__ started. x.shape: {x.shape}")

        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        print(f"[DEBUG Layer {layer_id}] QKV shapes - Q: {queries.shape}, K: {keys.shape}, V: {values.shape}")

        # Get the starting position BEFORE the cache increments it
        start_pos = self.cache.offset

        retrieved_kv_ids, keys, values, kv_positions = self.cache.update_and_fetch(queries, keys, values)
        mx.eval(keys, values, kv_positions)

        # Use the pre-increment offset for RoPE
        print(f"[DEBUG Layer {layer_id}] Cache fetch successful. K_all: {keys.shape}, V_all: {values.shape}")
        query_positions = mx.arange(start_pos, start_pos + L, dtype=mx.int32)


        queries = apply_rope(queries, query_positions, self.rope._freqs)
        keys = apply_rope(keys, kv_positions, self.rope._freqs)

        if self.n_heads != self.n_kv_heads:
            assert self.n_heads % self.n_kv_heads == 0, "num_heads must be a multiple of num_key_value_heads for GQA."
            repeat_factor = self.n_heads // self.n_kv_heads
            keys = mx.repeat(keys, repeats=repeat_factor, axis=1)
            values = mx.repeat(values, repeats=repeat_factor, axis=1)

        # Ignore the MLX default mask. Build a custom boolean keep-mask that
        # accounts for dynamic retrieved tokens appended to K_all.
        qL = queries.shape[2]
        kL = keys.shape[2]

        # Explicitly unmask all past/retrieved tokens, apply causal block only to new Q vs new K
        keep_mask = mx.ones((qL, kL), dtype=mx.bool_)
        causal_block = mx.arange(qL)[:, None] >= mx.arange(qL)[None, :]
        keep_mask[:, kL - qL:] = causal_block

        print(f"[DEBUG Layer {layer_id}] Calling scaled_dot_product_attention...")
        output, attn_scores = scaled_dot_product_attention(
            queries, keys, values, cache=self.cache, scale=self.scale, mask=keep_mask
        )

        # TRAP 2: Force evaluation of SDPA output.
        mx.eval(output, attn_scores)
        print(f"[DEBUG Layer {layer_id}] SDPA successful. Output: {output.shape}")

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output = self.o_proj(output)

        self.cache.reward(attn_scores, retrieved_kv_ids)
        self.cache.add_boundaries(output)

        if cache is not None and getattr(cache, 'keys', mx.array([])) is None:
            cache.keys = mx.zeros((B, self.n_kv_heads, 1, self.head_dim), dtype=queries.dtype)
            cache.values = mx.zeros((B, self.n_kv_heads, 1, self.head_dim), dtype=queries.dtype)
            cache.offset = 1

        print(f"[DEBUG Layer {layer_id}] Attention __call__ finished.")
        return output