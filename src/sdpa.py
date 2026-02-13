from typing import Optional, Tuple

import mlx.core as mx
from mlx.utils import tree_map


def quantized_scaled_dot_product_attention(
    queries: mx.array,
    q_keys: tuple[mx.array, mx.array, mx.array],
    q_values: tuple[mx.array, mx.array, mx.array],
    scale: float,
    mask: Optional[mx.array],
    group_size: int = 64,
    bits: int = 8,
) -> Tuple[mx.array, mx.array]:
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    n_repeats = n_q_heads // n_kv_heads

    queries *= scale

    if n_repeats > 1:
        queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
        q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
        q_values = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)

    scores = mx.quantized_matmul(
        queries, *q_keys, transpose=True, group_size=group_size, bits=bits
    )
    if mask is not None:
        if isinstance(mask, str):
            qL, kL = scores.shape[-2:]
            q_indices = mx.arange(kL - qL, kL)
            k_indices = mx.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores += mask
    scores = mx.softmax(scores, axis=-1)
    out = mx.quantized_matmul(
        scores, *q_values, transpose=False, group_size=group_size, bits=bits
    )

    if n_repeats > 1:
        out = mx.reshape(out, (B, n_q_heads, L, D))

    return out, scores


def manual_scaled_dot_product_attention(
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        scale: float,
        mask: Optional[mx.array],
        sinks: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    if sinks is not None:
        raise ValueError("Manual SDPA does not support attention sinks.")

    B, n_q_heads, L, D = queries.shape
    _, n_kv_heads, S, _ = keys.shape
    n_repeats = n_q_heads // n_kv_heads

    queries = queries * scale

    if n_repeats > 1:
        queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
        keys = mx.expand_dims(keys, axis=2)
        values = mx.expand_dims(values, axis=2)

    keys_t = keys.transpose(0, 1, 2, 4, 3) if n_repeats > 1 else keys.transpose(0, 1, 3, 2)

    # 1. UPCAST TO FLOAT32 TO PREVENT OVERFLOW
    scores = mx.matmul(queries.astype(mx.float32), keys_t.astype(mx.float32))

    if mask is not None:
        if isinstance(mask, str):
            qL, kL = scores.shape[-2:]
            q_indices = mx.arange(kL - qL, kL)
            k_indices = mx.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]

        if mask.dtype == mx.bool_:
            # 2. USE -1e9 SAFELY WITHIN FLOAT32
            scores = mx.where(mask, scores, -1e9)
        else:
            scores += mask

    # 3. SOFTMAX IN FLOAT32, CAST BACK AFTER
    scores = mx.softmax(scores, axis=-1).astype(queries.dtype)
    out = mx.matmul(scores, values.astype(mx.float32)).astype(queries.dtype)

    if n_repeats > 1:
        out = mx.reshape(out, (B, n_q_heads, L, D))

    return out, scores

def scaled_dot_product_attention(
    queries,
    keys,
    values,
    cache,
    scale: float,
    mask: Optional[mx.array],
    sinks: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    if hasattr(cache, "bits"):
        if sinks is not None:
            raise ValueError("Quantized SDPA does not support attention sinks.")
        return quantized_scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            group_size=cache.group_size,
            bits=cache.bits,
        )
    else:
        return manual_scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            sinks=sinks,
        )
