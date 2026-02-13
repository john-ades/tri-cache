
from typing import List, Union

import mlx.nn as nn
import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from transformers import PreTrainedTokenizer


def prefill_system_prompt(
        model: nn.Module,
        system_prompt_tokens: mx.array
):
    """
    Encodes the system prompt and runs it through the model.
    The modified Attention layers will capture the KV states into
    TriCache.global_kvs.
    """

    # 1. Prepare inputs
    input_tensor = mx.array(system_prompt_tokens)[None, :]  # Shape (1, L)

    # 2. Set Caches to Prefill Mode
    for layer in model.layers:
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'cache'):
            layer.self_attn.cache.is_global_prefill = True

    # 3. Forward pass (ignore output, we just want the side-effect of caching)
    model(input_tensor)

    # 4. Force evaluation to ensure cache is actually populated in memory
    # (MLX is lazy, so we need to eval something dependent on the computation)
    cache_states = []
    for layer in model.layers:
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'cache'):
            # Reset flag
            layer.self_attn.cache.is_global_prefill = False
            # Set offset to after global prefill
            layer.self_attn.cache.offset = layer.self_attn.cache.global_len
            # Collect state for eval
            cache_states.extend(layer.self_attn.cache.state)

    mx.eval(cache_states)
    print(f"System prompt prefilled. Processed {len(system_prompt_tokens)} tokens.")

def generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    system_prompt: str,
    prompt: Union[str, List[int]],
    verbose: bool = False,

    max_new_tokens: int = 256,
):

    system_prompt_tokens = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}], add_generation_prompt=False
    )

    # prefill cache global kvs with the system prompt
    pass

