from mlx_lm.models.llama import Model, ModelArgs

from src.cache import TriCache, CacheConfig
from src.llama.attn import Attention


def patch_model_attention(model: Model):
    args: ModelArgs = model.args

    cache_config = CacheConfig(
        hidden_size=args.hidden_size,
        head_dim=args.head_dim,
        n_heads=args.num_attention_heads,
        n_kv_heads=args.num_key_value_heads,
        local_len=1024,
        max_retrieval_search_len=50,
        top_k_key_representatives=5,
        cluster_min_token_split=8,
        cluster_max_token_split=512,
        max_memory_tokens=8_192
    )

    for i, layer in enumerate(model.layers):
        original_attn = layer.self_attn
        cache = TriCache(cache_config)
        attn = Attention(args, cache)

        # ADDED: Inject the layer index for debugging
        attn.layer_idx = i

        attn.update(original_attn.parameters())
        layer.self_attn = attn

    model.cache_classes = None