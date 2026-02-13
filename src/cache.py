# src/cache.py
from dataclasses import dataclass, field
from typing import Tuple, Any, Dict, List, Set

import mlx.core as mx
import numpy as np

from abit_clustering import ABITClustering, TreeNode


@dataclass
class CacheConfig:
    hidden_size: int  # Model hidden size (n_heads * head_dim)
    head_dim: int  # Dh, for KV shapes
    n_heads: int  # Total query heads
    n_kv_heads: int  # KV heads (may be < n_heads for GQA)
    local_len: int  # size of local window
    max_retrieval_search_len: int = 1024  # amount of post-norm keys to return on sim search
    top_k_key_representatives: int = 5  # the top highly influential tokens that can represent a cluster
    max_memory_tokens: int = 8_192  # max size of tokens for retrieval

    cluster_threshold_adjustment: float = 0.01
    cluster_window_size: int = 3
    cluster_min_token_split: int = 8
    cluster_max_token_split: int = 512
    cluster_split_tokens_tolerance: int = 5
    cluster_min_cluster_size: int = 8  # Establishes the minimum token count required for a sub-cluster to be considered valid after identifying potential split indices in the _build_tree method.


# TODO Q: Do we need a Quantized TriCache version too?
class TriCache:
    """
    Contains all caches for the attention layers
    """

    def __init__(self, config: CacheConfig):
        self.config = config
        self.offset = 0
        self.global_len = 0

        # State flag to determine if we are currently processing the system prompt
        self.is_global_prefill = False
        self.global_kvs: List[Tuple[mx.array, mx.array]] = []
        self.local_queue: List[int] = []

        # Absolute token positions used as int ids
        self.cache_tokens: Set[int] = set()
        self.cache_token_kv: Dict[int, Tuple[mx.array, mx.array]] = {}
        self.cache_token_accum_scores: Dict[int, float] = {}

        # cluster_id format = "{start_token_position}-{end_token_position}" or "UUID"?
        self.cache_clusters: Set[str] = set()
        self.cache_cluster_tokens: Dict[str, List[int]] = {}

        # ADDED: Initialize rep keys dict to fix potential AttributeError if referenced
        self.cache_cluster_rep_keys: Dict[str, Any] = {}  # Placeholder if needed logic exists

        # key format = "{cluster_id}/{relation_type}"
        self.cache_cluster_relations: Dict[str, Set[str]] = {}

        self.retrieval_key_index: Dict[
            int, str] = {}  # maps the index position (relative to retrieval_key_store) to the cluster id
        self.retrieval_key_store: list[
            mx.array] = []  # stores the post-norm key values. Use retrieval_key_index to get cluster id

        # enforce top_k_key_representatives <= config.cluster_min_cluster_size
        # creating clustering algo
        self.clustering = ABITClustering(
            threshold_adjustment=self.config.cluster_threshold_adjustment,
            window_size=self.config.cluster_window_size,
            min_split_tokens=self.config.cluster_min_token_split,
            max_split_tokens=self.config.cluster_max_token_split,
            split_tokens_tolerance=self.config.cluster_split_tokens_tolerance,
            min_cluster_size=self.config.cluster_min_cluster_size,
            max_tokens=int(round(self.config.local_len * 1.5)),
            # should be 1.5 x local_len for additional token padding
            rebuild_frequency=1
        )

    # --- Property to expose arrays for mx.eval() ---
    @property
    def state(self):
        """Returns a list of all arrays currently stored in the global cache."""
        arrays = []
        for k, v in self.global_kvs:
            arrays.append(k)
            arrays.append(v)
        return arrays

    def update_and_fetch(self, queries, keys, values):
        B, H_kv, T_new, Dh = keys.shape
        # print(f"[DEBUG] TriCache.update_and_fetch: T_new={T_new}, offset={self.offset}, prefill={self.is_global_prefill}")

        if B != 1:
            raise NotImplementedError("B=1 assumed.")

        if self.is_global_prefill:
            # Append new global keys/values
            self.global_kvs.append((keys, values))
            self.global_len += keys.shape[2]
            return [], keys, values, mx.arange(keys.shape[2], dtype=mx.int32)

        # Concat along length dimension (axis 2)
        if len(self.global_kvs) > 0:
            global_keys = mx.concatenate([k for k, v in self.global_kvs], axis=2)
            global_vals = mx.concatenate([v for k, v in self.global_kvs], axis=2)
        else:
            # Handle case where no system prompt was prefilled or empty
            global_keys = mx.zeros((1, H_kv, 0, Dh), dtype=keys.dtype)
            global_vals = mx.zeros((1, H_kv, 0, Dh), dtype=values.dtype)

        if T_new > self.config.local_len:
            raise ValueError(f"Input length {T_new} exceeds local_len {self.config.local_len}; chunk inputs.")

        new_token_ids = []
        for i in range(T_new):
            position = self.offset + i
            token_k = keys[0, :, i, :]  # (H_kv, Dh)
            token_v = values[0, :, i, :]
            # add token to cache
            self.cache_tokens.add(position)
            self.cache_token_kv[position] = (token_k, token_v)

            new_token_ids.append(position)

        self.local_queue.extend(new_token_ids)
        self.local_queue = self.local_queue[-self.config.local_len:]  # Trim to local_len
        self.offset += T_new

        # drop tokens, consolidate memory, preserve important memories
        # print("[DEBUG] TriCache: Calling _evict...")
        self._evict()

        # get recent local key/vals from cache
        # Concat along length dimension (axis 2)
        if self.local_queue:
            local_kvs = [self.cache_token_kv[token] for token in self.local_queue]
            local_keys = mx.expand_dims(mx.stack([k for k, v in local_kvs], axis=1), axis=0)
            local_vals = mx.expand_dims(mx.stack([v for k, v in local_kvs], axis=1), axis=0)
        else:
            local_keys = mx.zeros((1, H_kv, 0, Dh), dtype=keys.dtype)
            local_vals = mx.zeros((1, H_kv, 0, Dh), dtype=values.dtype)

        # Use queries to get relevant clusters from cache
        # print("[DEBUG] TriCache: Calling _retrieve...")
        retrieved_kv_ids, retrieved_keys, retrieved_vals = self._retrieve(queries)

        print(
            f"  [TriCache] Concat shapes -> Global: {global_keys.shape}, Retrieved: {retrieved_keys.shape}, Local: {local_keys.shape}")

        # combine
        K_all = mx.concatenate([global_keys, retrieved_keys, local_keys], axis=2)
        V_all = mx.concatenate([global_vals, retrieved_vals, local_vals], axis=2)

        # Positions for RoPE
        global_pos = mx.arange(self.global_len, dtype=mx.int32)
        retr_pos = mx.array(retrieved_kv_ids, dtype=mx.int32) if retrieved_kv_ids else mx.array([], dtype=mx.int32)
        local_pos = mx.array(self.local_queue, dtype=mx.int32)
        positions = mx.concatenate([global_pos, retr_pos, local_pos])

        return retrieved_kv_ids, K_all, V_all, positions

    def reward(self, attention_scores: mx.array, retrieved_kv_ids: List[int]):
        # print(f"[DEBUG] TriCache: reward called. Scores shape: {attention_scores.shape}")
        t_global = self.global_len
        t_retr = len(retrieved_kv_ids)
        t_local = len(self.local_queue)

        non_global_ids = retrieved_kv_ids + self.local_queue
        t_non_global = t_retr + t_local

        if t_non_global > 0:
            non_global_start = t_global
            non_global_end = t_global + t_non_global

            # Check bounds to prevent crash if shapes mismatch
            if attention_scores.shape[-1] < non_global_end:
                # This can happen if attention_scores is just for the new token vs new token?
                # Usually attention_scores is (B, H, Lq, K_all_len)
                pass

            non_global_scores = attention_scores[..., non_global_start:non_global_end]  # (B, H, Lq, T_non_global)
            rewards = mx.sum(non_global_scores, axis=(0, 1, 2))  # (T_non_global,)
            for i, token_id in enumerate(non_global_ids):
                self.cache_token_accum_scores[token_id] = self.cache_token_accum_scores.get(token_id, 0.0) + float(
                    rewards[i])

    def add_boundaries(self, out: mx.array):
        # Extract embeddings from the post-attention output for clustering
        # print("[DEBUG] TriCache: add_boundaries called")
        B, Lq, D = out.shape
        if B != 1:
            raise ValueError("Batch size must be 1 for prototyping purposes.")

        embeddings = np.array(out[0])  # Shape: (Lq, D)
        token_counts = np.ones(Lq, dtype=int)  # Each token counts as 1

        # Perform incremental clustering on the new embeddings
        # print("[DEBUG] TriCache: clustering.partial_fit...")
        self.clustering.partial_fit(embeddings, token_counts)
        # print(f"[DEBUG] TriCache: clustering done. Labels len: {len(self.clustering.labels_)}")

        # --- Map Local Clustering Labels to Global IDs ---
        # Calculate absolute token positions for the current clustering window
        current_window_len = len(self.clustering.labels_)

        # self.offset is the *next* token position, so the window ends at offset-1
        window_start_id = self.offset - current_window_len

        # Group current tokens by their temporary ABIT label
        #    We need to group them first to calculate their ranges.
        temp_grouped_clusters: Dict[int, List[int]] = {}

        # TODO Q: should we only include leaf clusters? or higher order clusters?

        for i, local_label in enumerate(self.clustering.labels_):
            abs_token_id = window_start_id + i
            if local_label not in temp_grouped_clusters:
                temp_grouped_clusters[local_label] = []
            temp_grouped_clusters[local_label].append(abs_token_id)

        # Convert temporary groups into Persistent Global Cluster IDs
        #    Format: "{start_token}-{end_token}"
        active_clusters: Dict[str, List[int]] = {}

        for local_label, tokens in temp_grouped_clusters.items():
            if not tokens:
                continue
            # Create a unique signature based on absolute position
            start_pos = tokens[0]
            end_pos = tokens[-1]
            cluster_uid = f"{start_pos}-{end_pos}"
            if cluster_uid in self.cache_clusters:
                # already exists
                continue

            active_clusters[cluster_uid] = tokens

        # Indexing
        #   Define boundary for the "local queue".
        #   Tokens/Clusters strictly older than this are candidates for long-term storage.
        local_window_start = self.offset - self.config.local_len

        clusters_to_index = []

        # TODO: Add sibling relations and parent references

        for cluster_uid, tokens in active_clusters.items():
            # Update the global registry with the current definition of this cluster
            self.cache_clusters.add(cluster_uid)
            self.cache_cluster_tokens[cluster_uid] = tokens

            # Check if this cluster has fully exited the "local active window"
            # We look at the *last* token in the cluster.
            last_token_in_cluster = tokens[-1]

            if last_token_in_cluster < local_window_start:
                # If we haven't calculated a representative key for this ID yet, index it.
                if cluster_uid not in self.cache_cluster_rep_keys:
                    clusters_to_index.append(cluster_uid)

        # If we have clusters that have fallen out of the local window, index them now
        if clusters_to_index:
            # print(f"[DEBUG] Indexing clusters: {clusters_to_index}")
            self._index_clusters(clusters_to_index)

    def _index_clusters(self, clusters):
        """
        Calculates and stores representative keys (mean key + top-k influential keys)
        for the specified clusters to enable retrieval.
        """
        # TODO Q: What is the best way to represent a cluster?

        for cluster_id in clusters:
            # Retrieve token IDs associated with this cluster
            token_ids = self.cache_cluster_tokens.get(cluster_id, [])
            if not token_ids:
                continue

            # Filter out tokens that might have been evicted (defensive check)
            valid_tokens = [t for t in token_ids if t in self.cache_token_kv]
            if not valid_tokens:
                continue

            # Mark as indexed
            self.cache_cluster_rep_keys[cluster_id] = True

            # --- Step 1: Mean Key Calculation ---

            # 1.1 Collect all keys: [N_tokens] list of (n_kv_heads, head_dim)
            # We extract just the Key (index 0) from the KV tuple
            raw_keys = [self.cache_token_kv[t][0] for t in valid_tokens]

            # Stack to shape (N_tokens, n_kv_heads, head_dim)
            keys_stack = mx.stack(raw_keys, axis=0)

            # 1.2 Normalize all keys (L2 norm along head_dim)
            # Adding epsilon to avoid division by zero
            norm = mx.norm(keys_stack, axis=-1, keepdims=True)
            keys_normalized = keys_stack / (norm + 1e-6)

            # 1.3 Calculate the mean key across tokens -> (n_kv_heads, head_dim)
            mean_key = mx.mean(keys_normalized, axis=0)

            # 1.4 Normalize the mean key
            mean_key_norm = mx.norm(mean_key, axis=-1, keepdims=True)
            mean_key = mean_key / (mean_key_norm + 1e-6)

            # 1.5 Store Mean Key in the retrieval store
            idx = len(self.retrieval_key_store)
            self.retrieval_key_store.append(mean_key)
            self.retrieval_key_index[idx] = cluster_id

            # --- Step 2: Top-K Representative Keys ---

            # 2.1 / 2.2 Identify top-k tokens based on accumulated attention scores
            scores = [self.cache_token_accum_scores.get(t, 0.0) for t in valid_tokens]

            # Zip scores with the *normalized* keys from step 1.2.
            # We want to store normalized keys for cosine similarity retrieval.
            # Using Python sort is efficient here for cluster-sized lists.
            scored_keys = sorted(
                zip(scores, keys_normalized),
                key=lambda x: x[0],
                reverse=True
            )

            # Select top-k (or fewer if cluster is small)
            k = self.config.top_k_key_representatives
            top_k_entries = scored_keys[:k]

            # 2.3 Store Top-K Keys in the retrieval store
            for _, key_tensor in top_k_entries:
                idx = len(self.retrieval_key_store)
                self.retrieval_key_store.append(key_tensor)
                self.retrieval_key_index[idx] = cluster_id

    def _retrieve(self, queries: mx.array):
        """
        queries size -> [1, queries, n_heads, Lq, head_dim]
        """
        if len(self.retrieval_key_store) == 0:
            empty = mx.zeros([1, self.config.n_kv_heads, 0, self.config.head_dim], dtype=queries.dtype)
            return [], empty, empty

        # Normalize queries
        q_norm = mx.norm(queries, axis=-1, keepdims=True)
        queries_norm = queries / q_norm

        # Compute representative query per kv head
        repeat_factor = self.config.n_heads // self.config.n_kv_heads
        queries_resh = queries_norm.reshape(queries.shape[0], self.config.n_kv_heads, repeat_factor,
                                            queries.shape[2], self.config.head_dim)
        rep_query = mx.mean(queries_resh, axis=[2, 3])  # [1, n_kv_heads, head_dim]
        rep_norm = mx.norm(rep_query, axis=-1, keepdims=True)
        rep_query = rep_query / rep_norm
        rep_query = rep_query[0]  # [n_kv_heads, head_dim]

        # Stack retrieval_key_store
        store_keys = mx.stack(self.retrieval_key_store, axis=0)  # [num_keys, n_kv_heads, head_dim]

        # Compute similarities
        dots_per_head = mx.sum(store_keys * rep_query[None, :, :], axis=-1)  # [num_keys, n_kv_heads]
        sim_scores = mx.mean(dots_per_head, axis=-1)  # [num_keys]

        # Get top indices sorted descending
        sorted_indices = mx.argsort(-sim_scores)[:self.config.max_retrieval_search_len]

        # Collect unique clusters without exceeding max_memory_tokens
        cluster_set: Set[str] = set()
        token_count = 0

        # Convert to python list for iteration
        sorted_indices_list = sorted_indices.tolist()

        for idx in sorted_indices_list:
            cl_id = self.retrieval_key_index.get(int(idx))
            if not cl_id or cl_id in cluster_set:
                continue
            cl_tokens = self.cache_cluster_tokens[cl_id]
            new_count = token_count + len(cl_tokens)
            if new_count > self.config.max_memory_tokens:
                break
            cluster_set.add(cl_id)
            # TODO: query cluster's neighbors OR other correlated clusters (based on attention)
            token_count = new_count

        # Build final list of token IDs (surviving tokens only)
        retrieved_kv_ids: List[int] = []
        for cl_id in cluster_set:
            cl_tokens = self.cache_cluster_tokens[cl_id]
            surviving = [t for t in cl_tokens if t in self.cache_token_kv]
            retrieved_kv_ids.extend(surviving)

        retrieved_kv_ids = sorted(set(retrieved_kv_ids))  # dedup (defensive) + sort by position

        if not retrieved_kv_ids:
            empty = mx.zeros([1, self.config.n_kv_heads, 0, self.config.head_dim], dtype=queries.dtype)
            return [], empty, empty

        # Fetch KV for surviving tokens
        ks = [self.cache_token_kv[tid][0] for tid in retrieved_kv_ids]  # each [n_kv_heads, head_dim]
        vs = [self.cache_token_kv[tid][1] for tid in retrieved_kv_ids]

        # Shape: [1, n_kv_heads, T_retr, head_dim] to match global_keys / local_keys
        retrieved_keys = mx.expand_dims(mx.stack(ks, axis=1), axis=0)
        retrieved_vals = mx.expand_dims(mx.stack(vs, axis=1), axis=0)

        return retrieved_kv_ids, retrieved_keys, retrieved_vals

    def _evict(self):
        """
        1) Rank tokens by (Accumulated Attention / Log(Age))
        2) Remove the lowest scoring tokens to free space.
        3) Prune clusters that become empty or too sparse.
        4) Rebuild the retrieval index to physically free memory for deleted clusters
        """

        # TODO Q: Should we also include least recently used score to prevent forgetting of recent events and punish
        #   unused high attention tokens

        # TODO: add buffer to prevent eviction on every generation step

        # 1. Identify Candidates (Protect Active Window)
        # We never evict tokens currently in the local active window (local_queue).
        protected_tokens = set(self.local_queue)
        candidates = [t for t in self.cache_tokens if t not in protected_tokens]

        if not candidates:
            return

        # 2. Calculate Scores with Hyperbolic Time Decay
        # Score = Accum_Attn / log(Age + e)
        # This allows "Needle in the Haystack" (old but high attention) to survive,
        # while efficiently pruning old low-attention noise.

        current_time = self.offset
        scored_candidates = []

        # Pre-calculate log base for efficiency
        # age + e (approx 2.718) ensures we never divide by zero or get negative logs
        EXP_CONST = 2.71828

        for t in candidates:
            # Token IDs are absolute positions, so Age = Current - ID
            age = current_time - t

            # Get accumulated attention (default to 0.0 if never attended)
            raw_score = self.cache_token_accum_scores.get(t, 0.0)

            # Apply Decay
            # Older tokens need significantly higher raw scores to survive.
            decay_factor = np.log(age + EXP_CONST)
            final_score = raw_score / decay_factor

            scored_candidates.append((final_score, t))

        # 3. Select Tokens to Remove
        # Sort ascending (lowest score first).
        # We remove enough tokens to get back down to max_memory_tokens exactly.
        num_to_remove = len(self.cache_tokens) - self.config.max_memory_tokens
        if num_to_remove <= 0:
            return

        scored_candidates.sort(key=lambda x: x[0])

        # Extract just the token IDs
        tokens_to_remove = [x[1] for x in scored_candidates[:num_to_remove]]
        tokens_to_remove_set = set(tokens_to_remove)

        # 4. Remove Tokens from Token Stores
        for t in tokens_to_remove:
            # Remove KV data (frees tensor memory)
            if t in self.cache_token_kv:
                del self.cache_token_kv[t]

            # Remove metadata
            if t in self.cache_token_accum_scores:
                del self.cache_token_accum_scores[t]

            self.cache_tokens.remove(t)

        # check all the clusters that belong recently dropped to tokens_to_remove
        # determine which clusters should be pruned

        # a cluster could be considered "weak" if the amount of surviving tokens is a threshold of
        #   the original cluster size

        """
        1) determine which clusters are "weak"
        1) If cluster "is weak" or has enough tokens removed either:
            - combine cluster with other relevant cluster within the same parent group with one of its siblings
            - if no siblings suitable then begin to index the parent cluster (if not already present) and drop the cluster
            - remove any tokens that were a part of the removed/consolidated clusters from cache

        :return: 
        """