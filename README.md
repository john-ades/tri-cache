# TriCache: Dynamic, Attention-Aware Context Management for MLX Large Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: MLX](https://img.shields.io/badge/Framework-MLX-blue.svg)](https://github.com/ml-explore/mlx)

**TriCache** is an experimental, training-free Key-Value (KV) cache architecture built on Apple's MLX framework. Designed to strictly bound memory consumption while preserving long-context reasoning, this project manipulates low-level Transformer attention mechanisms to implement a dynamic, retrieval-augmented memory hierarchy.

> **Note:** This repository is an active Work In Progress (WIP). The architectural framework is fully implemented, but dynamic graph compilation within the MLX runtime is currently under active debugging.

## 1. Abstract

Standard LLM inference suffers from $O(N^2)$ memory scaling due to the linear growth of the KV cache. To address this, TriCache replaces the contiguous memory buffer with a tripartite system: a static **Global** cache, a sliding **Local** window, and a clustered **Retrieval** block. By intercepting raw attention scores during generation, the system assigns value to historic tokens. When memory thresholds are breached, a hyperbolic time-decay algorithm evicts low-value tokens, preserving "needles in the haystack" while discarding contextual noise. 

## 2. Related Work

This architecture synthesizes and expands upon two recent advancements in long-context inference:

* **InfLLM (arXiv:2402.04617):** InfLLM introduces the concept of partitioning context into a localized syntactic window and distant, block-based memory units. TriCache adopts this topological separation, specifically utilizing local queues for immediate linguistic fluency while offloading older representations into a scalable vector store.
* **ReAttention / Attention-Aware Caching (arXiv:2407.15176):** Recognizing that token importance is highly skewed, ReAttention relies on historical attention metrics to determine retention. TriCache implements a similar feedback loop, directly routing output from a custom Scaled Dot-Product Attention (SDPA) kernel back into the memory manager.

## 3. Architectural Methodology

### 3.1 The Tripartite Cache Topology
The monolithic KV cache is structurally divided into three specialized sectors:
1. **Global Cache:** Dedicated entirely to the system prompt. Evaluated once during prefill and permanently fixed to anchor the model's behavioral constraints.
2. **Local Cache:** A strict $N$-token sliding window (e.g., $N=1024$). This guarantees that the most recent conversational context and immediate syntactic structures are always perfectly preserved.
3. **Retrieval Cache:** Once tokens exit the Local window, they are grouped using an incremental `ABITClustering` algorithm. Clusters are represented by their Mean Key and Top-K influential keys. During inference, the current queries perform a cosine similarity search against this store, dynamically fetching only the most relevant historical clusters.

### 3.2 Attention-Driven Eviction Policy
To maintain a strict upper bound on memory (`max_memory_tokens`), TriCache employs an active pruning mechanism. Tokens are scored based on their utility versus their age:

$$ Score = \frac{\sum A}{\log(T_{age} + e)} $$

Where $\sum A$ is the accumulated attention mass the token has received across its lifespan, and $T_{age}$ is its absolute temporal distance from the current generation step. This decay function heavily penalizes old, ignored tokens while protecting historically critical information.

## 4. Implementation Details

The current codebase successfully implements the core theoretical components:
* **Model Patching:** Dynamic overriding of the `mlx-community/Llama-3.2-3B-Instruct` attention layers with custom modules (`src/llama/patch.py`).
* **Custom SDPA:** Extracted and modified the MLX SDPA logic to expose raw attention matrices for token valuation (`src/sdpa.py`).
* **Non-Contiguous RoPE:** Custom rotary position embedding application to handle the disjointed positional IDs of concatenated Global, Retrieved, and Local tensors.
* **Vector Indexing & Clustering:** On-the-fly chunking and L2-normalized indexing of evicted tokens for semantic retrieval.

## 5. Current Limitations and Future Work

TriCache pushes against the boundaries of lazy-evaluated tensor frameworks. The following challenges are actively being addressed:

* **Dynamic Computation Graphs:** Because TriCache retrieves variable amounts of historical tokens per step, the $K$ and $V$ tensor shapes fluctuate dynamically. This currently causes graph compilation instability and shape mismatch errors within the MLX backend.
* **Complex Masking Topologies:** Enforcing standard causal masking for the Local window, while simultaneously applying a broadcasted "keep-mask" for disjointed Retrieval tokens, introduces complex matrix broadcasting challenges.
* **GQA Broadcasting:** Managing Grouped-Query Attention (GQA) ratios when selectively retrieving from specific KV heads requires further dimension-alignment refinement.

## 6. Minimal Usage

```python
from mlx_lm import load
from src.llama.generate import prefill_system_prompt
from src.llama.patch import patch_model_attention

# 1. Load the model
model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct")

# 2. Patch attention layers to use TriCache
patch_model_attention(model)

# 3. Prefill the Global Cache
sys_prompt = tokenizer.apply_chat_template([{"role": "system", "content": "You are a helpful AI."}])
prefill_system_prompt(model, sys_prompt)

# Generation logic (WIP) ...

```

## 7. License

This project is licensed under the **MIT License**.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.
