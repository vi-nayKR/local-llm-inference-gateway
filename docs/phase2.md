# 📘 Phase 2: Local SLM Inference Client & SSE Token Streaming Proxy

---

## 🎯 1. Overview & Objective

While **Phase 1** intercepts repetitive queries in $<5\text{ms}$ via semantic caching, **cache misses** still require invoking a Language Model for generation.

In traditional enterprise deployments, sending every cache miss to external commercial APIs (GPT-4o, Claude 3.5) introduces high recurring cloud costs, network egress latency, and compliance data-sharing concerns.

**Phase 2 Goal:** Build an **OpenAI-compatible inference proxy** capable of executing modern **Small Language Models (SLMs)** locally on commodity Commodity & Edge Hardware hardware, supporting:
1. **vLLM PagedAttention Continuous Batching** for high-throughput generation ($>120\text{ tokens/second}$).
2. **Real-Time Server-Sent Events (SSE)** token streaming (`stream: true`) with sub-50ms Time-To-First-Token (TTFT).
3. **Automatic Write-Back** to the Redis 8 vector semantic cache upon completion synthesis.

---

## 📐 2. Architectural Concepts: PagedAttention & Local SLM Tier

### A. The Key-Value (KV) Cache Problem
During autoregressive token generation, the model caches previous Keys and Values in GPU VRAM to avoid recomputing attention across earlier tokens.

In naive inference servers (e.g. vanilla Hugging Face):
- Memory must be allocated in **contiguous physical VRAM blocks** for the maximum possible sequence length (`max_seq_len = 4096`).
- Because user prompts have variable lengths, up to **60%–80% of allocated GPU memory is wasted on fragmentation and over-reservation**.

### B. vLLM PagedAttention Virtual Memory
Inspired by virtual memory paging in operating systems:
$$\text{Logical Blocks} \longrightarrow \text{Block Table Mapping} \longrightarrow \text{Non-Contiguous Physical GPU Pages}$$

- KV-cache is partitioned into fixed-size physical blocks (e.g. 16 tokens per block).
- Memory is allocated dynamically on-demand token-by-token.
- Enables **continuous batching** without head-of-line blocking, doubling serving throughput.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PAGEDATTENTION KV-CACHE MAPPING                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  Logical Sequence: [Token 0 ... 15] ──► Block Table ──► Physical Page #42  │
│  Logical Sequence: [Token 16 ... 31] ──► Block Table ──► Physical Page #107 │
│                                                                             │
│  ✓ 0% Memory Fragmentation  ✓ Dynamic Block Sharing  ✓ Continuous Batching  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### C. Small Language Model (SLM) Tier
Modern 1B–3B parameter models achieve benchmark reasoning comparable to older 13B–70B models while consuming minimal RAM:

| Model | Parameters | 4-bit VRAM Footprint | Throughput (Apple Silicon) | Primary Strength |
| :--- | :--- | :--- | :--- | :--- |
| **`Llama-3.2-1B-Instruct`** | `1.2 Billion` | **~1.2 GB RAM** | `155 tok/s` | Ultra-fast classification & intent parsing |
| **`Qwen-2.5-1.5B-Instruct`** | `1.5 Billion` | **~1.5 GB RAM** | `140 tok/s` | Superior coding, JSON, & math reasoning |
| **`Llama-3.2-3B-Instruct`** | `3.2 Billion` | **~2.2 GB RAM** | `95 tok/s` | Balanced complex reasoning & analysis |
| **`Phi-3.5-mini-instruct`** | `3.8 Billion` | **~2.8 GB RAM** | `85 tok/s` | Compliance parsing & logic queries |

---

## 🛠️ 3. Step-by-Step Code Walkthrough

### Step 1: Local Inference Client (`src/gateway/vllm_client.py`)
- **`generate_completion(messages, model, ...)`:**
  Executes non-streaming completions. Tracks execution time, computed token counts, and generation speed (tokens/sec).
- **`stream_completion(messages, model, ...)`:**
  Asynchronous generator emitting chunks conforming to the OpenAI SSE protocol:
  `data: {"choices": [{"delta": {"content": "..."}}]}` ending with `data: [DONE]`.

### Step 2: OpenAI-Compatible Router (`src/gateway/router.py`)
- **`/v1/models`:** Enumerates available Local SLM models for client compatibility.
- **`/v1/chat/completions`:**
  - Evaluates semantic cache first (unless `stream=True` or `bypass_cache=True`).
  - Dispatches to `inference_client` on cache miss.
  - Automatically writes new completions back to the semantic cache for subsequent queries.
- **`/v1/cache/stats` & `/v1/cache/clear`:** Real-time cache management endpoints.

---

## 🧪 4. How to Run & Verify Phase 2

### Command:
```bash
python3 tests/test_vllm_proxy.py
```

### Expected Output:
```text
..
----------------------------------------------------------------------
Ran 2 tests in 0.280s

OK
```

### What the Tests Verify:
1. `test_local_slm_generation`: Confirms non-streaming completion returns valid content, correct model identifier, and high tokens/second throughput.
2. `test_sse_token_streaming`: Asserts streaming generator outputs well-formed OpenAI chunks and terminates cleanly with `data: [DONE]\n\n`.

---

## 💡 5. Technical Questions & Architectural Explanations

### Q: Why run Small Language Models (SLMs) locally with vLLM instead of calling commercial cloud APIs?
> **Answer:** Running modern 1B–3B parameter SLMs locally via vLLM PagedAttention provides three major advantages:
> 1. **Zero External API Cost:** Eliminates per-token billing, saving $>60\%$ on high-volume production pipelines.
> 2. **Deterministic Latency & Privacy:** Retains all sensitive data on local infrastructure with no third-party data transmission, achieving $<50\text{ms}$ first-token latency.
> 3. **High Concurrency via Continuous Batching:** PagedAttention partitions KV-cache into non-contiguous physical pages, preventing memory fragmentation and sustaining $>120\text{ tokens/second}$ on commodity commodity hardware.
