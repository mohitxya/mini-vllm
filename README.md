# mini-vLLM

An educational LLM inference runtime built in Python/PyTorch to understand the internals of modern LLM serving systems.

Inspired by vLLM, but intentionally simplified for learning. Implements the core ideas behind autoregressive decoding, KV-cache inference, request scheduling, batching, streaming, and paged KV-cache memory management.

---

## Why I built this

Most LLM tutorials stop at:

```python
model.generate(...)
```

This project breaks that black box apart — implementing a miniature serving stack:

```
prompt → request → runtime/scheduler → backend → model forward pass → sampling → token → response
```

---

## Features

- Manual autoregressive decoding
- KV-cache based incremental decoding
- Request lifecycle tracking
- Naive sequential, round-robin, and continuous batching runtimes
- Greedy, temperature, and top-k sampling
- FastAPI blocking and streaming inference endpoints
- Simulated paged KV-cache allocator
- Benchmark suite for throughput and latency
- CPU and CUDA (Google Colab) support

---

## Architecture

```
Client / Script / API
        ↓
GenerationRequest
        ↓
Runtime / Scheduler
        ↓
Backend Interface
        ↓
HFBackend (PyTorch + Hugging Face)
```

The runtime is decoupled from the backend so scheduling logic can run on different execution backends (CPU, CUDA, ONNX, TensorRT).

---

## Runtime Modes

### 1. Naive Sequential
Processes each request fully before moving to the next. Lowest overhead, highest raw throughput baseline.
```
A A A A A → done
B B B B B → done
```

### 2. Round-Robin Scheduler
Interleaves requests token-by-token. Improves fairness and reduces average wait time.
```
A B C
A B C
...
```

### 3. Simplified Continuous Batching
Runs multiple active requests in one batched forward pass. Demonstrates dynamic batching and active-set management.
```
[A, B] → [A, B] → [B, C] → [C, D] → ...
```

> Note: This recomputes full sequences rather than using batched KV-cache decoding — intentional for clarity.

---

## KV Cache

**Without KV cache** — recomputes the entire prefix each step:
```
step 1: prompt
step 2: prompt + token1
step 3: prompt + token1 + token2
```

**With KV cache** — only processes the latest token:
```
step 1: full prompt → create KV cache
step 2: token1 + KV cache
step 3: token2 + KV cache
```

The project also includes a **simulated paged KV-cache allocator** that tracks fixed-size cache pages, request-to-page ownership, internal fragmentation, and page freeing on request completion. This is an educational simulator, not a replacement for production PagedAttention.

---

## Installation

```bash
git clone https://github.com/mohitxya/mini-vllm.git
cd mini-vllm

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Single-request decoding
python3 -m scripts.run_step_decode

# Round-robin scheduler
python3 -m scripts.run_round_robin

# Continuous batching
python3 -m scripts.run_continuous_batch

# Paged KV-cache simulation
python3 -m scripts.run_paged_cache_sim
```

---

## API Server

```bash
uvicorn mini_vllm.api.server:app
```

```bash
# Health check
curl http://127.0.0.1:8000/health

# Blocking generation
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A GPU is useful because", "max_new_tokens": 30, "strategy": "top_k", "temperature": 0.8, "top_k": 50}'

# Streaming generation
curl -N -X POST http://127.0.0.1:8000/generate_stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A GPU is useful because", "max_new_tokens": 30, "strategy": "top_k", "temperature": 0.8, "top_k": 50}'
```

---

## Benchmarks

```bash
# CPU
python3 -m benchmarks.benchmark_runtimes --device cpu --max-new-tokens 32 --max-batch-size 4

# CUDA
python3 -m benchmarks.benchmark_runtimes --device cuda --max-new-tokens 32 --max-batch-size 4
```

### Example Results (CPU, distilgpt2)

| Runtime | Tokens/sec | Avg Latency | Avg Wait | Notes |
|---|---|---|---|---|
| Naive sequential | 55.00 | 0.774s | 0.550s | Highest raw throughput |
| Round-robin KV cache | 49.65 | 1.127s | 0.060s | Much better fairness |
| Continuous batch (bs4) | 45.77 | 0.998s | 0.152s | Dynamic batching demo |

---

## Google Colab (GPU)

1. Runtime → Change runtime type → **GPU**
2. Run:

```python
!nvidia-smi

!git clone https://github.com/mohitxya/mini-vllm.git
%cd mini-vllm
!pip install -r requirements.txt

!python -m scripts.check_device

!python -m benchmarks.benchmark_runtimes --device cuda --max-new-tokens 32 --max-batch-size 4
```

---

## Limitations

This project is intentionally simplified for education. It does not implement:

- Production PagedAttention kernels
- True batched KV-cache decoding
- Tensor parallelism or distributed serving
- Optimized CUDA/Triton kernels
- Production-grade async request queues or admission control
- Large-model quantized serving