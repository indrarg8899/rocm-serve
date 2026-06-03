# Architecture Overview

## System Design

rocm-serve follows a three-layer architecture optimized for AMD MI300X GPUs:

```
┌─────────────────────────────────────────────────┐
│                HTTP / OpenAI API                 │
│         FastAPI + uvicorn (async)               │
├─────────────────────────────────────────────────┤
│              Request Scheduler                  │
│     Continuous Batching + Preemption            │
│     Priority Queue → Batch Assembly            │
├─────────────────────────────────────────────────┤
│              Inference Engine                   │
│  PagedAttention │ Tensor Parallel │ Quantize    │
│  KV-Cache       │ HIP Kernels     │ Dequant     │
├─────────────────────────────────────────────────┤
│           AMD ROCm / CDNA 3 Hardware            │
│  MI300X (192GB HBM3) │ Infinity Fabric         │
└─────────────────────────────────────────────────┘
```

## Request Flow

1. **HTTP Request** arrives at FastAPI endpoint
2. **Tokenization** via HuggingFace tokenizer
3. **Scheduler** adds request to waiting queue
4. **Batch Assembly** — scheduler fills batch up to `max_num_seqs` and `max_num_batched_tokens`
5. **Forward Pass** — PagedAttention kernel with batched KV-cache lookup
6. **Sampling** — Temperature, top-p, top-k, or greedy
7. **Streaming** — SSE chunks emitted per token
8. **Completion** — Final response returned

## Memory Management

rocm-serve uses PagedAttention to manage KV-cache memory:

- **Pre-allocated block pool** — Memory reserved at startup based on `gpu_memory_utilization`
- **Block size** — Configurable via `--block-size` (default: 16 tokens)
- **No internal fragmentation** — KV-cache blocks are exactly `block_size` tokens
- **Cross-request sharing** — Prefix caching reuses blocks for common prompts
- **Graceful preemption** — When memory is exhausted, longest-running request is preempted

## Tensor Parallelism

For multi-GPU setups (e.g., 4× MI300X for Llama-3-70B):

- Model weights are sharded across GPUs along the attention head dimension
- Custom HIP all-reduce kernels minimize inter-GPU communication
- Infinity Fabric provides 896 GB/s inter-GPU bandwidth on MI300X platform
- Communication is overlapped with computation via double-buffering
