# Benchmarking Guide

## Prerequisites

- AMD Instinct MI300X GPU(s) with ROCm 6.1+
- rocm-serve installed and running
- A model loaded (e.g., Llama-3-70B-Instruct)

## Running Benchmarks

### Quick Benchmark
```bash
# Start server
rocm-serve --model meta-llama/Llama-3-70B-Instruct --tensor-parallel 4

# Run built-in benchmark
python -m benchmarks.benchmark \
    --url http://localhost:8000/v1/chat/completions \
    --model Llama-3-70B-Instruct \
    --num-requests 100 \
    --concurrency 32 \
    --input-tokens 512 \
    --output-tokens 256
```

### Using Shell Scripts
```bash
bash benchmarks/llama3_70b_fp16.sh    # FP16 on 4x MI300X
bash benchmarks/llama3_70b_int4.sh    # AWQ INT4 on 2x MI300X
bash benchmarks/mixtral_8x22b.sh      # Mixtral on 4x MI300X
```

## Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Throughput** | Total tokens generated per second | Higher is better |
| **TTFT** | Time to First Token (latency) | <100ms |
| **TPS** | Tokens per second per sequence | Higher is better |
| **Inter-Token Latency** | Time between consecutive tokens | <20ms |
| **Requests/sec** | Completed requests per second | Higher is better |

## Environment

All benchmarks run with:
- GPU Memory Utilization: 0.92
- Max Sequences: 256
- Block Size: 16
- Sampling: greedy (temperature=0.0)
- Input: Synthetic prompts with configurable token count
