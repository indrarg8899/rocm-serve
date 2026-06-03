#!/bin/bash
# Benchmark: Llama-3-70B-Instruct INT4 on 2x MI300X
set -euo pipefail

MODEL="meta-llama/Llama-3-70B-Instruct"
TP=2
PORT=8000

rocm-serve \
    --model "$MODEL" \
    --tensor-parallel "$TP" \
    --quantization awq \
    --port "$PORT" \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 8192 \
    --log-level warning &

SERVER_PID=$!
sleep 30

python3 -m benchmarks.benchmark \
    --url "http://localhost:$PORT/v1/chat/completions" \
    --model "$MODEL" \
    --num-requests 100 \
    --concurrency 32 \
    --input-tokens 512 \
    --output-tokens 256 \
    --stream

kill $SERVER_PID 2>/dev/null || true
