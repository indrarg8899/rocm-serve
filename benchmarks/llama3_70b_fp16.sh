#!/bin/bash
# Benchmark: Llama-3-70B-Instruct FP16 on 4x MI300X
# Usage: bash llama3_70b_fp16.sh

set -euo pipefail

MODEL="meta-llama/Llama-3-70B-Instruct"
TP=4
PORT=8000
RESULTS_DIR="$(dirname "$0")/results"
mkdir -p "$RESULTS_DIR"

echo "=== Starting rocm-serve ==="
rocm-serve \
    --model "$MODEL" \
    --tensor-parallel "$TP" \
    --port "$PORT" \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 8192 \
    --block-size 16 \
    --log-level warning &

SERVER_PID=$!
sleep 30  # Wait for model loading

echo "=== Running benchmark ==="

# Throughput test: send 100 requests with concurrency 32
python3 -m benchmarks.benchmark \
    --url "http://localhost:$PORT/v1/chat/completions" \
    --model "$MODEL" \
    --num-requests 100 \
    --concurrency 32 \
    --input-tokens 512 \
    --output-tokens 256 \
    --stream \
    2>&1 | tee "$RESULTS_DIR/llama3_70b_fp16.json"

# Latency test: single request
echo "=== Latency test ==="
time curl -s "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Write a detailed essay about AMD CDNA 3 architecture, covering MI300X specs, HBM3 memory subsystem, and Infinity Fabric topology.\"}],
        \"max_tokens\": 1024,
        \"temperature\": 0.0
    }" > "$RESULTS_DIR/llama3_70b_fp16_latency.json"

kill $SERVER_PID 2>/dev/null || true
echo "=== Benchmark complete ==="
