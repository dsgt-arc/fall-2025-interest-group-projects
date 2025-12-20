#!/bin/bash
#SBATCH -J vllm
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH --nodes=1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH -q embers
#SBATCH -o logs/vllm/%j.out

set -ex

function find_free_port() {
    python3 - <<EOF
import socket
s = socket.socket()
s.bind(('', 0))
print(s.getsockname()[1])
s.close()
EOF
}

function wait_for_vllm() {
    local port=$1
    for i in {1..60}; do
        curl -sf http://localhost:$port/v1/models && return 0
        sleep 5
    done
    echo "vLLM failed to start"
    exit 1
}

MODEL=${MODEL:-"Qwen/Qwen3-0.6B"}
PORT=$(find_free_port)

hostname
nvidia-smi

echo "Starting vLLM on port $PORT"

uvx --with "vllm==0.4.3" python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --host 0.0.0.0 \
    --port $PORT \
    --gpu-memory-utilization 0.9 \
    --max-model-len 2048 &

SERVER_PID=$!
trap "kill $SERVER_PID" EXIT

wait_for_vllm $PORT

echo "Running inference test"
curl -s http://localhost:$PORT/v1/completions \
  -H "Content-Type: application/json" \
  -d "{
        \"model\": \"$MODEL\",
        \"prompt\": \"Write a haiku about Georgia Tech\",
        \"max_tokens\": 64
      }" | jq

echo "done"
