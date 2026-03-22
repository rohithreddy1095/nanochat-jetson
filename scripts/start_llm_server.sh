#!/bin/bash
# Start the LLM inference server using llama.cpp
# Compatible with ClawMesh's OpenAI API expectations
#
# Usage:
#   bash scripts/start_llm_server.sh
#   NANOCHAT_PORT=8000 bash scripts/start_llm_server.sh

set -e

LLAMA_CPP_DIR="/home/jetson/repo/llama.cpp"
MODEL="${LLAMA_CPP_DIR}/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
PORT="${NANOCHAT_PORT:-8000}"
HOST="${NANOCHAT_HOST:-0.0.0.0}"
CTX="${CTX_SIZE:-2048}"

# Check model exists
if [ ! -f "${MODEL}" ]; then
    echo "ERROR: Model not found at ${MODEL}"
    echo "Download it with:"
    echo "  wget -P ${LLAMA_CPP_DIR}/models https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    exit 1
fi

# Check binary exists
if [ ! -f "${LLAMA_CPP_DIR}/build/bin/llama-server" ]; then
    echo "ERROR: llama-server not found. Build llama.cpp first."
    exit 1
fi

# Drop caches to free memory (Jetson unified memory)
echo "Dropping kernel caches..."
echo "jetson_1095" | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null

echo "Starting llama-server on ${HOST}:${PORT}"
echo "Model: ${MODEL}"
echo "Context: ${CTX}"

exec "${LLAMA_CPP_DIR}/build/bin/llama-server" \
    -m "${MODEL}" \
    -ngl 99 \
    -c "${CTX}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --chat-template chatml
