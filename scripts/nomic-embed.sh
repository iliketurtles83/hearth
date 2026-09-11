#!/usr/bin/env bash
set -euo pipefail

LLAMA_CPP_DIR="${HOME}/Projects/llama.cpp"
LLAMA_SERVER="${LLAMA_CPP_DIR}/build/bin/llama-server"
MODEL_DIR="${LLAMA_CPP_DIR}/models"
EMBED_MODEL="nomic-embed-text-v1.5.Q4_K_M.gguf"
EMBED_VOCAB="ggml-vocab-nomic-bert-moe.gguf"

if [[ ! -x "${LLAMA_SERVER}" ]]; then
	echo "Error: llama-server not found or not executable at ${LLAMA_SERVER}" >&2
	exit 1
fi

if [[ ! -f "${MODEL_DIR}/${EMBED_MODEL}" ]]; then
	echo "Error: embedding model not found at ${MODEL_DIR}/${EMBED_MODEL}" >&2
	exit 1
fi

if [[ ! -f "${MODEL_DIR}/${EMBED_VOCAB}" ]]; then
	echo "Error: embedding vocab not found at ${MODEL_DIR}/${EMBED_VOCAB}" >&2
	exit 1
fi

echo "Starting nomic-embed-text on port 10001"

exec "${LLAMA_SERVER}" \
	-m "${MODEL_DIR}/${EMBED_MODEL}" \
	--alias nomic-embed-text \
	--n-gpu-layers 0 \
	--ctx-size 512 \
	--embedding \
	--pooling mean \
	--host 0.0.0.0 \
	--port 10001
