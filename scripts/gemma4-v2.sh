#!/usr/bin/env bash
set -euo pipefail

LLAMA_CPP_DIR="${HOME}/Projects/llama.cpp"
LLAMA_SERVER="${LLAMA_CPP_DIR}/build/bin/llama-server"
MODEL_DIR="${LLAMA_CPP_DIR}/models"
MODEL_FILE="gemma-4-E4B-it-UD-Q4_K_XL.gguf"

if [[ ! -x "${LLAMA_SERVER}" ]]; then
	echo "Error: llama-server not found or not executable at ${LLAMA_SERVER}" >&2
	exit 1
fi

if [[ ! -f "${MODEL_DIR}/${MODEL_FILE}" ]]; then
	echo "Error: model file not found at ${MODEL_DIR}/${MODEL_FILE}" >&2
	exit 1
fi


echo "Starting gemma-4 on port 10000"

exec "${LLAMA_SERVER}" \
	-m "${MODEL_DIR}/${MODEL_FILE}" \
	--alias gemma-4 \
	--main-gpu 0 \
	--n-gpu-layers 26 \
	--ctx-size 131072 \
	--flash-attn on \
	--threads 4 \
	-b 512 \
	-ub 512 \
	--cache-type-k q8_0 \
	--cache-type-v q8_0 \
	--jinja \
	--embedding \
	--host 0.0.0.0 \
	--port 10000