#!/usr/bin/env bash
set -euo pipefail

LLAMA_CPP_DIR="${HOME}/Projects/llama.cpp"
LLAMA_SERVER="${LLAMA_CPP_DIR}/build/bin/llama-server"
MODEL_DIR="${LLAMA_CPP_DIR}/models"
EMBED_MODEL="nomic-embed-text-v1.5.Q4_K_M.gguf"
# Bind address/port. Loopback by default; a Docker backend needs an address
# its containers can reach (e.g. NOMIC_HOST=<tailscale-ip> plus a firewall rule).
NOMIC_HOST="${NOMIC_HOST:-127.0.0.1}"
NOMIC_PORT="${NOMIC_PORT:-10001}"

if [[ ! -x "${LLAMA_SERVER}" ]]; then
	echo "Error: llama-server not found or not executable at ${LLAMA_SERVER}" >&2
	exit 1
fi

if [[ ! -f "${MODEL_DIR}/${EMBED_MODEL}" ]]; then
	echo "Error: embedding model not found at ${MODEL_DIR}/${EMBED_MODEL}" >&2
	exit 1
fi

echo "Starting nomic-embed-text on ${NOMIC_HOST}:${NOMIC_PORT}"

exec "${LLAMA_SERVER}" \
	-m "${MODEL_DIR}/${EMBED_MODEL}" \
	--alias nomic-embed-text \
	--n-gpu-layers 0 \
	--ctx-size 2048 \
	--batch-size 2048 \
	--ubatch-size 2048 \
	--threads 4 \
	--embedding \
	--pooling mean \
	--host "${NOMIC_HOST}" \
	--port "${NOMIC_PORT}"
