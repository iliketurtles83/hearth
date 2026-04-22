#!/usr/bin/env bash
# scripts/download-models.sh
# Downloads the openWakeWord ONNX models needed for wake-word detection.
# Run once from the repo root: bash scripts/download-models.sh

set -euo pipefail

MODELS_DIR="$(cd "$(dirname "$0")/.." && pwd)/backend/models"
mkdir -p "$MODELS_DIR"

BASE_OWW="https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"
BASE_COMMUNITY="https://github.com/fwartner/home-assistant-wakewords-collection/raw/main/en/computer"

declare -A MODELS=(
  ["melspectrogram.onnx"]="${BASE_OWW}/melspectrogram.onnx"
  ["embedding_model.onnx"]="${BASE_OWW}/embedding_model.onnx"
  ["computer_v2.onnx"]="${BASE_COMMUNITY}/computer_v2.onnx"
)

for filename in "${!MODELS[@]}"; do
  dest="${MODELS_DIR}/${filename}"
  if [[ -f "$dest" ]]; then
    echo "  [skip] ${filename} already exists"
  else
    echo "  [download] ${filename} ..."
    curl -fsSL --retry 3 -o "$dest" "${MODELS[$filename]}"
    echo "  [ok] ${filename}"
  fi
done

echo ""
echo "All models ready in: ${MODELS_DIR}"
