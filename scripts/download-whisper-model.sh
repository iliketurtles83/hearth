#!/usr/bin/env bash
# scripts/download-whisper-model.sh
#
# Downloads a faster-whisper (CTranslate2) model into a local, persistent
# directory so the backend loads it from disk instead of the HuggingFace cache
# at runtime. Mirrors scripts/download-tts-models.sh and
# scripts/download-models.sh (openWakeWord).
#
# The model is placed under backend/models/whisper/<model>/, which is
# bind-mounted into the backend container (./backend:/app). That means it
# survives rebuilds and is NOT fetched over the network on the first
# /transcribe call.
#
# Usage (from the repo root):
#   bash scripts/download-whisper-model.sh             # default: base.en
#   bash scripts/download-whisper-model.sh small.en    # any WHISPER_MODEL size
set -euo pipefail

MODEL="${1:-base.en}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${ROOT_DIR}/backend/models/whisper/${MODEL}"
REPO_ID="Systran/faster-whisper-${MODEL}"
BASE_URL="https://huggingface.co/${REPO_ID}/resolve/main"

# CTranslate2 files faster-whisper needs to load a model from a local directory.
FILES=(model.bin config.json tokenizer.json vocabulary.txt)

if [[ -f "${TARGET_DIR}/model.bin" ]]; then
  echo "  [skip] ${MODEL} already present at ${TARGET_DIR}"
  exit 0
fi

mkdir -p "${TARGET_DIR}"
echo "Downloading faster-whisper '${MODEL}' from ${REPO_ID} -> ${TARGET_DIR}"

for filename in "${FILES[@]}"; do
  dest="${TARGET_DIR}/${filename}"
  url="${BASE_URL}/${filename}"
  if command -v curl >/dev/null 2>&1; then
    echo "  [download] ${filename} ..."
    curl -fSL --retry 3 -o "${dest}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    echo "  [download] ${filename} ..."
    wget -q -O "${dest}" "${url}"
  else
    echo "Neither curl nor wget is available; cannot download ${filename}." >&2
    exit 1
  fi
  echo "  [ok] ${filename}"
done

echo ""
echo "faster-whisper '${MODEL}' ready in: ${TARGET_DIR}"
