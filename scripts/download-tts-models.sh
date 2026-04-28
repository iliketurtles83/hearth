#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$ROOT_DIR/backend/models/tts"

MODEL_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx"
VOICES_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

mkdir -p "$TARGET_DIR"

download() {
  local url="$1"
  local output="$2"

  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --output "$output" "$url"
    return
  fi

  if command -v wget >/dev/null 2>&1; then
    wget -O "$output" "$url"
    return
  fi

  echo "Neither curl nor wget is available; cannot download TTS models." >&2
  exit 1
}

download "$MODEL_URL" "$TARGET_DIR/kokoro-v1.0.int8.onnx"
download "$VOICES_URL" "$TARGET_DIR/voices-v1.0.bin"

echo "Downloaded Kokoro TTS assets to $TARGET_DIR"
