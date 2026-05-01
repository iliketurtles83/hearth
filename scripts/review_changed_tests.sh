#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f "backend/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source backend/.venv/bin/activate
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat <<'EOF'
Usage: bash scripts/review_changed_tests.sh [--base <ref>] [--dry-run] [--allow-known-failures]

Options:
  --base <ref>  Compare against merge-base with <ref> (default: origin/main if available; else HEAD~1)
  --dry-run     Print selected tests but do not run pytest
  --allow-known-failures
                Apply local deselection list from docs/review/KNOWN_FAILURES.txt

Examples:
  bash scripts/review_changed_tests.sh
  bash scripts/review_changed_tests.sh --base origin/main
  bash scripts/review_changed_tests.sh --dry-run
  bash scripts/review_changed_tests.sh --allow-known-failures
EOF
}

BASE_REF=""
DRY_RUN=false
ALLOW_KNOWN_FAILURES=false
KNOWN_FAILURES_FILE="docs/review/KNOWN_FAILURES.txt"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base)
      BASE_REF="${2:-}"
      if [[ -z "$BASE_REF" ]]; then
        echo "--base requires a ref" >&2
        exit 2
      fi
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --allow-known-failures)
      ALLOW_KNOWN_FAILURES=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$BASE_REF" ]]; then
  if git rev-parse --verify origin/main >/dev/null 2>&1; then
    BASE_REF="origin/main"
  else
    BASE_REF="HEAD~1"
  fi
fi

if ! git rev-parse --verify "$BASE_REF" >/dev/null 2>&1; then
  echo "Base ref not found: $BASE_REF" >&2
  exit 2
fi

merge_base="$(git merge-base HEAD "$BASE_REF")"
changed_files="$(git diff --name-only "$merge_base"...HEAD; git ls-files --others --exclude-standard)"

if [[ -z "$changed_files" ]]; then
  echo "No changed files detected relative to $BASE_REF."
  echo "Running focused baseline suite."
fi

declare -A selected_map=()
selected_tests=()

add_test() {
  local t="$1"
  if [[ -z "${selected_map[$t]+x}" ]]; then
    selected_map["$t"]=1
    selected_tests+=("$t")
  fi
}

# Default focused suite for ambiguous changes or no backend changes.
default_suite=(
  "backend/tests/test_auth.py"
  "backend/tests/test_router.py"
  "backend/tests/test_graph.py"
  "backend/tests/test_memory_isolation.py"
  "backend/tests/test_code_tool.py"
  "backend/tests/test_weather.py"
)

needs_default=false

while IFS= read -r file; do
  [[ -z "$file" ]] && continue
  case "$file" in
    backend/auth.py|backend/tests/test_auth.py)
      add_test "backend/tests/test_auth.py"
      ;;
    backend/router.py|backend/tests/test_router.py)
      add_test "backend/tests/test_router.py"
      ;;
    backend/graph.py|backend/tests/test_graph.py)
      add_test "backend/tests/test_graph.py"
      add_test "backend/tests/test_code_tool.py"
      ;;
    backend/memory.py|backend/tests/test_memory_isolation.py)
      add_test "backend/tests/test_memory_isolation.py"
      ;;
    backend/main.py|backend/tests/test_chat_sessions.py|backend/tests/test_chat_voice_metadata.py)
      add_test "backend/tests/test_chat_sessions.py"
      add_test "backend/tests/test_chat_voice_metadata.py"
      add_test "backend/tests/test_graph.py"
      ;;
    backend/tools/weather.py|backend/tests/test_weather.py)
      add_test "backend/tests/test_weather.py"
      ;;
    backend/tools/music.py|backend/tests/test_music.py)
      add_test "backend/tests/test_music.py"
      ;;
    backend/tools/code*.py|backend/tests/test_code_tool.py)
      add_test "backend/tests/test_code_tool.py"
      add_test "backend/tests/test_graph.py"
      ;;
    backend/tts/*|backend/tests/test_tts_*.py|backend/tests/test_tts_endpoint.py)
      add_test "backend/tests/test_tts_endpoint.py"
      add_test "backend/tests/test_tts_loader.py"
      add_test "backend/tests/test_tts_piper.py"
      add_test "backend/tests/test_tts_kokoro.py"
      ;;
    backend/requirements.txt|docker-compose.yml|backend/Dockerfile|.github/workflows/*)
      needs_default=true
      ;;
    backend/*)
      needs_default=true
      ;;
    *)
      ;;
  esac
done <<< "$changed_files"

if [[ ${#selected_tests[@]} -eq 0 || "$needs_default" == "true" ]]; then
  for t in "${default_suite[@]}"; do
    add_test "$t"
  done
fi

echo "Base ref: $BASE_REF"
echo "Merge-base: $merge_base"
echo "Selected tests (${#selected_tests[@]}):"
for t in "${selected_tests[@]}"; do
  echo "- $t"
done

if [[ "$DRY_RUN" == "true" ]]; then
  exit 0
fi

pytest_args=("${selected_tests[@]}" "-q")

if [[ "$ALLOW_KNOWN_FAILURES" == "true" ]]; then
  if [[ -f "$KNOWN_FAILURES_FILE" ]]; then
    while IFS= read -r line; do
      # Ignore comments and blank lines.
      [[ -z "${line// }" ]] && continue
      [[ "$line" =~ ^[[:space:]]*# ]] && continue
      pytest_args+=("--deselect" "$line")
    done < "$KNOWN_FAILURES_FILE"
    echo "Applying local known-failures deselection from $KNOWN_FAILURES_FILE"
  else
    echo "Known-failures file not found: $KNOWN_FAILURES_FILE"
  fi
fi

"$PYTHON_BIN" -m pytest "${pytest_args[@]}"
