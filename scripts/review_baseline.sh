#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f "backend/.venv/bin/activate" ]]; then
  # Use the project-local virtual environment when available.
  # shellcheck disable=SC1091
  source backend/.venv/bin/activate
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

run_step() {
  local title="$1"
  shift
  echo
  echo "==> ${title}"
  "$@"
}

run_step "Install backend dependencies" \
  "$PYTHON_BIN" -m pip install -q -r backend/requirements.txt

run_step "Install security tooling" \
  "$PYTHON_BIN" -m pip install -q pip-audit bandit

run_step "Run focused backend regression suite" \
  "$PYTHON_BIN" -m pytest \
    backend/tests/test_auth.py \
    backend/tests/test_router.py \
    backend/tests/test_graph.py \
    backend/tests/test_memory_isolation.py \
    backend/tests/test_code_tool.py \
    backend/tests/test_weather.py \
    -q

run_step "Run dependency vulnerability audit" \
  "$PYTHON_BIN" -m pip_audit -r backend/requirements.txt --progress-spinner off

if command -v gitleaks >/dev/null 2>&1; then
  run_step "Run secret scan" \
    gitleaks detect --no-banner --source . --redact --exit-code 1
else
  echo
  echo "==> Run secret scan"
  echo "Skipping local secret scan: gitleaks is not installed. CI still enforces this check."
fi

run_step "Run static security scan" \
  "$PYTHON_BIN" -m bandit -q -ll -r backend -x backend/tests,backend/.venv,backend/chroma,backend/models,backend/__pycache__

echo

echo "Review baseline checks completed successfully."
