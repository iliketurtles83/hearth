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

run_step "Run focused backend regression suite" \
  "$PYTHON_BIN" -m pytest \
    backend/tests/test_auth.py \
    backend/tests/test_router.py \
    backend/tests/test_graph.py \
    backend/tests/test_memory_isolation.py \
    backend/tests/test_weather.py \
    -q

if "$PYTHON_BIN" -m pip show pip-audit >/dev/null 2>&1; then
  run_step "Run dependency vulnerability audit" \
    "$PYTHON_BIN" -m pip_audit -r backend/requirements.txt --progress-spinner off
else
  echo
  echo "==> Run dependency vulnerability audit"
  echo "Skipping: pip-audit is not installed in the active environment."
fi

if command -v gitleaks >/dev/null 2>&1; then
  run_step "Run secret scan" \
    gitleaks detect --no-banner --source . --redact --exit-code 1
else
  echo
  echo "==> Run secret scan"
  echo "Skipping: gitleaks is not installed."
fi

if "$PYTHON_BIN" -m pip show bandit >/dev/null 2>&1; then
  run_step "Run static security scan" \
    "$PYTHON_BIN" -m bandit -q -ll -r backend -x backend/tests,backend/.venv,backend/chroma,backend/models,backend/__pycache__
else
  echo
  echo "==> Run static security scan"
  echo "Skipping: bandit is not installed in the active environment."
fi

echo

echo "Local baseline checks completed successfully."
