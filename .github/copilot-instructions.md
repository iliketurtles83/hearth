# Copilot Cloud Agent Onboarding (Hearth)

Trust this file first. Only search the repo when a detail here is missing or a documented command fails.

## What This Repo Is

Hearth is a local-first AI assistant. It combines:
- FastAPI backend (also serves static frontend).
- Streaming chat (SSE), wake-word/transcription, memory, music, and code-question tooling.
- LangGraph orchestration and local Ollama routing (optional Anthropic fallback).
- Docker Compose runtime with Caddy HTTPS edge.

Repo profile:
- Medium-sized monorepo, backend-centric.
- Languages: Python + plain JavaScript.
- Runtime: Docker image uses Python 3.11; local venvs may be 3.12/3.13.

## High-Value Paths

- `README.md` (ops/deploy guidance)
- `docs/PROJECT_CONTEXT.md` (architecture and roadmap)
- `docker-compose.yml` (service topology)
- `backend/main.py` (entrypoint, middleware, routes, startup checks)
- `backend/graph.py` (LangGraph state and nodes)
- `backend/router.py` (intent routing/model choice)
- `backend/memory.py` (SQLite + Chroma memory)
- `backend/music_fastpath.py` (deterministic pre-graph music routing)
- `backend/routes/` (auth/code-file/memory routers)
- `backend/tools/` (weather/music/code indexer tool modules)
- `backend/tts/engines/` (Piper/Kokoro)
- `backend/tests/` (API/graph/memory/tool/TTS/music/weather coverage)
- `scripts/review_changed_tests.sh` and `scripts/review_baseline.sh` (actual local gates)

Top-level tree (quick orientation):
- `README.md`, `docker-compose.yml`, `genres.txt`
- directories: `backend/`, `frontend/`, `scripts/`, `docs/`, `caddy/`, `mpd/`, `.github/`

README highlights (what matters most):
- Single-origin serving (backend serves UI and API).
- Docker-first deployment.
- Voice features need HTTPS context for browsers.
- Model download scripts are required when assets are missing.

## CI/Policy Reality Check

Do not assume workflow-enforced CI from repository files alone. Use local scripts as your validation baseline.

## Verified Environment

Validated executable:
- `/home/jack/assistant/.venv/bin/python`

Observed versions:
- Python 3.12.3
- pip 26.0.1
- pytest 9.0.3
- Docker 29.4.2
- Docker Compose v5.1.3

## Always-Use Command Sequence

Run from repo root unless noted.

1. Bootstrap dependencies:
```bash
/home/jack/assistant/.venv/bin/python -m pip install -q -r backend/requirements.txt
```
Result: works (about 0.6s on warm venv).

2. Fast test selection then execution:
```bash
bash scripts/review_changed_tests.sh --dry-run
bash scripts/review_changed_tests.sh
```
Result: script works; observed run had 12 failures in `backend/tests/test_memory_isolation.py` due `_FakeMemoryStore` API mismatch in selected test mix.

3. Optional local iteration with known deselections:
```bash
bash scripts/review_changed_tests.sh --allow-known-failures
```
Known deselections: `docs/review/KNOWN_FAILURES.txt`.

4. Baseline gate script:
```bash
bash scripts/review_baseline.sh
```
Result: focused regression passed (143 tests), pip-audit passed, gitleaks skipped if missing locally, Bandit can fail run (observed B608 medium findings in `backend/memory.py`).

5. Container validation/build:
```bash
docker compose config
docker compose build backend
```
Result: both pass; backend image build took about 57.7s.

6. Local backend run (from `backend/`):
```bash
/home/jack/assistant/.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8010
```
Observed failure: Chroma readonly DB error.

Reliable workaround:
```bash
mkdir -p /tmp/hearth-chroma
CHROMA_PATH=/tmp/hearth-chroma MEMORY_DB_PATH=/tmp/hearth-memory.db \
  /home/jack/assistant/.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8011
```
Result: startup succeeds.

7. Required model assets when missing:
```bash
bash scripts/download-models.sh
bash scripts/download-tts-models.sh
```

## Required Preconditions and Gotchas

- Always install requirements before tests.
- For non-Docker uvicorn, writable `CHROMA_PATH` and `MEMORY_DB_PATH` may be required.
- Voice/TTS validation requires model files under `backend/models/` and `backend/models/tts/`.
- `review_baseline.sh` uses `set -e`; Bandit findings stop the script and any chained commands.
- Local secret scan is skipped unless `gitleaks` is installed.
- Uvicorn is long-running; treat successful startup logs as pass, then stop it.

## Code-Change Validation Policy

- Keep edits small and module-local; update relevant tests.
- Minimum validation for backend changes: changed-tests script + baseline script.
- If touching startup/memory paths, run local uvicorn once (with writable overrides if needed).
- If touching Docker/Caddy/deploy files, run `docker compose config` and `docker compose build backend`.

## Architecture Constraints to Preserve

- Keep frontend API usage relative-path (single-origin contract).
- Keep deterministic music pre-router (`backend/music_fastpath.py`) in front of graph flow.
- Preserve workspace-root restrictions for code file operations.
- Preserve auth boundary behavior in `backend/main.py`.

## Security and privacy
- Local-first default. No data leaves the device unless the user triggers a
  cloud model call or an external tool.
- Never hardcode local paths, personal information, usernames, or device-specific details in code.
- Redact API keys, tokens, and personal data from all logs.