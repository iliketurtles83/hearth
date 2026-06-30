# Hearth — Agent Context

## What This Is

Hearth is a local-first personal AI assistant: FastAPI backend serving both UI and API, Caddy HTTPS edge, LangGraph orchestration, local Ollama inference with optional Anthropic fallback. Deployed via Docker Compose.

**Runtime**: Python 3.11 (Docker), 3.12/3.13 (local venv). Vanilla JS frontend.
**Deployment**: `docker compose up -d --build` (backend port 8000 is internal-only; Caddy on 443/80)

## High-Value Paths

- `backend/main.py` — entrypoint, middleware, routes, startup, `load_dotenv()` at top (before any local imports)
- `backend/graph.py` — LangGraph state graph + checkpointing + SqliteSaver
- `backend/intents.py` — deterministic intent classifier + model constants (routing is split across `intents.py`, `embedding_router.py`, `routing_config.py`)
- `backend/routing_config.py` — routing config object (env-derived)
- `backend/memory.py` — SQLite + ChromaDB hybrid memory
- `backend/music_fastpath.py` — deterministic pre-graph music routing
- `backend/embedding_router.py` — embedding-based router (exists; `router.py` does NOT)
- `backend/tools/` — weather, music, code indexer tool modules
- `backend/tts/` — pluggable TTS (Piper / Kokoro)
- `backend/routes/` — auth + memory tool routers only (no code-file router at module level)
- `backend/tests/` — 22 test files covering API, graph, memory, tool, TTS, music, weather
- `scripts/review_baseline.sh` — full local validation gate (pip-install → test → pip-audit → gitleaks → Bandit)
- `scripts/review_changed_tests.sh` — git-based targeted test selection
- `docs/architecture.md` — architecture decisions, model setup
- `docs/review/KNOWN_FAILURES.txt` — known-failures deselection list

## Commands (Copy-Paste Ready)

```bash
# Run all backend tests (in Docker):
docker compose exec -T backend sh -c 'cd /app && PYTHONPATH=/app python -m pytest -q'

# Run all backend tests (local, from repo root):
cd backend && python -m pytest -q

# Focused test selection (changed-files or default suite):
bash scripts/review_changed_tests.sh --dry-run
bash scripts/review_changed_tests.sh
bash scripts/review_changed_tests.sh --allow-known-failures

# Full local gate (must pass before PR):
bash scripts/review_baseline.sh

# Local dev server (no Docker):
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Writable-path workaround for local uvicorn (Chroma needs writable dir):
mkdir -p /tmp/hearth-chroma && CHROMA_PATH=/tmp/hearth-chroma MEMORY_DB_PATH=/tmp/hearth-memory.db uvicorn main:app --host 127.0.0.1 --port 8010

# Model assets:
bash scripts/download-models.sh
bash scripts/download-tts-models.sh

# Container validation:
docker compose config
docker compose build backend
```

## Gotchas

- **`.env` is gitignored** — create from `.env.example`. All env vars are read at import time, so `load_dotenv()` runs early in `main.py` before any local imports. Do not move it.
- **ChromaDB needs a writable path** — local `uvicorn` without Docker will fail if `CHROMA_PATH` points to a read-only location. Override it or use the writable-path workaround above.
- **Voice features require HTTPS** — browser secure context for `navigator.mediaDevices`. Use Caddy's HTTPS or `https://localhost`. Plain `http://localhost:8000` breaks mic/audio-worklet.
- **Tests must include `tools/` on PYTHONPATH** — when running tests outside Docker, `tools` is importable from `backend/`. In Docker the bind mount handles this.
- **`review_baseline.sh` uses `set -e`** — Bandit findings stop the script. Known false positives exist (B608 in `memory.py`); check `docs/review/KNOWN_FAILURES.txt`.
- **Known-failures deselection**: `docs/review/KNOWN_FAILURES.txt` — used by `--allow-known-failures` flag. CI always runs the full suite without deselections.
- **Code workspace path**: `CODE_WORKSPACE_ROOT` env var sets the writable workspace for code tool nodes. Inside Docker it's bound at `/code-workspace`.
- **Music import on first boot**: Beets auto-imports at `/music` if library is empty. Run manually: `docker compose exec backend sh -c 'cd /beets && beet import -A /music'`.
- **Model files** (`backend/models/*.onnx`, `backend/models/tts/*`) are gitignored but required at runtime. Download before first use.
- **`backend/router.py` does NOT exist** — routing is split across `intents.py`, `embedding_router.py`, and `routing_config.py`. Don't look for a single `router.py`.
- **`.gitignore` hides `.env`, `*.onnx`, `*.bin`, `*.db`, `*.sqlite`, `backend/chroma/`, `caddy/certs/`** — model/TTS assets, memory DBs, Chroma persistence, and mkcert certs are all gitignored.
- **Memory system has known flaws** (`backend/memory.py`):
  - Facts table lacks `UNIQUE(user_id, key)` constraint — concurrent runs can duplicate facts.
  - ChromaDB stores `key: value` strings (e.g., `"location: Tallinn"`), not original text — poor semantic recall.
  - LLM extraction uses `/api/generate` with a `system` prompt, which Ollama ignores — should use `/api/chat`.
  - Episodic summaries in `summaries` table are not filtered for sensitive data by `_is_sensitive`.
  - Mark summaries as `consolidated` in Phase 1 to prevent race-condition duplication during consolidation.

## Architecture Constraints

- Frontend always uses relative API paths — single-origin contract with FastAPI. Never serve UI from a separate dev server in production.
- `music_fastpath.py` sits in front of the graph for deterministic music commands. Don't route music through the graph.
- Code tool workspace-root restrictions (`resolve_workspace_path` + `realpath` prefix check) must be preserved for file-write safety.
- Session state is in-memory + cookie-scoped — lost on restart.
- `Dockerfile` is minimal (Python 3.11-slim, copy requirements → install → copy source → uvicorn). No layer optimization.
- Local secret scan in `review_baseline.sh` is skipped unless `gitleaks` is installed on the host.

## Code-Change Validation Policy

- Keep edits small and module-local; update relevant tests.
- Minimum validation for backend changes: `review_changed_tests.sh` → `review_baseline.sh`.
- If touching startup/memory paths, run local uvicorn once (with writable overrides if needed).
- If touching Docker/Caddy/deploy files, run `docker compose config` and `docker compose build backend`.

## Security and Privacy

- Local-first default. No data leaves the device unless the user triggers a cloud model call or an external tool.
- Never hardcode local paths, personal information, usernames, or device-specific details in code.
- Redact API keys, tokens, and personal data from all logs.
