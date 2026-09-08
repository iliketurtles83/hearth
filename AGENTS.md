# Hearth — Agent Context

## What This Is

Hearth is a local-first personal AI assistant: FastAPI backend serving both UI and API, Caddy HTTPS edge, LangGraph orchestration, local OpenAI-compatible inference (llama.cpp/gemma-4) with optional Anthropic fallback. Deployed via Docker Compose.

**Runtime**: Python 3.11 (Docker), 3.12/3.13 (local venv). Vanilla JS frontend.
**Deployment**: `docker compose up -d --build` (backend port 8000 is internal-only; Caddy on 443/80)

## High-Value Paths

- `backend/main.py` — entrypoint, middleware, route wiring, startup validation. `import load_env` is its first local import (import-order invariant). Memory and auth are lazy singletons: `get_memory_store()` / `get_auth_service()` (PEP 562 `__getattr__` keeps `main.memory_store` / `main.auth_service` working); a bare `import main` must not write state files.
- `backend/load_env.py` — single entry point that calls `load_dotenv()` once. Must be imported before any env-reading local module (routing_config, intents, embedding_router, memory, graph, auth, routes/*). A test or script that imports one of those modules standalone (without importing `main` or `load_env`) must set the relevant env vars itself.
- `backend/graph.py` — LangGraph state graph (6 nodes: history_loader → intent_classifier → memory_retrieval → tool_router → responder → memory_writer → END) + SqliteSaver checkpointing
- `backend/intents.py` — deterministic intent classifier + shared model constants (ROUTE_CONFIDENCE_THRESHOLD, CHAT_MODEL, CLOUD_MODEL)
- `backend/routing_config.py` — RoutingConfig dataclass loaded from env (singleton ROUTING_CONFIG)
- `backend/embedding_router.py` — embedding-based intent router (exemplar index + dual classifier: tool + dialogue). `backend/router.py` does NOT exist.
- `backend/memory.py` — SQLite + ChromaDB hybrid memory (MemoryStore). LLM extraction/consolidation call Ollama `/api/chat` — never `/api/generate`, which ignores the `system` prompt.
- `backend/hearth_prompt.txt` — persona/system prompt, loaded via `_load_hearth_prompt` (env var override + hardcoded fallback in `main.py`).
- `backend/music_fastpath.py` — deterministic pre-graph music routing (bypasses LLM entirely). Don't route music through the graph.
- `backend/auth.py` — scrypt-hashed auth with SQLite token store. Token format: 64-char hex.
- `backend/app_schemas.py` — Pydantic request/response schemas (ChatRequest, TTSRequest, CodeRequest, SessionSelectRequest)
- `backend/tools/` — weather, music, base tool modules. Dispatched via `tools.dispatch(tool_name, params)`.
- `backend/tools/weather.py` — Open-Meteo integration (no API key). Has `is_weather_reasoning()` for full vs fast path.
- `backend/tts/` — pluggable TTS (Piper / Kokoro via `TTS_ENGINE` env var). Engines in `tts/engines/`.
- `backend/routes/` — 8 factory route modules (auth, chat, code, health, memory_tool, session, tts, voice). Each exposes `create_*_router(services)` and is wired in `main.py` via a shared services dict — no module-level routers.
- `backend/tests/` — 25 test files covering API, graph, memory, tool, TTS, music, weather
- `scripts/review_baseline.sh` — full local validation gate (pip-install → focused tests → pip-audit → gitleaks → Bandit). Uses `set -e`.
- `scripts/review_changed_tests.sh` — git-based targeted test selection with file→test mapping logic.
- `scripts/download-whisper-model.sh` — faster-whisper model; without it `/transcribe` returns 503.
- `docs/review/KNOWN_FAILURES.txt` — local known-failures deselection list (applied via `--allow-known-failures`).
- `docs/review/SECURITY_CORRECTNESS_CHECKLIST.md` — per-PR security + correctness checklist
- `docs/review/ENFORCEMENT.md` — enforcement guide
- `docs/MEMORY.md`, `docs/PLAN-LLM-NATIVE-TOOL-CALLING.md` — design/plan docs; read before restructuring memory or tool-calling.
- `caddy/Caddyfile` — TLS termination on :443, HTTP→HTTPS redirect on :80, reverse_proxy to backend:8000
- `mpd/` — MPD config directory (mpd.conf)
- `config.yaml` — Beets config (non-interactive, no MusicBrainz lookups, copy: no, move: no)

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

# Writable-path workaround for local uvicorn (defaults write into backend/):
mkdir -p /tmp/hearth-local && CHROMA_PATH=/tmp/hearth-local/chroma MEMORY_DB_PATH=/tmp/hearth-local/memory.db GRAPH_CHECKPOINT_DB_PATH=/tmp/hearth-local/graph-checkpoints.sqlite AUTH_DB_PATH=/tmp/hearth-local/auth.db uvicorn main:app --host 127.0.0.1 --port 8010

# Model assets:
bash scripts/download-models.sh
bash scripts/download-tts-models.sh
bash scripts/download-whisper-model.sh

# Container validation:
docker compose config
docker compose build backend
```

## Gotchas

- **`.env` is gitignored** — create from `.env.example`. All env vars are read at import time, so `import load_env` (which calls `load_dotenv()`) must stay `main.py`'s first local import, ahead of any env-reading module. Do not reorder it.
- **ChromaDB needs a writable path** — local `uvicorn` without Docker will fail if `CHROMA_PATH` points to a read-only location. Override it or use the writable-path workaround above.
- **Voice features require HTTPS** — browser secure context for `navigator.mediaDevices`. Use Caddy's HTTPS or `https://localhost`. Plain `http://localhost:8000` breaks mic/audio-worklet.
- **Tests must include `tools/` on PYTHONPATH** — when running tests outside Docker, run from `backend/` so `tools` is importable. In Docker the bind mount handles this.
- **`review_baseline.sh` uses `set -e`** — Bandit findings stop the script (Bandit runs only if installed; pip-audit and gitleaks are skipped when absent). Known false positives exist (B608 in `memory.py`); check `docs/review/KNOWN_FAILURES.txt`.
- **`review_baseline.sh` runs a focused test subset** — not all 25 tests. It runs: test_auth, test_router, test_graph, test_memory_isolation, test_weather. The full suite is via `review_changed_tests.sh` or direct pytest.
- **`review_changed_tests.sh` maps changed files to tests** — e.g. `backend/main.py` → test_chat_sessions + test_chat_voice_metadata + test_graph. `backend/tts/*` → 4 TTS test files (endpoint, loader, piper, kokoro — not normalise). Falls back to a default suite for unmatched backend changes.
- **Known-failures deselection**: `docs/review/KNOWN_FAILURES.txt` — 4 `test_chat_sessions` cases plus 2 `test_music` cases that fail only under full-suite ordering (state leak). Used by `--allow-known-failures` (local only).
- **Music import on first boot**: Beets auto-imports at `/music` if library is empty. Run manually: `docker compose exec backend sh -c 'cd /beets && beet import -A /music'`.
- **Model files are gitignored but required at runtime** (`backend/models/*.onnx`, `backend/models/tts/*`, `backend/models/whisper/`). Download before first use. openWakeWord and faster-whisper load lazily (first WebSocket / first `/transcribe`), so the first voice request is slow. `_validate_startup()` only *warns* on missing models — it does not fail boot.
- **Startup chat-model warmup** (`CHAT_MODEL_WARMUP=true`) pre-loads gemma4:e4b (~80s) before serving `/health`; the compose healthcheck has a 120s `start_period` for this. Don't shorten it.
- **`backend/router.py` does NOT exist** — routing is split across `intents.py`, `embedding_router.py`, and `routing_config.py`. Don't look for a single `router.py`.
- **`.gitignore` hides `.env`, `*.onnx`, `*.mp3`, `*.bin`, `*.db`, `*.sqlite`, `*.sqlite3`, `backend/chroma/`, `caddy/certs/`, `mpd/mpdstate`, `mpd.pid`, `graph_checkpoints.*`, `backend/tests/artifacts/tts-benchmark.json`** — model/TTS assets, memory DBs, Chroma persistence, mkcert certs, and MPD state are all gitignored.
- **No CI exists**: `.github/` itself is gitignored, so workflows can't be committed. The `scripts/review_*.sh` scripts are the only gates. A local `.github/copilot-instructions.md` exists but is untracked.
- **CORS policy**: `CORS_ORIGINS` defaults to `*` (permissive for plain-HTTP dev). Set to exact Caddy origin(s) once HTTPS is in use; credentials are only allowed when it's not `*`. `SESSION_COOKIE_SECURE=true` when Caddy is the browser-facing edge.
- **Auth middleware ordering**: COOP/COEP middleware is added first, then AuthMiddleware, then CORSMiddleware. Auth checks use `Sec-Fetch-Mode: navigate` to distinguish browser navigations from API calls. Unprotected paths: `/health`, `/`, `/ws/wake`.
- **Graph fallback**: If the checkpointed graph is unavailable (lifespan failed), `main.py` lazily builds a no-checkpoint graph on first use and logs a warning. Conversation persistence is silently disabled in this case.

## Architecture Constraints

- Frontend always uses relative API paths — single-origin contract with FastAPI. Never serve UI from a separate dev server in production.
- `music_fastpath.py` sits in front of the graph for deterministic music commands. Don't route music through the graph.
- Code tool is code-question-only: the confirmation-gated file-write nodes were removed (see comment in `graph.py`). `/code` forces the code-question intent; there is no file-write path — don't reintroduce workspace path resolution.
- Session state is in-memory + cookie-scoped — lost on restart.
- `Dockerfile` is minimal (Python 3.11-slim, copy requirements → install → copy source → uvicorn). No layer optimization.
- Runtime state DBs live on the `hearth-data` volume at `/tmp/hearth/` inside the container (memory, chroma, checkpoints, auth) — not in the image.
- MPD audio output requires PulseAudio/PipeWire socket mount (`PULSE_SERVER` env var) for host audio from within the container.
- `review_baseline.sh` activates `backend/.venv` if present, otherwise uses `$PYTHON_BIN` or `python` (the changed-tests script also falls back to a root-level `.venv`).

## Code-Change Validation Policy

- Keep edits small and module-local; update relevant tests.
- Minimum validation for backend changes: `review_changed_tests.sh` → `review_baseline.sh`.
- If touching startup/memory paths, run local uvicorn once (with writable overrides if needed).
- If touching Docker/Caddy/deploy files, run `docker compose config` and `docker compose build backend`.

## Security and Privacy

- Local-first default. No data leaves the device unless the user triggers a cloud model call or an external tool.
- Never hardcode local paths, personal information, usernames, or device-specific details in code.
- Redact API keys, tokens, and personal data from all logs.
