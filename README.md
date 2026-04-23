# Local AI Assistant

Local-first personal AI assistant with streaming chat, wake-word voice input, hybrid memory (SQLite + Chroma), and model routing between local Ollama and optional Anthropic fallback.

## Stack

- FastAPI backend
- Ollama for local inference
- Anthropic API as optional cloud fallback
- openWakeWord for wake detection
- faster-whisper for transcription
- SQLite + ChromaDB memory layer
- Static frontend served by FastAPI
- Docker Compose deployment

## Architecture

- Browser clients connect to `http://<LAN-IP>:8000`
- FastAPI serves both UI and API from one origin
- Ollama runs as a separate container (`11434`)
- Backend mounts `frontend/` as static files

Main backend endpoints:

- `POST /chat` (SSE streaming)
- `POST /transcribe`
- `WS /ws/wake`
- `GET /health`
- `GET /memory`
- `DELETE /memory/{id}`
- `DELETE /memory`
- `GET /chat/sessions`
- `POST /chat/session/new`
- `POST /chat/session/select`
- `GET /chat/session/messages`
- `DELETE /chat/session` (reset current session messages)
- `DELETE /chat/sessions/{session_id}` (delete a specific session)

## Project Layout

```text
.
├── docker-compose.yml
├── backend/
│   ├── main.py
│   ├── router.py
│   ├── memory.py
│   ├── requirements.txt
│   ├── models/
│   └── Dockerfile
├── frontend/
│   ├── index.html
│   ├── message.js
│   ├── voice.js
│   ├── style.css
│   └── audio-processor.js
└── scripts/
    └── download-models.sh
```

## Prerequisites

- Docker + Docker Compose
- NVIDIA GPU + drivers (optional but recommended for local model speed)
- Linux host on LAN

## Environment Variables

Create `.env` in the repo root. Example:

```bash
# Model routing
MODEL_LOCAL=llama3.2
MODEL_CLOUD=claude-sonnet-4-20250514
ROUTE_CONFIDENCE_THRESHOLD=0.55

# Optional cloud fallback
ANTHROPIC_API_KEY=

# Session settings
CHAT_SESSION_COOKIE=assistant_session
CHAT_SESSION_IDLE_TTL_SECONDS=1800
CHAT_SESSION_MAX_ITEMS=200
CHAT_TOKEN_BUDGET=1500
CHAT_MAX_TURNS=24

# Ollama backend URL (inside Docker network)
OLLAMA_URL=http://ollama:11434

# Memory tuning
MEMORY_TOP_N=5
MEMORY_MIN_RELEVANCE_SCORE=0.28
# MEMORY_DB_PATH=/app/memory.db
# CHROMA_PATH=/app/chroma
```

Notes:

- If `ANTHROPIC_API_KEY` is empty, cloud fallback is disabled and local responses continue.
- Memory DB and Chroma data are stored under `backend/` by default.

## Run With Docker

From repo root:

```bash
docker compose up -d --build
```

Check health:

```bash
curl -s http://localhost:8000/health
```

Open UI:

- `http://localhost:8000`
- For LAN devices: `http://<your-lan-ip>:8000`

## Wake Word Models

Required ONNX model files in `backend/models/`:

- `computer_v2.onnx`
- `melspectrogram.onnx`
- `embedding_model.onnx`

If missing, run:

```bash
bash scripts/download-models.sh
```

## Local Development (Without Docker)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then open `http://localhost:8000`.

## Session Management (Phase 5)

Session state is in-memory and cookie-scoped.

- New session: `POST /chat/session/new`
- Switch session: `POST /chat/session/select`
- List sessions: `GET /chat/sessions`
- Get current session messages: `GET /chat/session/messages`
- Reset current session messages: `DELETE /chat/session`
- Delete one session: `DELETE /chat/sessions/{session_id}`

Behavior:

- Deleting the active session automatically creates a new session cookie.
- Sessions expire by idle TTL and may be evicted by max-capacity settings.

## Memory Layer (Phase 5)

Memory uses SQLite for structured storage and Chroma for semantic recall.

Tables:

- `facts`
- `preferences`
- `summaries`

Commands recognized in chat:

- `save this`
- `remember this`
- `remember <text>`
- `do not remember this`
- `forget <query>`

Safety behavior:

- Sensitive values (tokens/passwords/phone-like/address-like patterns) are blocked.
- Some location-history style entries require confirmation unless explicit save is requested.

## Troubleshooting

- `/health` fails:
  - Check backend logs: `docker compose logs -f backend`
- Wake word not triggering:
  - Confirm model files exist in `backend/models/`
  - Verify mic permissions in browser
- Cloud responses not used:
  - Ensure `ANTHROPIC_API_KEY` is set
  - Check route telemetry logs from backend
- Session list empty after restart:
  - Expected. Session store is currently in-process memory only.

## Current Status

- Phase 1: LAN-safe single-origin serving complete
- Phase 2: Wake-word pipeline complete (desktop/Linux/LAN browser)
- Phase 3: Chat context management in progress
- Phase 4: Intent routing complete
- Phase 5: SQLite/Chroma memory layer and sessions UI implemented

## Security Notes

- Do not commit `.env`
- Keep API keys in environment variables only
- Frontend uses relative API paths and is served by backend static mount
- File/path safety constraints are enforced in backend features as implemented
