# Local AI Assistant — Copilot Instructions

This project is a local-first personal AI assistant with voice activation, streaming chat,
tool integrations, and optional cloud model fallback. It runs entirely on-device via Docker
Compose and is designed to be reachable from all devices on the local network.

Stack: FastAPI backend · Ollama (local LLM) · Anthropic API (cloud fallback) ·
OpenWakeWord · faster-whisper · Piper TTS · SQLite · Browser UI

---

## Architecture overview

```
Browser / LAN client
  └── http://<LAN-IP>:8000          (single origin — FastAPI serves UI + API)
        ├── GET  /                   static frontend (mounted into backend container)
        ├── POST /chat               streaming chat endpoint
        ├── POST /transcribe         audio → text via faster-whisper
        ├── WS   /ws/wake            wake-word detection socket
        ├── POST /weather            weather tool
        ├── POST /music/*            music search + playback control
        └── GET  /health             uptime check

FastAPI backend
  ├── router.py                      intent classifier + model routing
  ├── memory.py                      SQLite memory layer
  ├── tools/weather.py               weather adapter
  ├── tools/music.py                 media indexer + playback
  └── tts.py                         pluggable TTS engine

Ollama (container)                   local model inference (GPU)
Anthropic API (external)             cloud fallback for complex queries
```

**Serving rule:** frontend is always served from the FastAPI origin. Never run a
separate dev server in production. Mount `./frontend` into the backend container and
serve it as static files. All browser fetch/WebSocket calls use relative paths
(`/chat`, `/ws/wake`, etc.) — no hardcoded `localhost` or port numbers anywhere in
runtime code.

---

## Development order

Features are ordered to minimize rework. Each phase builds on stable foundations
from the phase before it.

---

### Phase 1 — Stabilize LAN access and single-origin serving
**Status: complete**
**Estimate: 1–2 days**

Goal: The assistant is reachable from every device on the local network and
remains stable under normal use. Every subsequent feature depends on this.

Tasks:
- Serve frontend from FastAPI static mount — no separate origin.
- Replace every `http://localhost:*` in runtime JS with relative paths.
- Bind all services to `0.0.0.0` and verify Docker port mappings.
- Add `/health` endpoint and container health checks in `docker-compose.yml`.
- Add startup validation that emits clear logs on misconfiguration.
- Keep CORS permissive during development; restrict to LAN hosts when stable.

Acceptance:
- `http://<LAN-IP>:8000` loads the UI and chat works from a phone or tablet.
- Voice WebSocket connects and stays open without looped reconnect.
- A container restart surfaces import/runtime errors in logs rather than silently failing.

---

### Phase 2 — Harden wake-word voice input pipeline
**Status: in progress**
**Estimate: 1–2 days**
**Depends on: Phase 1**

Goal: Hands-free activation via "Computer, ..." with a stable wake-to-transcribe
loop and explicit state transitions.

Frontend state machine — enforce these transitions only, no others:
```
off → sleeping → recording → transcribing → sleeping
```

Tasks:
- On backend startup, validate that required ONNX model files exist and log
  clearly if any are missing — do not silently swallow import errors.
- Keep wake model filename and prediction key in sync with downloaded model assets.
- Add a short post-wake guard window (recommended: 1.5 s) to prevent retrigger
  while the user is still speaking.
- Log WebSocket close codes and reasons on every disconnect.
- Log wake score and threshold decisions at DEBUG level.
- Make microphone permission errors user-friendly and non-fatal — show a clear
  in-UI message rather than a console-only error.
- Add reconnect backoff on the frontend WebSocket (do not reconnect in a tight loop).

Acceptance:
- Clicking mic enters stable sleeping state.
- Saying wake phrase triggers exactly one capture/transcribe cycle then returns
  to sleeping.
- Transcribed text is sent to `/chat` automatically.
- No retrigger occurs while the user is speaking after the wake event.

---

### Phase 3 — Chat context management
**Status: next**
**Estimate: 1–2 days**
**Depends on: Phase 1**

Goal: Keep each active chat session coherent by passing recent conversational
context with every new user message, without introducing durable memory yet.

Tasks:
- Maintain a message history buffer for the current chat session.
- Pass the last N messages (or messages up to a token budget) to Ollama on
  each `/chat` request.
- Add bounded-context controls: max turns, token cap, and truncation strategy.
- Optionally summarize older in-session messages once history grows beyond
  thresholds, and include that summary in context.
- Store session context in memory or a lightweight session store (no SQLite
  requirement in this phase).
- Emit lightweight telemetry: context length, estimated token usage,
  truncation/summarization events.

Acceptance:
- Multi-turn follow-up questions resolve correctly within the same chat session.
- Context size remains bounded under long conversations.
- Session context is isolated per client/session and resettable.
- No dependency on persistent SQLite memory for this phase.

---

### Phase 4 — Smarter model routing with intent classification
**Status: in progress**
**Estimate: 1–2 days**
**Depends on: Phases 1, 3**

Goal: Local model handles all routine queries. Cloud model is invoked only when
genuinely needed. Failures degrade gracefully.

Intent categories:
- `quick-local` — factual, short, conversational
- `reasoning-heavy` — multi-step planning, analysis, architecture
- `external-data-needed` — weather, news, live data (tool path, not cloud)
- `memory-needed` — references to prior facts or user preferences

Tasks:
- Replace keyword heuristics in `router.py` with a lightweight intent classifier
  (prompt-based using the local model itself, or a small rule set with confidence
  scoring).
- Add a confidence threshold: if local model confidence is below threshold,
  escalate to cloud.
- Emit structured telemetry per request: route chosen, first-token latency,
  completion latency, error/fallback count.
- Keep a visible model badge in the UI showing which model handled the response.
- Fallback policy: if cloud is unavailable, return local response with a
  user-visible notice, never a silent failure.

Acceptance:
- Conversational and short prompts stay local.
- Complex planning and reasoning prompts route to cloud.
- Cloud unavailability degrades gracefully with a local response.

---

### Phase 5 — SQLite memory layer
**Estimate: 2–3 days**
**Depends on: Phase 4**

Goal: Assistant remembers user preferences and relevant facts across restarts.
This is the foundation for personalization, weather defaults, music preferences,
and conversational continuity.

Schema (start minimal, extend as needed):
```sql
facts        (id, key, value, source, created_at, expires_at)
preferences  (id, key, value, updated_at)
summaries    (id, session_id, summary, created_at)
```

Tasks:
- Implement write policy: save only high-value facts — explicit user statements,
  stated preferences, recurring intents. Never store secrets unless explicitly
  approved by the user.
- Implement retrieval policy: fetch top-N relevant items per query using keyword
  or embedding match; inject concise snippets into system/context prompt.
- Add CRUD endpoints:
  - `GET  /memory`         — list stored items
  - `DELETE /memory/{id}`  — remove one item
  - `DELETE /memory`       — clear all
- Surface basic memory viewer in UI.

Acceptance:
- Stated preferences survive container restart.
- User can view, delete individual items, and clear all memory from the UI.
- Memory snippets appear in context without bloating the prompt.

---

### Phase 6 — Weather tool
**Estimate: 1 day**
**Depends on: Phase 5**

Goal: First external tool integration. Validates the tool-routing architecture
cleanly before adding more complex tools.

Tasks:
- Build a weather provider adapter with a normalized response schema (do not
  leak provider-specific field names into the rest of the codebase).
- Route weather intents directly to the tool endpoint, then summarize through
  the model — do not pass raw API JSON to the user.
- Use the remembered default location from memory; support inline override
  ("weather in Tallinn").
- Return graceful error responses for API failures and offline conditions.

Acceptance:
- "What is the weather?" uses stored default location.
- "Weather in <city>" overrides default correctly.
- Output is concise and includes units.
- API failure returns a clear user-facing message, not a stack trace.

---

### Phase 7 — Local music library playback
**Estimate: 3–5 days**
**Depends on: Phases 2, 5**

Goal: Search and play music from a local collection by voice or text.

Tasks:
- Build a media indexer that scans configured folders and stores metadata in
  SQLite (title, artist, album, path, duration, genre).
- Add endpoints: search, play, pause, stop, next, queue, now-playing.
- Decide playback architecture early:
  - Browser playback — works for same-device sessions, simpler.
  - Backend audio daemon (e.g. MPD) — works for always-on host audio, more complex.
- Add confirmation patterns: "Playing <track> by <artist>".
- Add ambiguity resolution: if query matches multiple tracks, ask to clarify.

Acceptance:
- "Play <song or artist>" works end to end by voice and text.
- Pause, resume, next, and status commands work.
- Ambiguous matches prompt clarification rather than picking silently.

---

### Phase 8 — TNG-style voice output (TTS)
**Estimate: 2–4 days**
**Depends on: Phases 2, 4 (and ideally Phase 6 for tool response parity)**

Goal: Assistant responds with spoken audio in a clear, consistent female voice
aligned to TNG computer style.

Tasks:
- Implement TTS as a backend service with a pluggable engine interface so the
  underlying model can be swapped without changing callers.
- Start with Piper TTS; tune speaking rate, pitch, and prosody for a calm,
  authoritative delivery.
- Add brief/full spoken response modes — spoken output should be shorter than
  chat output by default.
- Return audio as a stream or file URL; auto-play in frontend with correct
  handling of browser autoplay constraints.
- Add barge-in behavior: a new wake phrase while TTS is playing stops playback
  and immediately starts listening.

Acceptance:
- Assistant speaks responses end to end with stable audio quality.
- Voice playback does not block normal chat use.
- Barge-in stops TTS and resumes wake-word listening within 500 ms.

---

### Phase 9 — LangGraph agent orchestration
**Estimate: 1–2 weeks**
**Depends on: all prior phases**

Goal: Replace procedural routing with a stateful LangGraph graph. Nodes handle
intent classification, memory retrieval, tool dispatch, and response generation.
Enables multi-step reasoning, tool chaining, and durable session continuity.

Graph nodes:
```
input → intent_classifier → memory_retrieval → tool_router
  ├── weather_tool
  ├── music_tool
  ├── code_tool
  └── chat_fallback
        └── responder → output
```

State shape:
```python
class AssistantState(TypedDict):
    messages: list[dict]
    intent: str
    memories: list[str]
    tool_result: str
    user_prefs: dict
    session_id: str
```

Tasks:
- Introduce LangGraph with SqliteSaver checkpointer for durable session state.
- Migrate existing router and tool logic into named graph nodes.
- Add ChromaDB for semantic memory retrieval alongside SQLite for exact recall.
- Implement ToolNode pattern for file read/write in the code assistant node.

Acceptance:
- "Computer, continue that function from yesterday" resolves via checkpointed state.
- All existing tools work through the graph without regression.
- Graph state is inspectable for debugging.

---

## Cross-cutting standards

### Security and privacy
- Local-first default. No data leaves the device unless the user triggers a
  cloud model call or an external tool.
- Explicit user consent before writing to persistent memory.
- Redact API keys, tokens, and personal data from all logs.
- Never commit `.env` to git.

### Reliability
- All services bind to `0.0.0.0` in Docker.
- Structured logs for: routing decisions, wake-word scores, transcription results,
  tool calls, and errors.
- Retry with exponential backoff for transient network failures.
- Standardized error response shape: `{ "error": str, "code": str, "retryable": bool }`.
- Container restarts must surface errors in logs — never swallow startup failures.

### Code conventions
- All frontend API calls use relative paths. No hardcoded hosts or ports in
  runtime code.
- New tools are added as modules under `backend/tools/` with a consistent
  interface: `async def run(params: dict) -> dict`.
- Environment variables are the only configuration mechanism. No config files
  that duplicate `.env` values.
- Type-annotate all Python function signatures.
- Keep `requirements.txt` up to date. Pin major versions once a phase is stable.

### Testing priorities
- Backend: `/chat`, `/transcribe`, `/ws/wake`, `/weather`, `/health`.
- Frontend: streaming markdown render, mic toggle state machine, WebSocket
  reconnect behavior.
- LAN matrix: verify from desktop browser, mobile browser, and tablet after
  every Phase 1–3 change.
- Add tests before marking any phase acceptance criteria as done.

---

## Immediate next sprint (Issues 1–7 in order)

| # | Issue | Estimate | Depends on |
|---|-------|----------|------------|
| 1 | Finish LAN-safe single-origin serving and health checks | 1–2 days | — | ✅ done |
| 2 | Replace all `localhost` frontend paths with relative paths | 2–4 hours | 1 | ✅ done |
| 3 | Harden wake-word pipeline: startup validation, guard window, WS logging | 1–2 days | 1, 2 |
| 4 | Add chat context management: bounded session buffer + token-aware context | 1–2 days | 1 |
| 5 | Upgrade router: intent categories, confidence scoring, telemetry | 1–2 days | 1, 4 |
| 6 | SQLite memory layer: schema, write/read policy, CRUD endpoints | 2–3 days | 5 |
| 7 | Weather tool: adapter, memory-backed default location, error handling | 1 day | 6 |

Each issue maps directly to a phase above. Use these as GitHub issue titles and
reference the phase section for full task and acceptance detail.
