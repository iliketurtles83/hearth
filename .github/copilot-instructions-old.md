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
**Status: complete (desktop/Linux/LAN browsers)**
**Estimate: 1–2 days**
**Depends on: Phase 1**

Goal: Hands-free activation via "Computer, ..." with a stable wake-to-transcribe
loop and explicit state transitions.

Frontend state machine — enforce these transitions only, no others:
```
off → sleeping → recording → transcribing → sleeping
```

Implementation notes (decisions made during build):
- **openWakeWord v0.6.0**: backbone models (`melspectrogram.onnx`, `embedding_model.onnx`)
  are no longer bundled in the PyPI package. Pass explicit `melspec_model_path` and
  `embedding_model_path` to `Model()`, pointing at `backend/models/`. Do not rely on
  the package's `resources/models/` directory — it is empty after v0.6.0.
- **Audio dtype**: pass raw `int16` numpy arrays to `model.predict()`. The library's
  melspectrogram step requires int16 PCM. Converting to float32 beforehand silently
  zeros all samples and the model sees only silence.
- **Utterance capture**: `@ricky0123/vad-web` was removed. It requires ORT/WASM assets
  that fail to load under `Cross-Origin-Embedder-Policy` headers. Utterance capture is
  now done entirely from the existing AudioWorklet frame stream using an RMS energy
  threshold (no additional CDN dependencies). Silence is detected after ~640 ms of
  sub-threshold energy following speech onset; max capture is 15 s.
- **COEP header**: set to `credentialless` (not `require-corp`) so cross-origin CDN
  resources load while SharedArrayBuffer remains enabled for the worklet.
- **Linux dual-mic fix**: pass explicit audio constraints (`sampleRate: 16000`,
  `channelCount: 1`, processing disabled) to `getUserMedia` to prevent PipeWire/ALSA
  from opening the device twice.
- **Android / mobile HTTPS**: `navigator.mediaDevices` is `undefined` on plain HTTP in
  mobile browsers. The UI shows a clear in-UI error. Deferred — fix requires HTTPS
  termination (nginx/Caddy with self-signed cert, or LAN reverse proxy). See backlog.

Tasks — all complete:
- ✅ Startup validation: log clearly if ONNX model files are missing.
- ✅ Pass explicit backbone model paths to openWakeWord `Model()` constructor.
- ✅ Keep wake model filename and prediction key in sync (`computer_v2`).
- ✅ Pass raw int16 to `model.predict()`; do not normalize to float32 beforehand.
- ✅ Post-wake guard window (1.5 s) to prevent retrigger.
- ✅ WebSocket close code/reason logged on every disconnect.
- ✅ Wake score logged at DEBUG level.
- ✅ Mic permission errors shown in-UI, non-fatal.
- ✅ Secure-context check with user-friendly message on non-HTTPS clients.
- ✅ Reconnect backoff on frontend WebSocket.
- ✅ Utterance capture via RMS threshold on existing worklet frames (no vad-web).

Acceptance:
- ✅ Clicking mic enters stable sleeping state.
- ✅ Saying "Computer," triggers exactly one capture/transcribe cycle then returns
  to sleeping.
- ✅ Transcribed text is sent to `/chat` automatically.
- ✅ No retrigger occurs while the user is speaking after the wake event.
- ⏳ Android voice — deferred pending HTTPS on LAN (see backlog).

---

### Phase 3 — Chat context management
**Status: in progress**
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
**Status: complete**
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

## Backlog

- **Android / mobile voice**: `navigator.mediaDevices` requires a secure context
  (HTTPS). Options: nginx/Caddy reverse proxy with self-signed cert on LAN;
  or a native Android companion app. Do not attempt on plain HTTP.

---

## Immediate next sprint

| # | Issue | Status | Estimate | Depends on |
|---|-------|--------|----------|------------|
| 1 | LAN-safe single-origin serving and health checks | ✅ done | 1–2 days | — |
| 2 | Replace all `localhost` frontend paths with relative paths | ✅ done | 2–4 hours | 1 |
| 3 | Harden wake-word pipeline (Phase 2) | ✅ done | 1–2 days | 1, 2 |
| 4 | Chat context management: bounded session buffer + token-aware context | 🔄 in progress | 1–2 days | 1 |
| 5 | Upgrade router: intent categories, confidence scoring, telemetry | ✅ done | 1–2 days | 1, 4 |
| 6 | SQLite memory layer: schema, write/read policy, CRUD endpoints | 🔲 | 2–3 days | 5 |
| 7 | Weather tool: adapter, memory-backed default location, error handling | 🔲 | 1 day | 6 |
| 8 | HTTPS on LAN (nginx/Caddy) to unblock Android/mobile voice | 🔲 backlog | 0.5 days | — |

Each numbered issue maps to a phase above. Use these as GitHub issue titles and
reference the phase section for full task and acceptance detail.
