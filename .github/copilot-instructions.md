# Local AI Assistant — Copilot Instructions

This project is a local-first personal AI assistant with voice activation, streaming chat,
tool integrations, and optional cloud model fallback. It runs entirely on-device via Docker
Compose and is designed to be reachable from all devices on the local network.

Stack: FastAPI backend · Ollama (local LLM + local coder) · Anthropic API (cloud fallback) ·
OpenWakeWord · faster-whisper · Piper TTS (pluggable — Kokoro is a viable swap) · SQLite ·
ChromaDB · LangGraph · Browser UI

---

## Architecture overview

```
Browser / LAN client  (HTTPS via Caddy — Phase 0b)
  └── https://<LAN-IP>                (single origin — FastAPI serves UI + API)
        ├── GET  /                    static frontend (mounted into backend container)
        ├── POST /chat                streaming chat endpoint
        ├── POST /transcribe          audio → text via faster-whisper
        ├── WS   /ws/wake             wake-word detection socket
        ├── POST /weather             weather tool
        ├── POST /music/*             music search + playback control
        ├── POST /code                code generation / editing tool
        └── GET  /health              uptime check

FastAPI backend
  ├── graph.py                        LangGraph graph definition (Phase 9+)
  ├── router.py                       intent classifier + model routing (pre-Phase 9)
  ├── memory.py                       SQLite + ChromaDB tiered memory layer
  ├── tools/weather.py                weather adapter
  ├── tools/music.py                  media indexer + playback
  ├── tools/code.py                   code generation + file read/write
  └── tts.py                          pluggable TTS engine (Piper → Kokoro swap-ready)

Ollama (container)                    local model inference (GPU)
  ├── <chat model>                    general assistant (e.g. mistral, llama3)
  └── <coder model>                   code-specialized model (e.g. qwen2.5-coder)
Anthropic API (external)              cloud fallback for complex queries

Caddy (container)                     HTTPS reverse proxy on LAN (Phase 0b)
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

Tasks — all complete:
- ✅ Serve frontend from FastAPI static mount — no separate origin.
- ✅ Replace every `http://localhost:*` in runtime JS with relative paths.
- ✅ Bind all services to `0.0.0.0` and verify Docker port mappings.
- ✅ Add `/health` endpoint and container health checks in `docker-compose.yml`.
- ✅ Add startup validation that emits clear logs on misconfiguration.
- ✅ Keep CORS permissive during development; restrict to LAN hosts when stable.

Acceptance:
- ✅ `https://<LAN-IP>` loads the UI and chat works from a phone or tablet.
- ✅ Voice WebSocket connects and stays open without looped reconnect.
- ✅ A container restart surfaces import/runtime errors in logs rather than silently failing.

---

### Phase 2 — Harden wake-word voice input pipeline
**Status: complete (desktop + Android/Linux/LAN browsers)**
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
  mobile browsers. Unblocked by Phase 0b (Caddy HTTPS).

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
- ✅ Android voice — validated after Phase 0b (Caddy HTTPS).

Acceptance:
- ✅ Clicking mic enters stable sleeping state.
- ✅ Saying "Computer," triggers exactly one capture/transcribe cycle then returns
  to sleeping.
- ✅ Transcribed text is sent to `/chat` automatically.
- ✅ No retrigger occurs while the user is speaking after the wake event.
- ✅ Android voice works on HTTPS clients.

---

### Phase 3 — HTTPS on LAN (Caddy)
**Status: complete**
**Estimate: 0.5 days**
**Depends on: Phase 1**

Goal: Terminate HTTPS on the LAN so that all clients — including Android and iOS —
can access microphone and other secure-context browser APIs. This also unblocks
several tightening browser API restrictions (AudioWorklet, WebAuthn, etc.) and
should be done before any further mobile work.

Tasks — all complete:
- ✅ Add a `caddy` service to `docker-compose.yml` that reverse-proxies to FastAPI on
  port 8000 and terminates TLS.
- ✅ Use a self-signed certificate for LAN use, or mkcert for a locally-trusted CA.
  Document the one-time mkcert install step clearly in the README.
- ✅ All internal services continue to communicate over plain HTTP inside Docker.
  Only the edge (Caddy → LAN client) is HTTPS.
- ✅ Update all documentation, README, and example URLs from `http://` to `https://`.
- ✅ CORS policy: restrict to the Caddy origin once HTTPS is stable; remove the
  permissive dev-time CORS rule.

Acceptance:
- ✅ `https://<LAN-IP>` loads the UI from a phone or tablet without certificate errors
  (after mkcert CA install on client devices).
- ✅ `navigator.mediaDevices` is defined and microphone permission works on Android/iOS.
- ✅ Wake-word WebSocket connects over `wss://` without issues.
- ✅ Phase 2 Android voice item is unblocked and closed.

---

### Phase 4 — Chat context management
**Status: complete**
**Estimate: 1–2 days**
**Depends on: Phase 1**

Goal: Keep each active chat session coherent by passing recent conversational
context with every new user message, without introducing durable memory yet.

Tasks — all complete:
- ✅ Maintain a message history buffer for the current chat session.
- ✅ Pass the last N messages (or messages up to a token budget) to Ollama on
  each `/chat` request.
- ✅ Add bounded-context controls: max turns, token cap, and truncation strategy.
- ✅ Summarize older in-session messages once history grows beyond thresholds,
  and include that summary in context. This is the seed of the episodic memory
  tier introduced formally in Phase 9.5.
- ✅ Store session context in memory or a lightweight session store (no SQLite
  requirement in this phase).
- ✅ Emit lightweight telemetry: context length, estimated token usage,
  truncation/summarization events.

Acceptance:
- ✅ Multi-turn follow-up questions resolve correctly within the same chat session.
- ✅ Context size remains bounded under long conversations.
- ✅ Session context is isolated per client/session and resettable.
- ✅ No dependency on persistent SQLite memory for this phase.

---

### Phase 5 — Inner-monologue routing (replaces keyword intent classifier)
**Status: complete**
**Estimate: 1–2 days**
**Depends on: Phases 1, 4**

Goal: Replace single-step intent classification with a brief inner-monologue
reasoning pass before routing. The model thinks out loud about what the user
needs before committing to a route. This is how state-of-the-art agentic systems
(o3, Claude extended thinking, Gemini 2.5) handle routing — not by classifying
into a fixed category, but by reasoning about the request first.

Design:

The inner monologue is a short chain-of-thought prompt (run locally) that produces
a structured routing decision. It is not the full response — it is a deliberate
pre-step that runs before model selection or tool dispatch.

```
User input
  └── inner_monologue (local model, ~200 token budget)
        ├── What is the user actually asking for?
        ├── Do I need memory? Which kind?
        ├── Do I need a tool? Which one?
        ├── Can local handle this, or does it need cloud?
        └── → structured routing decision (JSON)
              { "route": "local|cloud|tool", "tool": str|null,
                "needs_memory": bool, "confidence": float,
                "reasoning": str }
```

The `reasoning` field is logged for observability but never shown to the user.
The structured decision drives all subsequent dispatch.

Intent categories (produced by monologue, not hardcoded):
- `quick-local` — factual, short, conversational
- `reasoning-heavy` — multi-step planning, analysis, architecture
- `code` — code generation, debugging, explanation, file editing
- `external-data-needed` — weather, news, live data (tool path, not cloud)
- `memory-needed` — references to prior facts or user preferences

Tasks:
- ✅ Replace keyword heuristics in `router.py` with a lightweight intent classifier
  (prompt-based using the local model itself).
- ✅ Add a confidence: if local model confidence is below threshold,
  escalate to cloud.
- ✅ Upgrade classifier to a full inner-monologue reasoning pass that outputs a
  structured routing JSON rather than a single intent label.
- ✅ Route `code` intent to the local coder model (see Phase 10) rather than the
  general chat model or cloud. Code tasks almost never need cloud escalation.
- ✅ Emit structured telemetry per request: route chosen, model used, first-token
  latency, completion latency, error/fallback count.
- ✅ Keep a visible model badge in the UI showing which model handled the response.
- ✅ Fallback policy: if cloud is unavailable, return local response with a
  user-visible notice, never a silent failure.

Acceptance:
- ✅ Conversational and short prompts stay local.
- ✅ Complex planning and reasoning prompts route to cloud.
- ✅ Inner monologue reasoning pass replaces single-step classification.
- ✅ Code prompts route to the local coder model.
- ✅ Cloud unavailability degrades gracefully with a local response.

---

### Phase 6 — SQLite memory layer
**Status: complete**
**Estimate: 2–3 days**
**Depends on: Phase 5**

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
- ✅ Implement write policy: save only high-value facts — explicit user statements,
  stated preferences, recurring intents. Never store secrets unless explicitly
  approved by the user.
- ✅ Define and enforce a memory sensitivity matrix:
  - auto-save: clear non-sensitive facts/preferences (e.g. nickname, broad preferences)
  - confirm-first: borderline personal data (e.g. timeline/location history)
  - explicit-only/blocked: secrets, credentials, exact address, phone-like identifiers
- ✅ Implement retrieval policy: fetch top-N relevant items per query using SQLite
  for exact/keyword recall and ChromaDB for semantic recall; inject concise
  snippets into system/context prompt.
- ✅ Add explicit memory controls (deterministic commands):
  - "save this" / "remember this"
  - "do not remember this"
  - "forget X"
- ✅ Add CRUD endpoints:
  - `GET  /memory`         — list stored items
  - `DELETE /memory/{id}`  — remove one item
  - `DELETE /memory`       — clear all
- ✅ Surface basic memory viewer in UI.
- ✅ Surface memory write outcomes in UI and logs (`saved`, `blocked-sensitive`, `needs-confirmation`).
- ✅ Add lightweight chat sessions sidebar (new/switch/reset) before weather-tool work.
- ✅ Keep a strict memory boundary for future persona work: memory extraction and retrieval
  must be fact/policy-driven and not modified by stylistic "human side" rendering.

Acceptance:
- ✅ Stated preferences survive container restart.
- ✅ User can view, delete individual items, and clear all memory from the UI.
- ✅ Memory snippets appear in context without bloating the prompt.
- ✅ Explicit save/forget commands behave deterministically and return clear status.
- ✅ Sensitive and confirm-first classes follow policy matrix consistently.
- ✅ Sessions sidebar supports starting and switching chat sessions.
- ✅ Persona/tone customization (future phase) cannot alter stored facts or tool-derived values.

---

### Phase 7 — Weather tool
**Status: complete**
**Estimate: 1 day**
**Depends on: Phase 6**

Goal: First external tool integration. Validates the tool-routing architecture
cleanly before adding more complex tools.

Tasks:
- ✅ Build a weather provider adapter with a normalized response schema (do not
  leak provider-specific field names into the rest of the codebase).
- ✅ Route weather intents directly to the tool endpoint, then summarize through
  the model — do not pass raw API JSON to the user.
- ✅ Use the remembered default location from memory; support inline override
  ("weather in Tallinn").
- ✅ Return graceful error responses for API failures and offline conditions.

Implementation notes:
- **Open-Meteo** (no API key required) — two-step: geocode city → fetch forecast.
- Tool module system: `backend/tools/base.py` (`ToolResult`), `backend/tools/__init__.py`
  (registry + dispatch). Adding future tools: create module, call `register()`. No
  other files change.
- `memory.get_preference("default_location")` / `memory.set_preference()` added to
  `MemoryStore` for keyed preference lookup.
- `WEATHER_UNITS` (celsius/fahrenheit) and `WEATHER_TIMEOUT_MS` env vars.
- `POST /weather` direct endpoint for frontend + future LangGraph nodes.

Acceptance:
- ✅ "What is the weather?" uses stored default location.
- ✅ "Weather in <city>" overrides default correctly.
- ✅ Output is concise and includes units.
- ✅ API failure returns a clear user-facing message, not a stack trace.

---

### Phase 8 — Local music library playback
**Estimate: 3–5 days**
**Depends on: Phases 2, 6**

Goal: Provide backend music search and playback integration using MPD and the
Strawberry music database, exposing HTTP endpoints and voice and text commands
so users can search, queue, and control playback from the UI or via voice.

Implementation notes (decisions made during review):
- **Strawberry DB schema**: no `id` column; use `rowid` as integer primary key.
  File paths are stored in the `url` column as `file://` URIs with URL encoding.
  No `songs_fts` FTS table in production databases — use LIKE queries only.
- **Path rewrite**: Strawberry stores host-side absolute file:// URIs
  (e.g. `file:///media/jack/buffer/audio/artist/song.mp3`). The backend URL-decodes
  and strips the `MUSIC_PATH_HOST` prefix to produce MPD-relative paths
  (e.g. `artist/song.mp3`) which MPD resolves against its `music_directory`.
- **Ambiguity**: auto-pick the top-ranked LIKE result and log the confidence score
  server-side. Ranked candidate list is preserved in API response for future UX.
- **MPD resilience**: per-request fresh MPD connection with one reconnect attempt.
  On second failure, return retryable ToolResult — never crash or silently drop.
- **Strawberry lock handling**: open with `timeout=5, check_same_thread=False, uri=True`
  (read-only URI mode). Wrap all reads in `try/except sqlite3.OperationalError`
  and return `retryable=True` when locked (Strawberry holds write lock during scans).
- **Artist radio**: implemented in Phase 8 core as weighted-random by `playcount`,
  seeded for determinism. This is the Phase 8b fallback entry point — 8b calls
  `artist_radio()` directly when the rec engine is unavailable.

Tasks:
- Add MPD as a Docker Compose service with music folder mounted.
- Mount Strawberry DB directory read-only into backend container; set `STRAWBERRY_DB_PATH`.
- Implement Strawberry search in `backend/tools/music.py` using LIKE on title/artist/album,
  ranked by playcount. No FTS — use rowid as track ID.
- Implement MPD client in `backend/tools/music.py` with per-request connections and
  reconnect-once resilience: play(url), pause(), resume(), stop(), next(), queue(url),
  now_playing(), queue_view().
- Queue model: single song, multi-song (artist radio).
- Add path rewrite: URL-decode Strawberry `file://` URIs → strip `MUSIC_PATH_HOST`
  prefix → pass relative path to MPD.
- Implement `artist_radio(artist, n)` using weighted-random by playcount (Phase 8b dep).
- Wrap all Strawberry reads in `OperationalError` handling; return retryable errors.
- Add `GET /music/now_playing` endpoint (required by acceptance — add to tasks explicitly).
- Add `GET /music/queue` endpoint for queue visibility.

Environment variables:
  `STRAWBERRY_DB_PATH`      path to Strawberry sqlite db inside backend container
  `MPD_HOST`                default: mpd
  `MPD_PORT`                default: 6600
  `MPD_TIMEOUT`             seconds (default: 5)
  `MUSIC_PATH`              host music folder path (mounted into MPD container as /music)
  `MUSIC_PATH_HOST`         Strawberry URL prefix on host (e.g. /media/jack/buffer/audio)
  `MUSIC_PATH_CONTAINER`    MPD music_directory (default: /music — unused in backend)
  `MUSIC_SEARCH_LIMIT`      max search results (default: 20)
  `MUSIC_ARTIST_RADIO_N`    tracks for artist radio (default: 10)

Acceptance:
- **Endpoints:** `POST /music/search`, `POST /music/play`, `POST /music/queue`,
  `POST /music/control`, `GET /music/now_playing`, and `GET /music/queue` are all
  implemented and return standardized responses.
- **Search quality:** `/music/search` returns ranked results from Strawberry's `songs`
  table (LIKE on title/artist/album, ordered by playcount) for title/artist/album queries.
- **Queue model:** Supports single song and artist radio (multi-song by artist).
- **Playback controls:** `/music/play` starts playback via MPD; `/music/queue` appends
  tracks; `/music/control` supports `pause`, `resume`, `next`, and `stop`;
  `GET /music/now_playing` returns current track; `GET /music/queue` returns queued tracks.
- **Voice & UI:** "Play <song or artist>" works end-to-end via voice or text and updates
  the frontend player state including now-playing bar and queue panel.
- **Auto-pick:** Multiple search matches auto-pick the top-ranked result; confidence is
  logged server-side and included in the API response.
- **Artist radio:** "Play <artist>" with no exact match queues N songs by that artist,
  weighted-randomly by playcount. This is the concrete foundation for Phase 8b fallback.
- **Docker & config:** MPD runs as a Docker Compose service with the music folder mounted;
  `STRAWBERRY_DB_PATH`, `MUSIC_PATH_HOST`, and `MPD_HOST` are respected by the backend.
- **Path rewrite:** file:// URIs from Strawberry are decoded and the host prefix stripped
  to produce MPD-relative paths before any MPD add/play call.
- **MPD resilience:** ConnectionError on any command attempts one reconnect; if that fails,
  returns `retryable=True`. Container restart does not cause silent failure.
- **Strawberry lock:** OperationalError during a scan returns `retryable=True`, never crashes.
- **Errors:** All music endpoints return the standardized error shape
  (`error`, `code`, `retryable`) on failures.

---

### Phase 8b (1–2 days, do it when rec engine MVP is ready)
**Estimate: 1–2 days**
**Depends on: Phase 8, recommendation engine MVP**

Goal: Add a music recommendation tool that integrates with the recommendation engine

Tasks:
- Recommend intent route: "play something like X" or "play songs similar to X"
- Thin adapter: call rec engine HTTP endpoint → resolve results against Strawberry DB → queue in MPD
- Graceful fallback if rec engine is unavailable (queue artist radio from Strawberry instead)

Acceptance:
- "Play something like <song/artist>" returns relevant recommendations and queues them in MPD.

---

### Phase 9 — TNG-style voice output (TTS)
**Estimate: 2–4 days**
**Depends on: Phases 2, 4 (and ideally Phase 6 for tool response parity)**

Goal: Assistant responds with spoken audio in a clear, consistent voice
aligned to TNG computer style.

TTS engine choice: implement the pluggable interface first, then benchmark
Piper TTS and Kokoro (open-weight, higher expressiveness ceiling) before
committing to a default. The interface must make the underlying engine
swappable without touching callers. `TTS_ENGINE` env var selects the engine.

Tasks:
- Implement TTS as a backend service with a pluggable engine interface.
  Engines live in `backend/tts/engines/` with a common async `synthesize(text) → bytes`
  interface. `tts.py` selects by `TTS_ENGINE` env var.
- Benchmark Piper and Kokoro on the host hardware for latency and voice quality
  before Phase 8 is marked complete. Document the winner and why.
- Tune speaking rate, pitch, and prosody for a calm, authoritative delivery.
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
- Swapping `TTS_ENGINE` in `.env` changes the engine without code changes.

---

### Phase 10 — LangGraph orchestration
**Estimate: 1–2 weeks**
**Depends on: all prior phases**

Goal: Replace procedural routing with a stateful LangGraph graph. Nodes handle
inner-monologue planning, memory retrieval, tool dispatch, and response generation.
Enables multi-step reasoning, tool chaining, and durable session continuity.

The graph architecture is the foundation for all subsequent intelligence phases
(11 personality, 12 tiered memory, 13 sub-agents). Design the state shape and
node interfaces carefully — retrofitting them later is expensive.

Graph nodes:
```
input
  └── tone_probe              reads user affect from transcript (feeds Phase 9.5)
        └── planner           inner monologue → structured plan (replaces router.py)
              └── memory_retrieval   proactive: pre-loads context before dispatch
                    └── tool_router
                          ├── weather_tool
                          ├── music_tool
                          ├── code_tool          (Phase 10)
                          └── chat_fallback
                                └── responder    generates grounded response
                                      └── persona_renderer   (Phase 9.5)
                                            └── output
```

State shape — include all fields from day one even if not all are used yet:
```python
class AssistantState(TypedDict):
    messages: list[dict]          # full conversation history
    transcript: str               # raw STT output for current turn
    tone: str                     # detected user affect (calm, frustrated, excited…)
    plan: dict                    # planner output: route, tool, needs_memory, reasoning
    memories: list[str]           # retrieved memory snippets
    tool_result: str              # tool output for current turn
    user_prefs: dict              # current user preferences
    session_id: str               # durable session identifier
    persona: dict                 # active persona parameters (Phase 9.5)
```

Implementation notes:
- Use LangGraph v0.2+. Pin the exact version in `requirements.txt` immediately —
  the checkpointer interface changed significantly between v0.1 and v0.2. Test
  checkpoint resume explicitly before marking acceptance.
- Use `SqliteSaver` checkpointer for durable session state.
- Keep graph state inspectable: add a `GET /graph/state/{session_id}` debug endpoint.
- The `planner` node is the upgraded inner-monologue from Phase 4 — not a new
  concept, just the same reasoning pass now running as a named graph node.
- The `tone_probe` and `persona_renderer` nodes are stubs in Phase 9 and
  activated in Phase 9.5. Wire them in from the start to avoid graph refactoring.

Tasks:
- Introduce LangGraph with SqliteSaver checkpointer.
- Migrate existing router and tool logic into named graph nodes.
- Add `tone` and `persona` fields to `AssistantState` as stubs.
- Add `tone_probe` and `persona_renderer` as pass-through stubs.
- Expand ChromaDB integration for semantic memory retrieval tuning.
- Keep graph state inspectable for debugging.
- Ensure all existing tools work through the graph without regression.

Acceptance:
- Existing chat/tool behavior is preserved while routing is graph-driven.
- Graph state is inspectable via debug endpoint.
- Durable session continuity works through LangGraph checkpoints.
- Checkpoint resume is explicitly tested after container restart.
- Stub nodes for tone and persona are wired but pass-through.

---

### Phase 11 — Personality and affect layer
**Estimate: 1–2 weeks**
**Depends on: Phase 10**

Goal: Implement the "best of human qualities" vision — perception of user affect,
consistent personality, warmth, and intuition — as a bounded layer in the graph
that operates strictly post-reasoning and never touches facts or tool outputs.

This is the layer where the assistant acquires personality. The architecture keeps
it safe: the `persona_renderer` receives a factually-correct, grounded response
and applies stylistic transformation only. It cannot alter facts, tool results,
or memory. Tests enforce this boundary explicitly.

Design principles:
- **Personality is post-hoc.** The reasoning/memory/tool stack produces a correct
  answer. The persona renderer then shapes how that answer is expressed — warmth,
  pacing, acknowledgement of the user's state — without changing its content.
- **Affect is input, not output.** `tone_probe` reads the user's emotional register
  from their transcript (word choice, phrasing, urgency) and injects a `tone` field
  into state. The persona renderer uses this to modulate response style — not to
  manufacture emotions in return.
- **Consistency over performance.** The persona should feel like a stable character,
  not a chatbot performing friendliness. Tune system prompt and renderer to be
  calm, direct, and warm rather than effusive.

Nodes activated in this phase:

`tone_probe` (post-transcription):
- Classifies user affect from transcript: calm, curious, frustrated, excited,
  uncertain, urgent.
- Runs locally; output is a single label injected into `AssistantState.tone`.
- Feeds `persona_renderer` at the end of the graph.

`persona_renderer` (post-responder):
- Receives the grounded response text and `AssistantState.tone`.
- Applies a configurable persona system prompt that shapes voice and style.
- Adjusts response pacing and register to match detected user tone.
- Hard constraint: output must contain all facts and tool values from input.
  A test suite verifies no fact drift occurs through the renderer.
- Persona parameters live in `AssistantState.persona` and are user-configurable
  (name, communication style, formality, warmth level).

`proactive_retrieval` upgrade:
- `memory_retrieval` node is upgraded to begin pre-loading relevant memories
  in parallel with transcription, rather than waiting for the planner to finish.
  This is the "intuition" behaviour — relevant context is ready before the
  user finishes speaking.

TTS affect bridge (optional, if Phase 8 engine supports it):
- Pass `tone` to the TTS engine to modulate prosody — slightly warmer pacing for
  excited tones, calmer cadence for frustrated ones. Only implement if the chosen
  TTS engine exposes prosody controls.

Tasks:
- Activate `tone_probe` node: implement affect classifier (local model, ~50 token
  output), wire `tone` into state.
- Activate `persona_renderer` node: implement post-hoc style transform with
  configurable persona parameters.
- Add persona configuration to user preferences in memory layer.
- Add fact-drift test suite: verify persona renderer output contains all factual
  content from responder output.
- Upgrade `memory_retrieval` to proactive parallel pre-loading.
- Optionally bridge `tone` to TTS prosody if engine supports it.
- Add UI controls for persona configuration (name, style, warmth).

Acceptance:
- Responses feel consistent in personality across different topics and sessions.
- Tone probe correctly identifies frustrated, excited, and neutral user inputs.
- Persona renderer never drops or alters facts, tool results, or memory values
  (verified by automated tests).
- Proactive memory retrieval reduces perceived latency on memory-heavy queries.
- Persona parameters survive container restart (stored in preferences memory).

---

### Phase 12 — Tiered memory and automatic consolidation
**Estimate: 1–2 weeks**
**Depends on: Phase 11**

Goal: Upgrade the flat SQLite memory layer to a three-tier memory architecture
that automatically promotes important information from ephemeral session context
into durable long-term facts. This mirrors how human memory works — frequent,
important, or explicitly stated things become durable; the rest fades.

Memory tiers:

```
Tier 1 — Working memory (in-context)
  Current turn messages + retrieved snippets injected into the prompt.
  Managed by: AssistantState.messages + AssistantState.memories
  Lifetime: single turn

Tier 2 — Episodic memory (session summaries)
  Compressed summaries of past sessions. When a session ends or grows beyond
  the token budget, the model summarises the key points into an episodic record.
  Managed by: SQLite summaries table
  Lifetime: until explicitly forgotten or consolidated

Tier 3 — Semantic memory (long-term facts)
  Stable facts, preferences, and patterns extracted from episodic records
  by the consolidation process.
  Managed by: SQLite facts + ChromaDB vectors
  Lifetime: indefinite (user-deletable)
```

Consolidation process:
- A background task runs after each session ends (or on a schedule).
- It reads recent episodic records and extracts candidate facts using a local
  model prompt: "What stable facts about the user or their preferences can be
  extracted from this session?"
- Candidates are filtered through the sensitivity matrix (Phase 5) before write.
- Extracted facts are embedded into ChromaDB for semantic retrieval.
- The episodic record is retained for a configurable period then pruned.

This replaces the Phase 3 summarization stub (⏳) with a proper tiered design.

Tasks:
- Add `episodic` table to SQLite schema: `(id, session_id, summary, created_at, consolidated)`.
- Implement end-of-session summarization: local model compresses session into
  episodic record.
- Implement consolidation task: extracts semantic facts from episodic records,
  filters through sensitivity matrix, writes to facts + ChromaDB.
- Add consolidation schedule (configurable: on session end, daily, or manual).
- Expose episodic records in the memory viewer UI alongside facts.
- Add `GET /memory/episodic` endpoint.
- Upgrade `memory_retrieval` node to query all three tiers and rank results.

Acceptance:
- Facts stated in session A are retrievable by semantic search in session B
  without the user restating them.
- Sensitive items are not auto-consolidated (sensitivity matrix enforced).
- User can view episodic records and delete them individually.
- Consolidation task runs without blocking the main request path.
- Memory viewer shows tier labels (working / episodic / semantic) for each item.

---

### Phase 13 — Local code assistant on LangGraph
**Estimate: 1–2 weeks**
**Depends on: Phase 12**

Goal: Add a dedicated code assistant node to the LangGraph flow so code tasks use
the local coder model with project context, explicit write confirmation, and
voice-friendly interaction.

The code assistant node is built into the graph rather than as a standalone VS Code
extension. This is the right call: the agent already owns memory, session state,
tool access, and voice input. Voice-triggered code generation ("Computer, write a
FastAPI endpoint that calls the weather tool") with project context from memory is
substantially more capable than an isolated sidebar model.

Code tool design decisions:
- Model selection: pull a code-specialized model into Ollama alongside the chat
  model (for example `qwen2.5-coder`). Route `code` intents here by default.
- File access: implement `backend/tools/code.py` with workspace-root enforcement;
  never allow path traversal outside configured root.
- Context injection: fetch relevant file summaries/schemas/decisions from memory
  and inject them into coder prompts for code intents.
- Voice ergonomics: summarize generated code in spoken form, with full code shown
  in chat and optional write-to-disk flow.

Tool interface (consistent with other tools):
```python
# backend/tools/code.py
async def run(params: dict) -> dict:
    # params: { "task": str, "files": list[str], "context": str, "write": bool }
    # returns: { "code": str, "language": str, "written_to": str | None }
```

Endpoints:
```
POST /code               — generate or edit code (streaming)
GET  /code/files         — list workspace files
GET  /code/files/{path}  — read a file
PUT  /code/files/{path}  — write a file (requires confirmation flag)
```

Tasks:
- Pull and configure a code-specialized Ollama model; add `OLLAMA_CODER_MODEL` env var.
- Implement `code_tool` node with read/write support and confirmation-gated writes.
- Wire code context retrieval into memory_retrieval for code intents.
- Add voice confirmation flow for file writes.
- Update UI with syntax highlighting, copy button, and save-to-disk actions.

Acceptance:
- "Computer, write a function that does X" generates code with relevant project context.
- "Computer, continue that function from yesterday" resolves via checkpointed graph state.
- File writes require explicit confirmation before touching disk.
- Code tasks do not silently route to cloud unless local coder model is unavailable.

---

### Phase 14 — Sub-agent architecture
**Estimate: 2–4 weeks**
**Depends on: Phase 13**

Goal: For complex multi-step tasks, the assistant spawns specialist sub-agents
rather than handling everything in a single monolithic response. This enables
parallel execution, specialization, and self-correction through a critic loop.

This is the natural evolution once Phase 10 (code agent) is stable: the planner
node can decompose a complex request into sub-tasks and assign each to a specialist.

Sub-agent roles (initial set):
- `researcher` — retrieves information from memory, tools, and web search
- `coder` — generates, reviews, and edits code using the coder model
- `critic` — reviews proposed responses or code for correctness and completeness
- `summarizer` — compresses long content for injection into working memory

Orchestration pattern:
```
planner
  └── task_decomposer         breaks complex request into sub-tasks
        ├── researcher         runs in parallel
        ├── coder              runs in parallel
        └── critic             reviews outputs from other agents
              └── synthesizer  assembles final response from sub-agent outputs
```

Design principles:
- Sub-agents are LangGraph subgraphs, not separate processes. They share the
  parent graph's state and checkpointer.
- Each sub-agent has a maximum turn budget to prevent runaway loops.
- The critic agent can reject and re-run another agent's output, but only once
  per sub-task to prevent infinite loops.
- All sub-agent reasoning is logged and inspectable.
- Sub-agents do not have direct user-facing output — only the synthesizer does.

Tasks:
- Design sub-agent state schema and parent↔child state passing contract.
- Implement `task_decomposer` node: local model decomposes request into typed
  sub-tasks with assigned agent roles.
- Implement `researcher`, `coder`, and `critic` sub-agent subgraphs.
- Implement `synthesizer` node: assembles final response, resolves conflicts
  between sub-agent outputs.
- Add sub-agent observability: turn counts, agent assignments, critic rejections
  visible in debug endpoint.
- Add `MAX_AGENT_TURNS` env var; default to a safe low value (e.g. 5).

Acceptance:
- "Research X, then write a Python script that does Y based on what you find"
  executes researcher and coder in parallel then synthesizes.
- Critic rejects and re-runs a sub-task at most once.
- Sub-agent turn budget is enforced; runaway loops are impossible.
- Simple queries that don't need decomposition skip straight to the single-agent
  path without overhead.

---

## Cross-cutting standards

### Security and privacy
- Local-first default. No data leaves the device unless the user triggers a
  cloud model call or an external tool.
- HTTPS on LAN (Phase 0b) is required before any mobile deployment.
- Memory writes policy: implicit save is allowed for clear non-sensitive
  preferences/facts; explicit consent is required for sensitive items.
- Memory policy matrix is mandatory in implementation (auto-save vs confirm-first
  vs explicit-only/blocked) and must be user-visible.
- Persona renderer is prohibited from modifying facts or tool-derived values.
  This is enforced by automated tests, not convention.
- Explicit user confirmation before writing any file to disk (code tool).
- Code tool workspace root is enforced — no path traversal outside configured dir.
- Redact API keys, tokens, and personal data from all logs.
- Never commit `.env` to git.

### Reliability
- All services bind to `0.0.0.0` in Docker.
- Structured logs for: routing decisions, model selected, inner-monologue output,
  wake-word scores, transcription results, tool calls, sub-agent turns, and errors.
- Retry with exponential backoff for transient network failures.
- Standardized error response shape: `{ "error": str, "code": str, "retryable": bool }`.
- Container restarts must surface errors in logs — never swallow startup failures.
- LangGraph version must be pinned; test checkpoint resume after any version bump.

### Code conventions
- All frontend API calls use relative paths. No hardcoded hosts or ports in
  runtime code.
- New tools are added as modules under `backend/tools/` with a consistent
  interface: `async def run(params: dict) -> dict`.
- TTS engines live in `backend/tts/engines/` with a common
  `async def synthesize(text: str) -> bytes` interface. `TTS_ENGINE` env var selects.
- Environment variables are the only configuration mechanism. No config files
  that duplicate `.env` values.
- Type-annotate all Python function signatures.
- Keep `requirements.txt` up to date. Pin major versions once a phase is stable.
- `OLLAMA_CHAT_MODEL` and `OLLAMA_CODER_MODEL` are separate env vars — never
  hardcode model names in source.

### Testing priorities
- Backend: `/chat`, `/transcribe`, `/ws/wake`, `/weather`, `/code`, `/health`.
- Frontend: streaming markdown render, code block render + copy/save, mic toggle
  state machine, WebSocket reconnect behavior.
- Memory-specific: deterministic save/forget commands, sensitivity matrix behavior,
  retrieval relevance across new sessions, tier promotion correctness.
- **Persona-boundary safety (critical)**: style/tone variation tests must prove
  that factual content and tool-derived values are identical before and after
  persona rendering. No fact drift is acceptable. This test suite is mandatory
  before Phase 9.5 acceptance.
- Sub-agent tests: decomposition correctness, critic loop termination, turn budget
  enforcement.
- LAN matrix: verify from desktop browser, mobile browser, and tablet after
  every Phase 1–3 change and after Phase 0b.
- Add tests before marking any phase acceptance criteria as done.

---

## Immediate next sprint

| # | Issue | Status | Estimate | Depends on |
|---|-------|--------|----------|------------|
| 0b | HTTPS on LAN: Caddy reverse proxy, mkcert, wss:// | ✅ done | 0.5 days | 1 |
| 1 | LAN-safe single-origin serving and health checks | ✅ done | 1–2 days | — |
| 2 | Replace all `localhost` frontend paths with relative paths | ✅ done | 2–4 hours | 1 |
| 3 | Harden wake-word pipeline (Phase 2) | ✅ done | 1–2 days | 1, 2 |
| 4 | Chat context management: bounded session buffer + token-aware context | 🔄 in progress | 1–2 days | 1 |
| 4b | Upgrade router to inner-monologue reasoning pass (structured JSON output) | ✅ done | 1–2 days | 4 |
| 5 | Upgrade router: intent categories + `code` route, confidence scoring, telemetry | ✅ done | 1–2 days | 1, 4 |
| 6 | SQLite memory layer: schema, policy matrix, explicit controls, CRUD endpoints | ✅ done | 2–3 days | 5 |
| 7 | Chat sessions sidebar: create/switch/reset sessions UI | ✅ done | 1 day | 6 |
| 8 | Weather tool: adapter, memory-backed default location, error handling | ✅ done | 1 day | 6, 7 |
| 9 | Local music library tool: media indexer, search/playback endpoints, voice controls | 🔲 | 3–5 days | 2, 5 |
| 10 | TTS voice output: benchmark Piper vs Kokoro, pluggable engine, barge-in | 🔲 | 2–4 days | 2, 4 |
| 11 | LangGraph migration: graph orchestration with tone + persona stubs | 🔲 | 1–2 weeks | all prior |
| 12 | Local code assistant node on LangGraph (coder model, code tool, write-confirm) | 🔲 | 1–2 weeks | 11 |
| 13 | Personality and affect layer: tone_probe, persona_renderer, proactive retrieval | 🔲 | 1–2 weeks | 11 |
| 14 | Tiered memory: episodic records, consolidation task, semantic promotion | 🔲 | 1–2 weeks | 11 |
| 15 | Sub-agent architecture: decomposer, researcher, coder, critic, synthesizer | 🔲 | 2–4 weeks | 12 |

Each numbered issue maps to a phase above. Use these as GitHub issue titles and
reference the phase section for full task and acceptance detail.
