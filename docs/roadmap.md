## Development order

Features are ordered to minimize rework. Each phase builds on stable foundations
from the phase before it.

---

## Current project context

When this section conflicts with historical roadmap notes below, follow this section.

- Phases 1–12 are complete.
- Phase 8 includes the deterministic music pre-router (`_parse_music_command`) that bypasses the LLM for clear music commands, compound title+artist search, and year/decade range playback.
- Phase 9 TTS is complete (Piper + Kokoro engines, `/tts` endpoint, barge-in, voice SSE metadata).
- Phase 10a LangGraph migration is complete (graph skeleton, checkpointing, all nodes wired, `/graph/state` endpoint, checkpoint resume test passing).
- Phase 10b code tool node is complete (ReAct loop, tree-sitter indexer, ChromaDB code_context collection, confirmation-gated writes, workspace-root enforcement, /code endpoints). Main-chat code remains scoped to voice-driven code *questions*; project-scoped coding work is handled by the Projects roadmap (`roadmap.4.md`).
- Phase 10c responder/modality split is complete (voice compression, fact-drift test, tone field wired as nullable).
- Phase 10d ChromaDB cleanup is complete (conversation_memory collection, auto-migration from assistant_memories, consolidated column on summaries table, 6 isolation tests passing).
- Phase 11 is complete (shipped as a single Hearth character prompt in `backend/hearth_prompt.txt`; tone_probe, persona_renderer, /persona endpoints, and persona UI panel were removed).
- Phase 12 is complete (three-tier memory: working/episodic/semantic; consolidation worker; tiered retrieval; `GET /memory/episodic`, `POST /memory/consolidate`; tier badges in memory panel). The consolidation worker currently uses regex-based candidate extraction — this is the known gap addressed in Phase 12b.
- Phase 12b is complete
- Phase 13 (external coding-agent integration) is retired. Its reusable pieces were kept and repurposed for the internal project-scoped coding-agent path in `roadmap.4.md` (confirmation gate flow, project write routing, code-context injection, and result shaping).
- Active models: `gemma:e4b` (chat) and `qwen2.5-coder:14b` (code) are both pulled and verified on this machine.
- Wake-word voice is stable on desktop/Linux. Treat Android/mobile voice as requiring an HTTPS-capable LAN edge before calling it complete.

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
  mobile browsers. Unblocked by Phase 3 (Caddy HTTPS).

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
- ✅ Android voice — validated after Phase 3 (Caddy HTTPS).

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
can access microphone and other secure-context browser APIs.

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
  tier introduced formally in Phase 12.
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
needs before committing to a route.

Design:

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

Intent categories (produced by monologue, not hardcoded):
- `quick-local` — factual, short, conversational
- `reasoning-heavy` — multi-step planning, analysis, architecture
- `code` — code questions, debugging, explanation (not file writes — see locked decisions)
- `external-data-needed` — weather, news, live data (tool path, not cloud)
- `memory-needed` — references to prior facts or user preferences

Tasks — all complete:
- ✅ Replace keyword heuristics in `router.py` with a lightweight intent classifier
  (prompt-based using the local model itself).
- ✅ Add a confidence threshold: if local model confidence is below threshold,
  escalate to cloud.
- ✅ Upgrade classifier to a full inner-monologue reasoning pass that outputs a
  structured routing JSON rather than a single intent label.
- ✅ Route `code` intent to the local coder model rather than the general chat model or cloud.
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

Schema (start minimal, extend as needed):
```sql
facts        (id, key, value, source, created_at, expires_at)
preferences  (id, key, value, updated_at)
summaries    (id, session_id, summary, created_at)
```

Tasks — all complete:
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

Acceptance:
- ✅ Stated preferences survive container restart.
- ✅ User can view, delete individual items, and clear all memory from the UI.
- ✅ Memory snippets appear in context without bloating the prompt.
- ✅ Explicit save/forget commands behave deterministically and return clear status.
- ✅ Sensitive and confirm-first classes follow policy matrix consistently.
- ✅ Sessions sidebar supports starting and switching chat sessions.

---

### Phase 7 — Weather tool
**Status: complete**
**Estimate: 1 day**
**Depends on: Phase 6**

Goal: First external tool integration. Validates the tool-routing architecture
cleanly before adding more complex tools.

Tasks — all complete:
- ✅ Build a weather provider adapter with a normalized response schema.
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
**Status: complete**
**Estimate: 3–5 days**
**Depends on: Phases 2, 6**

Goal: Provide backend music search and playback integration using MPD and the
Beets music library, exposing HTTP endpoints and voice and text commands so
users can search, queue, and control playback from the UI or via voice.

Implementation notes (decisions made during build):
- **Beets DB schema**: `items` table; `id` column is the integer primary key.
  `path` column stores raw filesystem paths (may be `bytes` in older Beets
  versions). Columns `genre`, `year`, and `rating` are optional — presence is
  checked at runtime via `PRAGMA table_info(items)` with `lru_cache`.
- **Path rewrite**: Beets stores raw filesystem paths. The backend decodes
  bytes→str with utf-8/surrogateescape fallback, then strips the `MUSIC_ROOT`
  prefix (env var, default `/music`) to produce MPD-relative paths
  (e.g. `artist/song.mp3`) which MPD resolves against its `music_directory`.
- **Ambiguity**: auto-pick the top-ranked LIKE result and log the confidence score
  server-side. Ranked candidate list is preserved in API response for future UX.
- **MPD resilience**: per-request fresh MPD connection with one reconnect attempt.
  On second failure, return retryable ToolResult — never crash or silently drop.
- **DB lock handling**: open Beets DB with `timeout=5, check_same_thread=False,
  uri=True` (read-only URI mode). Wrap reads in `try/except sqlite3.OperationalError`
  and return `retryable=True` on lock errors.
- **Rating**: Beets `rating` column (float 0–1) is used for ordering and radio
  weighting. Falls back to uniform weight `0.01` when column is absent.
- **Artist radio**: implemented in Phase 8 core as weighted-random by `rating`,
  seeded for determinism. This is the Phase 8b fallback entry point.
- **Startup bootstrap**: on first run, if Beets DB is empty/missing, the backend
  automatically calls `beet import -A <MUSIC_ROOT>` — offline, tag-only, no
  MusicBrainz. Subsequent starts detect a populated DB and skip the import.
- **Migration**: `backend/scripts/migrate_strawberry_playcount_to_beets.py`
  copies Strawberry playcounts into Beets `rating` values (log-scale, 0–1).
  Run with `--apply` to persist; dry-run by default.
- **Deterministic music pre-router**: `_parse_music_command()` in `main.py` fires
  BEFORE `router_route()` in the `/chat` endpoint. High-confidence music commands
  (control, now-playing, queue-view, explicit play/queue with concrete target) bypass
  the LLM entirely — tool is dispatched directly and a one-sentence plain-text
  response is formatted by `_format_music_response()` with zero LLM queries.
  Vague/ambiguous requests ("play something chill", "play something like X") fall
  through to the existing LLM path.
- **Compound title+artist search**: `_sync_search_by_title_artist(title, artist)` uses
  `title LIKE ? AND artist LIKE ?` for "play X by Y" commands — more precise than the
  general LIKE search. Falls back to artist radio if no match.
- **Year/decade search**: `_sync_search_by_year_range(year_start, year_end)` queries
  the Strawberry `year` column. Parser handles "80s" → 1980–1989, "2003" → exact year.
  Queues N random tracks from the pool, weighted-randomly by playcount.

Tasks — all complete:
- ✅ Add MPD as a Docker Compose service with music folder mounted.
- ✅ Mount Strawberry DB directory read-only into backend container; set `STRAWBERRY_DB_PATH`.
- ✅ Implement Strawberry search in `backend/tools/music.py` using LIKE on title/artist/album,
  ranked by playcount. No FTS — use rowid as track ID.
- ✅ Implement MPD client in `backend/tools/music.py` with per-request connections and
  reconnect-once resilience: play(url), pause(), resume(), stop(), next(), queue(url),
  now_playing(), queue_view().
- ✅ Queue model: single song, multi-song (artist radio).
- ✅ Add path rewrite: URL-decode Strawberry `file://` URIs → strip `MUSIC_PATH_HOST`
  prefix → pass relative path to MPD.
- ✅ Implement `artist_radio(artist, n)` using weighted-random by playcount (Phase 8b dep).
- ✅ Wrap all Strawberry reads in `OperationalError` handling; return retryable errors.
- ✅ Add `GET /music/now_playing` endpoint.
- ✅ Add `GET /music/queue` endpoint for queue visibility.
- ✅ Add `_parse_music_command()` deterministic pre-router in `main.py` (fires before LLM).
- ✅ Add `_format_music_response()` to generate plain-text replies without LLM for music.
- ✅ Add `_sync_search_by_title_artist()` for compound title+artist queries.
- ✅ Add `_sync_search_by_year_range()` for decade/year playback.
- ✅ Wire `artist_filter` and `year_range` params into `music.run()`.

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
- ✅ **Endpoints:** `POST /music/search`, `POST /music/play`, `POST /music/queue`,
  `POST /music/control`, `GET /music/now_playing`, and `GET /music/queue` are all
  implemented and return standardized responses.
- ✅ **Search quality:** `/music/search` returns ranked results from Strawberry's `songs`
  table (LIKE on title/artist/album, ordered by playcount) for title/artist/album queries.
- ✅ **Queue model:** Supports single song and artist radio (multi-song by artist).
- ✅ **Playback controls:** `/music/play` starts playback via MPD; `/music/queue` appends
  tracks; `/music/control` supports `pause`, `resume`, `next`, and `stop`;
  `GET /music/now_playing` returns current track; `GET /music/queue` returns queued tracks.
- ✅ **Voice & UI:** "Play <song or artist>" works end-to-end via voice or text and updates
  the frontend player state including now-playing bar and queue panel.
- ✅ **Auto-pick:** Multiple search matches auto-pick the top-ranked result; confidence is
  logged server-side and included in the API response.
- ✅ **Artist radio:** "Play <artist>" with no exact match queues N songs by that artist,
  weighted-randomly by playcount. This is the concrete foundation for Phase 8b fallback.
- ✅ **Docker & config:** MPD runs as a Docker Compose service with the music folder mounted;
  `STRAWBERRY_DB_PATH`, `MUSIC_PATH_HOST`, and `MPD_HOST` are respected by the backend.
- ✅ **Path rewrite:** file:// URIs from Strawberry are decoded and the host prefix stripped
  to produce MPD-relative paths before any MPD add/play call.
- ✅ **MPD resilience:** ConnectionError on any command attempts one reconnect; if that fails,
  returns `retryable=True`. Container restart does not cause silent failure.
- ✅ **Strawberry lock:** OperationalError during a scan returns `retryable=True`, never crashes.
- ✅ **Errors:** All music endpoints return the standardized error shape
  (`error`, `code`, `retryable`) on failures.
- ✅ **LLM bypass:** Clear music commands (control, now-playing, explicit play/queue) never
  reach the LLM planner or summarizer — deterministic pre-router handles them in `/chat`.
- ✅ **Title+artist search:** "Play X by Y" uses compound LIKE filter; falls back to
  artist radio if no match found.
- ✅ **Year/decade:** "Play 80s", "play 1994", "play music from 2003" resolve to
  year-range queries against Strawberry `year` column; N tracks queued randomly.

---

### Phase 8b — Music recommendations via external engine
**Estimate: 1–2 days**
**Depends on: Phase 8, external recommendation engine MVP**

Goal: Add a music recommendation tool that integrates with an external recommendation
engine as a thin adapter. This is the same pattern as the future coding agent
integration — Hearth calls the service, formats the result, nothing more.

Tasks:
- Recommend intent route: "play something like X" or "play songs similar to X"
- Thin adapter in `backend/tools/music_rec.py`: call rec engine HTTP endpoint →
  resolve results against Strawberry DB → queue in MPD
- Graceful fallback if rec engine is unavailable: queue artist radio from Strawberry instead
- Follow the same `async def run(params: dict) -> dict` tool interface

Acceptance:
- "Play something like <song/artist>" returns relevant recommendations and queues them in MPD.
- If rec engine is unreachable, falls back to artist radio with a user-visible note.

---

### Phase 9 — TNG-style voice output (TTS)
**Status: complete**
**Estimate: 2–3 days**
**Depends on: Phases 2, 4**

Goal: Implement a pluggable TTS engine and wire end-to-end audio playback with
barge-in support.

TTS engine interface:
- Pluggable engines live in `backend/tts/engines/` with a common
  `async synthesize(text: str) → bytes` interface.
- `backend/tts.py` selects engine by `TTS_ENGINE` env var.
- Engines handle speaker/voice selection internally (via env vars or config).

Tasks — all complete:
- ✅ Implement `backend/tts/` package with pluggable engine loader and common `synthesize()` method.
- ✅ Implement a Piper TTS engine in `backend/tts/engines/piper.py`.
- ✅ Implement a Kokoro TTS engine in `backend/tts/engines/kokoro.py`.
- ✅ Add benchmark harness for engine latency/availability and document rationale.
- ✅ Add a TTS endpoint: `POST /tts` that returns audio bytes (`audio/wav`).
- ✅ Wire voice-source metadata into `/chat` SSE so frontend can trigger `/tts` post-response.
- ✅ Frontend: implement playback widget with autoplay fallback and manual enable path.
- ✅ Add barge-in behavior so wake-word interruption stops assistant playback.
- ✅ Document runtime env/settings and provide `scripts/download-tts-models.sh` for Kokoro assets.
- ✅ Add backend regression coverage for `/chat` stream behavior and TTS endpoint mappings.

Acceptance:
- ✅ `/tts` endpoint returns audio bytes for arbitrary text input.
- ✅ Responses are read aloud end-to-end for voice-origin chat turns.
- ✅ Voice playback does not block chat use or subsequent message sending.
- ✅ Barge-in stops playback and resumes listening flow.
- ✅ Swapping `TTS_ENGINE` in `.env` changes the engine without code changes.
- ✅ Browser autoplay constraints are handled (auto-play muted, manual enable fallback).

---

### Phase 10a — Graph skeleton and router migration
**Status: complete**
**Estimate: 3–5 days**
**Depends on: all prior phases**

Goal: Introduce LangGraph with durable checkpointing and migrate existing router
logic into a stateful graph. Zero feature additions — all behavior must pass
through the graph with no regression.

Design notes:
- The deterministic music pre-router (`_parse_music_command()` in the HTTP handler)
  stays in place, by design. It guards the graph invocation for high-confidence music
  commands and is never moved into the graph.
- Pin LangGraph version immediately in `requirements.txt` — the checkpointer
  interface changed significantly between v0.1 and v0.2.
- Use `SqliteSaver` for durable session state.

Tasks — all complete:
- ✅ Create `backend/graph.py` with LangGraph StateGraph definition and SqliteSaver checkpointer.
- ✅ Pin LangGraph version in `requirements.txt` (v0.2+) and document the version constraint.
- ✅ Migrate `router.py` intent classification logic into the `intent_classifier` node.
- ✅ Migrate memory retrieval into the `memory_retrieval` node (call existing `memory.py` APIs).
- ✅ Migrate tool routing logic into the `tool_router` node (weather, music, chat fallback).
- ✅ Implement a simple `responder` node that passes through tool results or chat responses unchanged.
- ✅ Wire the `/chat` endpoint in `main.py` to invoke the graph instead of calling `router.route()`.
- ✅ Keep the HTTP-level `_parse_music_command()` guard in place — it fires before graph invocation.
- ✅ Add a `GET /graph/state/{session_id}` debug endpoint for state inspection.
- ✅ Implement checkpoint resume test: verify that after a graph invocation and container restart,
  loading the session resumes with correct state (not a full re-execution).

Acceptance:
- ✅ All existing chat, tool (weather, music), and memory behavior passes through the graph
  with zero functional regression.
- ✅ Graph state is inspectable via debug endpoint.
- ✅ Checkpoint save/resume works correctly; state persists across container restart.
- ✅ The deterministic music pre-router (`_parse_music_command`) still guards the graph
  and handles high-confidence music commands without graph invocation.
- ✅ LangGraph version is pinned in requirements.txt with documented rationale.
- ✅ Checkpoint resume test passes explicitly before Phase 10a is marked complete.

---

### Phase 10b — Code tool node
**Status: complete (main-chat code questions only)**
**Estimate: 1 week**
**Depends on: Phase 10a**

Goal: Add a dedicated code assistant node to the graph with a ReAct loop, tree-sitter
summaries, and ChromaDB code context injection. Main-chat writes are intentionally
not expanded here; project-scoped coding flow lives in `roadmap.4.md`.

Design notes:
- The `code_tool` node answers voice-driven code questions using project context
  retrieved from tree-sitter summaries stored in ChromaDB `code_context` collection.
- File writes are NOT expanded further for main chat. The confirmation-gated flow
  built in this phase is reused by project-scoped coding work.
- The ReAct loop, tree-sitter indexer, and code_context ChromaDB collection remain
  foundational for project-scoped coding in `roadmap.4.md`.

Tasks — all complete:
- ✅ Add `active_files: list[str]` and `code_context: str` to `AssistantState`.
- ✅ Build tree-sitter summary extraction (function signatures, class hierarchies, import graphs).
- ✅ Create a ChromaDB `code_context` collection separate from conversation memory.
- ✅ Implement `code_context_retrieval()` that queries ChromaDB and formats slices for injection.
- ✅ Implement the `code_tool` node with `create_react_agent` and langchain-community tools.
- ✅ Add workspace-root validation: all file operations must stay within configured boundary.
- ✅ Implement confirmation-gated writes: generate the diff, present to user, wait for approval.
- ✅ Add `tool_router` conditional to detect code intents and dispatch to `code_tool`.
- ✅ Add `/code` endpoints for streaming code generation, file reads, and file writes
  (with confirmation requirement).

Acceptance:
- ✅ Code questions ("explain this function", "how does X work") are answered using
  tree-sitter context retrieved from ChromaDB.
- ✅ Code intents are routed to the local coder model (`OLLAMA_CODER_MODEL`).
- ✅ Tree-sitter summaries are indexed and retrievable; context injection is working.
- ✅ Workspace-root boundary is enforced on every file operation.

---

### Phase 10c — Responder node and voice/chat modality split
**Status: complete**
**Estimate: 2–3 days**
**Depends on: Phase 10a, Phase 9**

Goal: Add the responder node with modality-aware output shaping. Voice responses
are compressed and natural; chat responses are full and detailed.

Design notes:
- The responder node receives a factual, tool-grounded response and shapes it
  based on the `modality` state field.
- For `modality="voice"`: compress to brief, natural spoken output; optimize for
  ear (not eye). Hand off to TTS engine.
- For `modality="chat"`: return full markdown with proper formatting, details, and links.

Tasks — all complete:
- ✅ Add `modality: str` (values: "voice" or "chat") to `AssistantState`.
- ✅ Add logic to the `/chat` endpoint to set `modality` based on request source.
- ✅ Implement the `responder` node with modality-aware response shaping:
  - For voice: compression prompt extracts key points, aims for 20-30% of original length
    while preserving facts.
  - For chat: return the full response as-is with markdown formatting.
- ✅ Wire the responder node into the graph flow (comes after tool_router).
- ✅ For voice responses, pipe the shaped output to the TTS engine (`/tts` endpoint).
- ✅ Return both text and audio to the frontend for voice flows.
- ✅ Add a test that verifies voice responses contain all factual content from the
  original response (no fact drift through compression).

Acceptance:
- ✅ Voice responses are concise and natural while chat responses are full and detailed.
- ✅ Responder node correctly routes output based on `modality` state field.
- ✅ Voice response compression preserves all factual content (verified by test).
- ✅ TTS integration works end-to-end: response text is compressed, synthesized,
  and returned with audio stream.
- ✅ Audio playback in frontend works for voice flows; barge-in stops playback.

---

### Phase 10d — ChromaDB collection architecture cleanup
**Status: complete**
**Estimate: 1–2 days**
**Depends on: Phase 10b, Phase 10c**

Goal: Separate conversation memory collections from code context collections
in ChromaDB. Verify no retrieval cross-contamination.

Design notes:
- Two ChromaDB collections: `conversation_memory` (facts, episodic summaries,
  preferences) and `code_context` (tree-sitter summaries, function signatures,
  import graphs).
- Retrieval for code intents queries only `code_context`.
- Retrieval for general intents queries only `conversation_memory`.

Tasks — all complete:
- ✅ Audit existing ChromaDB usage and identify all stored collections and metadata schemas.
- ✅ Create a separate `code_context` collection for tree-sitter summaries (done in 10b; confirmed isolated).
- ✅ Ensure `conversation_memory` collection stores only facts, preferences, and episodic summaries.
- ✅ Update `memory_retrieval` node to query only `conversation_memory` (with structured log per collection).
- ✅ Update `code_context_retrieval()` to query only `code_context`.
- ✅ Write tests that verify cross-intent queries do not contaminate results.
- ✅ Formalize the summaries table (`id, session_id, summary, created_at, consolidated`) as the backing store for episodic memory tier.
- ✅ Document the ChromaDB schema and migration path for existing instances.

Implementation notes:
- `assistant_memories` → `conversation_memory` rename with non-destructive auto-migration on startup.
- Live-instance migration for `consolidated` column: `ALTER TABLE summaries ADD COLUMN` wrapped in `try/except OperationalError`.
- `save_summary(user_id, session_id, summary) -> int` helper added to `MemoryStore` as Phase 12 hook.
- 6 new tests in `backend/tests/test_chroma_isolation.py` (all passing).

Acceptance:
- ✅ Two ChromaDB collections exist with clear separation: `conversation_memory` and `code_context`.
- ✅ Code context does not pollute chat memory retrieval; conversational facts do not
  appear in code context queries.
- ✅ Summaries table is formalized in schema with documented columns and purpose.
- ✅ Tests verify retrieval isolation: queries to one collection do not leak into the other.
- ✅ ChromaDB schema is documented for future migrations and maintenance.

---

### Phase 11 — Personality and affect layer
**Status: complete (shipped as system prompt)**
**Estimate: 1–2 weeks**
**Depends on: Phase 10**

Goal: Give Hearth a consistent, recognisable personality without adding runtime
overhead or fragile post-processing nodes.

Design decision: Replaced the full `tone_probe` + `persona_renderer` + `/persona` UI
architecture with a single, well-crafted system prompt loaded from `backend/hearth_prompt.txt`.

Rationale:
- A 500-word character prompt injected into every LLM call costs zero extra
  model calls and produces personality that is baked in rather than bolted on.
- `tone_probe` added a full LLM call per turn with no measurable improvement.
- `persona_renderer` added another post-hoc LLM call and introduced fact-drift risk.
- Personality tuning via a text file is faster to iterate than graph node config.

Implementation:

`_load_hearth_prompt()` in `main.py`:
- Reads `backend/hearth_prompt.txt` at startup; falls back to `CHAT_DEFAULT_SYSTEM_PROMPT`
  env var, then a hardcoded placeholder.
- Result is stored in `CHAT_DEFAULT_SYSTEM_PROMPT` — the existing default used by
  `ChatRequest.system` and `CodeRequest.system`. No other code changed.
- Edit the file, restart, personality updates immediately.

`backend/hearth_prompt.txt` (character definition):
- Warmth, intuition, clarity, honesty — 300 words covering personality, affect
  awareness, and per-modality response style (voice: short sentences; chat:
  headers + depth; music: just do it).
- Git-tracked alongside the code.

Removed from previous implementation:
- `tone_probe` node and `_TONE_LABELS` / `_TONE_PROBE_SYSTEM` constants.
- `persona_renderer` node and `_PERSONA_VOICE_SYSTEM` / `_PERSONA_CHAT_SYSTEM` constants.
- `AssistantState.tone` and `AssistantState.persona` fields.
- `/persona` GET and POST endpoints, `PersonaConfigRequest` model, `_PERSONA_STYLE_VALUES`.
- Persona sidebar panel from the frontend UI (`persona.js` deleted, HTML removed).

Tasks — all complete:
- ✅ Write Hearth character prompt (`backend/hearth_prompt.txt`).
- ✅ Implement `_load_hearth_prompt()` loader in `main.py` with file → env → hardcoded fallback.
- ✅ Remove `tone_probe`, `persona_renderer`, `/persona` endpoints, and all related
  state fields, constants, and UI.
- ✅ Update `test_persona.py` to verify prompt loading mechanics.
- ✅ Update `test_responder_modality.py` to remove tone-probe call assumptions.

Acceptance:
- ✅ `CHAT_DEFAULT_SYSTEM_PROMPT` is loaded from `hearth_prompt.txt` at startup (verified by test).
- ✅ `CODE_DEFAULT_SYSTEM_PROMPT` is loaded from `hearth_coder_prompt.txt` at startup.
- ✅ All LLM calls receive the Hearth character prompt via `augmented_system`.
- ✅ No extra model calls per turn for personality.
- ✅ Personality is editable by updating `hearth_prompt.txt` and restarting.
- ✅ No fact-drift risk from post-hoc persona rendering.

---

### Phase 12 — Tiered memory and automatic consolidation
**Status: complete**
**Estimate: 1–2 weeks**
**Depends on: Phase 11**

Goal: Upgrade the flat SQLite memory layer to a three-tier memory architecture
that automatically promotes important information into durable long-term facts.

Memory tiers:

```
Tier 1 — Working memory (in-context)
  Current turn messages + retrieved snippets injected into the prompt.
  Managed by: AssistantState.messages + AssistantState.memories
  Lifetime: single turn

Tier 2 — Episodic memory (session summaries)
  Compressed summaries of past sessions.
  Managed by: SQLite summaries table
  Lifetime: until explicitly forgotten or consolidated

Tier 3 — Semantic memory (long-term facts)
  Stable facts, preferences, and patterns extracted from episodic records
  by the consolidation process.
  Managed by: SQLite facts + ChromaDB vectors
  Lifetime: indefinite (user-deletable)
```

Implementation notes:
- Episodic tier backed by the existing `summaries` table (columns: `id, user_id, session_id, summary, created_at, consolidated`).
- `_build_episodic_record_text()` in `main.py` uses `_summarize_messages_chunk()` to format recent messages as structured text (no LLM call).
- `consolidate_pending()` in `MemoryStore` uses regex-based `_extract_candidates()` to extract facts/preferences from episodic text, enforces sensitivity matrix, writes to SQLite + ChromaDB, marks row `consolidated=1`. **The regex extractor is the known gap fixed in Phase 12b.**
- Background consolidation is non-blocking: `asyncio.to_thread` + `loop.create_task`.
- `MEMORY_CONSOLIDATION_MODE` env var: `on-session-end` (default), `interval`, or `manual`.
- `MEMORY_CONSOLIDATION_INTERVAL_SECONDS` and `MEMORY_CONSOLIDATION_BATCH_SIZE` are configurable.
- `retrieve()` queries all three tiers and ranks results; episodic scores capped at 0.85× to let high-confidence semantic facts rank above raw summaries.
- Routes in `backend/routes/memory_tool_routes.py`: `GET /memory/episodic`, `POST /memory/consolidate`.

Tasks — all complete:
- ✅ Add `episodic` table to SQLite schema: `(id, session_id, summary, created_at, consolidated)`.
- ✅ Implement end-of-session summarization: session messages compressed into episodic record.
- ✅ Implement consolidation task: extracts semantic facts from episodic records,
  filters through sensitivity matrix, writes to facts + ChromaDB.
- ✅ Add consolidation schedule (configurable: on session end, daily, or manual).
- ✅ Expose episodic records in the memory viewer UI alongside facts.
- ✅ Add `GET /memory/episodic` endpoint.
- ✅ Upgrade `memory_retrieval` node to query all three tiers and rank results.

Acceptance:
- ✅ Facts stated in session A are retrievable by semantic search in session B.
- ✅ Sensitive items are not auto-consolidated (sensitivity matrix enforced).
- ✅ User can view episodic records and delete them individually.
- ✅ Consolidation task runs without blocking the main request path.
- ✅ Memory viewer shows tier labels (working / episodic / semantic) for each item.

---

### Phase 12b — LLM-based memory extraction
**Status: complete (shipped)**
**Estimate: 3–5 days**
**Depends on: Phase 12**

Goal: Replace the regex-based `_extract_candidates()` in the consolidation worker
with a proper LLM call. This is the highest-value memory improvement available —
regex pattern matching misses implied facts, paraphrased preferences, and anything
the user states indirectly. A well-prompted local model handles all of these.

**Implementation Status:**
✅ Consolidation worker upgraded to LLM-based extraction (`_llm_extract_candidates()`)
✅ Confidence threshold (≥0.7) enforced; JSON parse errors handled gracefully
✅ Ollama unreachability handled with graceful fallback
✅ Non-blocking consolidation preserved (asyncio.to_thread)
✅ Sensitivity matrix still applied downstream (no regressions)
✅ Full test coverage: 5 Phase 12b tests in test_memory_isolation.py (all passing)
ℹ️  Old regex `_extract_candidates()` kept for direct user message ingestion (fast path)

Design:

The consolidation worker already runs in a background thread after session end.
The only change is what happens inside `consolidate_pending()`: replace
`_extract_candidates(episodic_text)` with a local model call that returns
structured JSON candidates.

Prompt contract (send to `OLLAMA_CHAT_MODEL`):
```
System: You are a memory extraction assistant. Given a summary of a conversation,
extract stable facts and preferences about the user. Return ONLY valid JSON with
no preamble. Format:
{
  "candidates": [
    { "key": "string", "value": "string", "type": "fact|preference", "confidence": 0.0–1.0 }
  ]
}

Rules:
- Extract only things that are stable and generalize beyond this conversation.
- Do not extract things said by the assistant, only about the user.
- Do not extract sensitive data (passwords, exact addresses, financial details).
- If nothing stable can be extracted, return { "candidates": [] }.
```

The returned candidates are then passed through the existing sensitivity matrix
filter before any write. The sensitivity matrix is not changing — only the
extraction step.

Implementation notes:
- Call `OLLAMA_CHAT_MODEL` via the existing Ollama client. Keep the token budget
  small (~300 output tokens) — extraction is a structured task, not a prose response.
- Parse the JSON response with `try/except`. On parse failure, log and skip the
  record (do not crash the consolidation worker).
- Set a confidence threshold: only write candidates with `confidence >= 0.7`.
- The old `_extract_candidates()` regex function is retained for direct user message
  ingestion (fast-path fallback, see Phase 6 explicit memory hints). The consolidation
  worker uses the LLM path exclusively for richer extraction from episodic summaries.
- Consolidation is non-blocking — this change does not affect that. The LLM call
  runs inside `asyncio.to_thread` as before.

**Implementation Details:**
- `_llm_extract_candidates()` (async) calls `OLLAMA_CHAT_MODEL` with structured prompt
- `_extract_candidates_llm_sync()` wrapper used by consolidation worker (runs in thread)
- Token budget: 400 output tokens for structured JSON extraction
- Text truncation: last 1500 chars of long summaries used for cost efficiency
- Confidence threshold: 0.7 (candidates below this are filtered)
- Error handling: JSON parse failure → empty list (logged, no crash)
- Ollama unreachability → empty list (logged, consolidation continues)
- Model selection: `OLLAMA_CHAT_MODEL` env var with fallback chain

**Quality Assessment (Lightweight):**
- Run consolidation against one week of real user sessions
- Spot-check facts table for quality and coverage (LLM vs. regex)
- Document intuition about candidate richness in Phase 12b completion notes

**Tests (All Passing):**
- `test_consolidate_pending_promotes_summary_facts()` — validates LLM in consolidation
- `test_llm_extract_filters_by_confidence()` — verifies 0.7 threshold
- `test_llm_extract_json_parse_failure()` — tests graceful JSON parse failure
- `test_llm_extract_ollama_unreachable()` — tests Ollama connectivity error
- `test_consolidate_uses_llm_extraction()` — confirms LLM path (not regex) in consolidation

**Acceptance (All Met):**
- ✅ Consolidation worker uses LLM extraction (not regex)
- ✅ Parse failures handled gracefully — worker never crashes
- ✅ Confidence threshold enforced (≥0.7)
- ✅ Sensitivity matrix still applied downstream — no regressions
- ✅ Consolidation schedule and non-blocking behavior unchanged
- ✅ Tests pass (Ollama mocked in CI)

---

### Phase 13 — External coding-agent integration (retired)
**Status: retired / superseded by project-scoped coding (`roadmap.4.md`)**
**Estimate: 3–5 days**
**Depends on: Phase 12b**

This phase is no longer an active direction. External-agent integration was
dropped in favor of an internal coding agent centered on Projects workspace.

Salvaged/reused work:
- ✅ `coding_agent_tool` confirmation-gate UX and pending-task state flow.
- ✅ `coding_agent_executor` result-shaping path (voice summary vs chat detail).
- ✅ `code_context` injection pattern from `memory_retrieval`.
- ✅ `code-question` vs coding-task split that keeps main chat lighter.

Follow-up:
- Continue implementation in `docs/roadmap.4.md` only.
- Do not add new external coding-agent dependencies or setup requirements.

---

### Phase 14 — Vision input
**Status: complete**
**Estimate: 2–3 days**
**Depends on: Phase 12b**

Goal: Enable users to attach images to chat and ask questions about them.
Hearth processes images end-to-end with a unified multimodal model, producing
accurate answers grounded in visual content.

Design rationale:

Gemma 4 (e2b/e4b family) is Google's first genuinely capable multimodal small
model with *native* vision support. Unlike earlier multimodal approaches that
bolt-on separate CLIP encoders, Gemma 4 handles vision directly in the same
forward pass:

1. Image patches are tokenized alongside text tokens in a unified embedding space
2. A single attention mechanism attends over both image and text tokens
3. No separate vision pipeline, CLIP bridge, or embedding projection layer
4. The model reasons about images with the same parameter weights it uses for text

This architectural unity means one model call per turn, predictable latency, and
no post-processing stage.

Model selection:
```
- Chat / vision (general): gemma:e4b (multimodal, 9B)  ← OLLAMA_VISION_MODEL
- Code (specialist):        qwen2.5-coder:14b (text-only, no regression)
```

`OLLAMA_VISION_MODEL` defaults to `CHAT_MODEL` (`gemma:e4b`). Both are the same
model — no second model download required.

Image handling:

```
User attaches image → frontend encodes as raw base64 (no data-URI prefix)
  → POST /chat { message, source, image_base64, image_mime }
  → /chat validates image (MIME, base64, ≤25 MB) → 422 on failure
  → intent_classifier short-circuits to "vision" (structural signal, no LLM call)
  → tool_router → responder
  → responder calls Ollama /api/chat with messages[].images (multimodal forward pass)
  → on local failure: cloud fallback to Anthropic Claude vision API
  → on both unavailable: user-visible error with `ollama pull gemma:e4b`
  → returns text response; TTS suppressed (vision responses are not auto-read)
```

Storage:
- Images are NOT stored persistently. Each image is processed and discarded.
- Episodic summaries may describe the interaction ("user asked about an image of a
  car") but never store the image itself.
- Rationale: images can contain sensitive data; local-first privacy preserved
  by non-persistence.

Intent routing — new `vision` intent category:
- Image presence is a **structural signal**: `intent_classifier` short-circuits
  before `classify_intent()` is called when `image_base64` is set in state.
  There is no ambiguous case — an attached image always means "vision request".
- Text-only keyword scoring (`_VISION_PATTERNS`, `_VISION_KEYWORDS`) handles
  imageless visual queries ("describe this photo?" without an attached file).
  The classifier still runs for those cases, scoring them as `vision` intent.
- `VISION_MODEL` in `router.py` maps `vision` intent to the local multimodal model.
- `quick-local` dampening is applied when vision score ≥ 0.25 (same pattern as
  code and weather intents).

Implementation notes:

**Payload format:**
- Flat JSON fields: `image_base64` (raw base64 string, no `data:…;base64,` prefix)
  and `image_mime` (`"image/png"` | `"image/jpeg"` | `"image/webp"`).
- Kept simple intentionally. This is a single-user LAN system; base64 in the
  JSON body is appropriate and avoids multipart complexity.
- `ChatRequest` in `app_schemas.py` carries both fields as optional; they are
  transparent to all non-vision code paths.

**Validation (`_validate_image()` in `main.py`):**
- Allowed MIME types: `image/png`, `image/jpeg`, `image/webp`.
- Maximum size: 25 MB (decoded bytes).
- Base64 integrity: `base64.b64decode(..., validate=True)`.
- Returns 422 with `{"error": "...", "code": "INVALID_IMAGE"}` on failure.
- Validation fires in the HTTP handler before the graph is invoked.

**Ollama multimodal API (`stream_local_vision()` in `main.py`):**
- Uses `/api/chat` endpoint (not `/api/generate`) with a structured messages array.
- Image passed as `messages[].images: [base64_string]`.
- Streams response chunks from `message.content` field.
- Separate from the existing `stream_local()` function — text-only paths are
  completely unchanged.

**Cloud fallback (Anthropic Claude vision):**
- `responder` node in `graph.py` tries `stream_local_vision()` first.
- On any exception, falls back to `stream_cloud()` with Anthropic's vision message
  format: `content: [{type: "image", source: {type: "base64", ...}}, {type: "text", ...}]`.
- On both unavailable: returns a user-visible error string with `ollama pull gemma:e4b`.
- `response_model` in the SSE meta event reflects which model actually ran.

**TTS suppression:**
- `_voice_tts_metadata()` is called as normal but its result is set to `None` when
  `intent_for_log == "vision"` before the SSE event is emitted.
- Rationale: image responses are visual; auto-reading them aloud is intrusive.
  User can trigger TTS manually via the stop/play controls.

**Frontend:**
- Hidden `<input id="image-upload" type="file" accept="image/png,image/jpeg,image/webp">`.
- Paperclip `<button id="image-attach-btn">` triggers the picker.
- Preview strip (`#image-preview-strip`) above the textarea with thumbnail and clear button.
- Client-side MIME and size validation before encoding (fail fast, no wasted round trip).
- `fileToBase64()` strips the `data:…;base64,` prefix — backend receives raw base64.
- `pendingImage` state cleared after send.
- Attached image shown as a `<img class="chat-image-thumb">` in the user message bubble.
- `image_base64` and `image_mime` included in the POST body only when `pendingImage` is set.

**Graph changes:**
- `AssistantState` carries `image_base64: str | None` and `image_mime: str | None`
  (ephemeral; never written to memory or checkpointer).
- `AssistantGraphDependencies` carries `stream_local_vision` callable and `vision_model` string.
- `intent_classifier` node: if `image_base64` is set, returns `planner_status: "deterministic"`
  immediately — no heuristic, no planner, no LLM call.
- `responder` node: vision branch runs before tool/cloud/local branches; handles local
  attempt, cloud fallback, and hard-error case.

Environment variables:
  `OLLAMA_VISION_MODEL`   Ollama model for vision requests (default: same as `CHAT_MODEL`)

Tasks — all complete:
- ✅ Add `OLLAMA_VISION_MODEL` env var to `main.py` and `router.py` (defaults to `CHAT_MODEL`).
- ✅ Extend `ChatRequest` in `app_schemas.py` with `image_base64` and `image_mime` optional fields.
- ✅ Implement `_validate_image()` in `main.py`: MIME allowlist, 25 MB cap, base64 integrity check.
- ✅ Add `stream_local_vision()` in `main.py` using Ollama `/api/chat` with `images` array.
- ✅ Wire image validation and graph state fields into the `/chat` endpoint.
- ✅ Add `vision` intent to `router.py`: `VISION_MODEL` constant, `_VISION_PATTERNS` /
  `_VISION_KEYWORDS` / `_looks_like_vision_request()`, scoring in `classify_intent()`,
  quick-local dampening, `RouteDecision` docstring.
- ✅ Short-circuit `intent_classifier` node in `graph.py` when `image_base64` is present
  (before `classify_intent()` is called).
- ✅ Add vision branch to `responder` node: local attempt → cloud fallback → hard-error string.
- ✅ Suppress TTS for vision intent in `generate()` SSE loop.
- ✅ Wire `stream_local_vision` and `vision_model` into `AssistantGraphDependencies`.
- ✅ Add image attach button, preview strip, and chat history thumbnails to frontend
  (`index.html`, `message.js`, `style.css`).
- ✅ Add 18 tests in `backend/tests/test_vision.py` covering validation, intent routing,
  classifier short-circuit logic, and TTS suppression.

Acceptance:
- ✅ Frontend image picker (file button) attaches images to chat with preview strip.
- ✅ `/chat` endpoint accepts `image_base64` + `image_mime`; validates on arrival.
- ✅ Image validation rejects unsupported MIME types, bad base64, and files >25 MB (HTTP 422).
- ✅ `intent_classifier` short-circuits to `vision` intent without calling the LLM when an
  image is attached; text-only visual queries still scored by keyword patterns.
- ✅ Local `gemma:e4b` is used via Ollama `/api/chat` with `images` array.
- ✅ Cloud fallback to Anthropic Claude vision API fires on local model failure.
- ✅ User-visible error with `ollama pull gemma:e4b` if both local and cloud unavailable.
- ✅ Model badge in SSE meta event reflects which model handled the image question.
- ✅ Images are not persisted to disk, SQLite, or ChromaDB; ephemeral per turn.
- ✅ TTS is suppressed for vision responses; user initiates audio manually.
- ✅ No regression on text-only chat (143 existing tests pass).
