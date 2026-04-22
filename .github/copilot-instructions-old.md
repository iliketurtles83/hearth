# Local AI Assistant Roadmap and Build Instructions

This project runs as a local-first assistant with:
- FastAPI backend
- Ollama local model routing with optional cloud fallback
- Browser chat UI with streaming responses and wake-word voice pipeline
- Docker Compose deployment

Use the development order below to reduce rework and ensure each new feature builds on stable foundations.

## 1) Stabilize Core Platform and Local Network Access (in progress)

Goal:
- The assistant is reachable from other devices on the LAN and remains stable under normal use.

Why first:
- Every other feature depends on reliable connectivity, predictable routing, and backend uptime.

Implementation guidance:
- Keep backend and frontend served from the same FastAPI origin when possible.
- In Docker Compose, mount both backend and frontend into the backend container so static assets and AudioWorklet files are available.
- Confirm services bind to 0.0.0.0 and ports are exposed correctly.
- Prefer relative API paths in frontend requests (for example /chat, /transcribe, /ws/wake) to avoid host mismatch when accessed from phones/tablets on LAN.
- Keep CORS permissive during development, then restrict allow_origins to LAN hosts when stable.
- Add health checks and verify container restarts do not hide import/runtime errors.

Definition of done:
- Opening http://<LAN-IP>:8000 from another device loads the UI.
- Chat works from LAN clients.
- Voice WebSocket connects and remains open without repeated reconnect loops.

## 2) Complete Voice Input Wake Flow: "Computer, ..." (in progress)

Goal:
- Hands-free activation using a Star Trek style wake phrase and then speech capture/transcription.

Why second:
- Voice input is already partially implemented and should be hardened before adding voice output.

Implementation guidance:
- Keep wake-word model filename and prediction key in sync with downloaded model assets.
- Add backend startup validation:
	- Check required ONNX files exist.
	- Emit clear startup logs if files are missing.
- Add a short post-wake guard window to avoid retrigger while the user is speaking.
- In frontend state machine, enforce explicit transitions:
	- off -> sleeping -> recording -> transcribing -> sleeping
- Improve observability:
	- Log WebSocket close codes/reasons.
	- Log wake score threshold decisions at debug level.
- Keep microphone permission errors user-friendly and non-fatal.

Definition of done:
- Clicking mic enables stable sleeping state.
- Saying wake phrase triggers recording once.
- Transcribed text is sent automatically and returns to sleeping state.

## 3) Improve Model Routing for Complex Queries (in progress)

Goal:
- Automatically use stronger cloud model only when needed, while keeping local model as default.

Why third:
- Routing quality directly affects answer usefulness and cost; it should mature before adding memory and tools.

Implementation guidance:
- Replace simple keyword routing with a small intent classifier layer:
	- Categories: quick-local, reasoning-heavy, external-data-needed, personal-memory-needed.
- Add confidence scoring and fallback policy:
	- If local response quality is low or tool/data is required, escalate to cloud.
- Maintain transparent model badge in UI for user trust.
- Add latency and token telemetry:
	- route chosen
	- first token latency
	- completion latency
	- error/fallback count

Definition of done:
- Straightforward prompts stay local.
- Complex planning/reasoning prompts route to cloud.
- Failures degrade gracefully and still return a response path.

## 4) Add Assistant Memory (user profile + conversation memory)

Goal:
- Assistant remembers user preferences and relevant prior facts safely.

Why fourth:
- Memory is foundational for personalization, weather defaults, music preferences, and conversational continuity.

Implementation guidance:
- Use a two-tier memory design:
	- Short-term session memory: recent turns and active tasks.
	- Long-term user memory: durable profile, preferences, routines, constraints.
- Start simple with SQLite in backend:
	- tables for facts, preferences, interaction summaries, and retrieval metadata.
- Add memory write policy:
	- Save only high-value facts (preferences, recurring intents, explicit user statements).
	- Never store secrets unless explicitly approved.
- Add memory retrieval policy:
	- Retrieve top relevant items per query.
	- Inject concise memory snippets into system/context prompt.
- Add user controls:
	- view memory
	- delete memory items
	- clear all memory

Definition of done:
- Assistant can remember and reuse stated preferences across restarts.
- User can inspect and remove stored memory.

## 5) Enable Simple Utility Task: Weather

Goal:
- Assistant can answer current weather quickly and reliably.

Why fifth:
- This is the cleanest first external tool integration and validates tool-routing architecture.

Implementation guidance:
- Build a backend weather tool endpoint.
- Use user-approved location default from memory, with prompt override support.
- Choose one provider and normalize response schema.
- Add graceful error responses for API failures and offline mode.
- Route weather intents directly to tool then summarize through model.

Definition of done:
- "What is the weather" works with remembered default location.
- "Weather in <city>" overrides default.
- Output is concise, clear, and includes units.

## 6) Add Music Playback from Local Collection

Goal:
- Assistant can search and play music from local library by voice or text.

Why sixth:
- Depends on memory (favorites), intent routing, and reliable voice input.

Implementation guidance:
- Create a media indexer service:
	- Scan configured folders.
	- Store metadata in SQLite (title, artist, album, path, duration).
- Provide backend endpoints:
	- search tracks/albums/artists
	- play/pause/stop/next
	- queue management
- Choose playback architecture:
	- Browser playback for same-device sessions, or
	- Backend playback daemon for always-on host audio.
- Add confirmation patterns:
	- "Playing <track> by <artist>"
	- Clarify ambiguous matches.

Definition of done:
- User can say "Play <song/artist>" and hear audio.
- Assistant can stop/pause/resume and report what is currently playing.

## 7) Add TNG LCARS-style Female Voice Output (TTS)

Goal:
- Assistant responds with spoken audio in a voice aligned to TNG computer style.

Why seventh:
- Voice output should be added after core voice input, routing, and utility/tool flows are stable.

Implementation guidance:
- Implement TTS as a backend service with pluggable engines.
- Start with high-clarity female voice preset and tune:
	- speaking rate
	- pitch
	- prosody style
- Add optional "brief mode" and "full mode" response lengths for spoken output.
- Return audio stream or file URL to frontend and auto-play with user gesture constraints.
- Add interruption and barge-in behavior:
	- New wake phrase can stop current TTS and start listening.

Definition of done:
- Assistant can speak responses end-to-end from text output.
- Voice is intelligible, consistent, and does not block normal chat use.

## Cross-cutting standards

Security and privacy:
- Local-first default.
- Explicit user consent before storing personal memory.
- Redact sensitive data from logs.

Reliability:
- Structured logs for routing, wake-word, transcription, and tool usage.
- Retry/backoff for transient network failures.
- Clear user-facing errors.

Testing priorities:
- Backend API tests for chat, transcribe, wake socket, and weather tool.
- Frontend smoke tests for streaming markdown, mic toggle, and reconnect behavior.
- Manual LAN test matrix (desktop, phone, tablet).

## Immediate next sprint (recommended)

1. Finish LAN-safe frontend request paths and verify from a second device.
2. Harden wake-word pipeline with startup model checks and better close-code logging.
3. Upgrade router logic with confidence-based cloud escalation.
4. Introduce SQLite memory layer with minimal CRUD endpoints.
5. Add first weather tool integration using remembered default location.
