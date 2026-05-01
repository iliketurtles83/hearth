# Local AI Assistant — Copilot Instructions

This project is a local-first personal AI assistant with voice activation, streaming chat,
tool integrations, and optional cloud model fallback. It runs entirely on-device via Docker
Compose on a Linux machine with an NVIDIA RTX 3060 (12 GB VRAM) and is designed to be
reachable from devices on the local network.

Stack: FastAPI backend · Ollama (local LLM + local coder) · Anthropic API (cloud fallback) ·
OpenWakeWord · faster-whisper · Piper TTS (pluggable — Kokoro is a viable swap) · SQLite ·
ChromaDB · LangGraph · Browser UI

---

## Current state
- Phases 1–10a complete. Next: Phase 10b (code tool node), 10c (responder/modality), 10d (ChromaDB cleanup).
- Active models: `OLLAMA_CHAT_MODEL=gemma3:4b`, `OLLAMA_CODER_MODEL=qwen2.5-coder:14b`.
- Graph shape: `input → intent_classifier → memory_retrieval → tool_router → [weather|music|chat_fallback] → responder`.
- The deterministic music pre-router (`_parse_music_command`) fires in the HTTP handler before graph invocation — never move it into the graph.
- Serving rule: frontend is a static mount in FastAPI. No separate dev server. All API calls use relative paths.
- See PROJECT-CONTEXT.md for the full architecture and phase roadmap.

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
  `async def synthesize(text: str) -> bytes` interface. `TTS_ENGINE` env var selects via the `backend/tts` loader.
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
  persona rendering. No fact drift is acceptable.
- Add tests before marking any phase acceptance criteria as done.
