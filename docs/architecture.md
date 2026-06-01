# Hearth - Local AI Assistant — Project Context

Hearth is a local-first personal AI assistant with voice activation, streaming chat,
tool integrations, and optional cloud model fallback. It runs entirely on-device via Docker
Compose on a Linux machine with an NVIDIA RTX 3060 (12 GB VRAM) and is designed to be
reachable from devices on the local network.

Stack: FastAPI backend · Ollama (local LLM + local coder) · Anthropic API (cloud fallback) ·
OpenWakeWord · faster-whisper · Piper TTS (pluggable — Kokoro is a viable swap) · SQLite ·
ChromaDB · LangGraph · Browser UI

---

## Architecture overview

```
Browser / LAN client
  └── <single origin>                 (FastAPI serves UI + API; HTTPS edge when enabled)
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
  ├── tools/code.py                   code generation / file read (questions only, no writes)
  └── tts/                            pluggable TTS package (engines + loader)

Ollama (container)                    local model inference (GPU)
  ├── gemma:e4b                       general conversation, voice responses, persona anchor
  └── qwen2.5-coder:14b                code-specialized model for code questions
Anthropic API (external)              cloud fallback for complex queries

Reverse proxy / HTTPS edge            enabled when LAN/mobile deployment requires it
```

**Serving rule:** frontend is always served from the FastAPI origin. Never run a
separate dev server in production. Mount `./frontend` into the backend container and
serve it as static files. All browser fetch/WebSocket calls use relative paths
(`/chat`, `/ws/wake`, etc.) — no hardcoded `localhost` or port numbers anywhere in
runtime code.

## Model setup

- `OLLAMA_CHAT_MODEL=gemma:e4b` for general conversation, voice responses, and persona anchoring. Chosen for its natural prose quality and ability to hold a system-prompt persona. Acceptable reasoning at 4B; complex tasks fall through to cloud.
- `OLLAMA_CODER_MODEL=qwen2.5-coder:14b` for code questions and explanation. Do not route code tasks to cloud by default.
- Anthropic is fallback only when local confidence is low or the task exceeds local capability.
- Both local models hot-swap inside one Ollama container. Simultaneous residency is not realistic on 12 GB VRAM, so routing and UX must account for swap latency.
- Measured swap latency (2026-04-28, RTX 3060 NVMe): median 0.2–0.3 s after first load. Ollama keeps weights in system RAM after GPU eviction so repeat swaps are RAM to GPU re-pin only. First cold load from disk is about 2 s. Overall impact is imperceptible, so loading-state UX is low priority.


## Locked architecture decisions

- **Hearth implements its own project-scoped coding agent path.** Main-chat code stays `code-question` only; coding writes run through project-scoped `coding_agent_tool`/`coding_agent_executor` flows.
- The `code_tool` node stays in the graph for code-question routing in main chat. Keep that boundary clear: main chat answers code questions; project sessions handle coding work.
- The ReAct loop and tree-sitter indexer in `code_context` are core context infrastructure for both code questions and project coding sessions; continue evolving them as needed.
- The coding assistant is NOT a VS Code extension. Coding remains local-first within Hearth's own backend/runtime paths, not IDE plugin APIs.
- Sub-agent architecture (previously Phase 14) is shelved indefinitely. Do not implement task decomposer, critic, or synthesizer subgraphs. A single well-prompted agent with tool access is sufficient for Hearth's use cases.
- The deterministic music pre-router (`_parse_music_command()` in the HTTP handler) stays in place, by design. It guards the graph invocation for high-confidence music commands and is never moved into the graph.
- All frontend API calls use relative paths. No hardcoded hosts or ports in runtime code.
- New tools are modules under `backend/tools/` with interface: `async def run(params: dict) -> dict`.
- `OLLAMA_CHAT_MODEL` and `OLLAMA_CODER_MODEL` are separate env vars. Never hardcode model names in source.
- Structured logs for every routing decision, model selected, tool call, and error.
- Cloud fallback degrades gracefully with a user-visible notice, never silent failure.
- Android/mobile voice requires HTTPS-capable LAN edge support before it should be considered complete.

## Target LangGraph shape

```text
input → intent_classifier → memory_retrieval → tool_router
  ├── weather_tool
  ├── music_tool
  ├── code_tool        (code questions only; qwen2.5-coder:7b; no file writes)
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
    active_files: list[str]
    code_context: str
    modality: str          # "voice" or "chat"
```

---

