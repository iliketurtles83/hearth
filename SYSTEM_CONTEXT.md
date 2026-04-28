You are the AI developer for a local-first personal assistant project running on a Linux machine with an NVIDIA RTX 3060 (12 GB VRAM). The stack is FastAPI + Ollama + LangGraph + SQLite + ChromaDB, served as a single-origin app on the LAN via Docker Compose. The frontend is a browser UI with voice activation via OpenWakeWord ("Computer,..."), faster-whisper transcription, and Piper TTS output modeled on a TNG computer voice.

## Current state
- Phases 1–8 complete: LAN serving, single-origin frontend, wake-word pipeline stable on desktop/Linux, music playback (MPD + Strawberry), SQLite/ChromaDB memory, weather tool.
- Phases 9–14 not yet started.
- Active models: gemma3:4b (chat) and qwen2.5-coder:7b (code) — both pulled and verified on this machine.

## Model setup (Ollama)
- OLLAMA_CHAT_MODEL: gemma3:4b — general conversation, voice responses, personality anchor. Chosen for its natural prose quality and ability to hold a system-prompt persona. Acceptable reasoning at 4B; complex tasks fall through to cloud.
- OLLAMA_CODER_MODEL: qwen2.5-coder:7b — all code intents. Never send code tasks to cloud by default.
- Cloud fallback: Anthropic API (Claude), invoked only when local confidence is low or task exceeds local capability.
- Both models run in the same Ollama container. They hot-swap; simultaneous loading is not realistic at 12 GB VRAM.
- Measured swap latency (2026-04-28, RTX 3060 NVMe): median 0.2–0.3s after first load. Ollama keeps weights in system RAM after GPU eviction so repeat swaps are RAM→GPU re-pin only. First cold load from disk is ~2s. Overall: imperceptible — loading-state UX in Phase 10b is optional/low-priority.

## Architecture decisions locked in
- The coding assistant lives inside the LangGraph graph as a code_tool node, NOT as a VS Code extension. It inherits memory, session state, voice input, and tool access from the graph for free.
- The code node runs a ReAct loop (think → call tool → observe → repeat) using LangGraph's create_react_agent. Do not implement a single-shot code completion.
- Pre-built tools from langchain-community: ShellTool, ReadFileTool, WriteFileTool, PythonREPLTool. Do not reinvent these.
- Codebase context is built with tree-sitter (parse repo into file summaries, function signatures, import graphs) stored in ChromaDB. On code intents, memory retrieval injects relevant slices into the coder model's system prompt as code_context. This is what makes the local model competitive with cloud-backed editor agents.
- File writes require explicit user confirmation before touching disk. Voice flow: model summarizes what it wrote, user confirms, then write executes.
- Workspace root is enforced — no path traversal outside configured directory.
- ollama launch integrations (Claude Code, Codex, OpenCode, Hermes, OpenClaw) are terminal CLI tools irrelevant to this project. They cannot receive wake words, share graph state, or interact with the voice pipeline.

## LangGraph graph shape
input → intent_classifier → memory_retrieval → tool_router
  ├── weather_tool
  ├── music_tool
  ├── code_tool        (ReAct loop, qwen2.5-coder:7b, langchain-community tools)
  └── chat_fallback
        └── responder → output

State shape includes: messages, intent, memories, tool_result, user_prefs, session_id, active_files, code_context.

## Key constraints and rules
- All frontend API calls use relative paths. No hardcoded hosts or ports in runtime code.
- New tools are modules under backend/tools/ with interface: async def run(params: dict) -> dict
- OLLAMA_CHAT_MODEL and OLLAMA_CODER_MODEL are separate env vars. Never hardcode model names in source.
- Structured logs for every routing decision, model selected, tool call, and error.
- Cloud fallback degrades gracefully with a user-visible notice, never silent failure.
- Android/mobile voice requires HTTPS — deferred, pending LAN reverse proxy setup.

## Build order from here
Phase 9: TTS voice output with barge-in support (Piper engine, /tts endpoint, frontend playback widget).
Phase 10a: LangGraph graph skeleton and router migration (SqliteSaver checkpointer, no feature additions).
Phase 10b: Code tool node (ReAct loop, tree-sitter context, ChromaDB code_context, confirmation-gated writes).
Phase 10c: Responder node and voice/chat modality split.
Phase 10d: ChromaDB collection separation (conversation vs. code context).