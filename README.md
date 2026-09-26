# Hearth — Local-First Personal AI Assistant

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-orchestrated-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Docker Compose](https://img.shields.io/badge/docker--compose-v2+-2496ED.svg)](https://docs.docker.com/compose/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Hearth** is a private, local-first personal AI assistant built for home environments. It pairs local OpenAI-compatible inference (such as `llama.cpp` or Ollama) with stateful LangGraph orchestration, hybrid vector/relational memory, streaming voice input/output, multimodal vision, and local music management over MPD and Beets.

All traffic is served securely on your local network via an integrated Caddy reverse proxy with `mkcert` TLS termination, enabling secure-context browser APIs (`navigator.mediaDevices`, AudioWorklet) across desktop browsers, Android, and iOS devices without third-party certificate authority dependencies.

---

## Table of Contents

- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Layout](#project-layout)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
  - [1. Clone & Configure](#1-clone--configure)
  - [2. Generate LAN HTTPS Certificates](#2-generate-lan-https-certificates)
  - [3. Download Runtime Models](#3-download-runtime-models)
  - [4. Start the Application](#4-start-the-application)
  - [5. Trust the CA on Client Devices](#5-trust-the-ca-on-client-devices)
- [Core Capabilities](#core-capabilities)
  - [Streaming Chat & Checkpointed Sessions](#streaming-chat--checkpointed-sessions)
  - [Voice & Audio Pipeline](#voice--audio-pipeline)
  - [Multimodal Vision](#multimodal-vision)
  - [Hybrid Memory Layer](#hybrid-memory-layer)
  - [Local Music & MPD Integration](#local-music--mpd-integration)
  - [Code Understanding](#code-understanding)
- [Configuration Reference](#configuration-reference)
- [API Overview](#api-overview)
- [Local Development & Testing](#local-development--testing)
  - [Running Without Docker](#running-without-docker)
  - [Running the Test Suites](#running-the-test-suites)
  - [Security & Quality Baseline Gates](#security--quality-baseline-gates)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)

---

## Key Features

- 🔒 **Local-First & Private**: Runs entirely on your own hardware. Your prompts, memories, audio recordings, and images never leave your LAN unless you explicitly trigger optional cloud fallback (e.g. Anthropic Claude).
- 🛡️ **Zero-Friction LAN HTTPS**: Integrated Caddy edge proxy terminates TLS on port 443 using local `mkcert` certificates, ensuring microphone and media device APIs work flawlessly on mobile and desktop.
- 🎙️ **Complete Voice Pipeline**: Wake-word activation via [openWakeWord](https://github.com/dscripka/openWakeWord) ("Computer"), speech-to-text via [faster-whisper](https://github.com/guillaumekln/faster-whisper), and pluggable low-latency TTS ([Piper](https://github.com/rhasspy/piper) or [Kokoro](https://github.com/hexgrad/kokoro)) with automatic response compression and barge-in handling.
- 👁️ **Multimodal Vision**: Attach images to chat prompts with automatic payload validation and direct routing to local or cloud vision models.
- 🧠 **Hybrid Persistent Memory**: Combines SQLite (structured facts, user preferences, episodic session summaries) with ChromaDB vector embeddings for semantic recall and background "sleep" consolidation.
- 🧭 **Dual-Classifier Intent Routing**: Blends embedding-based intent routing (`nomic-embed-text`) with deterministic regex patterns to route between fast-paths, tools, local models, and cloud fallback.
- 🎵 **Local Music Automation**: Built-in [Beets](https://beets.io/) library metadata indexing and [MPD](https://www.musicpd.org/) playback control. Supports deterministic fast-path actions, genre/artist radio, queue management, volume adjustments, and queue shuffling.
- 💻 **Code Questioning**: Dedicated `/code` endpoint and graph intent for code review, explanation, and architectural questions.
- 🎨 **Responsive Web Interface**: Single-origin vanilla JS UI with light/dark theme toggle, real-time audio visualizers, session management, and settings menus.

---

## Architecture

```text
 Client Devices (Desktop / iOS / Android)
   │
   │  HTTPS :443 (LAN Edge, TLS terminated by Caddy via mkcert)
   ▼
┌─────────────────────────────────────────────────────────────┐
│ Caddy Reverse Proxy                                         │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP (Internal Docker network)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ FastAPI Backend (:8000, internal-only)                      │
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Pre-Graph Fastpaths                                     │ │
│ │  ├── Auth Middleware (scrypt tokens, cookie-scoped)     │ │
│ │  ├── Deterministic Music Fastpath (play/ctrl/shuffle)   │ │
│ │  └── Image Validation (multimodal gate)                 │ │
│ └────────────────────────────┬────────────────────────────┘ │
│                              │                              │
│ ┌────────────────────────────▼────────────────────────────┐ │
│ │ LangGraph Stateful Graph (SqliteSaver Checkpointing)    │ │
│ │                                                         │ │
│ │  [History Loader] ──► [Intent Classifier]               │ │
│ │                             │                           │ │
│ │  [Tool Router]    ◄── [Memory Retrieval]                │ │
│ │        │                                                │ │
│ │  [Responder]      ──► [Memory Writer] ──► END           │ │
│ └───────┬─────────────────────────────────────────────────┘ │
│         │                                                   │
│ ┌───────▼─────────────────────────────────────────────────┐ │
│ │ Tools & Subsystems                                      │ │
│ │  ├── Weather Tool (Open-Meteo geocoding & forecast)     │ │
│ │  ├── Music Tool (Beets SQLite index + MPD client)       │ │
│ │  ├── Hybrid Memory (SQLite relational + ChromaDB)       │ │
│ │  └── Audio/TTS (openWakeWord, Whisper, Piper/Kokoro)    │ │
│ └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────┬──────────────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
  Local Inference Server              Optional Cloud Fallback
  (llama.cpp / vLLM / Ollama)         (Anthropic Claude API)
```

---

## Project Layout

```text
.
├── docker-compose.yml          # Container stack orchestration (mpd, backend, caddy)
├── config.yaml                 # Beets configuration
├── caddy/
│   ├── Caddyfile               # Caddy proxy rules
│   └── certs/                  # Generated mkcert TLS certificates (gitignored)
├── backend/
│   ├── main.py                 # FastAPI application factory, lifespan, middleware
│   ├── load_env.py             # Environment bootstrap (dotenv loader)
│   ├── graph.py                # LangGraph state machine & node executors
│   ├── intents.py              # Heuristic intent classifier & shared model constants
│   ├── embedding_router.py     # Embedding-based exemplar classifier (tool + dialogue)
│   ├── routing_config.py       # Intent router dataclass & threshold settings
│   ├── music_fastpath.py       # Deterministic music parsing & response formatter
│   ├── memory.py               # SQLite + ChromaDB hybrid storage & consolidation
│   ├── memory_scheduler.py     # Background sleep-consolidation daemon
│   ├── auth.py                 # Scrypt token authentication & SQLite store
│   ├── app_schemas.py          # Pydantic request/response schemas
│   ├── hearth_prompt.txt       # Persona system prompt
│   ├── routes/                 # Factory route modules (chat, voice, auth, etc.)
│   ├── tools/                  # Dispatchable tools (music, weather, base)
│   ├── tts/                    # Pluggable TTS engines (Piper, Kokoro)
│   ├── models/                 # ONNX/Whisper model binaries (gitignored)
│   └── tests/                  # Pytest test suite (380+ tests)
├── frontend/
│   ├── index.html              # Single-page web UI
│   ├── message.js              # Streaming chat, markdown rendering, settings menu
│   ├── voice.js                # Wake-word socket & audio recorder handling
│   ├── auth.js                 # Authentication client logic
│   ├── theme.js                # Early theme initialization (avoids flash)
│   ├── audio-processor.js      # AudioWorklet processor for PCM capture
│   └── style.css               # Responsive styling (light & dark modes)
├── mpd/
│   └── mpd.conf                # Music Player Daemon configuration
├── scripts/
│   ├── download-models.sh      # Downloads wake-word ONNX models
│   ├── download-tts-models.sh  # Downloads Kokoro TTS model assets
│   ├── download-whisper-model.sh # Downloads faster-whisper base model
│   ├── review_baseline.sh      # Full verification gate (tests, audit, bandit)
│   └── review_changed_tests.sh # Targeted git-diff test runner
└── docs/                       # Architectural specs, review gates, checklists
```

---

## Prerequisites

- **Docker** and **Docker Compose v2+**
- **mkcert** ([installation guide](https://github.com/FiloSottile/mkcert))
- An OpenAI-compatible local inference server (e.g. [llama.cpp server](https://github.com/ggerganov/llama.cpp), [vLLM](https://github.com/vllm-project/vllm), or [Ollama](https://ollama.com/))
- *(Optional)* NVIDIA GPU with CUDA Container Toolkit for accelerated local inference

---

## Quick Start

### 1. Clone & Configure

```bash
git clone https://github.com/your-username/hearth.git
cd hearth
cp .env.example .env
```

Edit `.env` to configure your model names, LAN IP, and music paths:

```bash
# Model routing
OPENAI_CHAT_MODEL=gemma-4
OPENAI_BASE_URL=http://localhost:10000/v1
MODEL_CLOUD=claude-sonnet-4-20250514
ANTHROPIC_API_KEY=your_anthropic_api_key_if_desired

# Network & TLS
CORS_ORIGINS=https://192.168.1.50,https://localhost
SESSION_COOKIE_SECURE=true

# Music Paths
MUSIC_PATH=/path/to/your/music
BEETS_DB_DIR=/path/to/your/beets
```

### 2. Generate LAN HTTPS Certificates

Use `mkcert` to issue a local certificate covering `localhost`, `127.0.0.1`, and your host's LAN IP (e.g. `192.168.1.50`):

```bash
# Install local CA on the host
mkcert -install

# Generate certificates into caddy/certs/
mkdir -p caddy/certs
mkcert -cert-file caddy/certs/cert.pem -key-file caddy/certs/key.pem \
    localhost 127.0.0.1 192.168.1.50
```

### 3. Download Runtime Models

Fetch the wake-word, whisper transcription, and TTS model files:

```bash
bash scripts/download-models.sh
bash scripts/download-tts-models.sh
bash scripts/download-whisper-model.sh
```

### 4. Start the Application

Start the Docker Compose stack:

```bash
docker compose up -d --build
```

Verify that all containers are healthy:

```bash
docker compose ps
curl -sk https://localhost/health
```

### 5. Trust the CA on Client Devices

To enable microphone access on mobile browsers, install the root CA generated by `mkcert`:

1. Locate the CA file on your host:
   ```bash
   mkcert -CAROOT
   ```
2. Transfer `rootCA.pem` to your phone or client device:
   - **iOS**: AirDrop or email `rootCA.pem` → Profile Downloaded → Install → Settings → General → About → Certificate Trust Settings → Enable Full Trust.
   - **Android**: Settings → Security & Privacy → More Security Settings → Install from device storage → CA certificate.
3. Open `https://<YOUR-LAN-IP>` in your mobile browser.

---

## Core Capabilities

### Streaming Chat & Checkpointed Sessions

Hearth orchestrates interactions through a LangGraph state machine:
- **Conversation Continuity**: Every session is saved with `SqliteSaver` checkpointing, allowing users to switch or resume conversations without state loss.
- **Budget-Aware History**: Multi-turn history is dynamically budgeted against token constraints (`CHAT_TOKEN_BUDGET`), while older turns are automatically compacted into rolling session summaries.
- **Inner-Monologue & Reasoning**: When supported by the underlying model (such as via `OPENAI_THINK=true`), thinking tokens stream directly to the frontend and can be toggled in the UI.

### Voice & Audio Pipeline

Voice interaction runs over low-latency WebSockets and SSE:
- **Wake Word Detection**: Client streams audio through an `AudioWorklet` over `WS /ws/wake`. Detected using local ONNX wake-word models.
- **Speech-to-Text**: High-speed transcription via `faster-whisper`.
- **Speech Synthesis**: Synthesizes speech with Piper or Kokoro TTS. Spoken responses are intelligently compressed into conversational summaries so the assistant speaks concisely while full markdown is rendered on screen.
- **Barge-In**: Speaking while audio is playing immediately interrupts playback.

### Multimodal Vision

Attach images directly in the chat interface:
- **Structural Intent**: When an image is attached, the request is validated and automatically routed to the vision model (bypassing text-only fastpaths).
- **Format Validation**: Ensures valid MIME types (`image/png`, `image/jpeg`, `image/webp`) and payload size boundaries.

### Hybrid Memory Layer

Hearth maintains long-term memory using a dual-storage strategy:
- **Relational Storage (SQLite)**: Tracks discrete facts, user preferences, and conversation turns.
- **Vector Retrieval (ChromaDB)**: Embeds facts and summaries into a collection (`conversation_memory`) for semantic search.
- **Periodic "Sleep" Consolidation**: When the user is idle, a background daemon distills episodic turns into generalized facts and clears expired entries without stalling active chats.
- **In-Chat Controls**: Use natural commands like `"remember that I drink green tea"` or `"forget my location"`. Sensitive data (tokens, passwords) is automatically blocked.

### Local Music & MPD Integration

Listen to your personal music collection through MPD and Beets:
- **Deterministic Fast-Path**: Natural phrases bypass the LLM entirely for instant execution:
  - *"Play Bohemian Rhapsody"*
  - *"Queue some jazz"*
  - *"Shuffle my playlist"*
  - *"Set volume to 60"*
  - *"What's playing?"*
- **Library Updates**: Update your library from the web UI settings menu ("Update music library"), via `POST /music/beets/update`, or manually from the terminal:
  ```bash
  docker compose exec backend sh -c 'cd /beets && beet update /music && beet import -A /music'
  ```


- **Question Mode**: Hearth routes explanation requests (*"how does this algorithm work?"*, *"explain this error"*) through a code-optimized prompt.
- **Safe Guardrails**: Code execution and filesystem writes are strictly disallowed; requests are kept as safe, conversational programming explanations.

---

## Configuration Reference

Key variables configured in `.env`:

| Category | Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| **Inference** | `OPENAI_BASE_URL` | `http://localhost:10000/v1` | URL of the OpenAI-compatible inference server |
| | `OPENAI_CHAT_MODEL` | `gemma-4` | Model name for conversational chat |
| | `OPENAI_VISION_MODEL` | *(same as chat)* | Model name for vision requests |
| | `MODEL_CLOUD` | `claude-sonnet-4-20250514` | Optional Anthropic model for cloud fallback |
| | `ANTHROPIC_API_KEY` | `""` | API key for Anthropic fallback (optional) |
| | `CHAT_MODEL_WARMUP` | `true` | Pre-load chat model into VRAM on startup |
| **Router** | `ROUTER_EMBED_MODEL` | `nomic-embed-text` | Model used for embedding router exemplar classification |
| | `ROUTER_EMBEDDING_ENABLED` | `true` | Enable embedding-based intent classifier |
| | `ROUTE_CONFIDENCE_THRESHOLD` | `0.70` | Confidence required before cloud fallback escalation |
| **Memory** | `MEMORY_TOP_N` | `5` | Maximum memory hits injected into context |
| | `MEMORY_SLEEP_ENABLED` | `true` | Run background memory consolidation while idle |
| | `MEMORY_SLEEP_INTERVAL_SECONDS` | `900` | Consolidation loop heartbeat interval |
| **Audio & TTS** | `TTS_ENGINE` | `piper` | Active TTS engine (`piper` or `kokoro`) |
| | `PULSE_SOCKET` | `/run/user/1000/pulse/native` | Host PulseAudio/PipeWire socket for MPD audio |
| **Music** | `MUSIC_PATH` | `/path/to/music` | Host directory containing audio files |
| | `BEETS_DB_DIR` | `/path/to/beets` | Host directory containing Beets `library.db` |
| | `MPD_HOST` | `mpd` | Hostname of MPD daemon |
| **Network & Auth** | `CORS_ORIGINS` | `*` | Allowed CORS origins (set to exact LAN HTTPS address in prod) |
| | `SESSION_COOKIE_SECURE` | `false` | Send cookies only over HTTPS (`true` when Caddy terminates TLS) |

---

## API Overview

| Method | Path | Description |
| :--- | :--- | :--- |
| `POST` | `/chat` | SSE streaming chat endpoint (supports text, voice source, and images) |
| `POST` | `/code` | Intent-biased code question endpoint |
| `POST` | `/transcribe` | Transcribes multipart audio via faster-whisper |
| `POST` | `/tts` | Synthesizes speech to `audio/wav` via Piper/Kokoro |
| `WS` | `/ws/wake` | WebSocket stream for continuous wake-word detection |
| `GET` | `/health` | Application healthcheck |
| `GET` | `/memory` | List stored facts, preferences, and summaries |
| `DELETE`| `/memory/{id}` | Delete a specific memory item |
| `POST` | `/memory/consolidate` | Manually trigger episodic memory consolidation |
| `GET` | `/chat/sessions` | List active sessions for authenticated user |
| `POST` | `/chat/session/new` | Create a new chat session |
| `POST` | `/chat/session/select` | Switch the active session |
| `DELETE`| `/chat/sessions/{id}` | Delete a specific chat session |
| `GET` | `/music/now_playing` | Inspect current track, state, and volume |
| `GET` | `/music/queue` | View current MPD playback queue |
| `POST` | `/music/control` | Control MPD playback (`pause`, `resume`, `next`, `shuffle`, `set_volume`) |
| `POST` | `/music/beets/update` | Rescan and import new tracks into Beets library |
| `POST` | `/auth/register` | Register a new user account |
| `POST` | `/auth/login` | Authenticate and obtain session token |
| `POST` | `/auth/logout` | Invalidate current session token |
| `GET` | `/auth/me` | Return authenticated user details |

---

## Local Development & Testing

### Running Without Docker

To run the FastAPI server directly on the host machine:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Use custom paths to avoid writing state into source control:
mkdir -p /tmp/hearth-local
CHROMA_PATH=/tmp/hearth-local/chroma \
MEMORY_DB_PATH=/tmp/hearth-local/memory.db \
GRAPH_CHECKPOINT_DB_PATH=/tmp/hearth-local/graph-checkpoints.sqlite \
AUTH_DB_PATH=/tmp/hearth-local/auth.db \
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### Running the Test Suites

Run the full pytest suite (380+ tests):

```bash
# In Docker:
docker compose exec -T backend sh -c 'cd /app && PYTHONPATH=/app python -m pytest -q'

# Locally from repository root:
cd backend && python -m pytest -q
```

### Security & Quality Baseline Gates

Hearth includes automated local validation scripts that mirror the GitHub Actions CI pipeline:

```bash
# Run tests for files modified relative to origin/main:
bash scripts/review_changed_tests.sh

# Run the complete review gate (tests, pip-audit, gitleaks, bandit):
bash scripts/review_baseline.sh
```

---

## Troubleshooting

- **Microphone / Voice Not Working on Mobile**:
  - Web browsers require a *Secure Context* to access `navigator.mediaDevices`. Ensure you are connecting via `https://` and have installed the `mkcert` root CA on your mobile device (see [Trust the CA](#5-trust-the-ca-on-client-devices)).
- **Container Healthcheck Fails on Startup**:
  - The startup chat-model warmup (`CHAT_MODEL_WARMUP=true`) can take 60–90 seconds while weights load into VRAM. The Compose healthcheck has a 120s `start_period` for this reason. Check logs with `docker compose logs -f backend`.
- **No Sound from MPD**:
  - Ensure your host user's PulseAudio socket is reachable and `PULSE_SOCKET` is correctly set in `.env` (typically `/run/user/1000/pulse/native`). Verify permissions with `ls -la $PULSE_SOCKET`.
- **Missing Models Warning on Boot**:
  - Run `bash scripts/download-models.sh`, `bash scripts/download-tts-models.sh`, and `bash scripts/download-whisper-model.sh` to download the required assets.

---

## Acknowledgements

Hearth builds on top of an incredible ecosystem of open-source projects:

- [FastAPI](https://fastapi.tiangolo.com/) for high-performance async APIs.
- [LangChain & LangGraph](https://github.com/langchain-ai/langgraph) for robust state machine execution and checkpointing.
- [Caddy Server](https://caddyserver.com/) for automatic, zero-config reverse proxying.
- [openWakeWord](https://github.com/dscripka/openWakeWord) for efficient on-device wake-word detection.
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) for ultra-fast local transcription.
- [Piper](https://github.com/rhasspy/piper) & [Kokoro](https://github.com/hexgrad/kokoro) for neural text-to-speech.
- [ChromaDB](https://www.trychroma.com/) & [SQLite](https://www.sqlite.org/) for hybrid vector and relational memory.
- [Beets](https://beets.io/) & [Music Player Daemon (MPD)](https://www.musicpd.org/) for local audio management.
