# Parked Ideas & Future Backlog

This document captures architecture ideas and feature proposals that are parked for future implementation.

---

## 1. Dynamic Model Selector in Chat UI

**Status**: Parked

### Concept
Allow users to select which local LLM handles their request directly from the chat interface, rather than relying strictly on the environment variable default (`OPENAI_CHAT_MODEL`).

### Proposed Architecture

1. **Backend Endpoint**:
   - `GET /models`
   - Queries Ollama's local tags (`/api/tags`) or local llama.cpp endpoints to list available models.
   - Returns a list of installed models: e.g. `["gemma-4", "qwen2.5-coder:7b", "llama3.2:3b"]`.

2. **Schema & Graph Integration**:
   - Add `model: str | None = None` to `ChatRequest` in `backend/app_schemas.py`.
   - In `backend/graph.py`, pass `state.get("model")` through to the responder and `deps.stream_local(..., model_name=selected_model)`.

3. **Frontend UI**:
   - Add a subtle model dropdown button in `#input-inner` (next to the chat prompt input).
   - Fetch available models from `GET /models` on startup.
   - Allow user to switch models with the selection saved in `localStorage['hearth:selected_model']`.

---

## 2. Distributed Music Outputs — Home Assistant & Music Assistant Integration (Phases 2 & 3)

**Status**: Parked (Phase 1 Native Output Selection complete)

### Concept
Extend Hearth's audio output selector beyond local host speakers and the personal browser/phone stream to integrate with LAN smart speakers via **Home Assistant** and multi-room sync engines via **Music Assistant**.

### Roadmap

#### Phase 2: Home Assistant `media_player` Bridge
1. **Discovery & Auth**:
   - Configure `HASS_URL` and `HASS_TOKEN` (Long-Lived Access Token) in `.env`.
   - Query Home Assistant's REST/WebSocket API (`GET /api/states`) to discover all `media_player.*` entities (e.g. Google Nest, Sonos, smart TVs, Apple TV).
2. **Audio Delivery**:
   - When a Home Assistant player is selected in the Hearth output switcher (or via voice *"Play jazz on living room speaker"*), call HA's `media_player.play_media` service:
     - `media_content_id`: `https://<hearth-lan-ip>/music/stream` (or direct track URL)
     - `media_content_type`: `music`
3. **Transport Synchronization**:
   - Mirror pause/resume/stop/volume commands to the active HA `media_player` entity.

#### Phase 3: Music Assistant (MASS) Integration
1. **Core Concept**:
   - Music Assistant provides native player grouping, multi-room sample-accurate sync, crossfading, DSP, and direct support for Sonos, AirPlay, Cast, DLNA, and Squeezebox/Slimproto.
2. **Two-Way Integration**:
   - Option A: Hearth acts as an AI/voice frontend commanding Music Assistant's WebSocket API (`mass.players`, `mass.music.play`).
   - Option B: Expose Hearth's MPD stream as an external player provider / stream source inside Music Assistant.

