# LLM-Native Tool Calling Migration

> **STATUS: PARKED** — Abandoned for now to limit LLM calls and conserve GPU resources. The current regex/embedding routing stack remains in place. Revisit if the tradeoff is reconsidered. (2026-09-05)

## Problem

The current routing stack is three layers of regex/embedding classification that all fail at understanding intent:

```
regex fastpath → regex classifier → embedding router → graph → LLM
```

**Observed failures:**

- "play a game with me" → music fastpath regex matches "play" + "music" keywords, dispatches to MPD, returns "No tracks found"
- "I wish the weather would get better" → no keyword hit, falls through to LLM which can't fetch live weather data
- "what's the weather in Tallinn" → works (keyword hit), but only by coincidence of pattern matching

The fundamental flaw: regex can't understand intent, only match patterns. The embedding router adds another classification layer without solving the core problem.

## Proposed Architecture

```
[LLM with tool schemas] → decides: call tool or respond directly
```

One layer. The LLM itself acts as the router based on tool schemas and context — the approach modern architectures have shifted toward.

## Goals

1. Remove all regex-based routing (music fastpath, intent classifier, embedding router)
2. LLM-native tool calling via Ollama `/api/chat` with tool schemas
3. Strip auto-cloud routing (reasoning-heavy, code-question, `use_cloud`)
4. Keep cloud infrastructure for future manual toggle
5. Add model selector UI (dropdown to pick local model)
6. Fix badge display (show tool name for tool responses, model name for direct responses)

## What Changes

### New flow

```
User prompt
    │
    ▼
┌──────────────┐
│ LLM with     │  ← tool schemas (weather, music) passed in request
│ tool schemas │     model decides: call tool(s) or respond directly
└──────┬───────┘
       │
       ├── tool_call → execute tool(s) → format results → loop back
       │
       └── no tool → direct response streamed to client
```

### What stays

- Memory retrieval, session management, vision, TTS, voice
- Cloud fallback infrastructure (Anthropic client kept in place)
- Session checkpointing (LangGraph)

### Tradeoff

Every request goes through the LLM (even "pause", "stop", "next"). Slightly slower for simple commands (~200-500ms for tool call), but much more natural and consistent. No more regex edge cases.

## Files to Change

### Backend

1. **`intents.py`**
   - Remove intent classification logic (`classify_intent`, `route`, all pattern/keyword arrays)
   - Remove `reasoning-heavy`, `code-question` intent categories
   - Remove `use_cloud` from `RouteDecision`
   - Keep: `CHAT_MODEL`, `CLOUD_MODEL`, `VISION_MODEL` constants, `RouteDecision` dataclass (simplified)

2. **`embedding_router.py`**
   - Remove entirely (or strip to data classes only if tests depend on it)
   - No longer needed — LLM handles routing

3. **`routing_config.py`**
   - Remove `route_confidence_threshold`
   - Keep: `chat_token_budget`, `chat_max_turns`, `ollama_url`, router embed settings (may still be useful)

4. **`music_fastpath.py`**
   - Remove entirely (or keep as reference). No longer needed — LLM handles music commands via tool calling

5. **`graph.py`**
   - Replace `responder` node with tool-calling loop using Ollama `/api/chat`
   - Tool-calling loop: call model → if tool_calls: execute → format results → loop back → else: stream response
   - Remove `use_cloud` from `AssistantState`
   - Remove `cloud_model` and `stream_cloud` from `AssistantGraphDependencies`
   - Remove cloud responder path (`elif state.get("use_cloud")`)
   - Remove forced code path (`force_code`) and write-downgrade logic
   - Keep: vision cloud fallback (user-triggered image path)
   - Keep: memory retrieval, history loading, memory writing
   - Keep: vision path (image classification is deterministic)

6. **`main.py`**
   - Define tool schemas (weather, music) as JSON-serializable dicts for Ollama tool calling
   - Remove `stream_cloud` and `_late_stream_cloud`
   - Remove `stream_cloud`/`cloud_model` from `_make_graph_deps`
   - Add `model` field to `ChatRequest` schema
   - Add `GET /models` endpoint listing Ollama models (via `/api/tags`)
   - Pass model through the graph
   - Keep: Anthropic client and `stream_cloud` code (commented/kept for future manual toggle)
   - Keep: music fastpath import (or remove if we go fully LLM-native)

7. **`app_schemas.py`**
   - Add `model: str | None = None` to `ChatRequest`

### Frontend

8. **`index.html`**
   - Add model selector dropdown button in `#input-inner`, before the text input
   - Click opens list of available models, click selects

9. **`message.js`**
   - Fetch available models from `GET /models` on bootstrap
   - Model selector dropdown: click opens list, click selects, save preference to localStorage
   - Pass selected model in chat requests
   - Fix `appendModelBadge()`: show model name for direct responses, tool name for tool responses

10. **`style.css`**
    - Styles for model selector dropdown button and dropdown list

### Tests

11. **`tests/test_router.py`**
    - Remove reasoning-heavy + code-question tests
    - Remove `use_cloud` assertions
    - Update remaining tests for simplified intent classification

12. **`tests/test_graph.py`**
    - Remove `TEST_CLOUD_MODEL`, `_fake_stream_cloud`, `use_cloud`/`cloud_model`/`route_type` assertions
    - Update tests for tool-calling responder

13. **`tests/test_responder_modality.py`**
    - Remove cloud model references

14. **`tests/test_embedding_router.py`**
    - Remove entirely or update for stripped-down router

15. **`tests/test_chat_sessions.py`**
    - Update `route_type` assertions

16. **`tests/test_vision.py`**
    - Update `use_cloud` assertions

17. **New: `tests/test_models_endpoint.py`**
    - Test `GET /models` endpoint

## Implementation Order

1. **Backend: Remove routing layers** — strip intents, embedding router, music fastpath, auto-cloud
2. **Backend: Implement tool-calling responder** — new responder node with LLM tool calling loop
3. **Backend: Add `/models` endpoint and `model` field** — model selector support
4. **Frontend: Model selector UI** — dropdown, fetch models, pass model in requests
5. **Frontend: Fix badge display** — tool name vs model name
6. **Tests: Update all test files** — remove obsolete tests, update assertions, add new tests
7. **Integration: End-to-end validation** — test tool calling, model switching, badge display

## Notes

- Ollama supports tool calling via `/api/chat` when the model has been trained for it (Llama 3.1+, Mistral-large, Qwen2.5, Gemma 2/4)
- Gemma 4 supports tool calling — confirmed
- Tool schemas are passed as a `tools` array in the `/api/chat` request payload
- The model returns `tool_calls` in its response when it decides to use a tool
- Multiple tool calls can be returned in a single response (batch execution)
- The responder loop continues until the model returns a direct text response
