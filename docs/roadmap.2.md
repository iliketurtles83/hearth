# Hearth Refactor Plan — Agent Handover

## Objective

Eliminate the dual-state problem. `main.py` handles HTTP only. The graph owns all conversation reasoning. `MemoryStore` owns all persistent state. Frontend API shapes are preserved exactly — no frontend changes required.

---

## Constraints

- No data migration required — start clean
- No legacy compatibility — delete aggressively
- Frontend API response shapes must be preserved exactly
- Music fast-path stays in HTTP layer (correct architecture, do not move)
- Auth middleware stays in HTTP layer (correct architecture, do not move)

---

## Phase 1 — Demolition

Single commit. The system will be broken at the end of this phase. That is expected.

### 1.1 Delete from `main.py`

**Classes:**
- `_SessionRecord`
- `_SQLiteSessionStore`

**Module-level variables:**
- `_session_store_lock`
- `_session_store`
- `_code_write_lock` — moves into `AssistantGraphDependencies`
- `_pending_code_writes` — moves into `AssistantGraphDependencies`
- `_consolidation_loop_task` — consolidation moves into graph

**Functions:**
- `_cleanup_expired_sessions()`
- `_evict_oldest_sessions_if_needed()`
- `_session_owned_by()`
- `_get_or_create_session()`
- `_select_history_for_budget()` — duplicate of `graph.py` version
- `_normalize_summary_line()`
- `_summarize_messages_chunk()`
- `_truncate_summary()`
- `_build_episodic_record_text()`
- `_persist_session_episodic_snapshot()`
- `_spawn_episodic_persist_task()`
- `_consolidation_loop()`
- `_update_session_summary_if_needed()`
- `_build_local_prompt()` — duplicate of `graph.py` version
- `_augment_system_with_session_summary()` — duplicate of `graph.py` version
- `_augment_system_with_memories()` — duplicate of `graph.py` version
- `_should_inject_memory()` — duplicate of `graph.py` version
- `_session_preview_text()`
- `_list_sessions()`
- `_append_session_message()`

**Endpoints:**
- `GET /chat/sessions`
- `DELETE /chat/sessions/{session_id}`
- `POST /chat/session/new`
- `POST /chat/session/select`
- `DELETE /chat/session`
- `GET /chat/session/messages`

These endpoints are rebuilt in Phase 2 against `conversation_log`. The shapes are preserved.

### 1.2 Delete from `embedding_router.py`

Module-level globals that will be replaced by explicit dependency injection:
- `_router_cache`
- `_router_snapshot`
- `_router_error`
- `_router_lock`
- `get_embedding_router()`
- `get_embedding_router_snapshot()`
- `get_embedding_router_error()`
- `embedding_router_ready()`
- `warmup_embedding_router()`

Keep everything else — `build_embedding_router()`, all index/classifier classes, exemplars. Only the global singleton pattern is deleted.

### 1.3 Delete from `graph.py`

Remove all imports of the deleted globals:
```python
# delete these imports
from embedding_router import (
    get_embedding_router,          # deleted
    ...
)
```

Inside `intent_classifier` node, remove the `get_embedding_router()` call. Leave a `TODO` placeholder. This is wired in Phase 2.

### 1.4 Delete from `main.py` lifespan

Remove from `_graph_lifespan`:
- `warmup_embedding_router()` call
- `_app.state.embedding_router_ready` assignments
- `_app.state.embedding_router_error` assignments  
- `_app.state.embedding_router_snapshot` assignments
- `_consolidation_loop_task` start/cancel

Remove from `/health` endpoint:
- `embed_ready`, `embed_error`, `snapshot` fields — simplify to `{"status": "ok"}`

---

## Phase 2 — Construction

Four sequential build tasks. Each task is independently testable.

---

### Task 2.1 — `conversation_log` table in `memory.db`

Add to `MemoryStore._init_db()`:

```sql
CREATE TABLE IF NOT EXISTS conversation_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    ts          REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_convlog_session
    ON conversation_log(session_id);

CREATE INDEX IF NOT EXISTS idx_convlog_user_ts
    ON conversation_log(user_id, ts DESC);
```

Add these methods to `MemoryStore`:

**`log_turn(session_id, user_id, role, content) -> None`**
Append-only insert. No upsert, no conflict handling. Every call produces a new row.

**`get_session_turns(session_id, user_id, limit) -> list[dict]`**
```sql
SELECT role, content, ts
FROM conversation_log
WHERE session_id = ? AND user_id = ?
ORDER BY ts ASC
LIMIT ?
```
Returns `[{"role": str, "content": str, "ts": float}]`.

**`list_sessions(user_id) -> list[dict]`**
```sql
SELECT
    session_id,
    MIN(ts) as created_at,
    MAX(ts) as updated_at,
    COUNT(*) as message_count,
    (SELECT content FROM conversation_log c2
     WHERE c2.session_id = c1.session_id
     AND c2.user_id = ?
     AND c2.role = 'user'
     ORDER BY ts ASC LIMIT 1) as preview
FROM conversation_log c1
WHERE user_id = ?
GROUP BY session_id
ORDER BY updated_at DESC
```
Returns list matching existing `/chat/sessions` response shape exactly.

**`delete_session(session_id, user_id) -> str | None`**
```sql
DELETE FROM conversation_log WHERE session_id = ? AND user_id = ?
DELETE FROM summaries WHERE session_id = ? AND user_id = ?
```
Returns the `session_id` of the most recent remaining session for this user, or `None` if none remain.

**`reset_session(session_id, user_id) -> None`**
```sql
DELETE FROM conversation_log WHERE session_id = ? AND user_id = ?
DELETE FROM summaries WHERE session_id = ? AND user_id = ?
```
Clears content, same session_id survives.

---

### Task 2.2 — `memory_writer` node in `graph.py`

New terminal node. Every response-producing path flows through it before `END`.

```python
async def memory_writer(state: AssistantState) -> dict:
    user_id = state["user_id"]
    session_id = state["session_id"]
    message = state.get("message", "")
    response = state.get("response_text", "")

    # 1. Log the turn
    now = time.time()
    await asyncio.to_thread(
        deps.memory_store.log_turn,
        session_id, user_id, "user", message
    )
    if response:
        await asyncio.to_thread(
            deps.memory_store.log_turn,
            session_id, user_id, "assistant", response
        )

    # 2. Explicit memory commands + inline extraction
    # ingest_user_message handles: "remember X", "forget X",
    # "don't remember this", regex extraction of facts/preferences
    mem_result = await asyncio.to_thread(
        deps.memory_store.ingest_user_message,
        user_id,
        message,
        source=state.get("source", "text"),
    )

    # 3. Consolidation trigger
    # Count unconsolidated summaries. If above threshold,
    # run consolidation non-blocking.
    # Threshold replaces wall-clock schedule.
    unconsolidated = await asyncio.to_thread(
        deps.memory_store._count_unconsolidated, user_id
    )
    consolidation_threshold = int(
        os.getenv("MEMORY_CONSOLIDATION_THRESHOLD", "3")
    )
    if unconsolidated >= consolidation_threshold:
        asyncio.create_task(
            asyncio.to_thread(
                deps.memory_store.consolidate_pending,
                user_id,
                int(os.getenv("MEMORY_CONSOLIDATION_BATCH_SIZE", "50")),
            )
        )

    return {"memory_result": mem_result}
```

Add `_count_unconsolidated(user_id)` to `MemoryStore`:
```sql
SELECT COUNT(*) FROM summaries
WHERE user_id = ? AND consolidated = 0
```

Add `memory_result: dict` to `AssistantState`.

**Wire `memory_writer` into the graph:**

```python
# Add node
graph.add_node("memory_writer", memory_writer)

# All response-producing paths terminate here
graph.add_edge("responder", "memory_writer")
graph.add_edge("code_tool", "memory_writer")
graph.add_edge("write_executor", "memory_writer")
graph.add_edge("coding_agent_executor", "memory_writer")

# Confirmation gates do NOT go through memory_writer
# (they produce no response, just set state)
graph.add_edge("coding_agent_tool", END)

# memory_writer is the new terminal
graph.add_edge("memory_writer", END)
```

**Move `memory_result` SSE emission** from `main.py generate()` into the graph stream. `memory_writer` calls `writer({"memory": mem_result})` before returning. Remove the memory SSE block from `main.py`.

---

### Task 2.3 — `memory_retrieval` loads history from `MemoryStore`

Replace the injected history pattern with direct loading.

**In `graph.py` `memory_retrieval` node**, replace:
```python
history = list(state.get("history", []))
session_summary = str(state.get("session_summary", "") or "")
```

With:
```python
turns = await asyncio.to_thread(
    deps.memory_store.get_session_turns,
    state["session_id"],
    state["user_id"],
    limit=CHAT_MAX_TURNS * 2,
)
history = [{"role": t["role"], "content": t["content"]} for t in turns]

# session summary: most recent unconsolidated summary for this session
session_summary = await asyncio.to_thread(
    deps.memory_store.get_latest_session_summary,
    state["session_id"],
    state["user_id"],
)
```

Add `get_latest_session_summary(session_id, user_id) -> str` to `MemoryStore`:
```sql
SELECT summary FROM summaries
WHERE session_id = ? AND user_id = ?
ORDER BY created_at DESC LIMIT 1
```
Returns the summary string or empty string.

**Remove from `AssistantState`:**
- `history: list[dict]`
- `session_summary: str`

These are no longer injected — they're loaded inside the graph.

**Update `main.py` `graph_state` construction:**

```python
graph_state = {
    "user_id": user_id,
    "session_id": session_id,
    "message": request.message,
    "source": chat_source,
    "modality": "voice" if chat_source == "voice" else "chat",
    "system": effective_system,
    "image_base64": request.image_base64,
    "image_mime": request.image_mime,
}
```

Remove `"history"`, `"session_summary"` keys. They no longer exist in the handoff.

---

### Task 2.4 — Thread `embedding_router` through `AssistantGraphDependencies`

**In `graph.py`**, add to `AssistantGraphDependencies`:
```python
@dataclass
class AssistantGraphDependencies:
    memory_store: MemoryStore
    embedding_router: EmbeddingIntentRouter | None  # ADD
    stream_local: ...
    stream_cloud: ...
    stream_local_vision: ...
    tool_dispatch: ...
    chat_model: str
    cloud_model: str
    coder_model: str
    vision_model: str
    chroma_path: str
```

**In `graph.py` `intent_classifier` node**, replace:
```python
embed_router = get_embedding_router()
```
With:
```python
embed_router = deps.embedding_router
```

**In `main.py` `_graph_lifespan`**, replace the warmup block with:
```python
embed_router = None
embed_snapshot = None
try:
    embed_router, embed_snapshot = await build_embedding_router()
    log.info(
        "embedding_router.ready | model=%s dim=%d",
        embed_snapshot.model, embed_snapshot.dim,
    )
except Exception as exc:
    log.warning("embedding_router.failed | error=%s | using heuristic fallback", exc)

deps = _make_graph_deps(embedding_router=embed_router)
```

**Update `_make_graph_deps()`** to accept and pass through `embedding_router`.

**Update `/health` endpoint** to reflect actual status without the deleted state fields:
```python
@app.get("/health")
async def health():
    graph_ready = getattr(app.state, "assistant_graph", None) is not None
    return {
        "status": "ok" if graph_ready else "starting",
        "embedding_router": embed_router is not None,
    }
```

---

### Task 2.5 — Rebuild session endpoints

New thin endpoints in `main.py` (or a new `session_routes.py`) backed entirely by `MemoryStore` methods from Task 2.1.

All shapes match what the frontend currently expects exactly.

**`GET /chat/sessions`**
```python
@app.get("/chat/sessions")
async def list_chat_sessions(http_request: Request):
    user_id = http_request.state.user_id
    current_session_id = http_request.cookies.get(SESSION_COOKIE_NAME)
    sessions = memory_store.list_sessions(user_id)
    return JSONResponse({
        "sessions": sessions,
        "current_session_id": current_session_id,
    })
```

**`GET /chat/session/messages`**
```python
@app.get("/chat/session/messages")
async def get_session_messages(http_request: Request):
    user_id = http_request.state.user_id
    session_id = http_request.cookies.get(SESSION_COOKIE_NAME) or str(uuid4())
    turns = memory_store.get_session_turns(session_id, user_id, limit=500)
    response = JSONResponse({"session_id": session_id, "messages": turns})
    _set_session_cookie(response, session_id)
    return response
```

**`POST /chat/session/new`**
```python
@app.post("/chat/session/new")
async def new_session(http_request: Request):
    session_id = str(uuid4())
    response = JSONResponse({"ok": True, "session_id": session_id})
    _set_session_cookie(response, session_id)
    return response
```
No server state created. The session comes into existence when the first turn is logged by `memory_writer`.

**`POST /chat/session/select`**
```python
@app.post("/chat/session/select")
async def select_session(payload: SessionSelectRequest, http_request: Request):
    user_id = http_request.state.user_id
    sessions = memory_store.list_sessions(user_id)
    if not any(s["session_id"] == payload.session_id for s in sessions):
        return _error_response("Session not found", "SESSION_NOT_FOUND", False, 404)
    response = JSONResponse({"ok": True, "session_id": payload.session_id})
    _set_session_cookie(response, payload.session_id)
    return response
```

**`DELETE /chat/sessions/{session_id}`**
```python
@app.delete("/chat/sessions/{session_id}")
async def delete_session(session_id: str, http_request: Request):
    user_id = http_request.state.user_id
    current = http_request.cookies.get(SESSION_COOKIE_NAME)
    memory_store.delete_session(session_id, user_id)
    next_session_id = None
    if current == session_id:
        sessions = memory_store.list_sessions(user_id)
        next_session_id = sessions[0]["session_id"] if sessions else str(uuid4())
    payload = {"ok": True, "session_id": session_id}
    if next_session_id:
        payload["active_session_id"] = next_session_id
    response = JSONResponse(payload)
    if next_session_id:
        _set_session_cookie(response, next_session_id)
    return response
```

**`DELETE /chat/session`** (reset current session)
```python
@app.delete("/chat/session")
async def reset_session(http_request: Request):
    user_id = http_request.state.user_id
    session_id = http_request.cookies.get(SESSION_COOKIE_NAME) or str(uuid4())
    memory_store.reset_session(session_id, user_id)
    response = JSONResponse({"ok": True, "session_id": session_id})
    _set_session_cookie(response, session_id)
    return response
```

---

## Final `main.py` `/chat` Handler

After all tasks complete, the handler is:

```python
@app.post("/chat")
async def chat(request: ChatRequest, http_request: Request):
    user_id = http_request.state.user_id
    session_id = http_request.cookies.get(SESSION_COOKIE_NAME) or str(uuid4())
    chat_source = _normalize_chat_source(request.source)
    effective_system = request.system or CHAT_DEFAULT_SYSTEM_PROMPT

    # Music fast-path
    music_cmd = parse_music_command(request.message)
    if music_cmd is not None:
        music_cmd["prompt"] = request.message
        music_cmd["user_id"] = user_id
        return _music_fast_path_response(music_cmd, session_id)

    # Image validation
    image_error = _validate_image(request.image_base64, request.image_mime)
    if image_error:
        return JSONResponse({"error": image_error, "code": "INVALID_IMAGE"}, status_code=422)

    graph_state = {
        "user_id": user_id,
        "session_id": session_id,
        "message": request.message,
        "system": effective_system,
        "source": chat_source,
        "modality": "voice" if chat_source == "voice" else "chat",
        "image_base64": request.image_base64,
        "image_mime": request.image_mime,
    }

    graph_runner = getattr(app.state, "assistant_graph", _assistant_graph)

    async def generate():
        async for event in graph_runner.astream(
            graph_state,
            config=checkpoint_config(session_id),
            stream_mode="custom",
        ):
            if "meta" in event:
                yield f"data: {json.dumps(event['meta'])}\n\n"
            elif "text" in event:
                yield f"data: {json.dumps({'text': event['text']})}\n\n"
            elif "memory" in event:
                yield f"data: {json.dumps({'memory': event['memory']})}\n\n"
            elif "notice" in event:
                yield f"data: {json.dumps({'notice': event['notice']})}\n\n"
            elif event.get("fallback"):
                yield f"data: {json.dumps(event)}\n\n"
        yield "data: [DONE]\n\n"

    response = StreamingResponse(generate(), media_type="text/event-stream")
    _set_session_cookie(response, session_id)
    return response
```

---

## Completion Checklist

```
Phase 1 — Demolition
  [ ] _SessionRecord, _SQLiteSessionStore deleted
  [ ] All listed functions deleted from main.py
  [ ] All listed endpoints deleted from main.py
  [ ] embedding_router.py globals and warmup deleted
  [ ] graph.py imports of deleted globals cleaned up
  [ ] intent_classifier TODO placeholder left for router wiring

Phase 2 — Construction
  [ ] 2.1 conversation_log table + 5 MemoryStore methods
  [ ] 2.2 memory_writer node + graph edges rewired
  [ ] 2.3 memory_retrieval loads history from MemoryStore
  [ ] 2.3 history/session_summary removed from AssistantState
  [ ] 2.3 graph_state construction in main.py simplified
  [ ] 2.4 embedding_router in AssistantGraphDependencies
  [ ] 2.4 intent_classifier uses deps.embedding_router
  [ ] 2.4 lifespan builds router and passes through deps
  [ ] 2.5 All 5 session endpoints rebuilt against conversation_log
  [ ] /chat handler simplified to final form

Verification
  [ ] Send a message, check conversation_log has two rows (user + assistant)
  [ ] Reload page, history renders from conversation_log
  [ ] Create new session, send message, switch back to first session
  [ ] Delete session, confirm redirect to next session
  [ ] Say "remember my name is X", check facts table
  [ ] Say "forget my name", check facts table deleted
  [ ] Embedding router logs "embedding_router.ready" on startup
  [ ] Heuristic fallback logs correctly when Ollama is down
```

---

## What Does Not Change

- `auth.py` and `auth_routes.py` — untouched
- `music_fastpath.py` — untouched  
- `memory_tool_routes.py` — untouched
- `code_file_routes.py` — untouched
- All frontend files — untouched
- SSE event shapes consumed by frontend — identical
- `MemoryStore` existing methods — additive only, nothing modified
- `graph.py` node logic — only `memory_retrieval` and `intent_classifier` change internally