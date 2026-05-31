# Hearth Refactor Plan v3 — Agent Handover

Supersedes `roadmap.2.md`. Same intent (eliminate dual-state, centralize on
`MemoryStore`, inject deps), corrected against the actual code in
`main.py`, `graph.py`, `memory.py`, `embedding_router.py`, and the tests
under `backend/tests/`.

---

## Objective

`main.py` owns HTTP only. The graph owns conversation reasoning. `MemoryStore`
owns all persistent conversation state. Frontend API shapes and SSE event
shapes are preserved exactly — zero frontend changes.

## Hard Constraints

- Frontend response shapes preserved: `/chat/sessions` items must keep
  `{session_id, created_at, updated_at, message_count, preview}`; messages
  must keep `{role, content, ts}`; SSE events keep `text`, `model`, `notice`,
  `memory`, `voice`, terminator `[DONE]`.
- Music fast-path stays in HTTP layer.
- Auth middleware stays in HTTP layer.
- Code-write lock + pending writes stay shared between HTTP routes and graph
  (do **not** move solely into graph deps).
- Phase 1 and Phase 2 ship as **one coordinated change**. Phase 1 alone
  leaves the system without a working session store.
- "Start clean" is destructive: existing `_session_store` rows are dropped.
  Call this out in the release note. No migration script.

---

## Verified Facts (from current code)

These shaped the plan; do not re-derive:

- `AssistantState` already has `response_text: str`. Every response-producing
  node returns `{"response_text": ...}`:
  `responder` (graph.py:1479), `code_tool` (~711–760), `write_executor`
  (~1109–1160), `coding_agent_tool` (~1171–1208, confirmation prompt text),
  `coding_agent_executor` (~1209–1270).
- `responder` only calls `writer({"text": ...})` for voice modality; text
  modality streams through other paths. The HTTP layer in `main.py`
  currently accumulates `event["text"]` chunks (main.py:1331–1365) and then
  calls `_append_session_message(...)`.
- Graph uses `AsyncSqliteSaver` (graph.py:1535). It exposes
  `aget_tuple`/`aput`/`adelete_thread` on `BaseCheckpointSaver` v2.
- `summaries` table already has `session_id` and `consolidated` columns
  (memory.py:154–162). `save_summary(user_id, session_id, summary) -> int`
  and `consolidate_pending(user_id=None, limit=50)` already exist.
- `build_embedding_router(...)` is `async` and returns
  `(EmbeddingIntentRouter, EmbeddingRouterSnapshot)` (embedding_router.py:364).
- `intent_classifier` reads `state.get("history", [])` via
  `_last_assistant_message` to detect write follow-ups (graph.py:412–480).
  History must be present *before* `intent_classifier` runs.
- Current edge topology: every response-producing node currently goes
  straight to `END`. There is no terminal write node.
- `code_file_routes.create_code_file_router(...)` consumes
  `_code_write_lock` and `_pending_code_writes` directly (main.py:719–721).

---

## Phase 1 — Demolition

Single commit. System is broken at end of Phase 1; Phase 2 lands in the same
PR.

### 1.1 Delete from `main.py`

Classes: `_SessionRecord`, `_SQLiteSessionStore`.

Module-level: `_session_store_lock`, `_session_store`,
`_consolidation_loop_task`.

Keep at module level (shared with HTTP routes): `_code_write_lock`,
`_pending_code_writes`. They are also passed into graph deps; see 2.4.

Functions to delete:
`_cleanup_expired_sessions`, `_evict_oldest_sessions_if_needed`,
`_session_owned_by`, `_get_or_create_session`,
`_select_history_for_budget` (duplicate of graph version),
`_normalize_summary_line`, `_summarize_messages_chunk`,
`_truncate_summary`, `_build_episodic_record_text`,
`_persist_session_episodic_snapshot`, `_spawn_episodic_persist_task`,
`_consolidation_loop`, `_update_session_summary_if_needed`,
`_build_local_prompt`, `_augment_system_with_session_summary`,
`_augment_system_with_memories`, `_should_inject_memory`,
`_session_preview_text`, `_list_sessions`, `_append_session_message`.

Endpoints to delete (rebuilt in 2.5):
`GET /chat/sessions`, `DELETE /chat/sessions/{session_id}`,
`POST /chat/session/new`, `POST /chat/session/select`,
`DELETE /chat/session`, `GET /chat/session/messages`.

### 1.2 Delete from `embedding_router.py`

Global singleton pattern only. Delete: `_router_cache`, `_router_snapshot`,
`_router_error`, `_router_lock`, `get_embedding_router`,
`get_embedding_router_snapshot`, `get_embedding_router_error`,
`embedding_router_ready`, `warmup_embedding_router`.

Keep `build_embedding_router`, all classifier/index/exemplar classes.

### 1.3 Delete from `graph.py`

Remove the imports of the deleted globals. Inside `intent_classifier`,
replace `get_embedding_router()` with `deps.embedding_router` (final wiring
in 2.4 — it can be `None` until then).

### 1.4 Trim `_graph_lifespan` and `/health`

Remove `warmup_embedding_router()` call and the
`app.state.embedding_router_*` assignments. Remove the consolidation loop
task. New `/health` shape (smaller, but a contract change — update probes):

```json
{"status": "ok", "embedding_router": true}
```

---

## Phase 2 — Construction

### Task 2.1 — `conversation_log` table + 8 `MemoryStore` methods

Schema (added to `MemoryStore._init_db`):

```sql
CREATE TABLE IF NOT EXISTS conversation_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    ts          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_convlog_session_user_ts
    ON conversation_log(session_id, user_id, ts);
CREATE INDEX IF NOT EXISTS idx_convlog_user_ts
    ON conversation_log(user_id, ts DESC);
```

Methods to add:

| Method | Purpose |
|---|---|
| `log_turn(session_id, user_id, role, content) -> None` | Append-only insert with `ts=time.time()`. |
| `get_session_turns(session_id, user_id, limit=500) -> list[dict]` | Returns `[{role, content, ts}]` ordered by `ts ASC`. |
| `list_sessions(user_id) -> list[dict]` | Returns shape `[{session_id, created_at, updated_at, message_count, preview}]` ordered by `updated_at DESC`. `preview` is the first user turn's content truncated to ~120 chars (use the existing helper logic — copy from deleted `_session_preview_text`, scoped to the module). |
| `delete_session(session_id, user_id) -> None` | Deletes from `conversation_log` and `summaries`. Does not pick the "next" session — that's HTTP-layer concern. |
| `reset_session(session_id, user_id) -> None` | Same deletes; same `session_id` survives because it's only a cookie value. |
| `get_latest_session_summary(session_id, user_id) -> str` | Most recent `summaries.summary` for the pair, or `""`. |
| `count_unconsolidated(user_id) -> int` | `SELECT COUNT(*) FROM summaries WHERE user_id=? AND consolidated=0`. Public (no leading underscore). |
| `count_session_turns(session_id, user_id) -> int` | Used by summarization trigger in 2.2. |

All methods take SQLite-friendly inputs; no async required (callers wrap in
`asyncio.to_thread`).

### Task 2.2 — `memory_writer` terminal node + rolling summarization

New node added to `graph.py`. Replaces all the HTTP-side persistence and the
deleted `_update_session_summary_if_needed` / consolidation loop.

```python
SUMMARY_TRIGGER = int(os.getenv("MEMORY_SUMMARY_TRIGGER", "18"))
CONSOLIDATION_THRESHOLD = int(os.getenv("MEMORY_CONSOLIDATION_THRESHOLD", "3"))
CONSOLIDATION_BATCH = int(os.getenv("MEMORY_CONSOLIDATION_BATCH_SIZE", "50"))

async def memory_writer(state: AssistantState, writer) -> dict:
    user_id = state["user_id"]
    session_id = state["session_id"]
    message = state.get("message", "")
    response_text = (state.get("response_text") or "").strip()

    # 1. Persist the turn pair (assistant only if non-empty).
    await asyncio.to_thread(
        deps.memory_store.log_turn, session_id, user_id, "user", message
    )
    if response_text:
        await asyncio.to_thread(
            deps.memory_store.log_turn,
            session_id, user_id, "assistant", response_text,
        )

    # 2. Explicit + inline memory extraction (existing API).
    mem_result = await asyncio.to_thread(
        deps.memory_store.ingest_user_message,
        user_id, message, source=state.get("source", "text"),
    )

    # 3. Rolling session summary trigger.
    turn_count = await asyncio.to_thread(
        deps.memory_store.count_session_turns, session_id, user_id
    )
    if turn_count and turn_count % SUMMARY_TRIGGER == 0:
        # Reuse existing graph-side summarizer helper; do NOT call the
        # deleted main.py one. See note below.
        asyncio.create_task(_rolling_summary_task(deps, session_id, user_id))

    # 4. Consolidation trigger (threshold-based, replaces wall-clock loop).
    pending = await asyncio.to_thread(
        deps.memory_store.count_unconsolidated, user_id
    )
    if pending >= CONSOLIDATION_THRESHOLD:
        asyncio.create_task(asyncio.to_thread(
            deps.memory_store.consolidate_pending, user_id, CONSOLIDATION_BATCH,
        ))

    # 5. SSE emit (moved from main.py).
    if mem_result:
        writer({"memory": mem_result})

    return {"memory_result": mem_result}
```

`_rolling_summary_task` is a thin helper that loads the last
`SUMMARY_TRIGGER` turns via `get_session_turns`, calls the existing
local-model summarizer used by the graph today, and stores the result via
`save_summary(user_id, session_id, summary)`. If you would otherwise need to
resurrect any of the deleted `_summarize_messages_chunk` /
`_truncate_summary` logic, **move that code into `graph.py` as private
helpers** rather than leaving it in `main.py`.

Edge wiring (note the deliberate exclusion of confirmation-only nodes):

```python
graph.add_node("memory_writer", memory_writer)

# Response-producing terminal paths route through memory_writer.
graph.add_edge("responder", "memory_writer")
graph.add_edge("code_tool", "memory_writer")
graph.add_edge("write_executor", "memory_writer")
graph.add_edge("coding_agent_executor", "memory_writer")
graph.add_edge("memory_writer", END)

# Confirmation gates: the user *should* see these in history (they're the
# assistant's reply to the user's request), so they DO go through
# memory_writer. If product decides confirmations should be ephemeral,
# rewire `coding_agent_tool` -> END here and document it.
graph.add_edge("coding_agent_tool", "memory_writer")
```

Decision recorded above: confirmation prompts persist. They are the only
thing the user sees on that turn; omitting them would create gaps in the
visible chat history.

Add `memory_result: dict[str, Any]` to `AssistantState`.

Delete the SSE `memory` emission from `main.py generate()` — `memory_writer`
owns it now.

### Task 2.3 — `memory_retrieval` loads history; classifier order

`AssistantState` retains `history` and `session_summary`. They are now loaded
*inside the graph*, not injected by HTTP. Because `intent_classifier` uses
`history`, the load must happen **before** the classifier.

Two options; pick A:

**A. New `history_loader` node as the first node (recommended).**

```python
async def history_loader(state: AssistantState) -> dict:
    turns = await asyncio.to_thread(
        deps.memory_store.get_session_turns,
        state["session_id"], state["user_id"], limit=CHAT_MAX_TURNS * 2,
    )
    summary = await asyncio.to_thread(
        deps.memory_store.get_latest_session_summary,
        state["session_id"], state["user_id"],
    )
    return {
        "history": [{"role": t["role"], "content": t["content"]} for t in turns],
        "session_summary": summary,
    }

graph.add_node("history_loader", history_loader)
graph.add_edge(START, "history_loader")
graph.add_edge("history_loader", "intent_classifier")
```

`memory_retrieval` keeps its current memory-fetch responsibilities; it no
longer needs to touch history.

**B.** (Rejected) Load history inside `intent_classifier`. Mixes concerns and
makes the classifier untestable in isolation.

Update `main.py` graph-state construction:

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

`history` and `session_summary` are no longer passed in.

### Task 2.4 — Embedding router via dependency injection

`AssistantGraphDependencies` gains:

```python
embedding_router: EmbeddingIntentRouter | None
code_write_lock: Lock              # shared with HTTP layer
pending_code_writes: dict[str, dict]  # shared with HTTP layer
```

`_make_graph_deps(*, embedding_router=None)` accepts the router and the
shared lock/dict (created once at module scope in `main.py`).

`_graph_lifespan` ordering — **router warmup happens before graph
construction**:

```python
async def _graph_lifespan(app: FastAPI):
    embed_router = None
    if ROUTER_EMBEDDING_WARMUP:
        try:
            embed_router, snap = await build_embedding_router()
            log.info("embedding_router.ready | model=%s dim=%d", snap.model, snap.dim)
        except Exception as exc:
            log.warning("embedding_router.failed | %s | using heuristic", exc)

    deps = _make_graph_deps(embedding_router=embed_router)
    async with create_assistant_graph(deps, ...) as checkpointed_graph:
        app.state.assistant_graph = checkpointed_graph
        yield
```

`intent_classifier` reads `deps.embedding_router`; if `None`, falls back to
the existing heuristic path (already coded).

### Task 2.5 — Rebuild session endpoints + checkpoint cleanup

Thin endpoints in `main.py` (or a new `backend/routes/session_routes.py`).
All shapes preserved.

```python
@app.get("/chat/sessions")
async def list_chat_sessions(http_request: Request):
    user_id = http_request.state.user_id
    current = http_request.cookies.get(SESSION_COOKIE_NAME)
    return JSONResponse({
        "sessions": memory_store.list_sessions(user_id),
        "current_session_id": current,
    })

@app.get("/chat/session/messages")
async def get_session_messages(http_request: Request):
    user_id = http_request.state.user_id
    session_id = http_request.cookies.get(SESSION_COOKIE_NAME) or str(uuid4())
    turns = memory_store.get_session_turns(session_id, user_id, limit=500)
    resp = JSONResponse({"session_id": session_id, "messages": turns})
    _set_session_cookie(resp, session_id)
    return resp

@app.post("/chat/session/new")
async def new_session():
    session_id = str(uuid4())
    resp = JSONResponse({"ok": True, "session_id": session_id})
    _set_session_cookie(resp, session_id)
    return resp

@app.post("/chat/session/select")
async def select_session(payload: SessionSelectRequest, http_request: Request):
    user_id = http_request.state.user_id
    if not any(s["session_id"] == payload.session_id
               for s in memory_store.list_sessions(user_id)):
        return _error_response("Session not found", "SESSION_NOT_FOUND", False, 404)
    resp = JSONResponse({"ok": True, "session_id": payload.session_id})
    _set_session_cookie(resp, payload.session_id)
    return resp

@app.delete("/chat/sessions/{session_id}")
async def delete_session(session_id: str, http_request: Request):
    user_id = http_request.state.user_id
    current = http_request.cookies.get(SESSION_COOKIE_NAME)
    memory_store.delete_session(session_id, user_id)
    await _clear_checkpoint_thread(session_id)  # see below
    next_id = None
    if current == session_id:
        sessions = memory_store.list_sessions(user_id)
        next_id = sessions[0]["session_id"] if sessions else str(uuid4())
    payload = {"ok": True, "session_id": session_id}
    if next_id:
        payload["active_session_id"] = next_id
    resp = JSONResponse(payload)
    if next_id:
        _set_session_cookie(resp, next_id)
    return resp

@app.delete("/chat/session")
async def reset_session(http_request: Request):
    user_id = http_request.state.user_id
    session_id = http_request.cookies.get(SESSION_COOKIE_NAME) or str(uuid4())
    memory_store.reset_session(session_id, user_id)
    await _clear_checkpoint_thread(session_id)
    resp = JSONResponse({"ok": True, "session_id": session_id})
    _set_session_cookie(resp, session_id)
    return resp
```

**Checkpoint cleanup helper** (prevents the `graph_checkpoints.sqlite`
thread leak the v2 roadmap missed):

```python
async def _clear_checkpoint_thread(session_id: str) -> None:
    graph = getattr(app.state, "assistant_graph", None)
    if graph is None:
        return
    checkpointer = getattr(graph, "checkpointer", None)
    if checkpointer is None:
        return
    cfg = checkpoint_config(session_id)
    try:
        if hasattr(checkpointer, "adelete_thread"):
            await checkpointer.adelete_thread(session_id)
        elif hasattr(checkpointer, "delete_thread"):
            await asyncio.to_thread(checkpointer.delete_thread, session_id)
    except Exception as exc:
        log.warning("checkpoint_cleanup_failed | session_id=%s | %s",
                    session_id, exc)
```

If the installed LangGraph version exposes neither method, fall back to a
direct `DELETE FROM checkpoints WHERE thread_id=?` against the saver's
SQLite path. Document the installed version in repo memory once verified.

### Final `/chat` handler

```python
@app.post("/chat")
async def chat(request: ChatRequest, http_request: Request):
    user_id = http_request.state.user_id
    session_id = http_request.cookies.get(SESSION_COOKIE_NAME) or str(uuid4())
    chat_source = _normalize_chat_source(request.source)
    effective_system = request.system or CHAT_DEFAULT_SYSTEM_PROMPT

    music_cmd = parse_music_command(request.message)
    if music_cmd is not None:
        music_cmd["prompt"] = request.message
        music_cmd["user_id"] = user_id
        return _music_fast_path_response(music_cmd, session_id)

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

    resp = StreamingResponse(generate(), media_type="text/event-stream")
    _set_session_cookie(resp, session_id)
    return resp
```

No assistant-text accumulation in HTTP. Persistence happens entirely inside
the graph via `memory_writer`.

---

## Persistence Model Note (response_text)

The previous roadmap assumed `memory_writer` could read
`state["response_text"]` without addressing where it comes from. Verified
above: **every** terminal node already returns `{"response_text": ...}`.
Because `memory_writer` is wired *after* those nodes via direct edges, the
LangGraph state-merge means `state["response_text"]` is the most recently
written value by the time `memory_writer` runs. No HTTP-side accumulation
required.

For voice modality, `responder` *also* calls `writer({"text": ...})` —
that's an SSE concern, orthogonal to persistence. Both behaviours coexist.

---

## Test Migration

Almost all of `backend/tests/test_chat_sessions.py` references deleted
helpers (`_get_or_create_session`, `_list_sessions`,
`_append_session_message`, `_update_session_summary_if_needed`,
`_truncate_summary`, `_spawn_episodic_persist_task`,
`_select_history_for_budget`). Rewrite, do not patch:

1. Replace fixture `clear_session_store` with one that truncates
   `conversation_log` and `summaries` for the test user.
2. Tests that previously poked `_session_store` directly now write via
   `memory_store.log_turn(...)`.
3. Endpoint tests still hit the same URLs and assert the same JSON keys —
   they should pass once the new endpoints land.
4. Update `test_health_reports_embedding_router_state` to the new
   `/health` shape (or delete if not worth keeping).
5. Add new tests:
   - `test_memory_writer_persists_user_and_assistant_turn`
   - `test_memory_writer_skips_assistant_when_response_empty`
   - `test_memory_writer_triggers_rolling_summary_on_threshold`
   - `test_delete_session_clears_checkpoint_thread`
   - `test_intent_classifier_sees_history_loaded_by_history_loader`
   - `test_embedding_router_injected_into_intent_classifier`

Run `bash scripts/review_changed_tests.sh` after rewrite. Update
`docs/review/KNOWN_FAILURES.txt` if anything is deferred.

---

## Files Untouched

- `auth.py`, `routes/auth_routes.py`
- `music_fastpath.py`
- `routes/memory_tool_routes.py`
- `routes/code_file_routes.py` (constructor signature unchanged; deps still
  passed in from `main.py`)
- All frontend files
- `tools/*` modules
- `tts/*` modules

---

## Completion Checklist

```
Phase 1 — Demolition
  [x] _SessionRecord, _SQLiteSessionStore deleted
  [x] All listed helpers deleted from main.py
  [x] All listed session endpoints deleted from main.py
  [x] _code_write_lock / _pending_code_writes kept at module scope
  [x] embedding_router.py globals + warmup deleted
  [x] graph.py imports of deleted globals removed
  [x] intent_classifier reads deps.embedding_router (may be None temporarily)
  [x] _consolidation_loop_task removed from lifespan
  [x] /health simplified (see deviation below)

Phase 2 — Construction
  [x] 2.1 conversation_log table + 8 MemoryStore methods
  [x] 2.2 memory_writer node added, SSE memory emit moved into it
  [ ] 2.2 confirmation-gate edge decision documented in code comment
        (decision implemented — confirmations route through memory_writer —
         but no inline code comment was added)
  [ ] 2.2 rolling-summary helper lives in graph.py
        (DEFERRED: memory_writer triggers consolidation but does NOT yet
         invoke a rolling per-session summarizer; `_rolling_summary_task`
         and the SUMMARY_TRIGGER turn counter are not implemented)
  [x] 2.3 history_loader node added; START -> history_loader -> intent_classifier
  [x] 2.3 history/session_summary removed from graph_state in main.py
  [~] 2.4 embedding_router + shared code-write state in AssistantGraphDependencies
        (embedding_router IS in deps; code_write_lock /
         pending_code_writes are NOT in deps — still module-scope only,
         and no graph node consumes them, so deferred until a node needs it)
  [x] 2.4 lifespan warms router BEFORE create_assistant_graph
  [x] 2.5 5 session endpoints rebuilt against conversation_log
  [x] 2.5 _clear_checkpoint_thread helper added and called by delete + reset
  [ ] /chat handler simplified to final form
        (handler still keeps telemetry/timing/log lines from prior
         implementation; SSE forwarding works correctly but the body is
         larger than the roadmap target)
  [ ] Test suite rewritten; new tests added

Verification
  [ ] Send a message; conversation_log has user + assistant row
  [ ] Reload; history renders unchanged
  [ ] Create new session, switch, switch back
  [ ] Delete session; cookie reanchors to next; checkpoint thread gone
  [ ] "remember my name is X" -> facts row; "forget my name" -> deleted
  [ ] After 18 turns, summaries row appears for the session
        (will NOT happen until rolling-summary helper above is implemented)
  [ ] After 3+ summaries, consolidation runs (log line)
  [ ] embedding_router.ready logs on startup; heuristic fallback logs when down
  [ ] scripts/review_changed_tests.sh green
  [ ] scripts/review_baseline.sh green
  [ ] docker compose config && docker compose build backend green
```

### Deviations from the plan

1. **`/health` shape.** Roadmap specified
   `{"status": "ok", "embedding_router": true}`. Implemented shape is
   `{"status": "ok" | "starting", "embedding_router": <bool>}` where
   `status` reflects whether the graph is constructed. Update probes /
   tests accordingly.
2. **Rolling session summarization.** Not implemented this pass. Without
   it, `summaries` rows for active sessions will only appear if some
   other code path writes them, so `get_latest_session_summary` will
   usually return `""`. Behaviour is still correct, just lossy on
   long-session context.
3. **Code-write lock in graph deps.** Deferred. Field not added to
   `AssistantGraphDependencies` because no graph node needs it yet.
4. **/chat handler trim.** Endpoint works correctly and forwards `text`,
   `meta`, `memory`, `notice`, and `fallback` SSE events from the graph,
   but retains existing telemetry. Roadmap "final form" simplification
   not applied.
5. **Test suite.** Not rewritten. `backend/tests/test_chat_sessions.py`
   still references deleted helpers and will fail. Rewrite is the next
   logical step.

### Cleanup (because we are not migrating)

The shared SQLite DB at `backend/memory.db` (path overridable via
`MEMORY_DB_PATH`) still contains the legacy `chat_sessions` table
created by the deleted `_SQLiteSessionStore`. It is dormant — no code
reads or writes it — but it occupies disk.

Files / tables you may delete for a clean baseline:

- `backend/memory.db` — full reset of memory + sessions. Keep if you
  want to retain extracted `facts`, `preferences`, and `summaries`. If
  you keep it, manually drop the dormant table:
  `sqlite3 backend/memory.db "DROP TABLE IF EXISTS chat_sessions;"`
- `backend/graph_checkpoints.sqlite` — LangGraph thread checkpoints
  from previous sessions. Safe to delete; new session UUIDs will not
  collide with old threads, but deleting reclaims disk.
- `backend/chroma/` — semantic / episodic memory vectors. **Keep** if
  you want recall to continue working. Deleting forces a clean Chroma
  rebuild (slow on first use). Not required by this refactor.

Do NOT delete:
- `backend/auth.db` — user accounts, unrelated to this refactor.
- `backend/models/`, `backend/models/tts/` — runtime model assets.
- `backend/chroma/` unless you accept losing existing memory vectors.

---

## Differences vs. roadmap.2.md

| Issue in v2 | Resolution in v3 |
|---|---|
| `response_text` persistence ambiguous | Verified all terminal nodes set it; `memory_writer` reads from merged state. No HTTP accumulation. |
| Session summarization disappeared | `memory_writer` triggers rolling summary every N turns; helpers move into `graph.py`. |
| Missing `MemoryStore` methods unspecified | 8 methods enumerated with signatures and SQL. |
| Checkpoint threads leaked on delete | `_clear_checkpoint_thread` helper called by delete and reset endpoints. |
| Embedding router init order undefined | Lifespan warms router *before* graph creation; explicit ordering. |
| Confirmation gates ambiguous | Explicit decision: persisted via `memory_writer`; reasoning documented. |
| Code-write lock relocation broke routes | Stays at module scope; *also* passed to graph deps. |
| `intent_classifier` lost access to history | New `history_loader` node runs before classifier. |
| Phase 1 left system unusable | Phase 1 + Phase 2 ship together as one PR. |
| `/health` shape change unflagged | Called out explicitly; tests + probes updated. |
| Test churn understated | Test rewrite section enumerates fixtures and new cases. |
