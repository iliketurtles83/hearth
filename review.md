# Backend Code Review

Scope: `backend/` (main.py, graph.py, memory.py, intents.py, auth.py, routing_config.py,
music_fastpath.py, routes/, tools/). Issues are listed in descending priority. Line
references are approximate and may shift as the code changes.

---

## High Priority

### 1. `load_dotenv()` runs too late — `.env` is mostly ignored
**File:** [backend/main.py](backend/main.py#L95)

`load_dotenv()` is called *after* the top-of-file imports and *after* `memory_store` is
constructed:

```python
memory_store = MemoryStore(
    db_path=os.getenv("MEMORY_DB_PATH", _memory_db_default),
    chroma_path=os.getenv("CHROMA_PATH", _chroma_default),
)
load_dotenv()   # ← too late
```

By the time this runs, the following have already read the environment at import time and
captured defaults:
- `intents.py` → `CHAT_MODEL`, `CODER_MODEL`, `VISION_MODEL`, `CLOUD_MODEL`
- `routing_config.py` → `ROUTING_CONFIG` (ollama URL, token budgets, embedding flags)
- `memory_store` paths (`MEMORY_DB_PATH`, `CHROMA_PATH`)
- numerous module-level constants in `main.py`

Net effect: values placed in a `.env` file silently have no effect on model selection,
routing, or DB paths. Only `os.getenv(...)` calls that execute at *request* time (e.g.
inside graph nodes) pick up `.env`, producing inconsistent behaviour.

**Fix:** Call `load_dotenv()` as the very first executable statement, before any
`from intents/routing_config/... import` and before `MemoryStore(...)`.

---

### 2. Memory consolidation holds the DB lock during blocking LLM calls
**File:** [backend/memory.py](backend/memory.py#L885) (`consolidate_pending`)

`consolidate_pending` opens `with self._lock:` and then, inside the loop, calls
`self._extract_candidates_llm_sync(...)`, which performs a synchronous HTTP request to
Ollama with a 10s timeout — *while holding the lock*:

```python
with self._lock:
    ...
    for row in rows:
        candidates = self._extract_candidates_llm_sync(summary_text, source="consolidation")
```

Because `self._lock` guards every other SQLite read/write in `MemoryStore`, a single
consolidation pass can block all memory operations (history loads, turn logging,
retrieval) for up to `10s × number_of_summaries`. The whole request path stalls behind it.

This is launched fire-and-forget from `memory_writer`, so it competes directly with live
chat turns.

**Fix:** Perform LLM extraction *outside* the lock (gather candidates first, then take the
lock only for the DB writes), or batch/limit it. Consider not calling the LLM under the
lock at all.

---

### 3. SPA catch-all returns 401 for unauthenticated deep links
**Files:** [backend/main.py](backend/main.py#L1190) (catch-all), [backend/main.py](backend/main.py#L300) (`AuthMiddleware`)

The catch-all route serves `index.html` for client-side routing:

```python
@app.get("/{full_path:path}", include_in_schema=False)
async def _spa_catchall(full_path: str):
    return FileResponse(os.path.join(_frontend_dir, "index.html"))
```

But `AuthMiddleware` only exempts `/`, a fixed unprotected set, `/static`, the auth login
/register endpoints, and paths with a static-asset extension. Any other extensionless path
(e.g. `/settings`, `/chat/ui`, a refreshed deep link) is treated as protected and returns
a `401 UNAUTHORIZED` JSON body **instead of the SPA shell / login page**. This breaks
deep-linking and hard refreshes for unauthenticated users.

**Fix:** Allow `GET` HTML navigations to fall through to the SPA shell (e.g. exempt
non-API GET requests, or whitelist the catch-all), and let the frontend handle auth.

---

## Medium Priority

### 4. `/transcribe` is unauthenticated (resource-abuse vector)
**File:** [backend/main.py](backend/main.py#L290) (`_UNPROTECTED_PATHS`)

`/transcribe` is in `_UNPROTECTED_PATHS`, so any unauthenticated caller can upload audio
and force a Whisper transcription. Whisper inference is CPU/GPU heavy — this is an
unauthenticated compute-DoS surface with no rate limiting or size cap on the upload.

**Fix:** Require auth for `/transcribe`, or add size limits + rate limiting. At minimum,
cap `await audio.read()` size before writing the temp file.

### 5. `_extract_candidates_llm_sync` has a broken "running loop" branch
**File:** [backend/memory.py](backend/memory.py#L530)

```python
try:
    loop = asyncio.get_running_loop()
    return loop.run_until_complete(self._llm_extract_candidates(text, source))  # ← raises
except RuntimeError:
    loop = asyncio.new_event_loop()
    ...
```

`run_until_complete()` cannot be called on an already-running loop — it raises
`RuntimeError: This event loop is already running`. The code only works because callers
run it in a worker thread (no running loop), hitting the `except` branch. The `try` branch
is dead/incorrect and would fail loudly if ever reached from an async context.

**Fix:** Remove the misleading branch, or use `asyncio.run_coroutine_threadsafe` /
`anyio.from_thread` correctly for the in-loop case.

### 6. `stream_local` (text path) does not check the HTTP status
**File:** [backend/main.py](backend/main.py#L520) (`stream_local`)

The vision streamer calls `resp.raise_for_status()`, but the plain text streamer does not:

```python
async with client.stream("POST", f"{OLLAMA_URL}/api/generate", json={...}) as resp:
    async for line in resp.aiter_lines():
        data = json.loads(line)   # ← on a 4xx/5xx body, this throws an opaque error
```

If Ollama returns an error (model not pulled, bad request), the error body is parsed as if
it were a token stream, producing confusing `JSONDecodeError`/`KeyError` instead of a clean
failure message.

**Fix:** Add `resp.raise_for_status()` and handle non-200 explicitly, matching the vision path.

### 7. Fire-and-forget consolidation task can be GC'd / swallows errors
**File:** [backend/graph.py](backend/graph.py#L1565) (`memory_writer`)

```python
asyncio.create_task(
    asyncio.to_thread(deps.memory_store.consolidate_pending, user_id, consolidation_batch)
)
```

The task reference is discarded. Per CPython docs, tasks not referenced may be
garbage-collected before completion, and any exception raised inside is never logged. There
is no `add_done_callback` to surface failures.

**Fix:** Keep a reference (e.g. a module-level set) and attach a done-callback that logs
exceptions.

### 8. Music fast-path skips conversation logging
**Files:** [backend/main.py](backend/main.py#L660) (`/chat` music fast-path), [backend/graph.py](backend/graph.py#L1507) (`memory_writer`)

The deterministic music fast-path returns its own `StreamingResponse` and never runs the
graph, so `memory_writer` never logs the user turn or the assistant reply to
`conversation_log`. Consequences:
- Music interactions don't appear in `/chat/sessions` history or session previews.
- Follow-up context ("play another by them") has no record of what was played.

Confirm whether this is intentional. If continuity is desired, log the turn in the
fast-path before returning.

### 9. Duplicated path-safety / diff / confirm logic across modules
**Files:** [backend/graph.py](backend/graph.py#L360) and [backend/routes/code_file_routes.py](backend/routes/code_file_routes.py#L30)

`_resolve_workspace_path` / `_safe_resolve`, `_make_unified_diff`, and the write-confirm
flow are independently re-implemented in `graph.py` and `code_file_routes.py`. They can
drift apart (e.g. different traversal checks or diff formats), which is a correctness and
security hazard for the file-write surface.

**Fix:** Extract a single shared helper module (e.g. `tools/workspace.py`) and import it
in both places.

---

## Lower Priority

### 10. `pending_code_writes` grows unbounded (no TTL purge)
**File:** [backend/routes/code_file_routes.py](backend/routes/code_file_routes.py#L90)

Each unconfirmed `PUT /code/files/...` stores a pending entry with `created_at`, but
nothing ever purges stale entries. Abandoned confirmations accumulate in memory
indefinitely.

**Fix:** Evict entries older than a TTL on each access, or cap the dict size.

### 11. In-memory state breaks under multiple workers
**Files:** [backend/main.py](backend/main.py#L230) (`_pending_code_writes`), [backend/routes/code_file_routes.py](backend/routes/code_file_routes.py)

`_pending_code_writes` (and the lazily-loaded model singletons) are per-process. If the
backend is ever run with `uvicorn --workers N > 1`, a confirmation can land on a different
worker than the one that created the pending write, causing spurious
"Pending write not found" 404s. Document the single-worker assumption or move pending
writes to the SQLite store.

### 12. Expired auth tokens are never purged automatically
**File:** [backend/auth.py](backend/auth.py#L283) (`purge_expired_tokens`)

`purge_expired_tokens()` exists but is not called anywhere (no startup hook, no scheduled
job). Expired token rows accumulate forever. `verify_token` correctly rejects them, so this
is hygiene, not a security hole.

**Fix:** Call it on startup and/or periodically.

### 13. Module-level graph is built then immediately discarded
**File:** [backend/main.py](backend/main.py#L640)

```python
_assistant_graph = build_assistant_graph(_make_graph_deps())
```

A full (non-checkpointed) graph is constructed at import time and then replaced by the
checkpointed graph in `_graph_lifespan`. Besides the wasted work, it acts as a silent
fallback: if lifespan setup fails, requests quietly run on a **no-checkpoint** graph (no
conversation persistence) rather than failing loudly.

**Fix:** Build lazily, or make the fallback path log a clear warning when used.

### 14. Wake-word threshold is hardcoded
**File:** [backend/main.py](backend/main.py#L975)

`if score > 0.5:` — the detection threshold is a magic constant, not configurable via env,
despite other tunables being env-driven. Sensitivity can't be tuned per deployment without
a code change.

**Fix:** Read from `os.getenv("WAKEWORD_THRESHOLD", "0.5")`.

### 15. Embedding code-route vs heuristic intent mismatch
**File:** [backend/graph.py](backend/graph.py#L250) (`_decision_from_embedding`)

When the embedding router selects the `code` tool label, the write-vs-question split is
decided by the *heuristic* (`heuristic.intent == "code-write"`). The embedding classifier
and heuristic can disagree, so an embedding-confident code route may still be mislabeled by
a weak heuristic signal. Minor routing-accuracy concern.

### 16. Bandit B608 findings in memory.py are parameterized (verify)
**File:** [backend/memory.py](backend/memory.py#L575) (`_forget_by_query`), `list_items`

`review_baseline.sh` flags B608 (SQL-string construction). The flagged queries use bound
parameters (`?`) for all user-controlled values — the only f-string interpolation builds
`%{q}%` wildcards that are still passed *as parameters*, not concatenated into SQL. These
appear to be false positives but should be annotated/suppressed so the baseline gate stays
green (the script uses `set -e` and stops on Bandit findings).

---

## Notes / Observations (not bugs)

- CORS middleware is added last, so it is outermost and correctly handles preflight before
  `AuthMiddleware` — good.
- `auth.py` uses scrypt with OWASP-minimum parameters and `secrets.compare_digest` — good.
- Path-traversal checks in both write surfaces use `realpath` + prefix check — correct.
- WebSocket `/ws/wake` bypasses `BaseHTTPMiddleware` (Starlette limitation); it is in the
  unprotected set anyway, so no behaviour change — but be aware WS routes are never
  auth-checked by the current middleware.
