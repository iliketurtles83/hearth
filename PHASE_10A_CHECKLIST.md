# Phase 10a Execution Checklist

Goal: introduce LangGraph with durable checkpointing and migrate the current chat routing flow into a stateful graph without changing externally visible behavior.

## Phase 10a success criteria

- [ ] Existing chat behavior still works end to end.
- [ ] Existing weather and music tool behavior still works end to end.
- [ ] Memory retrieval and injection behavior still works.
- [ ] The deterministic music fast-path still bypasses graph invocation.
- [ ] LangGraph is pinned in `backend/requirements.txt` with rationale.
- [ ] `backend/graph.py` exists with `StateGraph` and SQLite checkpointer wiring (`AsyncSqliteSaver` for async-safe graph streaming).
- [ ] `/chat` uses the graph for non-fast-path requests.
- [ ] `GET /graph/state/{session_id}` returns inspectable state.
- [ ] Checkpoint resume is covered by an explicit test.

## Slice 1: freeze current behavior surface

Objective: lock in the contracts that Phase 10a must preserve before architectural changes begin.

Deliverables:

- [x] Inventory current `/chat` orchestration seams.
- [x] Confirm existing router test coverage.
- [x] Confirm existing chat-session and voice SSE coverage.
- [x] Add explicit regression test for the music fast-path bypass.
- [x] Identify missing graph/checkpoint tests.

Exit criteria:

- [x] SSE event contract is pinned by tests.
- [x] Session continuity assumptions are pinned by tests.
- [x] Music fast-path bypass is pinned by tests.
- [x] Known test gaps for graph migration are written down.

## Slice 2: add graph scaffolding

Objective: create the graph module and state schema while delegating behavior to existing code.

Deliverables:

- [x] Pin LangGraph dependency.
- [x] Add `backend/graph.py`.
- [x] Define `AssistantState`.
- [x] Add `intent_classifier` node.
- [x] Add `memory_retrieval` node.
- [x] Add `tool_router` node.
- [x] Add `responder` node.
- [x] Add graph factory and compiled graph entry point.

Exit criteria:

- [x] Graph can be invoked directly in tests.
- [x] Node outputs match legacy routing decisions for simple prompts.

## Slice 3: move chat orchestration behind the graph

Objective: reduce `backend/main.py` to request translation, session plumbing, and SSE transport.

Deliverables:

- [x] Add `_make_graph_deps()` with late-binding proxies (monkeypatch-compatible).
- [x] Initialize module-level `_assistant_graph` at import time via `build_assistant_graph`.
- [x] Route non-music `/chat` requests through the graph.
- [x] Keep the HTTP-level music fast-path in place ahead of graph invocation.
- [x] `intent_classifier` emits `{"meta": {...}}` for SSE metadata.
- [x] `responder` handles cloud fallback events.

Exit criteria:

- [x] `/chat` uses graph orchestration for non-fast-path requests.
- [x] SSE payload shape and event order stay stable (141 tests pass).

## Slice 4: add checkpointing and state inspection

Objective: make session state durable and inspectable.

Deliverables:

- [ ] Wire the SQLite checkpointer into the graph.
- [ ] Map session identity to graph checkpoint identity.
- [ ] Add `GET /graph/state/{session_id}`.
- [ ] Decide which session concerns remain outside the graph for Phase 10a.

Exit criteria:

- [ ] Graph state survives process restart.
- [ ] State can be inspected without replaying the entire request.

## Slice 5: validate migration and resume behavior

Objective: prove the migration is architectural only, not behavioral.

Deliverables:

- [ ] Add graph routing smoke tests.
- [ ] Add checkpoint resume test.
- [ ] Add graph debug endpoint test.
- [ ] Re-run focused chat, router, music, weather, and memory tests.

Exit criteria:

- [ ] Focused regression suite passes.
- [ ] Checkpoint resume behavior is explicit and green.

## Current assessment

- [x] Existing router behavior is well covered by `backend/tests/test_router.py`.
- [x] Existing chat SSE and session behavior is partly covered by `backend/tests/test_chat_sessions.py`.
- [x] The music fast-path exists in `backend/main.py` before `router_route()`.
- [x] Explicit music commands are now covered by a regression test that proves `router_route()` is bypassed.
- [x] Vague music prompts are now covered by a regression test that proves they still flow through the normal router path.
- [x] Graph scaffolding now exists in `backend/graph.py` with async-safe SQLite checkpoint wiring.
- [x] Graph invocation and custom chunk streaming are now covered by `backend/tests/test_graph.py`.
- [ ] No current tests mention checkpoint resume behavior yet.
- [x] No current test explicitly proves music fast-path bypasses later orchestration.

## Focused test targets for Slice 1

- [x] Add a test that a confident music command does not call `router_route()`.
- [x] Add a test that a confident music command dispatches the music tool directly and returns `[DONE]`.
- [x] Optionally add a test that vague prompts like `play something chill` do not take the fast-path.

## Slice 1 validation

- [x] `backend/.venv/bin/python -m pytest tests/test_chat_sessions.py`

## Slice 2 validation

- [x] `backend/.venv/bin/python -m pytest tests/test_graph.py`
- [x] `backend/.venv/bin/python -m pytest tests/test_graph.py tests/test_chat_sessions.py tests/test_router.py`
