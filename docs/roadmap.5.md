# Roadmap 5: Remove Coding Agent Integration

Date: 2026-06-04 (revised 2026-06-05)
Status: In Progress
Goal: Strip coding agent, project management, workspace indexing, and code file write infrastructure from Hearth. Keep read-only code-question intent routing using chat model only.

---

## Progress Review (As Of 2026-06-05)

### Completed

- Deleted backend/tools/coding_agent.py
- Deleted backend/tools/code_indexer.py
- Deleted backend/tools/workspace.py
- Deleted backend/projects.py
- Deleted backend/routes/project_routes.py
- Deleted backend/routes/code_file_routes.py
- Deleted backend/tests/test_coding_agent.py
- Deleted backend/tests/test_code_tool.py
- Deleted backend/tests/test_project_routes.py
- Deleted backend/tests/test_projects.py
- Deleted backend/tests/test_project_memory_scope.py
- Deleted backend/hearth_coder_prompt.txt
- Started removing project scope from backend/memory.py

### In Progress

- backend/memory.py has substantial project_id removal, but needs verification and cleanup pass.

### Not Started / Remaining

- backend/graph.py still contains coding agent nodes, project_id state, workspace root checks, and code indexer imports.
- backend/main.py still imports/uses project routes, code file routes, coder model, project context, and project-filtered session APIs.
- backend/intents.py still defines and selects CODER_MODEL.
- backend/app_schemas.py still includes project_id in request models.
- docker-compose.yml still includes CODE_WORKSPACE_ROOT env and bind mount.
- .env.example still includes code-workspace and coder-model-related vars.
- Tests still reference removed components (for example graph/router/chroma/persona tests).

### Newly Completed (2026-06-05)

- Completed Chunk 1 in backend/memory.py:
	- Verified no project_id/project-scope signatures or SQL branches remain.
	- Verified no project-related indexes/migration logic remain in MemoryStore.
	- MemoryStore API now consistently uses user/session-only signatures.
- Updated backend/tests/test_chroma_isolation.py to remove dependency on deleted code-indexer module while preserving collection-isolation assertions.
- Validation: backend/tests/test_memory_isolation.py and backend/tests/test_chroma_isolation.py both pass (23 passed).

- Completed Chunk 2 in backend/graph.py:
	- Removed coding-agent nodes and edges (coding_agent_tool, coding_agent_executor).
	- Removed project-scoped routing branches and confirm_agent_task handling.
	- Removed workspace-root/code-tool path from graph flow.
	- Updated memory_writer calls to non-project memory-store signatures.
	- Kept code-question intent, routed through normal local responder path.
- Updated backend/tests/test_graph.py to remove coding-agent/write-confirmation assumptions.
- Validation: backend/tests/test_graph.py and backend/tests/test_router.py both pass (31 passed).

- Completed Chunk 3 in backend/main.py and backend/app_schemas.py:
	- Removed project/code-file route imports and registrations.
	- Removed ProjectStore initialization and project-context resolver usage.
	- Removed CODE_WORKSPACE_ROOT startup/indexing logic from main startup validation.
	- Removed project_id/project_folder from /chat and /code graph state payloads.
	- Removed project_id query params from session endpoints and updated MemoryStore calls.
	- Removed project_id fields from request schemas.
- Validation: backend/tests/test_auth.py, backend/tests/test_graph.py, backend/tests/test_chat_voice_metadata.py (34 passed).
- Note: backend/tests/test_chat_sessions.py currently fails at setup expecting main._session_store, which is an existing test/module mismatch outside this chunk's direct changes.

- Completed Chunk 4 in backend/intents.py:
	- Removed CODER_MODEL constant from runtime intent routing.
	- Simplified _pick_local_model so code-question local routing now uses CHAT_MODEL.
	- Updated backend/main.py to stop importing/using CODER_MODEL and to pass CHAT_MODEL into graph dependency wiring.
	- Updated backend/tests/test_router.py assertions to reflect CHAT_MODEL routing for code-question intent.
- Validation: backend/tests/test_router.py, backend/tests/test_graph.py, backend/tests/test_auth.py (48 passed).

- Completed Chunk 5:
  - Removed CODE_WORKSPACE_ROOT env var and /code-workspace bind mount from docker-compose.yml.
  - Removed OLLAMA_CODER_MODEL, CODE_INDEX_PATHS, CODE_ENABLE_SHELL, CODE_ENABLE_REPL from .env.example.
  - Removed CODING_AGENT_URL and CODING_AGENT_TIMEOUT_SECONDS from .env.example.
  - Validated: `docker compose config` succeeds.
- Validation: `docker compose config` passes.

### Current Risk Snapshot

- Highest risk: compile/import failures due to deleted modules still referenced in main.py and graph.py.
- Medium risk: test failures from stale assertions tied to CODER_MODEL, project_id, and code_indexer.
- Medium risk: API contract change on session endpoints that currently accept project_id query params.

---

## Target End State

- No project CRUD APIs.
- No code file read/write/confirm APIs.
- No coding-agent runtime adapter.
- No workspace-root path dependency for chat flow.
- No project-scoped memory/session storage.
- Code-question intent remains, but response generation uses CHAT_MODEL path only.

---

## Action Plan In Chunks

### Chunk 1: Stabilize Memory Layer

Scope:
- Finish backend/memory.py cleanup so there is no project_id parameter, query branch, migration, or index tied to project scope.

Actions:
- Remove any leftover project_id signature/usage.
- Run syntax check and unit targets touching MemoryStore.

Exit criteria:
- No project_id references in backend/memory.py.
- MemoryStore methods compile and tests that import MemoryStore run.

### Chunk 2: Remove Graph Coding-Agent Pipeline

Scope:
- backend/graph.py

Actions:
- Remove AssistantState project_id/project_folder fields.
- Remove code context query path tied to code_indexer.
- Remove coding_agent_tool and coding_agent_executor nodes and edges.
- Remove confirm_agent_task routing branch.
- Remove workspace-root dependent system prompt additions.
- Update memory method calls to new non-project signatures.

Exit criteria:
- No references to coding_agent, code_indexer, confirm_agent_task, project_folder, project_id, or CODE_WORKSPACE_ROOT in backend/graph.py.
- Graph builds and basic routing tests pass.

### Chunk 3: Strip Main API Surface

Scope:
- backend/main.py and backend/app_schemas.py

Actions:
- Remove project/code-file route imports and registrations.
- Remove ProjectStore initialization and project context resolver.
- Remove CODE_WORKSPACE_ROOT startup/indexing logic.
- Remove project_id/project_folder from send and send/voice state payloads.
- Remove project_id query params from session routes.
- Remove project_id fields from ChatRequest and CodeRequest schemas.

Exit criteria:
- main.py imports resolve without deleted modules.
- Session route handlers compile with new MemoryStore signatures.
- Request schemas no longer expose project_id.

### Chunk 4: Simplify Intent Model Selection

Scope:
- backend/intents.py

Actions:
- Remove CODER_MODEL constant and code-intent model branch in _pick_local_model.
- Keep code-question detection, but map to CHAT_MODEL for local routing.

Exit criteria:
- No CODER_MODEL symbol in runtime code.
- Router behavior remains deterministic for code-question intent.

### Chunk 5: Remove Environment and Compose Surface

Scope:
- docker-compose.yml
- .env.example

Actions:
- Remove CODE_WORKSPACE_ROOT environment variable and /code-workspace bind mount.
- Remove coding-agent-specific env vars from .env.example.
- Keep unrelated model vars required for normal chat operation.

Exit criteria:
- No CODE_WORKSPACE_ROOT references in compose/env example.
- docker compose config succeeds.

### Chunk 6: Repair and Update Tests

Scope:
- backend/tests/

Actions:
- Update/remove tests tied to deleted files and removed symbols.
- Update graph test fakes that still include project-scoped MemoryStore signatures.
- Update router tests that assert CODER_MODEL behavior.
- Update persona/chroma tests that depend on deleted coder prompt and code indexer.

Exit criteria:
- Changed-tests selection passes.
- No test imports reference deleted modules.

### Chunk 7: Full Validation Gate

Scope:
- Repository-wide verification

Actions:
- Run:
	- bash scripts/review_changed_tests.sh --dry-run
	- bash scripts/review_changed_tests.sh
	- bash scripts/review_baseline.sh
- If relevant files changed during execution, re-run focused tests.

Exit criteria:
- Validation scripts pass or known failures are documented with rationale.

---

## Suggested Execution Order (Updated)

1. Chunk 1 (memory)
2. Chunk 2 (graph)
3. Chunk 3 (main + schemas)
4. Chunk 4 (intents)
5. Chunk 5 (compose/env)
6. Chunk 6 (tests)
7. Chunk 7 (validation)

---

## Live Tracking Checklist

- [x] Delete coding-agent/project files and tests listed in initial plan
- [x] Finish memory.py cleanup and verify signatures
- [x] Remove graph coding-agent pipeline and project scope
- [x] Remove main.py project/code-file route and state plumbing
- [x] Remove project_id from request schemas
- [x] Remove CODER_MODEL and simplify local model routing
- [x] Remove workspace env/mount surface from compose and env example
- [ ] Update and fix tests for new architecture
- [ ] Run changed-tests and baseline validation scripts

---

## Notes For Implementers

- Treat compile/import health first: main.py and graph.py should be made self-consistent before broad test runs.
- Keep code-question intent classification, but ensure it no longer triggers tooling paths.
- Expect API behavior change for clients passing project_id query params; this is intentional per Roadmap 5.
