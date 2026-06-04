# Hearth Roadmap v4 — Projects Workspace

**Goal:** Separate coding work from main chat entirely. Projects are
first-class workspaces — each backed by a filesystem folder, a coder model,
an indexed codebase, and project-scoped memory. The main chat classifier gets
smaller and less ambiguous as a direct result.

---

## Design principles

- The Projects section is a distinct UI mode, not a clever intent inside chat.
- Switching to a project switches model, indexer context, and memory scope
  simultaneously. No ambiguity at the classifier.
- `code-write` intent is fully removed from the main chat domain in Phase 3.
  `code-question` is the only code-related intent that remains in main chat,
  retained exclusively for voice-driven one-off queries ("explain this function",
  "what does X do") where opening a project would be disproportionate friction.
  This is a deliberate, documented residual coupling — not an oversight.
- The `coding_agent_tool` and `coding_agent_executor` graph nodes become
  project-only paths after Phase 3. They stay in the graph but are unreachable
  from the main chat `tool_router`. `code_tool` stays for `code-question` only.
- Every project folder is a plain directory under `CODE_WORKSPACE_ROOT`.
  Optionally a git repo. Hearth adds nothing that prevents using the folder
  with any other tool.
- Tree-sitter indexing is always on-demand (triggered at open); keeping a
  live inotify-based watcher is out of scope for this roadmap.

---

## Phase 1 — Backend: project registry and index API

**Scope:** New `projects` table, REST endpoints, indexer integration. No
graph changes. No frontend changes beyond stub removal.

### 1.1 Project registry (`backend/memory.py` or new `projects.py`)

Add a `projects` table to the auth SQLite DB (or a standalone
`projects.db`, same path group):

```sql
CREATE TABLE IF NOT EXISTS projects (
    id          TEXT PRIMARY KEY,          -- uuid4
    user_id     TEXT NOT NULL,
    name        TEXT NOT NULL,
    folder_name TEXT NOT NULL,             -- relative to CODE_WORKSPACE_ROOT
    description TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    opened_at   REAL                       -- last open; NULL = never
);
CREATE INDEX IF NOT EXISTS idx_projects_user
    ON projects(user_id, opened_at DESC);
```

Rules:
- `folder_name` is validated with the same `resolve_workspace_path` guard
  used in `code_file_routes.py`. Traversal is rejected at registration.
- A project record and its filesystem folder are created together. If the
  filesystem `mkdir` fails, the DB row is rolled back.
- Deleting a project removes the DB row. It does NOT delete the folder —
  destructive filesystem operations require an explicit separate confirmation.

`ProjectStore` methods:

| Method | Purpose |
|---|---|
| `create_project(user_id, name, folder_name, description) -> dict` | Insert row + `mkdir`. Return project dict. |
| `list_projects(user_id) -> list[dict]` | All projects for user, ordered by `opened_at DESC`. |
| `get_project(project_id, user_id) -> dict \| None` | Single project. |
| `delete_project(project_id, user_id) -> bool` | Row delete only. |
| `touch_opened(project_id, user_id) -> None` | Set `opened_at = now`. |

### 1.2 Project REST routes (`backend/routes/project_routes.py`)

```
POST   /projects                   create project
GET    /projects                   list projects for authenticated user
GET    /projects/{id}              get single project
DELETE /projects/{id}              delete project record (not folder)
POST   /projects/{id}/open         touch opened_at; return project + folder listing
POST   /projects/{id}/index        trigger tree-sitter index; return stats
GET    /projects/{id}/index/status query index state (idle | running | done | error)
GET    /projects/{id}/files        list workspace files (delegates to existing list_code_files logic)
```

`POST /projects/{id}/index` response:
```json
{"status": "started", "project_id": "...", "folder": "my-project"}
```

`GET /projects/{id}/index/status` response:
```json
{"status": "done", "files_indexed": 42, "chunks": 187, "duration_s": 1.4}
```

All routes require auth. `user_id` comes from `request.state.user_id`.

### 1.3 Per-project index isolation

The current `code_indexer.index_workspace` writes into a single
`code_context` ChromaDB collection, keyed only by file path. Projects need
isolated collections so switching projects doesn't pollute results.

Change: add a `collection_name` parameter (default `"code_context"`) to
`index_workspace` and `query_code_context`. Projects pass
`f"code_context_{project_id}"`. The collection is created lazily on first
index; deleted on project delete if it exists.

`start_background_index` gains a `collection_name` kwarg. The index route
calls it and stores `(project_id, status, stats)` in a module-level dict
guarded by a lock — same pattern as `_pending_code_writes`.

Index status fields:
```python
{
    "status": "idle" | "running" | "done" | "error",
    "files_indexed": int,
    "chunks": int,
    "duration_s": float,
    "error": str | None,
    "started_at": float | None,
}
```

### 1.4 Optional git init

At project creation time, accept `"git_init": true` in the request body.
If set, run `git init <folder>` via `subprocess.run` with a strict allow-list:
- Only `["git", "init", str(resolved_path)]` — no shell=True, no user-supplied args.
- Log stdout/stderr; non-zero exit is a warning, not a 500 (project is still created).
- Expose `"git": true|false` in the project dict (check for `.git/` existence at
  list/get time, not stored in DB — source of truth is the filesystem).

No further git integration in this phase (commits, diffs, push — out of scope).

### 1.5 Project memory scope

Add a `project_id` column (nullable) to `conversation_log` and `summaries`
tables. When `project_id` is set:
- `log_turn`, `get_session_turns`, `list_sessions` all filter by `project_id`.
- `retrieve` and ChromaDB queries use the project's `code_context_{id}` collection
  instead of the default `conversation_memory` collection.
- Main chat sessions have `project_id = NULL`. No cross-contamination.

This is additive — existing rows without `project_id` continue to work.

Migration: `ALTER TABLE conversation_log ADD COLUMN project_id TEXT;` and
same for `summaries`. SQLite adds NULL for existing rows automatically.

---

## Phase 2 — Frontend: Projects panel

**Scope:** Activate the Projects button, render project list, switch context
on open. No new dependencies — plain JS like the rest of the frontend.

### 2.1 Activate the Projects section button

Remove `disabled` attribute and `"Projects coming soon."` note from
`index.html`. The button already has `id="sidebar-section-projects"`.

### 2.2 Projects panel (sidebar)

When "Projects" is active the sidebar shows:

```
[ + New Project ]
─────────────────
▸ my-api              (git)  2d ago
▸ hearth-frontend           opened now
▸ scripts-scratch            3w ago
```

Each row shows: name, git badge if `.git/` present, last opened time.
Clicking a row opens the project (calls `POST /projects/{id}/open`).

"New Project" opens an inline form: name, optional description, optional
folder name (defaults to slugified name), git init checkbox.

### 2.3 Project workspace view (main area)

Opening a project switches the main area from chat to a two-panel coding view:

```
┌──────────────────┬────────────────────────────────────────┐
│  File tree       │  Chat / coding input                   │
│  (left panel)    │                                        │
│                  │  [Index status bar]                    │
│  my-api/         │                                        │
│  ├── main.py     │  Ask about this project...             │
│  ├── tests/      │                                        │
│  └── README.md   │  [  Send  ]  [ Re-index ]              │
└──────────────────┴────────────────────────────────────────┘
```

File tree: rendered from `GET /projects/{id}/files`. Clicking a file sends
`GET /code/files/{path}` and displays content in a read-only code view.

Index status bar: shown after open; polls `GET /projects/{id}/index/status`
until `done` or `error`. Shows a spinner while `running`.

"Re-index" button calls `POST /projects/{id}/index`. Useful after adding
new files.

### 2.4 Model indicator

Show the active model name (from the coder model env var) in the project
header. This is a static string fetched from `GET /health` or a new
`GET /projects/config` endpoint that returns `{"coder_model": "..."}`.

### 2.5 Back to chat

A "← Back to Chat" button in the project header restores the normal chat
view and disconnects the project context from subsequent messages. Project
chats remain inside the project panel — they are not surfaced in the main
chat history and do not contaminate the global session list.

### 2.6 Per-project chat sessions

Each project maintains its own list of chat sessions. When re-entering a
project the last active session resumes automatically (session ID persisted
in `localStorage` per project). Users can start a new session or switch
between past sessions without leaving the project view.

**Session list UI** (left sidebar, below the file tree):

```
[ + New Chat ]
──────────────────────────────
▸ Add error handling to router   today
  Plan index isolation strategy  yesterday
  Initial project setup          3d ago
```

Each row shows an auto-generated name (first ~50 chars of the opening
message) and a relative timestamp. Clicking a row loads that session's
history.

**Backend additions:**
- `GET /sessions?project_id={id}` — returns sessions scoped to the project,
  ordered by most recent activity. Each record includes `session_id`,
  `name` (first message excerpt), and `last_active`.
- `PATCH /sessions/{session_id}` with `{"name": "..."}` — allows renaming
  a session.
- `DELETE /sessions/{session_id}` — removes all `conversation_log` rows for
  that session (scoped to `user_id`). Does not affect `project_memory`.

---

## Phase 3 — Graph: project-aware coding context

**Scope:** Wire the project context into the graph so that in-project chat
uses the right ChromaDB collection, skips general-memory retrieval, and
routes directly to the coder model.

### 3.1 `AssistantState` additions

```python
project_id:      str   # "" for main chat
project_folder:  str   # resolved absolute path, "" for main chat
project_mode:    str   # "plan" | "code"; "" for main chat
```

All three set by the HTTP chat handler when the request carries a
`project_id` field (new optional field in `ChatRequest`). `project_mode`
defaults to `"code"` when `project_id` is non-empty and the field is
absent from the request (safe default — plan/code toggle is a Phase 3.5
addition and older clients omit it).

### 3.2 `memory_retrieval` node

If `state["project_id"]` is non-empty:
- Skip `conversation_memory` ChromaDB retrieval.
- Use `query_code_context(query, chroma_path, collection_name=f"code_context_{project_id}")`.
- Return project code context as `state["code_context"]`.

### 3.3 `intent_classifier` shortcut

If `project_id` is set, skip intent classification entirely. For now,
unconditionally return `code-write` so the full coding pipeline is active:

```python
{"intent": "code-write", "confidence": 1.0, "route_type": "coding_agent"}
```

The project UI is already a coding context — do not spend tokens
re-classifying. This is the key simplification for the main chat classifier.

**Note:** Phase 3.5 replaces this with a mode-aware branch that returns
`code-question` routing when `project_mode == "plan"`. The shortcut
structure stays identical; only the returned intent changes.

### 3.4 `tool_router` shortcut

If `project_id` is set, route directly to `coding_agent_tool`. The
`coding_agent_tool` node already handles both questions and write tasks;
the distinction is the user's phrasing, not graph routing.

### 3.5 Model selection

When `project_id` is set, `deps.coder_model` is used unconditionally.
No heuristic, no embedding check. One model per project session.

### 3.6 Remove `code-write` from the main chat intent domain

This is the primary decoupling step. Once projects can accept coding work,
`code-write` has no valid main-chat path and should be cleanly excised:

- **`intents.py`**: remove `code-write` from the heuristic keyword sets and
  from the embedding exemplar list. `_is_code_intent` becomes
  `intent == "code-question"` only (or is removed entirely if no longer used).
- **`embedding_router.py`**: remove `code-write` label from the exemplar
  corpus. Rebuild/re-index the embedding router if exemplars are pre-compiled.
- **`intent_classifier` node (`graph.py`)**: remove the branch that can
  return `code-write`. If the heuristic or embedding fires on a write-like
  message and `project_id` is absent, downgrade to `code-question` (answer
  the question, do not attempt a write) and emit a notice: "To edit files,
  open a project first."
- **`tool_router` node (`graph.py`)**: remove the edge to `coding_agent_tool`
  from the main chat path. The node stays in the graph; it is only reachable
  from the `project_id` shortcut added in 3.4.

Acceptance signal: `grep -r 'code-write' backend/intents.py backend/graph.py`
returns no matches outside the project-path shortcut added in 3.3.

### 3.7 Isolate `coding_agent_*` nodes as project-only

`coding_agent_tool` and `coding_agent_executor` remain in the compiled graph
for project sessions, but the graph edge topology must make them unreachable
from the main chat path:

- Remove the `coding_agent_tool` conditional edge from `tool_router`'s
  normal (non-project) branch.
- Add a guard at the top of `coding_agent_tool`: if `project_id` is empty,
  return an error state immediately rather than forwarding to the external
  agent. Belt-and-suspenders against future topology mistakes.
- `code_tool` (questions only) remains a normal `tool_router` edge for
  `code-question` in main chat. It is not touched by this phase.

---

## Phase 3.5 — Per-project sessions and plan/code modes

**Scope:** Activate multi-session UX within each project and introduce two
explicit interaction modes. Both build directly on Phase 3 infrastructure
(`project_id` in state, classifier shortcut, coding-agent isolation) with no
new graph nodes required.

### 3.5.1 Per-project chat sessions

The `conversation_log.project_id` column (Phase 1) already scopes messages
correctly. This section activates the multi-session UX within a project.

**Backend:**
- Confirm `GET /sessions?project_id={id}` returns per-project sessions
  ordered by most recent activity, with `session_id`, auto-generated `name`
  (first 50 chars of opening message), and `last_active` timestamp.
- Add `PATCH /sessions/{session_id}` — accepts `{"name": "..."}` to let
  users rename sessions. Scoped to `user_id`.
- Add `DELETE /sessions/{session_id}` — removes all `conversation_log` rows
  for that session (scoped to `user_id`). Does not touch `project_memory`.
  Requires confirmation from the UI before calling.

**Frontend:**
- Project panel left sidebar gains a collapsible **Chats** section below
  the file tree:
  ```
  [ + New Chat ]
  ──────────────────────────────
  ▸ Add error handling to router   today
    Plan index isolation strategy  yesterday
    Initial project setup          3d ago
  ```
- Active session highlighted; clicking a past row loads its history via
  `GET /chat/history?session_id={id}`.
- "New Chat" generates a fresh `session_id` (UUID) and clears the chat panel.
- Session name is editable inline (double-click); calls `PATCH /sessions/{id}`.
- Active `session_id` persisted in `localStorage` per project so re-opening
  a project resumes the last active session.

### 3.5.2 Plan / Code mode

Two explicit modes per project, toggled in the project header. Mode is a
project-level preference (not per-session) and is persisted in `localStorage`.

| | Plan | Code |
|---|---|---|
| Purpose | Discuss, review, architect, question | Write, create, edit files |
| Graph path | `code_tool` (questions) | `coding_agent_tool` → `coding_agent_executor` |
| File writes | Never | With confirmation gate |
| Header badge | **Plan** | **Code** |
| Input placeholder | "Ask about this project, review code, plan changes…" | "Describe the change to make…" |

**Backend — `ChatRequest`:**

Add optional field `project_mode: str` (`"plan"` | `"code"`, default `"code"`
when absent). The HTTP handler passes it into `AssistantState` (already added
in 3.1).

**Backend — `intent_classifier` shortcut (replaces Phase 3.3 placeholder):**

```python
if state["project_id"]:
    if state.get("project_mode") == "plan":
        return {"intent": "code-question", "confidence": 1.0, "route_type": "code_tool"}
    else:  # "code" or default
        return {"intent": "code-write", "confidence": 1.0, "route_type": "coding_agent"}
```

Plan mode reuses the existing `code_tool` node (already in the graph for
`code-question`). No new node needed.

**Backend — `coding_agent_tool` guard (extends Phase 3.7):**

Extend the existing empty-`project_id` guard to also reject calls where
`project_mode == "plan"`:

```python
if not state.get("project_id"):
    return error_state("coding_agent_tool requires a project context")
if state.get("project_mode") == "plan":
    return error_state(
        "File writes are disabled in Plan mode. Switch to Code mode to make changes."
    )
```

Belt-and-suspenders: the classifier shortcut already prevents this path in
Plan mode; the guard catches any future topology mistake.

**Frontend:**

- Two-segment toggle `[ Plan | Code ]` in the project header, right of the
  model indicator.
- Default is **Code** mode on first project open.
- Mode stored in `localStorage` keyed by `project_id`.
- Mode sent with every chat request as `"project_mode"` in the JSON body.
- In Plan mode: "Confirm write" button is hidden from any pending
  `coding_agent` tool outputs (defensively, since the backend guard should
  prevent reaching that state).

---

## Phase 4 — Project memory

**Scope:** Per-project memory that survives across sessions: key facts,
decisions, patterns the assistant learns about the project.

### 4.1 `project_memory` table

```sql
CREATE TABLE IF NOT EXISTS project_memory (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id   TEXT NOT NULL,
    user_id      TEXT NOT NULL,
    key          TEXT NOT NULL,
    value        TEXT NOT NULL,
    source       TEXT NOT NULL DEFAULT 'inferred',  -- 'inferred' | 'explicit'
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_projmem_key
    ON project_memory(project_id, user_id, key);
```

### 4.2 Extraction

After each coding turn (in `memory_writer`), run the same LLM extraction
pass already used for chat memory, but scoped to the project:

Prompt addition: "Extract facts specific to this project: architecture
decisions, tech stack, file conventions, known issues. Do not extract
personal facts."

Results are upserted into `project_memory` keyed by
`(project_id, user_id, key)`.

### 4.3 Injection

At the start of each in-project session, load top-N project memory entries
and prepend them to the system prompt alongside the code context:

```
You are working on project "{name}".
Known facts about this project:
- stack: FastAPI + SQLite + ChromaDB
- test runner: pytest, always use .venv/bin/python
- ...
```

This gives the assistant instant project orientation without requiring the
user to re-explain every session.

### 4.4 Memory panel

The existing memory panel (`/memory`) gains a project filter: when in a
project, show project memory alongside global memory. Tag each entry
with its scope (global / project).

---

## Phase 5 — Polish and best practices

Small improvements that collectively make the coding agent substantially
more useful. Implement in any order within this phase.

### 5.1 Diff preview before writes

The confirmation gate (`coding_agent_tool` → `coding_agent_executor`) already
exists. Enhance: after the agent produces its output, render the unified diff
in the chat bubble before the "Confirm / Cancel" buttons. Use the existing
`make_unified_diff` helper.

### 5.2 Structured file-change log

After a confirmed write, append an entry to a `file_changes` table:

```sql
CREATE TABLE IF NOT EXISTS file_changes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    change_type TEXT NOT NULL,   -- 'write' | 'create' | 'delete'
    ts          REAL NOT NULL
);
```

Exposed via `GET /projects/{id}/changes` for audit purposes.

### 5.3 Auto-index on file write

When `coding_agent_executor` completes and reports `files_changed`, trigger
`start_background_index` scoped to the project. The UI's status bar polls
and shows "Indexing..." briefly. Index is never stale after a write.

### 5.4 `.hearthignore`

Support a `.hearthignore` file in the project root (same syntax as
`.gitignore`) to exclude files from indexing. Parse with the `pathspec`
library (already a transitive dep via chromadb/langchain ecosystem; add
explicitly to `requirements.txt` if absent). Fall back to the existing
`_IGNORE_DIRS` frozenset if the file is absent.

### 5.5 Language coverage

Add tree-sitter parsers for:
- TypeScript (`tree-sitter-typescript`) — already in `_EXTENSION_MAP` as `javascript`; add a real TS grammar.
- Rust (`tree-sitter-rust`) — common secondary language.
- Go (`tree-sitter-go`) — optional, add if the packages are available.

Each is a soft dep — the fallback regex path already handles missing parsers.

### 5.6 Stale index warning

If `opened_at` is more than 24 h ago and the index status is `done`, show
a soft warning in the status bar: "Index may be stale — Re-index?" Do not
block the user; it's advisory only.

### 5.7 Project README preview

If `README.md` exists at the project root, render it (via the existing
`marked.js` already loaded in the UI) in a collapsible panel above the
chat input. Collapsed by default after the first open; state stored in
`localStorage`.

---

## Hard constraints (carry forward from prior roadmaps)

- `resolve_workspace_path` / `WorkspacePathError` must guard every path
  that comes from the frontend. No exceptions.
- No `shell=True` anywhere in subprocess calls.
- All frontend API calls remain relative-path. No hardcoded hosts or ports.
- `CODE_WORKSPACE_ROOT` is still the single root. Projects live inside it.
  A project cannot escape it.
- Single-worker assumption for `_pending_code_writes` and the new index
  status dict is documented. Do not add workers until state moves to SQLite.
- Cloud model is not used for in-project coding unless the user explicitly
  triggers it. Coder model only.

---

## What this buys the main chat classifier

After Phase 3 ships (3.3–3.7 complete), the main chat classifier's domain
shrinks to:

| Intent | Path | Note |
|---|---|---|
| `chat` | `responder` | unchanged |
| `code-question` | `code_tool` | voice one-offs only; no writes |
| `weather` | `weather_tool` | unchanged |
| `music` | fast-path (pre-graph) | unchanged |

`code-write` is fully excised. `coding_agent_tool` and `coding_agent_executor`
are project-only graph paths, never touched by a main-chat message.

**Remaining residual coupling:** `code-question` in main chat. This is
intentional — forcing a user to open a project just to ask "what does this
function do" is unnecessary friction, especially for voice. If this ever
becomes a maintenance problem, the clean path is to add a lightweight
"ask about code" mode to the project panel and then drop `code-question`
from main chat entirely. That is out of scope for this roadmap.

**Net effect on the classifier:**
- Ambiguous `code-write` branch (heuristic + embedding disagreement) gone.
- Embedding exemplar corpus is smaller and more precise.
- `intent_classifier` token budget is reduced (no write-vs-question split
  logic to run).
- Routing confidence for remaining intents improves.

---

## Acceptance criteria (per phase)

### Phase 1
- [ ] `POST /projects` creates folder + DB row; rejects traversal paths.
- [ ] `POST /projects/{id}/index` triggers background index; status endpoint
      returns correct lifecycle states.
- [ ] `conversation_log` and `summaries` rows carry `project_id`; existing
      rows (NULL) still work for main chat.
- [ ] `git init` branch: succeeds when git is present; logs and continues when not.

### Phase 2
- [ ] Projects section button is enabled; clicking it shows project list.
- [ ] Creating a project from the UI creates the folder and appears in the list.
- [ ] Opening a project shows file tree and index status bar.
- [ ] Re-index button triggers index and status bar updates to `done`.
- [ ] "Back to Chat" restores normal chat view.

### Phase 3
- [ ] In-project chat uses `code_context_{project_id}` collection.
- [ ] `intent_classifier` is not called when `project_id` is set.
- [ ] Coder model is used for all in-project turns, confirmed in logs.
- [ ] `grep -r 'code-write' backend/intents.py backend/graph.py` returns no
      matches outside the project shortcut block.
- [ ] A main-chat message with write-like phrasing and no `project_id` receives
      a `code-question` response with an "open a project" notice — no write
      is attempted.
- [ ] `coding_agent_tool` with an empty `project_id` returns an error state
      without dispatching any coding task.

### Phase 3.5
- [ ] `GET /sessions?project_id={id}` returns only sessions for that project,
      ordered by most recent activity.
- [ ] "New Chat" button in the project panel creates a fresh session; chat
      panel clears and the new session appears at the top of the list.
- [ ] Switching to a past session loads its history without leaving the project view.
- [ ] Session rename (`PATCH /sessions/{id}`) updates the sidebar label.
- [ ] `DELETE /sessions/{id}` removes all conversation rows; other sessions
      in the same project are unaffected.
- [ ] Mode toggle switches between Plan and Code; selection persists across
      project re-opens (via `localStorage`).
- [ ] In Plan mode, a message requesting a file write receives a `code_tool`
      response (no write attempted); backend logs confirm `code_tool` node was used.
- [ ] In Code mode, the full `coding_agent_tool` → `coding_agent_executor`
      pipeline is reachable and confirmation gate appears as normal.
- [ ] `coding_agent_tool` called with `project_mode == "plan"` returns the
      "disabled in Plan mode" error without dispatching any coding task.

### Phase 4
- [ ] Project facts extracted after coding turns appear in `project_memory`.
- [ ] System prompt for in-project session includes project facts.
- [ ] Memory panel shows project-scoped entries when in a project.

### Phase 5
- [ ] Diff preview renders in confirmation bubble before write.
- [ ] `file_changes` table populated after confirmed writes.
- [ ] Index triggered automatically after write; UI status bar reflects it.
