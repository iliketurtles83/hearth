# Memory Architecture

> **Status (2026-09-07):** Phase 1 and the Phase 2 "sleep consolidation + knowledge graph" increment are implemented. Remaining Phase 2 backlog: `tool_events`, working memory, fact versioning.

## The Problem

Memory is currently stored in a way that makes it hard to recall accurately:

- **Hash embeddings produce zero semantic recall** — "my name is Alice" and "I'm called Alice" have different vectors
- **ChromaDB stores `"key: value"` strings** — too compressed for meaningful vector similarity
- **LLM extraction uses `/api/generate`** — Ollama ignores system prompts on that endpoint
- **Facts silently overwrite** — no versioning, no provenance, no audit trail
- **No fact decay** — ephemeral information ("I'm in Paris this week") stored identically to permanent facts
- **Episodic summaries unfiltered for sensitive data** — can leak into structured facts during consolidation
- **Threshold-based consolidation** — triggers mid-conversation, adds latency, can duplicate on crash recovery

## The Goal

Memory that is **accurate**, **recalled at the right time**, and **useful for retrieve-and-report**. Honest about what it knows and what it doesn't.

---

## Memory Layers

| Layer | What | Format | Lifespan |
|-------|------|--------|----------|
| **Episodic** | Raw conversation turns + tool-call transcripts | Structured log (session_id, turn_id, role, content, timestamp) | Permanent, retrieved in windows |
| **Semantic** | Stable facts about you, your preferences, your projects | Knowledge graph (entity → relation → entity) with provenance | Persistent, with decay |
| **Working** | Active goals, current plan state, recent tool results | In-memory dict, scoped to session | Lost on restart (recovered from episodic) |

---

## Storage Schema

### Existing Tables (with modifications)

**`facts`** — Stable facts with versioning and provenance:
```sql
CREATE TABLE facts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    source     TEXT NOT NULL,
    version    INTEGER NOT NULL DEFAULT 1,
    confidence REAL NOT NULL DEFAULT 1.0,
    created_at REAL NOT NULL,
    updated_at REAL,
    expires_at REAL,
    sensitive  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_facts_user_id ON facts(user_id);
CREATE UNIQUE INDEX idx_facts_user_key ON facts(user_id, key);
```

**`fact_history`** — Version trail for every fact:
```sql
CREATE TABLE fact_history (
    id         INTEGER PRIMARY KEY,
    fact_id    INTEGER NOT NULL,
    value      TEXT NOT NULL,
    version    INTEGER NOT NULL,
    source     TEXT NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY (fact_id) REFERENCES facts(id)
);
```

**`preferences`** — User preferences (add versioning):
```sql
CREATE TABLE preferences (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    version    INTEGER NOT NULL DEFAULT 1,
    updated_at REAL NOT NULL,
    sensitive  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_preferences_user_id ON preferences(user_id);
CREATE UNIQUE INDEX idx_preferences_user_key ON preferences(user_id, key);
```

**`summaries`** — Episodic summaries (add sensitive filter):
```sql
CREATE TABLE summaries (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      TEXT NOT NULL,
    session_id   TEXT NOT NULL,
    summary      TEXT NOT NULL,
    created_at   REAL NOT NULL,
    consolidated INTEGER NOT NULL DEFAULT 0,
    sensitive    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_summaries_user_id ON summaries(user_id);
CREATE INDEX idx_summaries_session_id ON summaries(session_id);
```

### New Tables

**`entities`** — Semantic IDs for known things (user-scoped for isolation):
```sql
CREATE TABLE entities (
    user_id    TEXT NOT NULL,
    id         TEXT NOT NULL,          -- "person:alice", "city:helsinki", "project:hearth"
    type       TEXT NOT NULL,          -- "person", "city", "project", "organization"
    attributes TEXT,                   -- JSON: {"name": "Alice", "age": 30}
    created_at REAL NOT NULL,
    PRIMARY KEY (user_id, id)
);
```

**`relations`** — Knowledge graph edges (user-scoped):
```sql
CREATE TABLE relations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    subject    TEXT NOT NULL,          -- entity reference (e.g. "person:alice")
    predicate  TEXT NOT NULL,          -- relation type (e.g. "lives_in")
    object     TEXT NOT NULL,          -- entity reference (e.g. "city:helsinki")
    confidence REAL NOT NULL DEFAULT 1.0,
    source     TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL,
    expires_at REAL
);
CREATE UNIQUE INDEX idx_relations_user_spo ON relations(user_id, subject, predicate, object);
CREATE INDEX idx_relations_user_object ON relations(user_id, object);
CREATE INDEX idx_relations_user_predicate ON relations(user_id, predicate);
```

**`tool_events`** — Structured tool-call transcripts (music, calendar, todo — NOT weather):
```sql
CREATE TABLE tool_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    turn_id     INTEGER NOT NULL,
    tool_name   TEXT NOT NULL,
    params      TEXT,                 -- JSON: {"query": "play radiohead"}
    result      TEXT,                 -- JSON: {"track": "Creep", "artist": "Radiohead"}
    timestamp   REAL NOT NULL
);
CREATE INDEX idx_tool_events_session ON tool_events(session_id);
CREATE INDEX idx_tool_events_tool ON tool_events(tool_name);
```

---

## Memory Flow

### Storage Phase (per turn)

1. **Log turn** to `conversation_log` (existing behavior)
2. **Extract explicit memory** via regex — works fine for direct patterns ("my name is X", "remember this: Y")
3. **Log tool events** — ~~capture music/calendar/todo calls as structured events~~ (deferred to Phase 2)
4. **Store in knowledge graph** — ~~if extraction produces entity/relation triples, upsert into `entities` and `relations`~~ (deferred to Phase 2)
5. **Update ChromaDB** — store **original text** (the turn or summary), not `"key: value"` ✅

### Consolidation Phase (scheduled)

**Trigger (implemented)**: The interval "sleep" scheduler (`memory_scheduler.py`) wakes every `MEMORY_SLEEP_INTERVAL_SECONDS` and runs a consolidation pass for users idle for at least `MEMORY_SLEEP_IDLE_SECONDS` — so the heavy LLM pass runs off the hot path. Also available manually via `POST /memory/consolidate`. The old per-turn threshold trigger is disabled by default (`MEMORY_CONSOLIDATION_THRESHOLD=0`) but re-enables if set >0.

**Process**:
1. **Read pending** — unconsolidated summaries ✅
2. **LLM extraction** — one `/api/chat` call per summary returning facts, preferences, and triples ✅
3. **Merge into graph** — upsert facts/prefs (with `expires_at` from `ttl_days`) **and** upsert `entities`+`relations` triples ✅ (versioning/conflict-resolution still deferred)
4. **Conflict resolution** — still deferred (upsert overwrites on `(user_id, key)`)
5. **Mark analyzed** — flag processed summaries ✅
6. **Rebuild index** — ChromaDB upserted on each write ✅
7. **Decay** — prune facts whose `expires_at` has passed (and drop their vectors) ✅

### Recall Phase (per query)

**Layered retrieval** — query multiple sources in parallel:

| Source | Query | Purpose | Max items | Status |
|--------|-------|---------|-----------|--------|
| **Knowledge graph** | Relation matching on query terms | "Who do I know?", "What am I working on?" | 5 | ✅ Implemented |
| **Semantic (embeddings)** | Vector similarity on original text | "What did we discuss about X last week?" | 3-5 | ✅ Implemented |
| **Episodic (recent)** | Last N turns | Conversation continuity | Context budget allows | ✅ Existing |
| **Working** | In-memory lookup | Current session state | N/A | ❌ Undefined |

**Injection into LLM**:
- If intent classifier says `memory-needed`, inject all top hits
- Graph-recall triple hits are merged into the same hit list and injected via `_augment_system_with_memories` ✅
- Otherwise, inject only the session summary

---

## Critical Fixes (Phase 1)

### 1.1 Replace hash embeddings with real model embeddings ✅

`HashEmbedingFunction` → `OllamaEmbeddingFunction` using `nomic-embed-text` via Ollama (same model `embedding_router.py` already uses: `ROUTER_EMBED_MODEL`).

**Implementation**: `_ollama_embed_sync()` helper in `memory.py` (note: `embedding_router.py` has its own separate `ollama_embed_text()` — not shared). `OllamaEmbeddingFunction` implements ChromaDB's `EmbeddingFunction` interface with a simple in-process cache (a plain `dict`; no LRU eviction). Graceful fallback to hash-based embedding if Ollama is unreachable. `MemoryStore._maybe_recreate_collection()` detects dimension mismatch on startup and recreates the ChromaDB collection so the new embedder takes effect without manual cleanup.

**Gotcha**: Existing ChromaDB data was embedded with 192-dim hash vectors. The dimension mismatch detection triggers a silent collection drop on first boot after this change — old vectors are lost. New vectors are 768-dim (nomic-embed-text).

### 1.2 Fix LLM extraction to use `/api/chat` ✅

Changed from `client.post(f"{ollama_url}/api/generate", json=payload)` with `"prompt"` + `"system"` to `client.post(f"{ollama_url}/api/chat", json=payload)` with `"messages"` array containing `{"role": "system", ...}` and `{"role": "user", ...}`.

Response parsing changed from `data["response"]` to `data["message"]["content"]`.

Ollama's `/api/generate` does not respect system prompts — this is why extraction is noisy.

### 1.3 Store original text in ChromaDB, not `"key: value"` ✅

ChromaDB now stores the original user message (`source_message` in `ingest_user_message`, `summary_text` in `consolidate_pending`). The `key` and `value` are stored in ChromaDB metadata for structured lookup.

The structured fact stays in SQLite; ChromaDB is purely a retrieval index with the full context.

`retrieve()` reads key/value from ChromaDB metadata instead of parsing the document text with `partition(":")`. The token-overlap filter gating Chroma results was removed — Chroma now participates fully in the merged ranking.

---

## Entity ID Convention

Semantic IDs enable cross-session entity matching without a lookup table:

- **Persons**: `person:{name_slug}` — e.g., `person:alice`
- **Cities**: `city:{name_slug}` — e.g., `city:helsinki`
- **Projects**: `project:{name_slug}` — e.g., `project:hearth`
- **Organizations**: `org:{name_slug}` — e.g., `org:mozilla`
- **Generic**: `{type}:{slug}` — e.g., `song:creep`, `book:dune`

Slug generation: lowercase, strip non-alphanumeric, replace spaces with hyphens.

---

## Tool-Call Memory

Capture structured tool events for:
- **Music** — what was played, queued, searched
- **Calendar** — events created, queried, modified
- **Todo** — tasks created, completed, modified

Do NOT capture:
- **Weather** — ephemeral, not worth storing as a structured event
- **Code generation** — output is in the conversation log, not stable knowledge

---

## Consolidation Configuration

| Env Var | Default | Description | Status |
|---------|---------|-------------|--------|
| `MEMORY_SLEEP_ENABLED` | `true` | Start the interval "sleep" consolidation job | ✅ Active (`memory_scheduler.py`) |
| `MEMORY_SLEEP_INTERVAL_SECONDS` | `900` | Seconds between scheduler heartbeats | ✅ Active |
| `MEMORY_SLEEP_IDLE_SECONDS` | `1800` | Seconds a user must be idle before consolidation | ✅ Active |
| `MEMORY_CONSOLIDATION_BATCH_SIZE` | `50` | Max summaries to process per user per pass | ✅ Active |
| `MEMORY_CONSOLIDATION_THRESHOLD` | `0` | Per-turn trigger; `0` = disabled (sleep job handles it), set >0 to re-enable | ✅ Active (opt-in) |

---

## Phase 1 Implementation Notes

**Files changed**: `backend/memory.py`, `backend/tests/test_memory_isolation.py`

**Migration path**: On first boot after deployment, `MemoryStore._maybe_recreate_collection()` detects that the existing ChromaDB collection was built with 192-dim hash embeddings while the runtime embedder produces 768-dim vectors. It silently drops and recreates the `conversation_memory` collection. Old vectors are lost; new vectors are generated from original text on next retrieval.

**No schema migrations needed**: Phase 1 changes don't add or modify any SQLite tables. The existing `facts`, `preferences`, `summaries`, and `conversation_log` tables are used as-is.

**Test coverage**: All 18 memory isolation tests pass. 14 graph tests and 7 responder modality tests pass. 4 pre-existing failures in `test_embedding_router.py` and `test_music.py` are unrelated to this change.

---

## Phase 2 (implemented) — Sleep consolidation + knowledge graph

**Files changed**: `backend/memory.py`, `backend/memory_scheduler.py`, `backend/main.py`, `backend/graph.py`, `backend/tests/test_memory_graph.py`, `backend/tests/test_memory_scheduler.py`, `.env.example`

- **Unified extraction**: `_llm_extract_memory` makes one `/api/chat` call per summary returning `facts`, `preferences`, and `triples` (tolerates the legacy `candidates` shape). Facts carry an optional `ttl_days`.
- **Knowledge graph**: `entities` (user-scoped semantic IDs) + `relations` (triple edges) tables. `run_sleep_pass` upserts triples; `retrieve()` merges a graph-recall source (relation matching on query terms, scoped per user) into the existing hybrid ranking.
- **Fact decay**: `ttl_days` → `facts.expires_at`; the sleep pass prunes expired facts and drops their Chroma vectors.
- **Sleep scheduler**: `memory_scheduler.py` runs a plain asyncio loop (started/cancelled from `_graph_lifespan`) that, on each heartbeat, consolidates users who are idle and have pending summaries. No external scheduler dependency.
- **Per-turn trigger**: `MEMORY_CONSOLIDATION_THRESHOLD` now defaults to `0` (disabled); the sleep job is the primary consolidator.

**Migration**: `entities`/`relations` are created with `CREATE TABLE IF NOT EXISTS` in `_init_db`. Runtime state DBs are gitignored/ephemeral, so no data migration is required.

**Tests**: `test_memory_graph.py` (triples + graph recall + decay + legacy tolerance) and `test_memory_scheduler.py` (config, enable/disable, idle gate, cancellation). Existing consolidation/isolation tests still pass via the tolerant extractor.

---

## Phase 2 Backlog

Remaining (still deferred):

- [ ] **`tool_events` table** — Structured music/calendar/todo capture
- [ ] **Fact versioning** — `fact_history` table + conflict resolution
- [ ] **Working memory** — In-memory dict, population strategy, injection
- [ ] **Sensitive data filtering** — Improve beyond regex (LLM extraction can still leak)

Completed in this pass:

- [x] **`entities` + `relations` tables** — Knowledge graph schema (user-scoped)
- [x] **Entity/relation extraction prompt** — Unified `/api/chat` prompt (facts + preferences + triples)
- [x] **Idle timer consolidation** — `memory_scheduler.py` interval + idle-gated background job
- [x] **Fact decay** — Sleep pass prunes facts past `expires_at` (set from `ttl_days`)

---

## What Stays the Same

- Regex extraction for direct messages — works fine for explicit patterns
- Session summaries — rolling summaries every N turns, stored in `summaries` table
- Intent-based memory injection — `_should_inject_memory` logic stays
- SQLite as canonical store — ChromaDB remains the retrieval index
- Single-flight consolidation lock — prevents concurrent consolidation passes
