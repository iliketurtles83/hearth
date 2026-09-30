"""Memory storage and retrieval.

Storage contract:
- SQLite is the canonical store for structured memory state (facts, preferences,
    summaries, and consolidation state).
- Chroma is a retrieval index for semantic recall and is populated from canonical
    memory content; it is not the source of truth for persisted user memory.
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
import json
import asyncio
import logging
from dataclasses import dataclass
from threading import Lock
from typing import Any

import chromadb
import httpx
import numpy as np
from chromadb.api.types import Documents, EmbeddingFunction

log = logging.getLogger("assistant.memory")


@dataclass
class MemoryCandidate:
    table: str  # facts | preferences
    key: str
    value: str
    source: str
    ttl_days: int | None = None


@dataclass
class MemoryTriple:
    subject: str
    predicate: str
    obj: str
    confidence: float = 1.0


@dataclass
class MemoryExtraction:
    candidates: list[MemoryCandidate]
    triples: list[MemoryTriple]


# ── LLM-based memory extraction (Phase 12b) ──────────────────────────────────────
# System prompt for memory extraction LLM. Instructs the model to extract stable
# facts and preferences from episodic summaries, returning structured JSON with
# confidence scores. Low-confidence extractions (< 0.7) are filtered downstream.
MEMORY_EXTRACTOR_SYSTEM = """You are a memory extraction assistant that distills a conversation
summary into durable semantic memory for a personal assistant.

Extract only information that is:
- Stable and generalizable (not ephemeral chat content)
- About the user (not the assistant)
- Non-sensitive (no passwords, exact addresses, financial details, identifiers)

Return ONLY valid JSON with no preamble or explanation. Format:
{
  "facts": [
    { "key": "string", "value": "string", "confidence": 0.0-1.0, "ttl_days": null or integer }
  ],
  "preferences": [
    { "key": "string", "value": "string", "confidence": 0.0-1.0 }
  ],
  "triples": [
    { "subject": "type:slug", "predicate": "string", "object": "string", "confidence": 0.0-1.0 }
  ]
}

Rules:
- "ttl_days" is null for permanent facts (identity, stable traits) and a small integer
  (1-30) for time-bound facts (trips, deadlines, "this week").
- Triple "subject" and "object" use semantic ids: "person:alice", "city:helsinki",
  "project:hearth", "org:mozilla", or a plain phrase for values.
- Use concise relation predicates like lives_in, works_on, favorite_thing, met, owns, wants.
- Prefer triples for relationships between named entities; use facts/preferences for
  flat key/value statements.
- Only include items with confidence >= 0.7.

If nothing stable can be extracted, return { "facts": [], "preferences": [], "triples": [] }."""


@dataclass
class MemoryCommand:
    action: str
    query: str | None = None


_COLLECTION_NAME = "conversation_memory"
_REINDEX_BATCH = 64


class EmbeddingUnavailableError(RuntimeError):
    """Raised when the embedding endpoint cannot produce a vector."""


def _embed_endpoint() -> tuple[str, str]:
    base_url = (os.getenv("OPENAI_EMBED_BASE_URL") or os.getenv("OPENAI_BASE_URL", "http://localhost:10001/v1")).rstrip("/")
    model = os.getenv("ROUTER_EMBED_MODEL", "nomic-embed-text")
    return base_url, model


def _openai_embed_sync(text: str, *, base_url: str, model: str) -> list[float] | None:
    """Synchronous embedding call via OpenAI-compatible /v1/embeddings.

    Returns None when the endpoint is unreachable or returns no vector.
    """
    try:
        payload = {"model": model, "input": text}
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(f"{base_url}/embeddings", json=payload)
            resp.raise_for_status()
            data = resp.json()
        # OpenAI format: {"data": [{"embedding": [...]}]}
        embedding_list = data.get("data", [{}])[0].get("embedding", [])
        vector = np.asarray(embedding_list, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0:
            return None
        return vector.tolist()
    except Exception:
        return None


class OpenAIEmbeddingFunction(EmbeddingFunction):
    """ChromaDB embedding function backed by local OpenAI-compatible endpoint (nomic-embed-text).

    Embeddings are cached per instance.  If the endpoint is unreachable this
    raises EmbeddingUnavailableError rather than returning a substitute vector:
    non-semantic vectors must never enter the semantic index.  SQLite is the
    canonical store, so MemoryStore defers indexing and retries later.
    """

    def __init__(self) -> None:
        self._cache: dict[str, list[float]] = {}

    def _embed_one(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        base_url, model = _embed_endpoint()
        vec = _openai_embed_sync(text, base_url=base_url, model=model)
        if vec is None:
            raise EmbeddingUnavailableError("embedding endpoint unavailable")
        self._cache[text] = vec
        return vec

    def __call__(self, input: Documents) -> list[list[float]]:
        return [self._embed_one(t) for t in input]


class MemoryStore:
    def __init__(self, db_path: str | None = None, chroma_path: str | None = None) -> None:
        root = os.path.dirname(__file__)
        db_default = os.path.join(root, "memory.db")
        chroma_default = os.path.join(root, "chroma")
        self.db_path = db_path or os.getenv("MEMORY_DB_PATH", db_default)
        self.chroma_path = chroma_path or os.getenv("CHROMA_PATH", chroma_default)
        self.top_n = int(os.getenv("MEMORY_TOP_N", "5"))
        self.min_relevance_score = float(os.getenv("MEMORY_MIN_RELEVANCE_SCORE", "0.28"))

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.chroma_path, exist_ok=True)

        self._lock = Lock()
        self._consolidation_lock = Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Improve concurrency characteristics for mixed read/write workload.
        # Some deployments mount the DB file read-only; in that case treat these
        # pragmas as best-effort and continue with SQLite defaults.
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.OperationalError as exc:
            if "readonly" in str(exc).lower():
                log.warning(
                    "memory.sqlite_pragmas_skipped | db=%s reason=readonly",
                    self.db_path,
                )
            else:
                raise
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._init_db()

        self._embedder = OpenAIEmbeddingFunction()
        self._chroma = chromadb.PersistentClient(path=self.chroma_path)
        self._collection = self._chroma.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=self._embedder,
        )
        # Set when a Chroma write was deferred; cleared once the index catches up.
        self._index_stale = False
        self._reconcile_index()

    def _stored_embedding_dim(self) -> int | None:
        """Return the dimension of vectors already in the collection, or None if empty."""
        got = self._collection.get(limit=1, include=["embeddings"])
        embeddings = got.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return None
        return len(embeddings[0])

    def _reconcile_index(self) -> None:
        """Bring the Chroma index in line with SQLite, the canonical store.

        The collection is recreated only when its stored vectors have a
        different dimension from the live embedder's (i.e. the embedding model
        changed); then every fact/preference row missing from Chroma is indexed.
        If the embedder is unreachable (e.g. still warming up at boot) the index
        is left untouched and rows are picked up by a later reconcile.
        """
        base_url, model = _embed_endpoint()
        probe = _openai_embed_sync("probe", base_url=base_url, model=model)
        if probe is None:
            log.warning("memory.index_reconcile_skipped | reason=embedder_unavailable")
            self._index_stale = True
            return

        try:
            stored_dim = self._stored_embedding_dim()
            if stored_dim is not None and stored_dim != len(probe):
                log.info(
                    "memory.collection_recreate | reason=embedding_dimension_mismatch "
                    "stored_dim=%d new_dim=%d count=%d",
                    stored_dim, len(probe), self._collection.count(),
                )
                self._chroma.delete_collection(name=_COLLECTION_NAME)
                self._collection = self._chroma.get_or_create_collection(
                    name=_COLLECTION_NAME,
                    embedding_function=self._embedder,
                )
            self._index_missing_rows()
            self._index_stale = False
        except Exception as exc:
            log.warning("memory.index_reconcile_failed | error=%s", type(exc).__name__)
            self._index_stale = True

    def _index_missing_rows(self) -> None:
        """Index every live fact/preference row that has no vector in Chroma.

        The original source message is not kept in SQLite, so rebuilt entries
        embed "key: value".  Caller holds _lock (or is __init__).
        """
        now = time.time()
        rows = self._conn.execute(
            """
            SELECT 'facts:' || id AS memory_id, 'facts' AS table_name,
                   user_id, key, value, source, created_at AS ts
            FROM facts
            WHERE expires_at IS NULL OR expires_at > ?
            UNION ALL
            SELECT 'preferences:' || id, 'preferences',
                   user_id, key, value, 'preference', updated_at
            FROM preferences
            """,
            (now,),
        ).fetchall()
        if not rows:
            return

        indexed = set(self._collection.get(include=[])["ids"])
        missing = [r for r in rows if r["memory_id"] not in indexed]
        for i in range(0, len(missing), _REINDEX_BATCH):
            batch = missing[i : i + _REINDEX_BATCH]
            self._collection.upsert(
                ids=[r["memory_id"] for r in batch],
                documents=[f"{str(r['key']).replace('_', ' ')}: {r['value']}" for r in batch],
                metadatas=[
                    {
                        "table": r["table_name"],
                        "key": r["key"],
                        "value": r["value"],
                        "source": r["source"],
                        "user_id": r["user_id"],
                        "created_at": r["ts"],
                        "consent_status": "reindexed",
                    }
                    for r in batch
                ],
            )
        if missing:
            log.info("memory.index_rebuilt | indexed=%d", len(missing))

    _SENSITIVE_SECRET_PATTERNS = [
        r"\b(api[_-]?key|token|password|secret|passwd|bearer)\b",
        r"\bsk-[a-z0-9]{16,}\b",
        r"\bghp_[a-z0-9]{20,}\b",
    ]
    _SENSITIVE_PHONE_PATTERNS = [
        r"\b\+?\d[\d\s().-]{7,}\d\b",
    ]
    _SENSITIVE_ADDRESS_PATTERNS = [
        r"\b\d+\s+[a-z0-9\s]+\s+(street|st|road|rd|avenue|ave|lane|ln|drive|dr|boulevard|blvd)\b",
    ]
    _CONFIRM_FIRST_PATTERNS = [
        r"\b(i\s+lived\s+in|used\s+to\s+live\s+in|lived\s+at)\b",
        r"\b(during|between|from\s+\d{4}\s+to\s+\d{4}|in\s+\d{4})\b",
    ]

    def _init_db(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.executescript(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id    TEXT NOT NULL,
                    key        TEXT NOT NULL,
                    value      TEXT NOT NULL,
                    source     TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL,
                    sensitive  INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS preferences (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id    TEXT NOT NULL,
                    key        TEXT NOT NULL,
                    value      TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    sensitive  INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS summaries (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id      TEXT NOT NULL,
                    session_id   TEXT NOT NULL,
                    summary      TEXT NOT NULL,
                    created_at   REAL NOT NULL,
                    consolidated INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS conversation_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT NOT NULL,
                    user_id     TEXT NOT NULL,
                    role        TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    ts          REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_facts_user_id
                    ON facts(user_id);
                CREATE INDEX IF NOT EXISTS idx_preferences_user_id
                    ON preferences(user_id);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_preferences_user_key
                    ON preferences(user_id, key);
                CREATE INDEX IF NOT EXISTS idx_summaries_user_id
                    ON summaries(user_id);
                CREATE INDEX IF NOT EXISTS idx_summaries_session_id
                    ON summaries(session_id);
                CREATE INDEX IF NOT EXISTS idx_convlog_session_user_ts
                    ON conversation_log(session_id, user_id, ts);
                CREATE INDEX IF NOT EXISTS idx_convlog_user_ts
                    ON conversation_log(user_id, ts DESC);

                CREATE TABLE IF NOT EXISTS entities (
                    user_id    TEXT NOT NULL,
                    id         TEXT NOT NULL,
                    type       TEXT NOT NULL,
                    attributes TEXT,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (user_id, id)
                );

                CREATE TABLE IF NOT EXISTS relations (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id    TEXT NOT NULL,
                    subject    TEXT NOT NULL,
                    predicate  TEXT NOT NULL,
                    object     TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    source     TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL,
                    expires_at REAL
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_relations_user_spo
                    ON relations(user_id, subject, predicate, object);
                CREATE INDEX IF NOT EXISTS idx_relations_user_object
                    ON relations(user_id, object);
                CREATE INDEX IF NOT EXISTS idx_relations_user_predicate
                    ON relations(user_id, predicate);

                CREATE TABLE IF NOT EXISTS session_titles (
                    user_id    TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    title      TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (user_id, session_id)
                );
                CREATE INDEX IF NOT EXISTS idx_session_titles_user
                    ON session_titles(user_id);
                """
            )
            # Live-instance migration: add 'consolidated' column if it doesn't exist yet.
            try:
                self._conn.execute(
                    "ALTER TABLE summaries ADD COLUMN consolidated INTEGER NOT NULL DEFAULT 0"
                )
                self._conn.commit()
            except sqlite3.OperationalError:
                pass  # column already exists

            # Live-instance migration: enforce one fact per (user_id, key).
            # Keep the most recent row for each key so unique index creation succeeds.
            # Read-only DB mounts cannot apply migrations; skip them gracefully.
            try:
                self._conn.execute(
                    """
                    DELETE FROM facts
                    WHERE id NOT IN (
                        SELECT MAX(id)
                        FROM facts
                        GROUP BY user_id, key
                    )
                    """
                )
                self._conn.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_user_key
                    ON facts(user_id, key)
                    """
                )
            except sqlite3.OperationalError as exc:
                if "readonly" in str(exc).lower():
                    log.warning(
                        "memory.migration_skipped | db=%s migration=facts_unique_index reason=readonly",
                        self.db_path,
                    )
                else:
                    raise
            self._conn.commit()

    def log_turn(self, session_id: str, user_id: str, role: str, content: str) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO conversation_log (session_id, user_id, role, content, ts)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, user_id, role, content, now),
            )
            self._conn.commit()

    def get_session_turns(self, session_id: str, user_id: str, limit: int = 500) -> list[dict[str, Any]]:
        safe_limit = max(1, int(limit))
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT role, content, ts
                FROM conversation_log
                WHERE session_id = ? AND user_id = ?
                ORDER BY ts ASC
                LIMIT ?
                """,
                (session_id, user_id, safe_limit),
            ).fetchall()
        return [
            {
                "role": str(r["role"]),
                "content": str(r["content"]),
                "ts": float(r["ts"]),
            }
            for r in rows
        ]

    def list_sessions(self, user_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    c1.session_id AS session_id,
                    MIN(c1.ts) AS created_at,
                    MAX(c1.ts) AS updated_at,
                    COUNT(*) AS message_count,
                    MAX(st.title) AS title,
                    (
                        SELECT c2.content
                        FROM conversation_log c2
                        WHERE c2.session_id = c1.session_id
                          AND c2.user_id = ?
                          AND c2.role = 'user'
                        ORDER BY c2.ts ASC
                        LIMIT 1
                    ) AS preview
                FROM conversation_log c1
                LEFT JOIN session_titles st
                    ON st.session_id = c1.session_id AND st.user_id = c1.user_id
                WHERE c1.user_id = ?
                GROUP BY c1.session_id
                ORDER BY updated_at DESC
                """,
                (user_id, user_id),
            ).fetchall()
        return [
            {
                "session_id": str(r["session_id"]),
                "created_at": float(r["created_at"]),
                "updated_at": float(r["updated_at"]),
                "message_count": int(r["message_count"]),
                "title": str(r["title"]) if r["title"] is not None else "",
                "preview": str(r["preview"] or "")[:120],
            }
            for r in rows
        ]

    def delete_session(self, session_id: str, user_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM conversation_log WHERE session_id = ? AND user_id = ?",
                (session_id, user_id),
            )
            self._conn.execute(
                "DELETE FROM summaries WHERE session_id = ? AND user_id = ?",
                (session_id, user_id),
            )
            self._conn.execute(
                "DELETE FROM session_titles WHERE session_id = ? AND user_id = ?",
                (session_id, user_id),
            )
            self._conn.commit()

    def reset_session(self, session_id: str, user_id: str) -> None:
        self.delete_session(session_id, user_id)

    def set_session_title(self, session_id: str, user_id: str, title: str) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO session_titles (user_id, session_id, title, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, session_id)
                DO UPDATE SET title = excluded.title, updated_at = excluded.updated_at
                """,
                (user_id, session_id, title, now),
            )
            self._conn.commit()

    def clear_session_title(self, session_id: str, user_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM session_titles WHERE session_id = ? AND user_id = ?",
                (session_id, user_id),
            )
            self._conn.commit()

    def session_exists(self, session_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM conversation_log WHERE session_id = ? LIMIT 1",
                (session_id,),
            ).fetchone()
        return row is not None

    def session_exists_for_user(self, session_id: str, user_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT 1
                FROM conversation_log
                WHERE session_id = ? AND user_id = ?
                LIMIT 1
                """,
                (session_id, user_id),
            ).fetchone()
        return row is not None

    def get_latest_session_summary(
        self,
        session_id: str,
        user_id: str,
    ) -> str:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT summary
                FROM summaries
                WHERE session_id = ? AND user_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (session_id, user_id),
            ).fetchone()
        return str(row["summary"]) if row and row["summary"] is not None else ""

    def count_unconsolidated(self, user_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT COUNT(*) AS c
                FROM summaries
                WHERE user_id = ? AND consolidated = 0
                """,
                (user_id,),
            ).fetchone()
        return int(row["c"] if row is not None else 0)

    def count_session_turns(self, session_id: str, user_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT COUNT(*) AS c
                FROM conversation_log
                WHERE session_id = ? AND user_id = ?
                """,
                (session_id, user_id),
            ).fetchone()
        return int(row["c"] if row is not None else 0)

    def get_last_activity(self, user_id: str) -> float | None:
        """Timestamp of the user's most recent logged turn, or None if unseen.

        Used by the sleep scheduler's idle gate: a user is only consolidated
        once they have been inactive for at least the idle threshold.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT MAX(ts) AS m FROM conversation_log WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if row is None or row["m"] is None:
            return None
        return float(row["m"])

    def list_users_with_pending(self) -> list[str]:
        """Distinct user_ids that have at least one unconsolidated summary."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT user_id FROM summaries WHERE consolidated = 0 ORDER BY user_id"
            ).fetchall()
        return [str(r["user_id"]) for r in rows]

    def _is_sensitive(self, text: str) -> bool:
        t = text.lower()
        for p in self._SENSITIVE_SECRET_PATTERNS + self._SENSITIVE_PHONE_PATTERNS + self._SENSITIVE_ADDRESS_PATTERNS:
            if re.search(p, t, re.IGNORECASE):
                return True
        return False

    def _requires_confirmation(self, message: str, key: str) -> bool:
        text = message.lower()
        if key != "location" and "location" not in key:
            return False
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in self._CONFIRM_FIRST_PATTERNS)

    def _parse_memory_command(self, message: str) -> MemoryCommand | None:
        m = message.strip()
        lower = m.lower()
        if re.fullmatch(r"(do\s+not\s+remember\s+this|don't\s+remember\s+this)", lower):
            return MemoryCommand(action="skip")
        if re.fullmatch(r"(save\s+this|remember\s+this)", lower):
            return MemoryCommand(action="save_previous")
        remember_match = re.fullmatch(r"remember\s+(.+)", m, flags=re.IGNORECASE)
        if remember_match and remember_match.group(1).strip().lower() != "this":
            return MemoryCommand(action="remember_payload", query=remember_match.group(1).strip())
        forget_match = re.fullmatch(r"forget\s+(.+)", m, flags=re.IGNORECASE)
        if forget_match:
            return MemoryCommand(action="forget", query=forget_match.group(1).strip())
        return None

    def _extract_candidates(self, message: str, source: str) -> list[MemoryCandidate]:
        """Extract memory candidates using regex pattern matching (fallback for direct ingestion).

        Phase 12b: This regex extractor is retained for direct user message ingestion via
        ingest_user_message() to maintain a fast, lightweight path for explicit memory
        commands ("Remember that..." hints). The consolidation worker uses LLM-based
        extraction (_llm_extract_candidates) for richer candidate discovery from episodic
        summaries.

        Regex patterns handle:
        - Explicit preferences: "my favorite X is Y", "I prefer X", "default Y is Z"
        - Explicit facts: "my name is X", "I live in Y", "I work on Z", "I lived in W"
        - Explicit memory hints: "remember X: Y" (parsed as-is for high-value facts)

        This path is intentionally simple and fast (no LLM call). For semantic extraction
        from episodic text, use _llm_extract_candidates() instead.
        """
        m = message.strip()
        lower = m.lower()
        out: list[MemoryCandidate] = []

        # Preferences
        pref_rules = [
            (r"\bmy favorite ([a-z ]{2,30}) is ([^.!?]{1,80})", "favorite_{0}"),
            (r"\bi prefer ([^.!?]{1,80})", "preference"),
            (r"\bdefault ([a-z ]{2,30}) is ([^.!?]{1,80})", "default_{0}"),
        ]
        for pattern, key_tpl in pref_rules:
            match = re.search(pattern, m, re.IGNORECASE)
            if not match:
                continue
            if len(match.groups()) == 2:
                left = re.sub(r"\s+", "_", match.group(1).strip().lower())
                key = key_tpl.format(left)
                value = match.group(2).strip()
            else:
                key = key_tpl
                value = match.group(1).strip()
            out.append(MemoryCandidate(table="preferences", key=key, value=value, source=source))

        # Facts
        fact_rules = [
            (r"\bmy name is ([^.!?]{1,80})", "name"),
            (r"\bi live in ([^.!?]{1,80})", "location"),
            (r"\bmy location is ([^.!?]{1,80})", "location"),
            (r"\bi lived in ([^.!?]{1,140})", "location_history"),
            (r"\bi work on ([^.!?]{1,120})", "work_context"),
        ]
        for pattern, key in fact_rules:
            match = re.search(pattern, m, re.IGNORECASE)
            if match:
                out.append(MemoryCandidate(table="facts", key=key, value=match.group(1).strip(), source=source))

        # Explicit memory hint for high-value facts.
        if "remember" in lower and len(m) <= 320 and ":" in m:
            key_part, value = m.split(":", 1)
            raw_key = key_part.strip().lower()[:48]
            # If key is just "remember"/"remembered"/"note", extract the real key from the value.
            # e.g. "Remember: my location is Stockholm" -> key="location", value="Stockholm"
            # e.g. "Remember: my favorite food is ramen" -> key="favorite_food", value="ramen"
            if raw_key in ("remember", "remembered", "note", "remember that"):
                # Try to extract from the value using the same patterns.
                value = value.strip()
                for pattern, key in [
                    (r"\bmy name is ([^.!?]{1,80})", "name"),
                    (r"\bi live in ([^.!?]{1,80})", "location"),
                    (r"\bmy location is ([^.!?]{1,80})", "location"),
                    (r"\bmy favorite ([a-z ]{2,30}) is ([^.!?]{1,80})", None),
                    (r"\bi prefer ([^.!?]{1,80})", None),
                ]:
                    match = re.search(pattern, value, re.IGNORECASE)
                    if match:
                        if match.lastindex == 2:
                            left = re.sub(r"\s+", "_", match.group(1).strip().lower())
                            raw_key = f"favorite_{left}"
                            value = match.group(2).strip()
                        else:
                            raw_key = key
                            value = match.group(1).strip()
                        break
                else:
                    # Fallback: use the whole value as a note.
                    raw_key = "note"
            else:
                # Normalize common key patterns.
                raw_key = re.sub(r"^(remember|remembered|note|remember that)\s*", "", raw_key)
                raw_key = re.sub(r"^my\s+", "", raw_key)
                raw_key = raw_key.strip()
                if raw_key in ("location", "location is", "where i live"):
                    raw_key = "location"
                elif raw_key in ("favorite food", "favorite food is"):
                    raw_key = "favorite_food"
                elif raw_key in ("name", "name is"):
                    raw_key = "name"
            out.append(MemoryCandidate(table="facts", key=raw_key, value=value.strip()[:240], source=source))

        # Deduplicate by table/key/value.
        unique: dict[tuple[str, str, str], MemoryCandidate] = {}
        for c in out:
            unique[(c.table, c.key, c.value)] = c
        return list(unique.values())

    async def _llm_extract_memory(self, text: str, source: str) -> MemoryExtraction:
        """Extract semantic memory (facts, preferences, triples) via OpenAI-compatible /v1/chat/completions.

        Runs a single structured extraction call and returns a MemoryExtraction.
        Gracefully returns an empty MemoryExtraction on endpoint unreachability,
        network error, or JSON parse failure so consolidation never crashes.
        """
        if not text or not text.strip():
            return MemoryExtraction(candidates=[], triples=[])

        # Truncate very long summaries to reduce token cost (last 1500 chars typically
        # contain the most recent, highest-value facts).
        text = text.strip()[-1500:] if len(text.strip()) > 1500 else text.strip()

        openai_url = (os.getenv("OPENAI_EMBED_BASE_URL") or os.getenv("OPENAI_BASE_URL", "http://localhost:10001/v1")).rstrip("/")
        chat_model = (
            os.getenv("OPENAI_CHAT_MODEL")
            or os.getenv("MODEL_LOCAL")
            or "llama3.2"
        )

        payload = {
            "model": chat_model,
            "messages": [
                {"role": "system", "content": MEMORY_EXTRACTOR_SYSTEM},
                {"role": "user", "content": text},
            ],
            "stream": False,
            "response_format": {"type": "json_object"},
            "max_tokens": 2048,
            "temperature": 0.1,
        }

        try:
            timeout = 120.0  # gemma4:e4b needs 50-70s with full system prompt
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(f"{openai_url}/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
                raw_response = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except httpx.ConnectError as e:
            log.warning(
                "memory.llm_extract | extraction_failed=endpoint_unreachable error=%s",
                str(e),
            )
            return MemoryExtraction(candidates=[], triples=[])
        except (httpx.TimeoutException, httpx.RequestError) as e:
            log.warning(
                "memory.llm_extract | extraction_failed=network_error error=%s",
                str(e),
            )
            return MemoryExtraction(candidates=[], triples=[])
        except Exception as e:
            log.error(
                "memory.llm_extract | extraction_failed=unexpected error=%s",
                str(e),
            )
            return MemoryExtraction(candidates=[], triples=[])

        try:
            parsed = json.loads(raw_response)
            if not isinstance(parsed, dict):
                raise ValueError("expected a JSON object")
        except (json.JSONDecodeError, ValueError) as e:
            log.warning(
                "memory.llm_extract | extraction_failed=json_parse_error raw=%s error=%s",
                raw_response[:200],
                str(e),
            )
            return MemoryExtraction(candidates=[], triples=[])

        return self._parse_extraction(parsed, source)

    async def _llm_extract_candidates(self, text: str, source: str) -> list[MemoryCandidate]:
        """Backwards-compatible key/value extractor; delegates to the unified pass."""
        return (await self._llm_extract_memory(text, source)).candidates

    def _candidate_from_dict(self, item: Any, table: str, source: str) -> MemoryCandidate | None:
        if not isinstance(item, dict):
            return None
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < 0.7:
            return None
        key = str(item.get("key", "")).strip()
        value = str(item.get("value", "")).strip()
        if not key or not value:
            return None
        ttl = item.get("ttl_days")
        ttl_days: int | None = None
        if isinstance(ttl, (int, float)) and not isinstance(ttl, bool) and ttl > 0:
            ttl_days = int(ttl)
        return MemoryCandidate(
            table=table,
            key=key[:48],
            value=value[:240],
            source=source,
            ttl_days=ttl_days,
        )

    def _triple_from_dict(self, item: Any) -> MemoryTriple | None:
        if not isinstance(item, dict):
            return None
        try:
            confidence = float(item.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < 0.7:
            return None
        subject = str(item.get("subject", "")).strip()
        predicate = str(item.get("predicate", "")).strip()
        obj = str(item.get("object", "")).strip()
        if not subject or not predicate or not obj:
            return None
        return MemoryTriple(
            subject=subject[:64],
            predicate=predicate[:64],
            obj=obj[:120],
            confidence=confidence,
        )

    def _parse_extraction(self, parsed: dict[str, Any], source: str) -> MemoryExtraction:
        """Parse a unified extraction payload into a MemoryExtraction.

        Tolerates the legacy {"candidates": [...]} shape so that existing
        callers and tests using the old schema keep working: a "type" of
        "preference" maps to the preferences table, everything else to facts.
        """
        has_new_schema = any(k in parsed for k in ("facts", "preferences", "triples"))
        candidates: list[MemoryCandidate] = []
        triples: list[MemoryTriple] = []

        if not has_new_schema:
            for item in parsed.get("candidates", []):
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get("type", "fact")).lower()
                table = "preferences" if item_type == "preference" else "facts"
                cand = self._candidate_from_dict(item, table, source)
                if cand:
                    candidates.append(cand)
        else:
            for item in parsed.get("facts", []) or []:
                cand = self._candidate_from_dict(item, "facts", source)
                if cand:
                    candidates.append(cand)
            for item in parsed.get("preferences", []) or []:
                cand = self._candidate_from_dict(item, "preferences", source)
                if cand:
                    candidates.append(cand)
            for item in parsed.get("triples", []) or []:
                triple = self._triple_from_dict(item)
                if triple:
                    triples.append(triple)

        log.debug(
            "memory.llm_extract | candidates=%d triples=%d",
            len(candidates),
            len(triples),
        )
        return MemoryExtraction(candidates=candidates, triples=triples)

    def _extract_candidates_llm_sync(self, text: str, source: str) -> list[MemoryCandidate]:
        """Synchronous wrapper for _llm_extract_candidates().

        Runs the async LLM extraction in the current event loop (or creates one).
        Used by consolidate_pending() which runs in asyncio.to_thread().

        Args:
            text: Episodic summary to extract from.
            source: Origin label for audit trail.

        Returns:
            List of MemoryCandidate objects, or empty list on error.
        """
        # This wrapper is always invoked from a worker thread (e.g. via
        # asyncio.to_thread in consolidate_pending), so there is no running event
        # loop here. Run the coroutine on a fresh loop owned by this thread.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._llm_extract_candidates(text, source))
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def _extract_memory_sync(self, text: str, source: str) -> MemoryExtraction:
        """Synchronous wrapper for _llm_extract_memory().

        Invoked from a worker thread (asyncio.to_thread) where no event loop is
        running, so run the coroutine on a fresh loop owned by this thread.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._llm_extract_memory(text, source))
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def _upsert_chroma(self, memory_id: str, text: str, metadata: dict[str, Any]) -> None:
        """Index one memory row.  Caller holds _lock.

        Fails closed: if the embedder is down the row stays SQLite-only and is
        indexed by the next successful reconcile.
        """
        try:
            self._collection.upsert(ids=[memory_id], documents=[text], metadatas=[metadata])
        except Exception as exc:
            log.warning(
                "memory.index_deferred | id=%s error=%s", memory_id, type(exc).__name__
            )
            self._index_stale = True
            return
        if self._index_stale:
            try:
                self._index_missing_rows()
                self._index_stale = False
            except Exception as exc:
                log.warning("memory.index_retry_failed | error=%s", type(exc).__name__)

    def _forget_by_query(self, user_id: str, query: str) -> int:
        q = query.strip().lower()
        if not q:
            return 0

        ids: list[str] = []
        with self._lock:
            cur = self._conn.cursor()
            rows = cur.execute(
                """
                SELECT id, 'facts' AS table_name FROM facts
                WHERE user_id = ? AND (lower(key) LIKE ? OR lower(value) LIKE ?)
                UNION ALL
                SELECT id, 'preferences' AS table_name FROM preferences
                WHERE user_id = ? AND (lower(key) LIKE ? OR lower(value) LIKE ?)
                """,
                (user_id, f"%{q}%", f"%{q}%", user_id, f"%{q}%", f"%{q}%"),
            ).fetchall()  # nosec B608 - query uses bound parameters; wildcard pattern is parameterized.

            for row in rows:
                table = row["table_name"]
                row_id = int(row["id"])
                if table == "facts":
                    cur.execute("DELETE FROM facts WHERE id = ? AND user_id = ?", (row_id, user_id))
                    ids.append(f"{table}:{row_id}")
                elif table == "preferences":
                    cur.execute("DELETE FROM preferences WHERE id = ? AND user_id = ?", (row_id, user_id))
                    ids.append(f"{table}:{row_id}")
            self._conn.commit()

        if ids:
            try:
                self._collection.delete(ids=ids)
            except Exception:
                pass
        return len(rows)

    def ingest_user_message(self, user_id: str, message: str, source: str = "chat", previous_user_message: str | None = None) -> dict[str, Any]:
        command = self._parse_memory_command(message)

        if command and command.action == "skip":
            return {
                "status": "do-not-remember",
                "saved": [],
                "blocked": [],
                "needs_confirmation": [],
                "candidates": 0,
                "explicit": True,
            }

        if command and command.action == "forget":
            deleted = self._forget_by_query(user_id, command.query or "")
            return {
                "status": "forgot",
                "deleted": deleted,
                "saved": [],
                "blocked": [],
                "needs_confirmation": [],
                "candidates": 0,
                "explicit": True,
            }

        if command and command.action == "remember_payload":
            source_message = command.query or ""
            explicit_requested = True
        else:
            explicit_requested = bool(command and command.action == "save_previous")
            source_message = previous_user_message.strip() if explicit_requested and previous_user_message else message

        if explicit_requested and not source_message:
            return {
                "status": "no-target",
                "saved": [],
                "blocked": [],
                "needs_confirmation": [],
                "candidates": 0,
                "explicit": True,
            }

        candidates = self._extract_candidates(source_message, source)
        if explicit_requested and not candidates:
            candidates.append(
                MemoryCandidate(
                    table="facts",
                    key="note",
                    value=source_message[:240],
                    source=source,
                )
            )

        saved: list[str] = []
        blocked: list[str] = []
        needs_confirmation: list[str] = []

        with self._lock:
            cur = self._conn.cursor()
            now = time.time()
            for c in candidates:
                fact_text = f"{c.key}: {c.value}"
                if self._is_sensitive(fact_text):
                    blocked.append(fact_text)
                    continue

                if self._requires_confirmation(source_message, c.key) and not explicit_requested:
                    needs_confirmation.append(fact_text)
                    continue

                if c.table == "preferences":
                    cur.execute(
                        """
                        INSERT INTO preferences (user_id, key, value, updated_at, sensitive)
                        VALUES (?, ?, ?, ?, 0)
                        ON CONFLICT(user_id, key) DO UPDATE
                            SET value = excluded.value, updated_at = excluded.updated_at
                        """,
                        (user_id, c.key, c.value, now),
                    )
                    # lastrowid is stale when ON CONFLICT takes the UPDATE branch.
                    pref_row = cur.execute(
                        "SELECT id FROM preferences WHERE user_id = ? AND key = ? LIMIT 1",
                        (user_id, c.key),
                    ).fetchone()
                    row_id = int(pref_row["id"])
                    memory_id = f"preferences:{row_id}"
                else:
                    cur.execute(
                        """
                        INSERT INTO facts (user_id, key, value, source, created_at, expires_at, sensitive)
                        VALUES (?, ?, ?, ?, ?, NULL, 0)
                        ON CONFLICT(user_id, key)
                            DO UPDATE SET
                                value = excluded.value,
                                source = excluded.source,
                                created_at = excluded.created_at
                        """,
                        (user_id, c.key, c.value, c.source, now),
                    )
                    fact_row = cur.execute(
                        "SELECT id FROM facts WHERE user_id = ? AND key = ? LIMIT 1",
                        (user_id, c.key),
                    ).fetchone()
                    row_id = int(fact_row["id"])
                    memory_id = f"facts:{row_id}"

                saved.append(memory_id)
                self._upsert_chroma(
                    memory_id,
                    source_message,
                    {
                        "table": c.table,
                        "key": c.key,
                        "value": c.value,
                        "source": c.source,
                        "user_id": user_id,
                        "created_at": now,
                        "consent_status": "explicit" if explicit_requested else "implicit",
                    },
                )

            self._conn.commit()

        status = "none"
        if saved:
            status = "saved"
        elif blocked and not needs_confirmation:
            status = "blocked-sensitive"
        elif needs_confirmation and not blocked:
            status = "needs-confirmation"
        elif blocked and needs_confirmation:
            status = "mixed-blocked-confirm"

        return {
            "status": status,
            "saved": saved,
            "blocked": blocked,
            "needs_confirmation": needs_confirmation,
            "candidates": len(candidates),
            "explicit": explicit_requested,
            "source_message": source_message,
        }

    def get_preference(self, user_id: str, key: str) -> str | None:
        """Return the stored preference value for *key* scoped to *user_id*, or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM preferences WHERE user_id = ? AND key = ? LIMIT 1",
                (user_id, key),
            ).fetchone()
        return str(row["value"]) if row else None

    def set_preference(self, user_id: str, key: str, value: str) -> None:
        """Upsert a preference by (user_id, key).  Overwrites any existing value."""
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO preferences (user_id, key, value, updated_at, sensitive)
                VALUES (?, ?, ?, ?, 0)
                ON CONFLICT(user_id, key)
                    DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (user_id, key, value, now),
            )
            self._conn.commit()

    def save_summary(
        self,
        user_id: str,
        session_id: str,
        summary: str,
    ) -> int:
        """Persist an episodic session summary.  Returns the new row id.

        The ``consolidated`` flag is left at 0 (False).  Phase 12's consolidation
        process will set it to 1 once the summary has been promoted to long-term
        semantic memory (SQLite facts + ChromaDB conversation_memory).
        """
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO summaries (user_id, session_id, summary, created_at, consolidated)
                VALUES (?, ?, ?, ?, 0)
                """,
                (user_id, session_id, summary, now),
            )
            self._conn.commit()
            return int(cur.lastrowid)  # type: ignore[arg-type]

    def _tier_for_table(self, table: str) -> str:
        if table in {"facts", "preferences"}:
            return "semantic"
        if table == "summaries":
            return "episodic"
        return "working"

    def list_items(self, user_id: str, limit: int = 200, offset: int = 0) -> dict[str, Any]:
        with self._lock:
            cur = self._conn.cursor()
            rows = cur.execute(
                """
                SELECT id, 'facts' AS table_name, key, value, source, created_at AS ts, 1 AS consolidated
                FROM facts WHERE user_id = ?
                UNION ALL
                SELECT id, 'preferences' AS table_name, key, value, '' AS source, updated_at AS ts, 1 AS consolidated
                FROM preferences WHERE user_id = ?
                UNION ALL
                SELECT id, 'summaries' AS table_name, session_id AS key, summary AS value, '' AS source, created_at AS ts, consolidated
                FROM summaries WHERE user_id = ?
                ORDER BY ts DESC
                LIMIT ? OFFSET ?
                """,
                (user_id, user_id, user_id, limit, offset),
            ).fetchall()  # nosec B608 - static SQL with bound parameters only.

            total = cur.execute(
                """
                SELECT (
                    (SELECT COUNT(*) FROM facts WHERE user_id = ?) +
                    (SELECT COUNT(*) FROM preferences WHERE user_id = ?) +
                    (SELECT COUNT(*) FROM summaries WHERE user_id = ?)
                ) AS total
                """,
                (user_id, user_id, user_id),
            ).fetchone()["total"]

        items = [
            {
                "id": f"{r['table_name']}:{r['id']}",
                "table": r["table_name"],
                "tier": self._tier_for_table(r["table_name"]),
                "key": r["key"],
                "value": r["value"],
                "source": r["source"],
                "consolidated": bool(r["consolidated"]),
                "ts": r["ts"],
            }
            for r in rows
        ]
        return {"items": items, "total": int(total), "limit": limit, "offset": offset}

    def list_episodic(
        self,
        user_id: str,
        limit: int = 200,
        offset: int = 0,
        consolidated: bool | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            cur = self._conn.cursor()
            if consolidated is None:
                rows = cur.execute(
                    """
                    SELECT id, session_id, summary, created_at, consolidated
                    FROM summaries
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (user_id, limit, offset),
                ).fetchall()

                total = cur.execute(
                    """
                    SELECT COUNT(*) AS total
                    FROM summaries
                    WHERE user_id = ?
                    """,
                    (user_id,),
                ).fetchone()["total"]
            else:
                consolidated_flag = 1 if consolidated else 0
                rows = cur.execute(
                    """
                    SELECT id, session_id, summary, created_at, consolidated
                    FROM summaries
                    WHERE user_id = ? AND consolidated = ?
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (user_id, consolidated_flag, limit, offset),
                ).fetchall()

                total = cur.execute(
                    """
                    SELECT COUNT(*) AS total
                    FROM summaries
                    WHERE user_id = ? AND consolidated = ?
                    """,
                    (user_id, consolidated_flag),
                ).fetchone()["total"]

        items = [
            {
                "id": f"summaries:{r['id']}",
                "table": "summaries",
                "tier": "episodic",
                "key": r["session_id"],
                "value": r["summary"],
                "source": "",
                "consolidated": bool(r["consolidated"]),
                "ts": r["created_at"],
            }
            for r in rows
        ]
        return {"items": items, "total": int(total), "limit": limit, "offset": offset}

    def run_sleep_pass(self, user_id: str, limit: int = 50) -> dict[str, int]:
        """Run a consolidated "sleep" pass for one user.

        Promotes pending episodic summaries into semantic memory — facts,
        preferences, and knowledge-graph triples — then prunes expired facts.
        The blocking LLM extraction runs without holding the store lock; the
        single-flight _consolidation_lock keeps concurrent passes from
        double-processing the same summaries.
        """
        if not self._consolidation_lock.acquire(blocking=False):
            return {"processed": 0, "promoted": 0, "blocked": 0, "triples": 0, "decayed": 0}

        now = time.time()
        try:
            # Phase 1: read pending summaries under the lock, then release it so the
            # blocking LLM extraction does not stall other memory operations.
            with self._lock:
                rows = self._conn.execute(
                    """
                    SELECT id, user_id, session_id, summary
                    FROM summaries
                    WHERE user_id = ? AND consolidated = 0
                    ORDER BY created_at ASC
                    LIMIT ?
                    """,
                    (user_id, limit),
                ).fetchall()

            # Phase 2: run the (blocking) LLM extraction for every summary WITHOUT
            # holding the lock.
            extracted: list[tuple[int, str, str, MemoryExtraction]] = []
            for row in rows:
                summary_id = int(row["id"])
                summary_text = str(row["summary"] or "")
                summary_user_id = str(row["user_id"])
                extraction = self._extract_memory_sync(summary_text, source="consolidation")
                extracted.append((summary_id, summary_user_id, summary_text, extraction))

            # Phase 3: apply all DB + Chroma + graph writes under the lock (no LLM).
            processed = 0
            promoted = 0
            blocked = 0
            triples_stored = 0
            with self._lock:
                cur = self._conn.cursor()
                for summary_id, summary_user_id, summary_text, extraction in extracted:
                    for c in extraction.candidates:
                        fact_text = f"{c.key}: {c.value}"
                        if self._is_sensitive(fact_text):
                            blocked += 1
                            continue

                        if c.table == "preferences":
                            cur.execute(
                                """
                                INSERT INTO preferences (user_id, key, value, updated_at, sensitive)
                                VALUES (?, ?, ?, ?, 0)
                                ON CONFLICT(user_id, key)
                                    DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                                """,
                                (summary_user_id, c.key, c.value, now),
                            )
                            pref_row = cur.execute(
                                "SELECT id FROM preferences WHERE user_id = ? AND key = ? LIMIT 1",
                                (summary_user_id, c.key),
                            ).fetchone()
                            memory_id = f"preferences:{int(pref_row['id'])}"
                        else:
                            expires_at = now + (c.ttl_days * 86400) if c.ttl_days else None
                            cur.execute(
                                """
                                INSERT INTO facts (user_id, key, value, source, created_at, expires_at, sensitive)
                                VALUES (?, ?, ?, ?, ?, ?, 0)
                                ON CONFLICT(user_id, key)
                                    DO UPDATE SET
                                        value = excluded.value,
                                        source = excluded.source,
                                        created_at = excluded.created_at,
                                        expires_at = excluded.expires_at
                                """,
                                (summary_user_id, c.key, c.value, c.source, now, expires_at),
                            )
                            fact_row = cur.execute(
                                "SELECT id FROM facts WHERE user_id = ? AND key = ? LIMIT 1",
                                (summary_user_id, c.key),
                            ).fetchone()
                            memory_id = f"facts:{int(fact_row['id'])}"

                        self._upsert_chroma(
                            memory_id,
                            summary_text,
                            {
                                "table": c.table,
                                "key": c.key,
                                "value": c.value,
                                "source": "consolidation",
                                "user_id": summary_user_id,
                                "created_at": now,
                                "consent_status": "consolidated",
                                "from_summary_id": summary_id,
                            },
                        )
                        promoted += 1

                    for t in extraction.triples:
                        if self._apply_triple(cur, summary_user_id, t, now):
                            triples_stored += 1

                    cur.execute("UPDATE summaries SET consolidated = 1 WHERE id = ?", (summary_id,))
                    processed += 1

                decayed = self._decay_user(cur, user_id, now)
                self._conn.commit()

            return {
                "processed": int(processed),
                "promoted": int(promoted),
                "blocked": int(blocked),
                "triples": int(triples_stored),
                "decayed": int(decayed),
            }
        finally:
            self._consolidation_lock.release()

    def _apply_triple(self, cur: sqlite3.Cursor, user_id: str, t: MemoryTriple, now: float) -> bool:
        """Upsert a knowledge-graph triple (plus its endpoint entities) for a user."""
        for ent_id in (t.subject, t.obj):
            if ":" in ent_id:
                ent_type, _, slug = ent_id.partition(":")
                if slug:
                    cur.execute(
                        "INSERT OR IGNORE INTO entities (user_id, id, type, attributes, created_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (user_id, ent_id, ent_type or "thing", json.dumps({"name": slug}), now),
                    )
        cur.execute(
            """
            INSERT INTO relations (user_id, subject, predicate, object, confidence, source, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, subject, predicate, object)
                DO UPDATE SET confidence = excluded.confidence, updated_at = excluded.updated_at
            """,
            (user_id, t.subject, t.predicate, t.obj, t.confidence, "consolidation", now, now),
        )
        return True

    def _decay_user(self, cur: sqlite3.Cursor, user_id: str, now: float) -> int:
        """Prune facts whose expires_at has passed (and drop their vectors).

        Returns the number of facts removed. Caller holds _lock and commits.
        """
        rows = cur.execute(
            "SELECT id FROM facts WHERE user_id = ? AND expires_at IS NOT NULL AND expires_at < ?",
            (user_id, now),
        ).fetchall()
        chroma_ids: list[str] = []
        for r in rows:
            row_id = int(r["id"])
            cur.execute("DELETE FROM facts WHERE id = ? AND user_id = ?", (row_id, user_id))
            chroma_ids.append(f"facts:{row_id}")
        if chroma_ids:
            try:
                self._collection.delete(ids=chroma_ids)
            except Exception:
                pass
        return len(chroma_ids)

    def consolidate_pending(self, user_id: str | None = None, limit: int = 50) -> dict[str, int]:
        """Backwards-compatible consolidation entrypoint.

        Forwards to run_sleep_pass (which now also stores knowledge-graph
        triples and prunes expired facts). With user_id=None it runs a pass for
        every user that has pending summaries.
        """
        if user_id is not None:
            return self.run_sleep_pass(user_id, limit)
        total = {"processed": 0, "promoted": 0, "blocked": 0, "triples": 0, "decayed": 0}
        for uid in self.list_users_with_pending():
            result = self.run_sleep_pass(uid, limit)
            for key in total:
                total[key] += int(result.get(key, 0))
        return total

    def delete_item(self, user_id: str, memory_id: str) -> bool:
        if ":" not in memory_id:
            return False
        table, raw_id = memory_id.split(":", 1)
        if table not in {"facts", "preferences", "summaries"}:
            return False
        if not raw_id.isdigit():
            return False

        deleted = False
        with self._lock:
            cur = self._conn.cursor()
            # user_id guard prevents cross-user deletion.
            if table == "facts":
                cur.execute("DELETE FROM facts WHERE id = ? AND user_id = ?", (int(raw_id), user_id))
            elif table == "preferences":
                cur.execute("DELETE FROM preferences WHERE id = ? AND user_id = ?", (int(raw_id), user_id))
            else:
                cur.execute("DELETE FROM summaries WHERE id = ? AND user_id = ?", (int(raw_id), user_id))
            deleted = cur.rowcount > 0
            self._conn.commit()

        if deleted and table in {"facts", "preferences"}:
            try:
                self._collection.delete(ids=[memory_id])
            except Exception:
                pass
        return deleted

    def clear_all(self, user_id: str) -> dict[str, int]:
        """Delete all memory for *user_id* only."""
        with self._lock:
            cur = self._conn.cursor()
            counts = {
                "facts": cur.execute("SELECT COUNT(*) AS c FROM facts WHERE user_id = ?", (user_id,)).fetchone()["c"],
                "preferences": cur.execute("SELECT COUNT(*) AS c FROM preferences WHERE user_id = ?", (user_id,)).fetchone()["c"],
                "summaries": cur.execute("SELECT COUNT(*) AS c FROM summaries WHERE user_id = ?", (user_id,)).fetchone()["c"],
                "entities": cur.execute("SELECT COUNT(*) AS c FROM entities WHERE user_id = ?", (user_id,)).fetchone()["c"],
                "relations": cur.execute("SELECT COUNT(*) AS c FROM relations WHERE user_id = ?", (user_id,)).fetchone()["c"],
            }
            cur.execute("DELETE FROM facts WHERE user_id = ?", (user_id,))
            cur.execute("DELETE FROM preferences WHERE user_id = ?", (user_id,))
            cur.execute("DELETE FROM summaries WHERE user_id = ?", (user_id,))
            cur.execute("DELETE FROM entities WHERE user_id = ?", (user_id,))
            cur.execute("DELETE FROM relations WHERE user_id = ?", (user_id,))
            self._conn.commit()

        # Remove this user's vectors from Chroma (post-filter by metadata).
        try:
            self._collection.delete(where={"user_id": user_id})
        except Exception:
            pass
        return {k: int(v) for k, v in counts.items()}

    def _query_terms(self, query: str) -> list[str]:
        return [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2][:10]

    def _token_overlap(self, query_terms: list[str], text: str) -> int:
        if not query_terms:
            return 0
        haystack = text.lower()
        return sum(1 for t in query_terms if t in haystack)

    def _keyword_rank(self, query: str, items: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
        terms = self._query_terms(query)
        if not terms:
            return []

        ranked: list[dict[str, Any]] = []
        for item in items:
            key = str(item.get("key", "")).lower()
            value = str(item.get("value", ""))
            text = f"{key} {value}".lower()
            overlap = self._token_overlap(terms, text)
            if overlap <= 0:
                continue

            score = overlap / float(len(terms))
            if any(t == key for t in terms):
                score += 0.15
            score = min(1.0, score)

            ranked.append(
                {
                    "id": item["id"],
                    "table": item.get("table", ""),
                    "tier": item.get("tier", "semantic"),
                    "key": item.get("key", ""),
                    "value": item.get("value", ""),
                    "text": f"{item.get('key', '')}: {item.get('value', '')}",
                    "score": float(score),
                    "source": "sqlite",
                }
            )

        ranked.sort(key=lambda r: r["score"], reverse=True)
        return ranked[:top_n]

    def _graph_recall(
        self,
        user_id: str,
        terms: list[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Knowledge-graph recall: match the user's relations against query terms.

        Returns scored triple hits (source="graph", type="triple"). Scoped to
        the user so relations never leak across accounts.
        """
        if not terms:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT subject, predicate, object, confidence FROM relations WHERE user_id = ?",
                (user_id,),
            ).fetchall()

        hits: list[dict[str, Any]] = []
        for r in rows:
            subject = str(r["subject"])
            predicate = str(r["predicate"])
            obj = str(r["object"])
            text = f"{subject} {predicate} {obj}".lower()
            overlap = self._token_overlap(terms, text)
            if overlap <= 0:
                continue
            try:
                confidence = float(r["confidence"])
            except (TypeError, ValueError):
                confidence = 1.0
            score = min(1.0, (overlap / float(len(terms))) * confidence)
            hits.append(
                {
                    "id": f"relation:{subject}|{predicate}|{obj}",
                    "table": "relations",
                    "tier": "semantic",
                    "key": predicate,
                    "value": f"{subject} {predicate} {obj}",
                    "text": f"{subject} {predicate} {obj}",
                    "score": float(score),
                    "source": "graph",
                    "type": "triple",
                }
            )

        hits.sort(key=lambda h: float(h["score"]), reverse=True)
        return hits[:limit]

    def retrieve(
        self,
        user_id: str,
        query: str,
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        raw_query = query.strip()
        if len(raw_query) < 2:
            return []

        limit = top_n or self.top_n
        query_terms = self._query_terms(raw_query)
        listed_all = self.list_items(user_id, limit=300, offset=0)["items"]

        semantic_items = [
            item
            for item in listed_all
            if item.get("table") in {"facts", "preferences"}
        ]
        episodic_items = [
            item
            for item in listed_all
            if item.get("table") == "summaries"
        ]

        sqlite_sem_hits = self._keyword_rank(raw_query, semantic_items, limit * 2)
        sqlite_epi_hits = self._keyword_rank(raw_query, episodic_items, limit)
        for hit in sqlite_epi_hits:
            hit["score"] = float(hit["score"]) * 0.85
            hit["source"] = "sqlite-episodic"

        chroma_hits: list[dict[str, Any]] = []
        try:
            result = self._collection.query(
                query_texts=[raw_query],
                n_results=limit * 2,
                where={"user_id": user_id},
            )
            ids = (result.get("ids") or [[]])[0]
            docs = (result.get("documents") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]
            metadatas = (result.get("metadatas") or [[]])[0]
            for idx, doc, dist, meta in zip(ids, docs, distances, metadatas):
                score = 1.0 / (1.0 + float(dist))
                table = str(idx).split(":", 1)[0] if ":" in str(idx) else "facts"
                # Use the current SQLite value for text (not the stale ChromaDB document)
                # so the system prompt always reflects the latest stored fact/preference.
                sqlite_item = next(
                    (item for item in semantic_items if item["id"] == idx),
                    None,
                )
                text = sqlite_item.get("value", "") if sqlite_item else doc
                chroma_hits.append(
                    {
                        "id": idx,
                        "table": table,
                        "tier": "semantic",
                        "key": str(meta.get("key", "")),
                        "value": str(meta.get("value", "")),
                        "text": text,
                        "score": score,
                        "source": "chroma",
                    }
                )
        except Exception:
            chroma_hits = []

        graph_hits = self._graph_recall(user_id, query_terms, limit)

        merged: dict[str, dict[str, Any]] = {}
        for hit in sqlite_sem_hits + sqlite_epi_hits + chroma_hits + graph_hits:
            if float(hit.get("score", 0.0)) < self.min_relevance_score:
                continue

            existing = merged.get(hit["id"])
            if not existing or float(hit["score"]) > float(existing["score"]):
                merged[hit["id"]] = hit

        merged_list = sorted(merged.values(), key=lambda h: float(h["score"]), reverse=True)[:limit]
        return merged_list
