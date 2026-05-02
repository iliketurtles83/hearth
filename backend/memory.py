import os
import re
import sqlite3
import hashlib
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any

import chromadb
import numpy as np
from chromadb.api.types import Documents, EmbeddingFunction


@dataclass
class MemoryCandidate:
    table: str  # facts | preferences
    key: str
    value: str
    source: str


@dataclass
class MemoryCommand:
    action: str
    query: str | None = None


class HashEmbeddingFunction(EmbeddingFunction):
    """Deterministic lightweight embedding suitable for local semantic recall."""

    def __init__(self, dim: int = 192):
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            # Use a stable hash so embedding positions are deterministic across restarts.
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:8], byteorder="big", signed=False) % self.dim
            vec[idx] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec.tolist()

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
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

        self._embedder = HashEmbeddingFunction()
        self._chroma = chromadb.PersistentClient(path=self.chroma_path)
        self._migrate_collection_name()
        self._collection = self._chroma.get_or_create_collection(
            name="conversation_memory",
            embedding_function=self._embedder,
        )

    def _migrate_collection_name(self) -> None:
        """One-time migration: copy assistant_memories → conversation_memory then delete the old collection.

        Safe to call on every startup — becomes a no-op once migration is done.
        """
        import logging as _logging
        _log = _logging.getLogger("assistant.memory")
        try:
            existing_names = [c.name for c in self._chroma.list_collections()]
        except Exception:
            return

        has_old = "assistant_memories" in existing_names
        has_new = "conversation_memory" in existing_names

        if not has_old:
            return  # nothing to migrate

        if has_old and has_new:
            _log.warning(
                "memory: both 'assistant_memories' and 'conversation_memory' exist — "
                "using 'conversation_memory' as authoritative; 'assistant_memories' left intact"
            )
            return

        # Copy all documents from old collection into new collection.
        old_col = self._chroma.get_collection(name="assistant_memories", embedding_function=self._embedder)
        new_col = self._chroma.get_or_create_collection(name="conversation_memory", embedding_function=self._embedder)

        try:
            result = old_col.get(include=["documents", "metadatas"])
            ids = result.get("ids") or []
            if ids:
                new_col.upsert(
                    ids=ids,
                    documents=result.get("documents") or [],
                    metadatas=result.get("metadatas") or [],
                )
            self._chroma.delete_collection("assistant_memories")
            _log.info(
                "memory: migrated %d documents from 'assistant_memories' to 'conversation_memory'",
                len(ids),
            )
        except Exception as exc:
            _log.error("memory: migration failed (%s) — leaving both collections intact", exc)

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
            (r"\bi lived in ([^.!?]{1,140})", "location_history"),
            (r"\bi work on ([^.!?]{1,120})", "work_context"),
        ]
        for pattern, key in fact_rules:
            match = re.search(pattern, m, re.IGNORECASE)
            if match:
                out.append(MemoryCandidate(table="facts", key=key, value=match.group(1).strip(), source=source))

        # Explicit memory hint for high-value facts.
        if "remember" in lower and len(m) <= 320 and ":" in m:
            key, value = m.split(":", 1)
            out.append(MemoryCandidate(table="facts", key=key.strip().lower()[:48], value=value.strip()[:240], source=source))

        # Deduplicate by table/key/value.
        unique: dict[tuple[str, str, str], MemoryCandidate] = {}
        for c in out:
            unique[(c.table, c.key, c.value)] = c
        return list(unique.values())

    def _upsert_chroma(self, memory_id: str, text: str, metadata: dict[str, Any]) -> None:
        self._collection.upsert(ids=[memory_id], documents=[text], metadatas=[metadata])

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
            ).fetchall()

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
                text = f"{c.key}: {c.value}"
                if self._is_sensitive(text):
                    blocked.append(text)
                    continue

                if self._requires_confirmation(source_message, c.key) and not explicit_requested:
                    needs_confirmation.append(text)
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
                    row_id = cur.lastrowid
                    memory_id = f"preferences:{row_id}"
                else:
                    cur.execute(
                        """
                        INSERT INTO facts (user_id, key, value, source, created_at, expires_at, sensitive)
                        VALUES (?, ?, ?, ?, ?, NULL, 0)
                        """,
                        (user_id, c.key, c.value, c.source, now),
                    )
                    row_id = cur.lastrowid
                    memory_id = f"facts:{row_id}"

                saved.append(memory_id)
                self._upsert_chroma(
                    memory_id,
                    text,
                    {
                        "table": c.table,
                        "key": c.key,
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

    def save_summary(self, user_id: str, session_id: str, summary: str) -> int:
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
            ).fetchall()

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
        where_sql = "WHERE user_id = ?"
        args: list[Any] = [user_id]
        if consolidated is not None:
            where_sql += " AND consolidated = ?"
            args.append(1 if consolidated else 0)

        with self._lock:
            cur = self._conn.cursor()
            rows = cur.execute(
                f"""
                SELECT id, session_id, summary, created_at, consolidated
                FROM summaries
                {where_sql}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                tuple(args + [limit, offset]),
            ).fetchall()

            total = cur.execute(
                f"""
                SELECT COUNT(*) AS total
                FROM summaries
                {where_sql}
                """,
                tuple(args),
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

    def consolidate_pending(self, user_id: str | None = None, limit: int = 50) -> dict[str, int]:
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            if user_id:
                rows = cur.execute(
                    """
                    SELECT id, user_id, session_id, summary
                    FROM summaries
                    WHERE user_id = ? AND consolidated = 0
                    ORDER BY created_at ASC
                    LIMIT ?
                    """,
                    (user_id, limit),
                ).fetchall()
            else:
                rows = cur.execute(
                    """
                    SELECT id, user_id, session_id, summary
                    FROM summaries
                    WHERE consolidated = 0
                    ORDER BY created_at ASC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

            processed = 0
            promoted = 0
            blocked = 0
            for row in rows:
                summary_id = int(row["id"])
                summary_text = str(row["summary"] or "")
                summary_user_id = str(row["user_id"])
                candidates = self._extract_candidates(summary_text, source="consolidation")

                for c in candidates:
                    text = f"{c.key}: {c.value}"
                    if self._is_sensitive(text):
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
                        cur.execute(
                            """
                            INSERT INTO facts (user_id, key, value, source, created_at, expires_at, sensitive)
                            VALUES (?, ?, ?, ?, ?, NULL, 0)
                            """,
                            (summary_user_id, c.key, c.value, c.source, now),
                        )
                        memory_id = f"facts:{int(cur.lastrowid)}"

                    self._upsert_chroma(
                        memory_id,
                        text,
                        {
                            "table": c.table,
                            "key": c.key,
                            "source": "consolidation",
                            "user_id": summary_user_id,
                            "created_at": now,
                            "consent_status": "consolidated",
                            "from_summary_id": summary_id,
                        },
                    )
                    promoted += 1

                cur.execute("UPDATE summaries SET consolidated = 1 WHERE id = ?", (summary_id,))
                processed += 1

            self._conn.commit()

        return {
            "processed": int(processed),
            "promoted": int(promoted),
            "blocked": int(blocked),
        }

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
            }
            cur.execute("DELETE FROM facts WHERE user_id = ?", (user_id,))
            cur.execute("DELETE FROM preferences WHERE user_id = ?", (user_id,))
            cur.execute("DELETE FROM summaries WHERE user_id = ?", (user_id,))
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

    def retrieve(self, user_id: str, query: str, top_n: int | None = None) -> list[dict[str, Any]]:
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
            for idx, doc, dist in zip(ids, docs, distances):
                score = 1.0 / (1.0 + float(dist))
                table = str(idx).split(":", 1)[0] if ":" in str(idx) else "facts"
                key, _, value = str(doc).partition(":")
                chroma_hits.append(
                    {
                        "id": idx,
                        "table": table,
                        "tier": "semantic",
                        "key": key.strip(),
                        "value": value.strip(),
                        "text": doc,
                        "score": score,
                        "source": "chroma",
                    }
                )
        except Exception:
            chroma_hits = []

        merged: dict[str, dict[str, Any]] = {}
        for hit in sqlite_sem_hits + sqlite_epi_hits + chroma_hits:
            overlap = self._token_overlap(query_terms, str(hit.get("text", "")))
            if overlap == 0 and hit.get("source") == "chroma":
                continue
            if float(hit.get("score", 0.0)) < self.min_relevance_score and overlap < 2:
                continue

            existing = merged.get(hit["id"])
            if not existing or float(hit["score"]) > float(existing["score"]):
                merged[hit["id"]] = hit

        merged_list = sorted(merged.values(), key=lambda h: float(h["score"]), reverse=True)[:limit]
        return merged_list
