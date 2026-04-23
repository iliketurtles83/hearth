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
    def __init__(self) -> None:
        root = os.path.dirname(__file__)
        db_default = os.path.join(root, "memory.db")
        chroma_default = os.path.join(root, "chroma")
        self.db_path = os.getenv("MEMORY_DB_PATH", db_default)
        self.chroma_path = os.getenv("CHROMA_PATH", chroma_default)
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
        self._collection = self._chroma.get_or_create_collection(
            name="assistant_memories",
            embedding_function=self._embedder,
        )

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
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL,
                    sensitive INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    sensitive INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            self._conn.commit()

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

    def _forget_by_query(self, query: str) -> int:
        q = query.strip().lower()
        if not q:
            return 0

        ids: list[str] = []
        with self._lock:
            cur = self._conn.cursor()
            rows = cur.execute(
                """
                SELECT id, 'facts' AS table_name FROM facts
                WHERE lower(key) LIKE ? OR lower(value) LIKE ?
                UNION ALL
                SELECT id, 'preferences' AS table_name FROM preferences
                WHERE lower(key) LIKE ? OR lower(value) LIKE ?
                """,
                (f"%{q}%", f"%{q}%", f"%{q}%", f"%{q}%"),
            ).fetchall()

            for row in rows:
                table = row["table_name"]
                row_id = int(row["id"])
                cur.execute(f"DELETE FROM {table} WHERE id = ?", (row_id,))
                if table in {"facts", "preferences"}:
                    ids.append(f"{table}:{row_id}")
            self._conn.commit()

        if ids:
            try:
                self._collection.delete(ids=ids)
            except Exception:
                pass
        return len(rows)

    def ingest_user_message(self, message: str, source: str = "chat", previous_user_message: str | None = None) -> dict[str, Any]:
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
            deleted = self._forget_by_query(command.query or "")
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
                        INSERT INTO preferences (key, value, updated_at, sensitive)
                        VALUES (?, ?, ?, 0)
                        """,
                        (c.key, c.value, now),
                    )
                    row_id = cur.lastrowid
                    memory_id = f"preferences:{row_id}"
                else:
                    cur.execute(
                        """
                        INSERT INTO facts (key, value, source, created_at, expires_at, sensitive)
                        VALUES (?, ?, ?, ?, NULL, 0)
                        """,
                        (c.key, c.value, c.source, now),
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

    def list_items(self, limit: int = 200, offset: int = 0) -> dict[str, Any]:
        with self._lock:
            cur = self._conn.cursor()
            rows = cur.execute(
                """
                SELECT id, 'facts' AS table_name, key, value, source, created_at AS ts
                FROM facts
                UNION ALL
                SELECT id, 'preferences' AS table_name, key, value, '' AS source, updated_at AS ts
                FROM preferences
                UNION ALL
                SELECT id, 'summaries' AS table_name, session_id AS key, summary AS value, '' AS source, created_at AS ts
                FROM summaries
                ORDER BY ts DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()

            total = cur.execute(
                """
                SELECT (
                    (SELECT COUNT(*) FROM facts) +
                    (SELECT COUNT(*) FROM preferences) +
                    (SELECT COUNT(*) FROM summaries)
                ) AS total
                """
            ).fetchone()["total"]

        items = [
            {
                "id": f"{r['table_name']}:{r['id']}",
                "table": r["table_name"],
                "key": r["key"],
                "value": r["value"],
                "source": r["source"],
                "ts": r["ts"],
            }
            for r in rows
        ]
        return {"items": items, "total": int(total), "limit": limit, "offset": offset}

    def delete_item(self, memory_id: str) -> bool:
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
            cur.execute(f"DELETE FROM {table} WHERE id = ?", (int(raw_id),))
            deleted = cur.rowcount > 0
            self._conn.commit()

        if deleted and table in {"facts", "preferences"}:
            try:
                self._collection.delete(ids=[memory_id])
            except Exception:
                pass
        return deleted

    def clear_all(self) -> dict[str, int]:
        with self._lock:
            cur = self._conn.cursor()
            counts = {
                "facts": cur.execute("SELECT COUNT(*) AS c FROM facts").fetchone()["c"],
                "preferences": cur.execute("SELECT COUNT(*) AS c FROM preferences").fetchone()["c"],
                "summaries": cur.execute("SELECT COUNT(*) AS c FROM summaries").fetchone()["c"],
            }
            cur.execute("DELETE FROM facts")
            cur.execute("DELETE FROM preferences")
            cur.execute("DELETE FROM summaries")
            self._conn.commit()

        try:
            self._chroma.delete_collection("assistant_memories")
        except Exception:
            pass
        self._collection = self._chroma.get_or_create_collection(
            name="assistant_memories",
            embedding_function=self._embedder,
        )
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

            # Normalize overlap so sqlite scores can be merged with chroma scores.
            score = overlap / float(len(terms))
            if any(t == key for t in terms):
                score += 0.15
            score = min(1.0, score)

            ranked.append(
                {
                    "id": item["id"],
                    "text": f"{item['key']}: {item['value']}",
                    "score": float(score),
                    "source": "sqlite",
                }
            )

        ranked.sort(key=lambda r: r["score"], reverse=True)
        return ranked[:top_n]

    def retrieve(self, query: str, top_n: int | None = None) -> list[dict[str, Any]]:
        raw_query = query.strip()
        if len(raw_query) < 2:
            return []

        limit = top_n or self.top_n
        query_terms = self._query_terms(raw_query)
        listed = [
            item
            for item in self.list_items(limit=300, offset=0)["items"]
            if item.get("table") in {"facts", "preferences"}
        ]
        sqlite_hits = self._keyword_rank(raw_query, listed, limit * 2)

        chroma_hits: list[dict[str, Any]] = []
        try:
            result = self._collection.query(query_texts=[raw_query], n_results=limit * 2)
            ids = (result.get("ids") or [[]])[0]
            docs = (result.get("documents") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]
            for idx, doc, dist in zip(ids, docs, distances):
                score = 1.0 / (1.0 + float(dist))
                chroma_hits.append({"id": idx, "text": doc, "score": score, "source": "chroma"})
        except Exception:
            chroma_hits = []

        merged: dict[str, dict[str, Any]] = {}
        for hit in sqlite_hits + chroma_hits:
            overlap = self._token_overlap(query_terms, str(hit.get("text", "")))
            if overlap == 0 and hit.get("source") == "chroma":
                continue
            if float(hit.get("score", 0.0)) < self.min_relevance_score and overlap < 2:
                continue

            existing = merged.get(hit["id"])
            if not existing or hit["score"] > existing["score"]:
                merged[hit["id"]] = hit

        merged_list = sorted(merged.values(), key=lambda h: h["score"], reverse=True)[:limit]
        return merged_list


memory_store = MemoryStore()
