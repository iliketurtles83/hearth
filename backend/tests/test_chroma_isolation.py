"""Phase 10d — ChromaDB collection isolation tests.

Verifies:
1. MemoryStore uses the 'conversation_memory' collection (not 'assistant_memories').
2. Code context indexed into 'code_context' does not appear in chat memory retrieval.
3. Chat facts stored in 'conversation_memory' do not appear in code context queries.
4. The summaries table has the 'consolidated' column (Phase 12 prerequisite).
5. Existing 'assistant_memories' data is auto-migrated to 'conversation_memory' on init.
"""
from __future__ import annotations

import pathlib
import sqlite3
import sys
import textwrap

import pytest

# Ensure the backend package is importable when pytest is run from the repo root.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from memory import MemoryStore


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def store(tmp_path):
    return MemoryStore(
        db_path=str(tmp_path / "memory.db"),
        chroma_path=str(tmp_path / "chroma"),
    )


# ── 1. Collection name ────────────────────────────────────────────────────────


def test_conversation_memory_collection_name(store):
    """MemoryStore must use 'conversation_memory', never 'assistant_memories'."""
    assert store._collection.name == "conversation_memory"


# ── 2. Code context does not bleed into chat retrieval ────────────────────────


def test_code_context_not_in_chat_retrieval(tmp_path):
    """Content indexed into 'code_context' must not appear in store.retrieve()."""
    pytest.importorskip("chromadb", reason="chromadb not installed")

    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()

    # Index a distinctive code snippet into the code_context collection.
    from tools.code_indexer import index_workspace

    src = tmp_path / "src"
    src.mkdir()
    (src / "secret_function.py").write_text(
        textwrap.dedent("""\
            def xyzzy_unique_sentinel_function(arg: int) -> int:
                \"\"\"This name should never appear in chat retrieval.\"\"\"
                return arg * 42
        """)
    )
    indexed = index_workspace(str(tmp_path), str(chroma_dir))
    assert indexed >= 1, "Expected at least one document indexed"

    # Now create a MemoryStore backed by the same chroma path.
    mem = MemoryStore(
        db_path=str(tmp_path / "memory.db"),
        chroma_path=str(chroma_dir),
    )

    # Retrieve using the exact code symbol name — must not surface code content.
    hits = mem.retrieve("alice", "xyzzy_unique_sentinel_function")
    for hit in hits:
        assert "xyzzy_unique_sentinel_function" not in hit.get("text", ""), (
            f"Code context leaked into chat retrieval: {hit}"
        )


# ── 3. Chat facts do not bleed into code context queries ──────────────────────


def test_chat_memory_not_in_code_context(tmp_path):
    """Facts stored in 'conversation_memory' must not appear in query_code_context()."""
    pytest.importorskip("chromadb", reason="chromadb not installed")

    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()

    mem = MemoryStore(
        db_path=str(tmp_path / "memory.db"),
        chroma_path=str(chroma_dir),
    )

    # Plant a distinctive fact in conversation_memory.
    mem.ingest_user_message("alice", "My favourite_unique_sentinel_fact is blue cheese.")

    # Query the code_context collection — the chat fact must not appear.
    from tools.code_indexer import query_code_context

    results = query_code_context("favourite_unique_sentinel_fact", str(chroma_dir), n=5)
    combined = "\n".join(results)
    assert "favourite_unique_sentinel_fact" not in combined, (
        f"Chat fact leaked into code context results: {combined!r}"
    )


# ── 4. summaries table has 'consolidated' column ─────────────────────────────


def test_summaries_has_consolidated_column(tmp_path):
    """The summaries table must include the 'consolidated' column for Phase 12."""
    mem = MemoryStore(
        db_path=str(tmp_path / "memory.db"),
        chroma_path=str(tmp_path / "chroma"),
    )
    rows = mem._conn.execute("PRAGMA table_info(summaries)").fetchall()
    col_names = [r[1] for r in rows]  # PRAGMA returns (cid, name, type, notnull, dflt, pk)
    assert "consolidated" in col_names, (
        f"'consolidated' column missing from summaries table; got: {col_names}"
    )


# ── 4b. save_summary() persists a row with consolidated=0 ────────────────────


def test_save_summary_stores_row(tmp_path):
    mem = MemoryStore(
        db_path=str(tmp_path / "memory.db"),
        chroma_path=str(tmp_path / "chroma"),
    )
    row_id = mem.save_summary("alice", "sess-001", "Discussed the weather in London.")
    assert isinstance(row_id, int) and row_id > 0

    row = mem._conn.execute(
        "SELECT * FROM summaries WHERE id = ?", (row_id,)
    ).fetchone()
    assert row is not None
    assert row["session_id"] == "sess-001"
    assert row["summary"] == "Discussed the weather in London."
    assert row["consolidated"] == 0


# ── 5. Auto-migration: assistant_memories → conversation_memory ───────────────


def test_migration_from_assistant_memories(tmp_path):
    """Pre-existing 'assistant_memories' data is copied to 'conversation_memory' on init."""
    pytest.importorskip("chromadb", reason="chromadb not installed")
    import chromadb as _chromadb
    import numpy as np
    from chromadb.api.types import Documents, EmbeddingFunction

    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()

    # Create the legacy collection and insert a document directly.
    class _TrivialEmbedder(EmbeddingFunction):
        def __call__(self, input: Documents) -> list[list[float]]:
            return [[0.1] * 192 for _ in input]

    client = _chromadb.PersistentClient(path=str(chroma_dir))
    old_col = client.get_or_create_collection(
        name="assistant_memories", embedding_function=_TrivialEmbedder()
    )
    old_col.upsert(
        ids=["facts:1"],
        documents=["name: migration_test_sentinel"],
        metadatas=[{"user_id": "alice"}],
    )
    del old_col, client  # close the client so the new one can open the same path

    # Construct MemoryStore — migration must run automatically.
    mem = MemoryStore(
        db_path=str(tmp_path / "memory.db"),
        chroma_path=str(chroma_dir),
    )

    # The old collection must be gone.
    existing_names = [c.name for c in mem._chroma.list_collections()]
    assert "assistant_memories" not in existing_names, (
        f"Old 'assistant_memories' collection still exists after migration: {existing_names}"
    )
    assert "conversation_memory" in existing_names

    # The document must be retrievable from the new collection.
    result = mem._collection.get(ids=["facts:1"])
    assert result["ids"] == ["facts:1"], "Migrated document not found in conversation_memory"
    assert "migration_test_sentinel" in (result["documents"] or [""])[0]
