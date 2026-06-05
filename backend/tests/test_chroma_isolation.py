"""Phase 10d — ChromaDB collection isolation tests.

Verifies:
1. MemoryStore uses the 'conversation_memory' collection (not 'assistant_memories').
2. Code context indexed into 'code_context' does not appear in chat memory retrieval.
3. Chat facts stored in 'conversation_memory' do not appear in code context queries.
4. The summaries table has the 'consolidated' column (Phase 12 prerequisite).
"""
from __future__ import annotations

import pathlib
import sqlite3
import sys

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
    """Content in a separate 'code_context' collection must not appear in store.retrieve()."""
    pytest.importorskip("chromadb", reason="chromadb not installed")

    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()

    # Create a MemoryStore backed by this chroma path.
    mem = MemoryStore(
        db_path=str(tmp_path / "memory.db"),
        chroma_path=str(chroma_dir),
    )

    # Write a distinctive symbol into a separate collection. This simulates code
    # indexing without depending on removed code-indexer modules.
    code_context = mem._chroma.get_or_create_collection(
        name="code_context",
        embedding_function=mem._embedder,
    )
    code_context.upsert(
        ids=["code:1"],
        documents=["def xyzzy_unique_sentinel_function(arg: int) -> int: return arg * 42"],
        metadatas=[{"path": "src/secret_function.py"}],
    )

    # Retrieve using the exact code symbol name — must not surface code content.
    hits = mem.retrieve("alice", "xyzzy_unique_sentinel_function")
    for hit in hits:
        assert "xyzzy_unique_sentinel_function" not in hit.get("text", ""), (
            f"Code context leaked into chat retrieval: {hit}"
        )


# ── 3. Chat facts do not bleed into code context queries ──────────────────────


def test_chat_memory_not_in_code_context(tmp_path):
    """Facts stored in 'conversation_memory' must not appear when querying 'code_context'."""
    pytest.importorskip("chromadb", reason="chromadb not installed")

    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()

    mem = MemoryStore(
        db_path=str(tmp_path / "memory.db"),
        chroma_path=str(chroma_dir),
    )

    # Plant a distinctive fact in conversation_memory.
    mem.ingest_user_message("alice", "My favourite_unique_sentinel_fact is blue cheese.")

    code_context = mem._chroma.get_or_create_collection(
        name="code_context",
        embedding_function=mem._embedder,
    )
    code_context.upsert(
        ids=["code:2"],
        documents=["def unrelated_helper(x): return x + 1"],
        metadatas=[{"path": "src/unrelated.py"}],
    )

    results = code_context.query(query_texts=["favourite_unique_sentinel_fact"], n_results=5)
    combined = "\n".join((results.get("documents") or [[]])[0])
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


