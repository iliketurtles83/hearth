from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from memory import MemoryStore
from tools.code_indexer import index_workspace


@pytest.fixture()
def store(tmp_path):
    return MemoryStore(
        db_path=str(tmp_path / "memory.db"),
        chroma_path=str(tmp_path / "chroma"),
    )


def test_project_id_columns_added_for_legacy_db(tmp_path):
    db_path = tmp_path / "legacy.db"
    chroma_path = tmp_path / "chroma"

    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            summary TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE TABLE conversation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            ts REAL NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()

    mem = MemoryStore(db_path=str(db_path), chroma_path=str(chroma_path))

    summary_cols = {r["name"] for r in mem._conn.execute("PRAGMA table_info(summaries)").fetchall()}
    conv_cols = {r["name"] for r in mem._conn.execute("PRAGMA table_info(conversation_log)").fetchall()}

    assert "project_id" in summary_cols
    assert "project_id" in conv_cols


def test_project_scoped_session_turns_and_lists(store):
    # Main chat (project_id NULL)
    store.log_turn("sess-main", "alice", "user", "hello main")
    # Project chat
    store.log_turn("sess-proj", "alice", "user", "hello project", project_id="proj-1")

    main_turns = store.get_session_turns("sess-main", "alice")
    proj_turns = store.get_session_turns("sess-proj", "alice", project_id="proj-1")

    assert len(main_turns) == 1
    assert main_turns[0]["content"] == "hello main"
    assert len(proj_turns) == 1
    assert proj_turns[0]["content"] == "hello project"

    # No cross-contamination between scopes.
    assert store.get_session_turns("sess-main", "alice", project_id="proj-1") == []
    assert store.get_session_turns("sess-proj", "alice") == []

    main_sessions = store.list_sessions("alice")
    proj_sessions = store.list_sessions("alice", project_id="proj-1")

    assert [s["session_id"] for s in main_sessions] == ["sess-main"]
    assert [s["session_id"] for s in proj_sessions] == ["sess-proj"]


def test_project_scoped_summaries_and_counts(store):
    store.save_summary("alice", "sess-main", "main summary")
    store.save_summary("alice", "sess-proj", "project summary", project_id="proj-1")

    assert store.get_latest_session_summary("sess-main", "alice") == "main summary"
    assert store.get_latest_session_summary("sess-proj", "alice", project_id="proj-1") == "project summary"

    # Scope isolation.
    assert store.get_latest_session_summary("sess-main", "alice", project_id="proj-1") == ""
    assert store.get_latest_session_summary("sess-proj", "alice") == ""

    assert store.count_unconsolidated("alice") == 1
    assert store.count_unconsolidated("alice", project_id="proj-1") == 1


def test_project_retrieve_uses_project_code_context_collection(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    project_dir = workspace / "proj"
    project_dir.mkdir()

    marker = "projectscope_marker_fn"
    (project_dir / "feature.py").write_text(
        f"def {marker}(x: int) -> int:\n    return x + 1\n",
        encoding="utf-8",
    )

    db_path = tmp_path / "memory.db"
    chroma_path = tmp_path / "chroma"
    chroma_path.mkdir()

    # Index only this project into its scoped collection.
    indexed = index_workspace(
        str(workspace),
        str(chroma_path),
        index_paths=["proj"],
        collection_name="code_context_proj-1",
    )
    assert indexed >= 1

    mem = MemoryStore(db_path=str(db_path), chroma_path=str(chroma_path))
    mem.ingest_user_message("alice", "remember projectscope_private_fact: secret-value")

    project_hits = mem.retrieve("alice", marker, project_id="proj-1")
    assert any(marker in str(hit.get("text", "")) for hit in project_hits)

    # In project mode, retrieval is scoped to code_context_{project_id}, not
    # conversation_memory semantic facts.
    project_fact_hits = mem.retrieve("alice", "projectscope_private_fact", project_id="proj-1")
    assert all("projectscope_private_fact" not in str(hit.get("text", "")) for hit in project_fact_hits)
