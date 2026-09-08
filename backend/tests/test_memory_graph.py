"""Tests for the lightweight knowledge graph + sleep-pass decay.

Covers: triple extraction -> entities/relations, user-scoped graph recall in
retrieve(), fact TTL + decay, and legacy candidates-schema tolerance.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from memory import MemoryStore  # noqa: E402


@pytest.fixture(autouse=True)
def mock_openai_embed_sync(monkeypatch):
    """Keep the embedder deterministic (no endpoint) and uniform-dimensioned."""
    monkeypatch.setattr("memory._openai_embed_sync", lambda *a, **kw: [0.0] * 768)


@pytest.fixture
def store(tmp_path):
    return MemoryStore(
        db_path=str(tmp_path / "memory.db"),
        chroma_path=str(tmp_path / "chroma"),
    )


def _patch_ollama_chat(monkeypatch, mock_response):
    from unittest.mock import AsyncMock, MagicMock

    async def mock_post_fn(*args, **kwargs):
        resp = MagicMock()
        resp.json = MagicMock(return_value=mock_response)
        resp.raise_for_status = MagicMock()
        return resp

    client = MagicMock()
    client.post = mock_post_fn
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    monkeypatch.setattr("httpx.AsyncClient", lambda *a, **kw: client)


def test_run_sleep_pass_stores_triples_and_graph_recall(store, monkeypatch):
    _patch_ollama_chat(monkeypatch, {"message": {"content": json.dumps({
        "facts": [{"key": "location", "value": "Tallinn", "confidence": 0.9}],
        "preferences": [],
        "triples": [
            {"subject": "person:alice", "predicate": "lives_in", "object": "city:tallinn", "confidence": 0.9},
            {"subject": "person:alice", "predicate": "works_on", "object": "project:hearth", "confidence": 0.9},
        ],
    })}})
    store.log_turn("sess-1", "alice", "user", "hi")
    store.save_summary("alice", "sess-1", "User lives in Tallinn and works on hearth")

    stats = store.run_sleep_pass("alice", limit=10)
    assert stats["processed"] == 1
    assert stats["promoted"] >= 1
    assert stats["triples"] == 2

    with store._lock:
        rels = store._conn.execute(
            "SELECT subject, predicate, object FROM relations WHERE user_id = ?", ("alice",)
        ).fetchall()
    assert len(rels) == 2

    hits = store.retrieve("alice", "where does alice live")
    graph_hits = [h for h in hits if h.get("source") == "graph"]
    assert graph_hits
    assert any("lives_in" in h["text"] for h in graph_hits)


def test_graph_recall_scoped_to_user(store, monkeypatch):
    _patch_ollama_chat(monkeypatch, {"message": {"content": json.dumps({
        "facts": [],
        "preferences": [],
        "triples": [
            {"subject": "person:alice", "predicate": "lives_in", "object": "city:tallinn", "confidence": 0.9},
        ],
    })}})
    store.save_summary("alice", "sess-1", "User lives in Tallinn")
    store.run_sleep_pass("alice", limit=10)

    bob_hits = store.retrieve("bob", "where does alice live")
    assert not any(h.get("source") == "graph" for h in bob_hits)


def test_run_sleep_pass_sets_ttl_and_decays_expired(store, monkeypatch):
    _patch_ollama_chat(monkeypatch, {"message": {"content": json.dumps({
        "facts": [{"key": "trip", "value": "in Paris this week", "confidence": 0.9, "ttl_days": 1}],
        "preferences": [],
        "triples": [],
    })}})
    store.save_summary("alice", "sess-1", "User is in Paris this week")
    stats = store.run_sleep_pass("alice", limit=10)
    assert stats["promoted"] >= 1

    with store._lock:
        row = store._conn.execute(
            "SELECT expires_at FROM facts WHERE user_id = ? AND key = 'trip'", ("alice",)
        ).fetchone()
    assert row is not None
    assert row["expires_at"] is not None and row["expires_at"] > time.time()

    # Force the fact to be expired, then re-run the pass (no pending summaries).
    with store._lock:
        store._conn.execute(
            "UPDATE facts SET expires_at = ? WHERE user_id = ? AND key = 'trip'",
            (time.time() - 5, "alice"),
        )
        store._conn.commit()
    stats2 = store.run_sleep_pass("alice", limit=10)
    assert stats2["decayed"] >= 1

    with store._lock:
        gone = store._conn.execute(
            "SELECT id FROM facts WHERE user_id = ? AND key = 'trip'", ("alice",)
        ).fetchone()
    assert gone is None


def test_run_sleep_pass_tolerates_legacy_candidates_schema(store, monkeypatch):
    _patch_ollama_chat(monkeypatch, {"message": {"content": json.dumps({
        "candidates": [{"key": "name", "value": "Alice", "type": "fact", "confidence": 0.9}],
    })}})
    store.save_summary("alice", "sess-1", "My name is Alice")
    stats = store.run_sleep_pass("alice", limit=10)
    assert stats["processed"] == 1
    assert stats["promoted"] >= 1
