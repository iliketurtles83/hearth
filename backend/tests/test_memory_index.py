"""Chroma index lifecycle: survives restarts, rebuilds on model change, fails closed.

SQLite is canonical; the Chroma collection is a derived index that must never
be wiped on a normal restart or filled with non-semantic fallback vectors.
"""
import hashlib

import pytest

from memory import MemoryStore


class _FakeEmbedder:
    """Deterministic stand-in for the embedding endpoint with a switchable dim."""

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim
        self.up = True

    def __call__(self, text, *, base_url, model):
        if not self.up:
            return None
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        return [((seed[i % len(seed)] + i) % 17) / 17.0 + 0.01 for i in range(self.dim)]


@pytest.fixture
def embedder(monkeypatch):
    fake = _FakeEmbedder()
    monkeypatch.setattr("memory._openai_embed_sync", fake)
    return fake


def _open(tmp_path) -> MemoryStore:
    return MemoryStore(
        db_path=str(tmp_path / "memory.db"),
        chroma_path=str(tmp_path / "chroma"),
    )


def _memory_row_count(store: MemoryStore) -> int:
    facts = store._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    prefs = store._conn.execute("SELECT COUNT(*) FROM preferences").fetchone()[0]
    return facts + prefs


def test_index_survives_restart(tmp_path, embedder):
    store = _open(tmp_path)
    store.ingest_user_message("alice", "My name is Alice and I love cats.")
    count = store._collection.count()
    assert count > 0

    reopened = _open(tmp_path)
    assert reopened._collection.count() == count


def test_embedder_down_at_boot_keeps_index(tmp_path, embedder):
    store = _open(tmp_path)
    store.ingest_user_message("alice", "My name is Alice and I love cats.")
    count = store._collection.count()

    embedder.up = False
    reopened = _open(tmp_path)
    assert reopened._collection.count() == count
    assert reopened._index_stale


def test_dimension_change_rebuilds_from_sqlite(tmp_path, embedder):
    store = _open(tmp_path)
    store.ingest_user_message("alice", "My name is Alice and I love cats.")
    rows = _memory_row_count(store)
    assert rows > 0

    embedder.dim = 384
    reopened = _open(tmp_path)
    assert reopened._stored_embedding_dim() == 384
    assert reopened._collection.count() == rows


def test_write_while_embedder_down_is_deferred_then_indexed(tmp_path, embedder):
    store = _open(tmp_path)

    embedder.up = False
    store.ingest_user_message("alice", "My name is Alice and I love cats.")
    assert _memory_row_count(store) > 0
    assert store._collection.count() == 0
    assert store._index_stale

    embedder.up = True
    store.ingest_user_message("bob", "My name is Bob and I love dogs.")
    assert store._collection.count() == _memory_row_count(store)
    assert not store._index_stale


def test_updated_preference_keeps_its_vector_id(tmp_path, embedder):
    store = _open(tmp_path)
    store.ingest_user_message("alice", "I prefer dark mode.")
    # Advance the connection's lastrowid past the preferences table, as chat logging does.
    for _ in range(5):
        store._conn.execute(
            "INSERT INTO conversation_log (session_id, user_id, role, content, ts) VALUES ('s', 'alice', 'user', 'hi', 0)"
        )
    store.ingest_user_message("alice", "I prefer light mode.")

    pref_ids = {f"preferences:{r[0]}" for r in store._conn.execute("SELECT id FROM preferences")}
    vector_ids = {i for i in store._collection.get(include=[])["ids"] if i.startswith("preferences:")}
    assert len(pref_ids) == 1
    assert vector_ids == pref_ids
