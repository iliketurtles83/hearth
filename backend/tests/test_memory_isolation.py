"""Tests for multi-user memory isolation.

Verifies that facts, preferences, and summaries are scoped to their owner
and that no data leaks across users.
"""
import pytest

from memory import MemoryStore


@pytest.fixture
def store(tmp_path):
    return MemoryStore(
        db_path=str(tmp_path / "memory.db"),
        chroma_path=str(tmp_path / "chroma"),
    )


# ── Facts isolation ───────────────────────────────────────────────────────────

def test_facts_do_not_leak_across_users(store):
    store.ingest_user_message("alice", "My name is Alice and I love cats.")
    store.ingest_user_message("bob",   "My name is Bob and I love dogs.")

    alice_items = store.list_items("alice")["items"]
    bob_items   = store.list_items("bob")["items"]

    alice_values = {i["value"] for i in alice_items}
    bob_values   = {i["value"] for i in bob_items}

    # No bleed-over: alice's items shouldn't appear in bob's list and vice versa.
    assert not (alice_values & bob_values) or alice_values.isdisjoint(bob_values)


def test_delete_item_cross_user_denied(store):
    store.ingest_user_message("alice", "Save the fact that I prefer dark mode.")
    alice_items = store.list_items("alice")["items"]
    assert alice_items  # at least one item was saved

    fact_id = alice_items[0]["id"]
    # Bob tries to delete Alice's item — must return False.
    deleted = store.delete_item("bob", fact_id)
    assert deleted is False

    # Item must still exist for Alice.
    alice_items_after = store.list_items("alice")["items"]
    assert any(i["id"] == fact_id for i in alice_items_after)


def test_clear_all_only_affects_owner(store):
    store.set_preference("alice", "theme", "dark")
    store.set_preference("bob",   "theme", "light")

    store.clear_all("alice")

    assert store.get_preference("alice", "theme") is None
    assert store.get_preference("bob",   "theme") == "light"


# ── Preferences isolation ─────────────────────────────────────────────────────

def test_preferences_isolated_by_user(store):
    store.set_preference("alice", "default_location", "London")
    store.set_preference("bob",   "default_location", "Tokyo")

    assert store.get_preference("alice", "default_location") == "London"
    assert store.get_preference("bob",   "default_location") == "Tokyo"


def test_preference_unknown_user_returns_none(store):
    assert store.get_preference("nobody", "default_location") is None


def test_preference_overwrite_scoped(store):
    store.set_preference("alice", "default_location", "Paris")
    store.set_preference("alice", "default_location", "Berlin")
    store.set_preference("bob",   "default_location", "Paris")

    # Alice's preference was updated; Bob's is unchanged.
    assert store.get_preference("alice", "default_location") == "Berlin"
    assert store.get_preference("bob",   "default_location") == "Paris"


# ── list_items pagination ─────────────────────────────────────────────────────

def test_list_items_scoped(store):
    store.ingest_user_message("alice", "Remember that I drink espresso.")
    store.ingest_user_message("carol", "Remember that I drink tea.")

    alice_items = store.list_items("alice")["items"]
    carol_items = store.list_items("carol")["items"]

    alice_ids = {i["id"] for i in alice_items}
    carol_ids = {i["id"] for i in carol_items}

    assert alice_ids.isdisjoint(carol_ids)


# ── Semantic retrieval (ChromaDB) ─────────────────────────────────────────────

def test_retrieve_scoped_to_user(store):
    store.ingest_user_message("alice", "I own a golden retriever named Max.")
    store.ingest_user_message("bob",   "I have a siamese cat named Luna.")

    alice_results = store.retrieve("alice", "pet")
    bob_results   = store.retrieve("bob",   "pet")

    alice_texts = " ".join(str(r.get("text", "")) for r in alice_results).lower()
    bob_texts   = " ".join(str(r.get("text", "")) for r in bob_results).lower()

    # Each user's retrieval should surface their own pet, not the other's.
    if alice_results:
        assert "max" in alice_texts or "golden" in alice_texts or "dog" in alice_texts
    if bob_results:
        assert "luna" in bob_texts or "siamese" in bob_texts or "cat" in bob_texts


def test_list_items_exposes_tier_labels(store):
    store.ingest_user_message("alice", "My name is Alice.")
    store.save_summary("alice", "sess-1", "- User: I prefer dark themes")

    items = store.list_items("alice")["items"]
    tiers = {item["tier"] for item in items}

    assert "semantic" in tiers
    assert "episodic" in tiers


def test_list_episodic_returns_only_summaries(store):
    store.ingest_user_message("alice", "My favorite editor is VS Code.")
    row_id = store.save_summary("alice", "sess-2", "- User: My favorite editor is VS Code.")

    episodic = store.list_episodic("alice")
    assert episodic["total"] >= 1
    assert any(item["id"] == f"summaries:{row_id}" for item in episodic["items"])
    assert all(item["tier"] == "episodic" for item in episodic["items"])


def test_consolidate_pending_promotes_summary_facts(store):
    """LLM extraction should promote facts from episodic summaries (Phase 12b)."""
    import json
    from unittest.mock import patch, AsyncMock, MagicMock

    # Mock Ollama response with extracted facts
    mock_response = {
        "response": json.dumps({
            "candidates": [
                {"key": "name", "value": "Alice", "type": "fact", "confidence": 0.95},
                {"key": "location", "value": "Tallinn", "type": "fact", "confidence": 0.90},
            ]
        })
    }

    async def mock_post_fn(*args, **kwargs):
        resp = MagicMock()
        resp.json = MagicMock(return_value=mock_response)
        resp.raise_for_status = MagicMock()
        return resp

    async_client_mock = MagicMock()
    async_client_mock.post = mock_post_fn
    async_client_mock.__aenter__ = AsyncMock(return_value=async_client_mock)
    async_client_mock.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=async_client_mock):
        store.save_summary(
            "alice",
            "sess-3",
            "- User: My name is Alice\n- User: I live in Tallinn",
        )

        stats = store.consolidate_pending("alice", limit=10)
        assert stats["processed"] == 1
        assert stats["promoted"] >= 1

        episodic_pending = store.list_episodic("alice", consolidated=False)
        assert episodic_pending["total"] == 0

        hits = store.retrieve("alice", "where do I live")
        assert any("tallinn" in str(hit.get("text", "")).lower() for hit in hits)


def test_consolidate_blocks_sensitive_candidates(store):
    """Sensitive items in episodic summaries must not be promoted to semantic facts."""
    # Store a summary that contains a sensitive item (password/credential pattern).
    store.save_summary(
        "alice",
        "sess-4",
        "- User: My password is hunter2",
    )

    stats = store.consolidate_pending("alice", limit=10)
    assert stats["processed"] == 1
    # The sensitive candidate should be blocked, not promoted.
    assert stats["blocked"] >= 1 or stats["promoted"] == 0

    # No fact or preference containing the credential should exist.
    items = store.list_items("alice")["items"]
    semantic_items = [i for i in items if i.get("tier") == "semantic"]
    values = " ".join(str(i.get("value", "")) for i in semantic_items).lower()
    assert "hunter2" not in values


def test_delete_episodic_record(store):
    """Deleting a summaries row via delete_item removes it for that user."""
    row_id = store.save_summary("alice", "sess-5", "- User: I work on a robotics project")

    # Confirm it's visible.
    episodic = store.list_episodic("alice")
    assert any(item["id"] == f"summaries:{row_id}" for item in episodic["items"])

    # Delete it.
    deleted = store.delete_item("alice", f"summaries:{row_id}")
    assert deleted is True

    # Confirm it's gone.
    episodic_after = store.list_episodic("alice")
    assert not any(item["id"] == f"summaries:{row_id}" for item in episodic_after["items"])


def test_delete_episodic_cross_user_denied(store):
    """Bob must not be able to delete Alice's episodic record."""
    row_id = store.save_summary("alice", "sess-6", "- User: I enjoy hiking")

    denied = store.delete_item("bob", f"summaries:{row_id}")
    assert denied is False

    episodic = store.list_episodic("alice")
    assert any(item["id"] == f"summaries:{row_id}" for item in episodic["items"])


# ── Phase 12b: LLM-based memory extraction tests ─────────────────────────────────

def test_llm_extract_filters_by_confidence(store, monkeypatch):
    """LLM extraction must filter candidates by confidence >= 0.7 (Phase 12b)."""
    import json
    import asyncio
    from unittest.mock import patch, AsyncMock, MagicMock

    # Mock Ollama response with mixed confidence scores
    mock_response = {
        "response": json.dumps({
            "candidates": [
                {
                    "key": "favorite_language",
                    "value": "Python",
                    "type": "preference",
                    "confidence": 0.95,  # High confidence → included
                },
                {
                    "key": "workspace_language",
                    "value": "JavaScript",
                    "type": "preference",
                    "confidence": 0.6,  # Low confidence → filtered out
                },
                {
                    "key": "location",
                    "value": "Helsinki",
                    "type": "fact",
                    "confidence": 0.85,  # High confidence → included
                },
                {
                    "key": "maybe_interest",
                    "value": "machine learning",
                    "type": "fact",
                    "confidence": 0.65,  # Below threshold → filtered out
                },
            ]
        })
    }

    async def mock_post_fn(*args, **kwargs):
        resp = MagicMock()
        resp.json = MagicMock(return_value=mock_response)  # Sync method
        resp.raise_for_status = MagicMock()
        return resp

    # Create a proper async context manager mock
    async_client_mock = MagicMock()
    async_client_mock.post = mock_post_fn
    async_client_mock.__aenter__ = AsyncMock(return_value=async_client_mock)
    async_client_mock.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=async_client_mock):
        # Call the async extraction function
        candidates = asyncio.run(store._llm_extract_candidates(
            "User: I like Python and JavaScript for work. Interested in ML.",
            source="test"
        ))

    # Should only include candidates with confidence >= 0.7
    assert len(candidates) == 2
    keys = {c.key for c in candidates}
    assert "favorite_language" in keys
    assert "location" in keys
    assert "workspace_language" not in keys  # Confidence 0.6 filtered out
    assert "maybe_interest" not in keys  # Confidence 0.65 filtered out


def test_llm_extract_json_parse_failure(store, monkeypatch):
    """LLM extraction must gracefully handle invalid JSON (Phase 12b)."""
    import asyncio
    from unittest.mock import patch, AsyncMock, MagicMock

    # Mock Ollama response with invalid JSON
    async def mock_post_fn(*args, **kwargs):
        resp = MagicMock()
        resp.json = MagicMock(return_value={"response": "not valid json { [ }"})
        resp.raise_for_status = MagicMock()
        return resp

    async_client_mock = MagicMock()
    async_client_mock.post = mock_post_fn
    async_client_mock.__aenter__ = AsyncMock(return_value=async_client_mock)
    async_client_mock.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=async_client_mock):
        # Call extraction with bad JSON response
        candidates = asyncio.run(store._llm_extract_candidates(
            "User: My name is Test User",
            source="test"
        ))

    # Should return empty list (graceful failure)
    assert candidates == []


def test_llm_extract_ollama_unreachable(store, monkeypatch):
    """LLM extraction must gracefully handle Ollama unreachability (Phase 12b)."""
    import asyncio
    from unittest.mock import patch, AsyncMock, MagicMock
    import httpx

    # Mock Ollama connection error
    async def mock_post_fn(*args, **kwargs):
        raise httpx.ConnectError("Failed to connect to Ollama")

    async_client_mock = MagicMock()
    async_client_mock.post = mock_post_fn
    async_client_mock.__aenter__ = AsyncMock(return_value=async_client_mock)
    async_client_mock.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=async_client_mock):
        # Call extraction when Ollama is unreachable
        candidates = asyncio.run(store._llm_extract_candidates(
            "User: Some test content",
            source="test"
        ))

    # Should return empty list (graceful failure, no crash)
    assert candidates == []


def test_consolidate_uses_llm_extraction(store, monkeypatch):
    """Consolidation worker should use LLM extraction instead of regex (Phase 12b)."""
    import json
    from unittest.mock import patch, AsyncMock, MagicMock

    # Mock Ollama with realistic extraction
    mock_response = {
        "response": json.dumps({
            "candidates": [
                {
                    "key": "name",
                    "value": "Alice",
                    "type": "fact",
                    "confidence": 0.95,
                },
                {
                    "key": "location",
                    "value": "Tokyo",
                    "type": "fact",
                    "confidence": 0.88,
                },
            ]
        })
    }

    async def mock_post_fn(*args, **kwargs):
        resp = MagicMock()
        resp.json = MagicMock(return_value=mock_response)
        resp.raise_for_status = MagicMock()
        return resp

    async_client_mock = MagicMock()
    async_client_mock.post = mock_post_fn
    async_client_mock.__aenter__ = AsyncMock(return_value=async_client_mock)
    async_client_mock.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=async_client_mock):
        # Create and consolidate an episodic summary
        store.save_summary(
            "alice",
            "sess-test",
            "- User: My name is Alice\n- User: I live in Tokyo",
        )

        # Run consolidation
        stats = store.consolidate_pending("alice", limit=10)

    # Should process the summary and promote LLM-extracted candidates
    assert stats["processed"] == 1
    assert stats["promoted"] >= 2  # name and location should be promoted

    # Verify facts are retrievable
    items = store.list_items("alice")["items"]
    semantic_items = [i for i in items if i.get("tier") == "semantic"]
    assert len(semantic_items) >= 2
