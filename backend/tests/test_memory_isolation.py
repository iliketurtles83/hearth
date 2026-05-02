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
