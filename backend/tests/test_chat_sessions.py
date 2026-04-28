from __future__ import annotations

import json
import os
import sys
import tempfile
import types
from types import SimpleNamespace

import pytest


if "musicpd" not in sys.modules:
    fake_musicpd = types.ModuleType("musicpd")

    class _FakeMPDClient:
        def connect(self, host: str, port: int) -> None:
            return None

        def disconnect(self) -> None:
            return None

    class _FakeConnectionError(Exception):
        pass

    fake_musicpd.MPDClient = _FakeMPDClient
    fake_musicpd.ConnectionError = _FakeConnectionError
    sys.modules["musicpd"] = fake_musicpd


_tmp_dir = tempfile.mkdtemp(prefix="assistant-chat-tests-")
os.environ["MEMORY_DB_PATH"] = os.path.join(_tmp_dir, "memory.db")
os.environ["CHROMA_PATH"] = os.path.join(_tmp_dir, "chroma")
os.environ["AUTH_DB_PATH"] = os.path.join(_tmp_dir, "auth.db")


import main  # noqa: E402


def _request(user_id: str, session_id: str | None = None):
    cookies = {}
    if session_id:
        cookies[main.SESSION_COOKIE_NAME] = session_id
    return SimpleNamespace(state=SimpleNamespace(user_id=user_id), cookies=cookies)


def _json_body(response) -> dict:
    return json.loads(response.body.decode("utf-8"))


@pytest.fixture(autouse=True)
def clear_session_store(monkeypatch):
    main._session_store.clear()
    yield
    main._session_store.clear()


def test_get_or_create_session_scoped_to_user():
    alice_session_id, alice_created = main._get_or_create_session("alice", None)
    bob_session_id, bob_created = main._get_or_create_session("bob", alice_session_id)

    assert alice_created is True
    assert bob_created is True
    assert bob_session_id != alice_session_id
    assert main._session_store[alice_session_id]["user_id"] == "alice"
    assert main._session_store[bob_session_id]["user_id"] == "bob"


def test_list_sessions_only_returns_owned_sessions():
    alice_session_id, _ = main._get_or_create_session("alice", None)
    bob_session_id, _ = main._get_or_create_session("bob", None)

    main._append_session_message(alice_session_id, "user", "alice prompt")
    main._append_session_message(bob_session_id, "user", "bob prompt")

    alice_sessions = main._list_sessions("alice")
    bob_sessions = main._list_sessions("bob")

    assert [item["session_id"] for item in alice_sessions] == [alice_session_id]
    assert [item["session_id"] for item in bob_sessions] == [bob_session_id]


@pytest.mark.asyncio
async def test_list_chat_sessions_hides_foreign_current_session_cookie():
    alice_session_id, _ = main._get_or_create_session("alice", None)

    response = await main.list_chat_sessions(_request("bob", alice_session_id))
    payload = _json_body(response)

    assert payload["sessions"] == []
    assert payload["current_session_id"] is None


@pytest.mark.asyncio
async def test_select_chat_session_denies_other_users_session():
    alice_session_id, _ = main._get_or_create_session("alice", None)

    response = await main.select_chat_session(
        main.SessionSelectRequest(session_id=alice_session_id),
        _request("bob"),
    )

    assert response.status_code == 404
    assert _json_body(response)["code"] == "SESSION_NOT_FOUND"


@pytest.mark.asyncio
async def test_delete_chat_session_denies_other_users_session():
    alice_session_id, _ = main._get_or_create_session("alice", None)

    response = await main.delete_chat_session(alice_session_id, _request("bob"))

    assert response.status_code == 404
    assert alice_session_id in main._session_store


@pytest.mark.asyncio
async def test_get_chat_session_messages_reanchors_stale_foreign_cookie():
    alice_session_id, _ = main._get_or_create_session("alice", None)
    main._append_session_message(alice_session_id, "user", "secret alice context")

    response = await main.get_chat_session_messages(_request("bob", alice_session_id))
    payload = _json_body(response)

    assert payload["session_id"] != alice_session_id
    assert payload["messages"] == []
    assert main._session_store[payload["session_id"]]["user_id"] == "bob"


@pytest.mark.asyncio
async def test_reset_chat_session_clears_messages_and_summary():
    session_id, _ = main._get_or_create_session("alice", None)
    main._append_session_message(session_id, "user", "hello")
    main._append_session_message(session_id, "assistant", "hi")
    main._session_store[session_id]["summary"] = "older summary"
    main._session_store[session_id]["summary_message_count"] = 2

    response = await main.reset_chat_session(_request("alice", session_id))
    payload = _json_body(response)

    assert payload["cleared_messages"] == 2
    assert main._session_store[session_id]["messages"] == []
    assert main._session_store[session_id]["summary"] == ""
    assert main._session_store[session_id]["summary_message_count"] == 0


def test_select_history_for_budget_respects_turn_cap(monkeypatch):
    monkeypatch.setattr(main, "CHAT_MAX_TURNS", 2)
    monkeypatch.setattr(main, "CHAT_TOKEN_BUDGET", 10_000)

    messages = [
        {"role": "user" if idx % 2 == 0 else "assistant", "content": f"message {idx}"}
        for idx in range(8)
    ]

    selected, _, truncated, _ = main._select_history_for_budget(
        messages=messages,
        system="system",
        current_user_message="current prompt",
        summary_text="",
    )

    assert [item["content"] for item in selected] == [f"message {idx}" for idx in range(4, 8)]
    assert truncated is True


def test_update_session_summary_if_needed_summarizes_older_messages(monkeypatch):
    monkeypatch.setattr(main, "CHAT_SUMMARY_KEEP_RECENT_MESSAGES", 2)
    monkeypatch.setattr(main, "CHAT_SUMMARY_TRIGGER_MESSAGES", 4)
    monkeypatch.setattr(main, "CHAT_SUMMARY_MAX_CHARS", 500)

    session_id, _ = main._get_or_create_session("alice", None)
    for idx in range(6):
        role = "user" if idx % 2 == 0 else "assistant"
        main._append_session_message(session_id, role, f"message {idx}")

    updated, summarized_count, summary_len = main._update_session_summary_if_needed(session_id)
    session = main._session_store[session_id]

    assert updated is True
    assert summarized_count == 4
    assert summary_len > 0
    assert "message 0" in session["summary"]
    assert "message 3" in session["summary"]
    assert session["summary_message_count"] == 4


def test_truncate_summary_keeps_tail_within_limit(monkeypatch):
    monkeypatch.setattr(main, "CHAT_SUMMARY_MAX_CHARS", 24)

    summary = "line 1\nline 2\nline 3\nline 4"
    truncated = main._truncate_summary(summary)

    assert len(truncated) <= 24
    assert truncated.endswith("line 4")