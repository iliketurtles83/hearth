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


def _route_endpoint(path: str, method: str):
    for route in main.app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            return route.endpoint
    raise AssertionError(f"Route not found: {method} {path}")


async def _read_sse_events(streaming_response) -> list[str]:
    events: list[str] = []
    async for chunk in streaming_response.body_iterator:
        if isinstance(chunk, bytes):
            text = chunk.decode("utf-8")
        else:
            text = str(chunk)
        for line in text.splitlines():
            if line.startswith("data: "):
                events.append(line[len("data: "):])
    return events


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


def test_list_sessions_sorted_by_updated_at_descending():
    first_id, _ = main._get_or_create_session("alice", None)
    second_id, _ = main._get_or_create_session("alice", None)

    main._append_session_message(first_id, "user", "older")
    main._append_session_message(second_id, "user", "newer")

    sessions = main._list_sessions("alice")
    assert [item["session_id"] for item in sessions] == [second_id, first_id]


def test_get_or_create_existing_session_does_not_touch_updated_at():
    session_id, _ = main._get_or_create_session("alice", None)
    original_updated_at = main._session_store[session_id]["updated_at"]

    returned_id, created = main._get_or_create_session("alice", session_id)

    assert returned_id == session_id
    assert created is False
    assert main._session_store[session_id]["updated_at"] == original_updated_at


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
async def test_get_chat_session_messages_does_not_touch_updated_at_for_existing_session():
    session_id, _ = main._get_or_create_session("alice", None)
    main._append_session_message(session_id, "user", "hello")
    original_updated_at = main._session_store[session_id]["updated_at"]

    response = await main.get_chat_session_messages(_request("alice", session_id))
    payload = _json_body(response)

    assert payload["session_id"] == session_id
    assert main._session_store[session_id]["updated_at"] == original_updated_at


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


@pytest.mark.asyncio
async def test_reset_chat_session_schedules_episodic_persist(monkeypatch):
    session_id, _ = main._get_or_create_session("alice", None)
    main._append_session_message(session_id, "user", "keep this")

    captured: list[tuple[str, str]] = []

    def _fake_spawn(sid: str, _snapshot: dict, reason: str):
        captured.append((sid, reason))

    monkeypatch.setattr(main, "_spawn_episodic_persist_task", _fake_spawn)

    await main.reset_chat_session(_request("alice", session_id))

    assert captured == [(session_id, "session_reset")]


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


def test_chat_request_default_system_uses_config_constant():
    req = main.ChatRequest(message="hello")
    assert req.system == main.CHAT_DEFAULT_SYSTEM_PROMPT


def test_required_wake_models_reflect_runtime_constants(monkeypatch):
    monkeypatch.setattr(main, "WAKEWORD_MODEL_FILE", "wake-custom.onnx")
    monkeypatch.setattr(main, "OWW_MELSPEC_MODEL_FILE", "melspec-custom.onnx")
    monkeypatch.setattr(main, "OWW_EMBEDDING_MODEL_FILE", "embed-custom.onnx")

    required = main._required_wake_models()

    assert required == [
        "wake-custom.onnx",
        "melspec-custom.onnx",
        "embed-custom.onnx",
    ]


@pytest.mark.asyncio
async def test_chat_stream_includes_done_and_voice_metadata_for_voice_source(monkeypatch):
    async def _fake_router_route(_message: str):
        return SimpleNamespace(
            intent="quick-local",
            confidence=0.99,
            use_cloud=False,
            model=main.CHAT_MODEL,
            planner_status="ok",
            needs_memory=False,
            tool=None,
            reasoning_summary="",
        )

    async def _fake_stream_local(_request, model_name=None):
        yield "hello"
        yield " world"

    ingest_sources: list[str] = []

    monkeypatch.setattr(main, "router_route", _fake_router_route)
    monkeypatch.setattr(main, "stream_local", _fake_stream_local)
    monkeypatch.setattr(main.memory_store, "retrieve", lambda *_args, **_kwargs: [])

    def _fake_ingest(*_args, **kwargs):
        ingest_sources.append(kwargs.get("source", ""))
        return {
            "status": "none",
            "saved": [],
            "blocked": [],
            "needs_confirmation": [],
            "deleted": 0,
            "explicit": False,
        }

    monkeypatch.setattr(main.memory_store, "ingest_user_message", _fake_ingest)

    response = await main.chat(
        main.ChatRequest(message="hello", source="voice"),
        _request("alice"),
    )
    events = await _read_sse_events(response)

    assert events[0] != "[DONE]"
    assert json.loads(events[0])["model"] == main.CHAT_MODEL
    text_payloads = [json.loads(event).get("text", "") for event in events if event != "[DONE]"]
    combined_text = "".join(text_payloads)
    assert "hello" in combined_text
    assert "world" in combined_text
    assert any("voice" in json.loads(event) for event in events if event != "[DONE]")
    assert events[-1] == "[DONE]"
    assert ingest_sources == ["voice"]


@pytest.mark.asyncio
async def test_chat_stream_omits_voice_metadata_for_text_source(monkeypatch):
    async def _fake_router_route(_message: str):
        return SimpleNamespace(
            intent="quick-local",
            confidence=0.99,
            use_cloud=False,
            model=main.CHAT_MODEL,
            planner_status="ok",
            needs_memory=False,
            tool=None,
            reasoning_summary="",
        )

    async def _fake_stream_local(_request, model_name=None):
        yield "text only"

    monkeypatch.setattr(main, "router_route", _fake_router_route)
    monkeypatch.setattr(main, "stream_local", _fake_stream_local)
    monkeypatch.setattr(main.memory_store, "retrieve", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        main.memory_store,
        "ingest_user_message",
        lambda *_args, **_kwargs: {
            "status": "none",
            "saved": [],
            "blocked": [],
            "needs_confirmation": [],
            "deleted": 0,
            "explicit": False,
        },
    )

    response = await main.chat(
        main.ChatRequest(message="hello", source="text"),
        _request("alice"),
    )
    events = await _read_sse_events(response)

    assert any(json.loads(event).get("text") == "text only" for event in events if event != "[DONE]")
    assert not any("voice" in json.loads(event) for event in events if event != "[DONE]")
    assert events[-1] == "[DONE]"


@pytest.mark.asyncio
async def test_chat_music_fast_path_bypasses_router_and_dispatches_music_tool(monkeypatch):
    async def _unexpected_router(_message: str):
        raise AssertionError("router_route should not run for explicit music commands")

    dispatched: list[tuple[str, dict]] = []

    async def _fake_dispatch(tool_name: str, params: dict):
        dispatched.append((tool_name, dict(params)))
        return main.ToolResult(
            ok=True,
            data={
                "action": "play",
                "track": {"title": "Battery", "artist": "Metallica"},
            },
        )

    monkeypatch.setattr(main, "router_route", _unexpected_router)
    monkeypatch.setattr(main.tools, "dispatch", _fake_dispatch)

    response = await main.chat(
        main.ChatRequest(message="play Battery by Metallica", source="text"),
        _request("alice"),
    )
    events = await _read_sse_events(response)

    assert json.loads(events[0]) == {"model": "music", "intent": "music", "confidence": 1.0}
    assert json.loads(events[1])["text"] == 'Now playing: "Battery" by Metallica.'
    assert events[-1] == "[DONE]"
    assert dispatched == [
        (
            "music",
            {
                "action": "play",
                "query": "battery",
                "artist_filter": "metallica",
                "prompt": "play Battery by Metallica",
                "user_id": "alice",
            },
        )
    ]


@pytest.mark.asyncio
async def test_chat_music_fast_path_formats_genre_multi_track_response(monkeypatch):
    async def _unexpected_router(_message: str):
        raise AssertionError("router_route should not run for explicit music commands")

    async def _fake_dispatch(_tool_name: str, _params: dict):
        return main.ToolResult(
            ok=True,
            data={
                "action": "play",
                "track": {"title": "As The Pages Burn", "artist": "Oratory"},
                "tracks": [
                    {"title": "As The Pages Burn", "artist": "Oratory"},
                    {"title": "End of All Hope", "artist": "Nightwish"},
                ],
                "genre": "heavy metal",
            },
        )

    monkeypatch.setattr(main, "router_route", _unexpected_router)
    monkeypatch.setattr(main.tools, "dispatch", _fake_dispatch)

    response = await main.chat(
        main.ChatRequest(message="play Heavy Metal", source="text"),
        _request("alice"),
    )
    events = await _read_sse_events(response)

    assert json.loads(events[1])["text"] == "Now playing: 2 Heavy Metal tracks."


@pytest.mark.asyncio
async def test_chat_vague_music_prompt_uses_music_fastpath(monkeypatch):
    music_dispatch_calls: list[tuple[str, dict]] = []

    async def _unexpected_music_dispatch(tool_name: str, params: dict):
        music_dispatch_calls.append((tool_name, params))
        raise AssertionError("deterministic music fastpath should not run for vague prompts")

    class _FakeGraph:
        async def astream(self, _state, config=None, stream_mode=None):
            yield {
                "meta": {
                    "model": main.CHAT_MODEL,
                    "intent": "quick-local",
                    "confidence": 0.70,
                    "route_type": "local",
                    "needs_memory": False,
                    "tool": None,
                    "planner_status": "fallback",
                    "reasoning_summary": "",
                }
            }
            yield {"text": "handled by graph"}

    monkeypatch.setattr(main.tools, "dispatch", _unexpected_music_dispatch)
    monkeypatch.setattr(main.app.state, "assistant_graph", _FakeGraph(), raising=False)
    monkeypatch.setattr(main.memory_store, "retrieve", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        main.memory_store,
        "ingest_user_message",
        lambda *_args, **_kwargs: {
            "status": "none",
            "saved": [],
            "blocked": [],
            "needs_confirmation": [],
            "deleted": 0,
            "explicit": False,
        },
    )

    response = await main.chat(
        main.ChatRequest(message="play something chill", source="text"),
        _request("alice"),
    )
    events = await _read_sse_events(response)

    assert music_dispatch_calls == []
    assert json.loads(events[0])["model"] == main.CHAT_MODEL
    assert any(json.loads(event).get("text") == "handled by graph" for event in events if event != "[DONE]")
    assert events[-1] == "[DONE]"


@pytest.mark.asyncio
async def test_chat_graph_stream_uses_session_id_as_checkpoint_thread(monkeypatch):
    session_id, _ = main._get_or_create_session("alice", None)
    captured: dict[str, object] = {}

    class _FakeGraph:
        async def astream(self, _state, config=None, stream_mode=None):
            captured["config"] = config
            captured["stream_mode"] = stream_mode
            yield {
                "meta": {
                    "model": main.CHAT_MODEL,
                    "intent": "quick-local",
                    "confidence": 0.99,
                    "route_type": "local",
                    "needs_memory": False,
                    "tool": None,
                    "planner_status": "ok",
                    "reasoning_summary": "",
                }
            }
            yield {"text": "checkpoint-bound"}

    monkeypatch.setattr(main, "_assistant_graph", _FakeGraph())
    if hasattr(main.app.state, "assistant_graph"):
        delattr(main.app.state, "assistant_graph")
    monkeypatch.setattr(
        main.memory_store,
        "ingest_user_message",
        lambda *_args, **_kwargs: {
            "status": "none",
            "saved": [],
            "blocked": [],
            "needs_confirmation": [],
            "deleted": 0,
            "explicit": False,
        },
    )

    response = await main.chat(
        main.ChatRequest(message="hello", source="text"),
        _request("alice", session_id),
    )
    events = await _read_sse_events(response)

    assert captured["stream_mode"] == "custom"
    assert captured["config"] == main.checkpoint_config(session_id)
    assert any(json.loads(event).get("text") == "checkpoint-bound" for event in events if event != "[DONE]")
    assert events[-1] == "[DONE]"


@pytest.mark.asyncio
async def test_get_graph_state_returns_snapshot_for_owned_session(monkeypatch):
    session_id, _ = main._get_or_create_session("alice", None)

    class _FakeGraph:
        async def aget_state(self, _config):
            return SimpleNamespace(
                values={"intent": "quick-local", "route_type": "local"},
                next=("responder",),
                metadata={"checkpointed": True},
            )

    monkeypatch.setattr(main, "_assistant_graph", _FakeGraph())
    if hasattr(main.app.state, "assistant_graph"):
        delattr(main.app.state, "assistant_graph")

    response = await main.get_graph_state(session_id, _request("alice", session_id))
    payload = _json_body(response)

    assert payload["session_id"] == session_id
    assert payload["state"]["intent"] == "quick-local"
    assert payload["next"] == ["responder"]
    assert payload["metadata"]["checkpointed"] is True


@pytest.mark.asyncio
async def test_get_graph_state_denies_foreign_session():
    session_id, _ = main._get_or_create_session("alice", None)

    response = await main.get_graph_state(session_id, _request("bob", None))

    assert response.status_code == 404
    assert _json_body(response)["code"] == "SESSION_NOT_FOUND"


@pytest.mark.asyncio
async def test_list_episodic_memory_endpoint_returns_user_rows():
    main.memory_store.save_summary("alice", "sess-ep-1", "- User: I like coffee")

    list_episodic_memory = _route_endpoint("/memory/episodic", "GET")

    response = await list_episodic_memory(
        _request("alice"),
        limit=200,
        offset=0,
        consolidated=None,
    )
    payload = _json_body(response)

    assert payload["total"] >= 1
    assert any(item["tier"] == "episodic" for item in payload["items"])


@pytest.mark.asyncio
async def test_consolidate_memory_endpoint_runs_for_current_user():
    main.memory_store.save_summary("alice", "sess-ep-2", "- User: My name is Alice")

    consolidate_memory = _route_endpoint("/memory/consolidate", "POST")

    response = await consolidate_memory(_request("alice"))
    payload = _json_body(response)

    assert payload["ok"] is True
    assert payload["stats"]["processed"] >= 1