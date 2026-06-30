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
def clear_session_store():
    main.memory_store._conn.execute("DELETE FROM conversation_log")
    main.memory_store._conn.execute("DELETE FROM summaries")
    main.memory_store._conn.commit()
    yield
    main.memory_store._conn.execute("DELETE FROM conversation_log")
    main.memory_store._conn.execute("DELETE FROM summaries")
    main.memory_store._conn.commit()


def test_list_sessions_scoped_to_user():
    alice_session_id = "sess-alice"
    bob_session_id = "sess-bob"

    main.memory_store.log_turn(alice_session_id, "alice", "user", "alice prompt")
    main.memory_store.log_turn(bob_session_id, "bob", "user", "bob prompt")

    alice_sessions = main.memory_store.list_sessions("alice")
    bob_sessions = main.memory_store.list_sessions("bob")

    assert len(alice_sessions) == 1
    assert alice_sessions[0]["session_id"] == alice_session_id
    assert len(bob_sessions) == 1
    assert bob_sessions[0]["session_id"] == bob_session_id


@pytest.mark.asyncio
async def test_list_chat_sessions_hides_foreign_current_session_cookie():
    main.memory_store.log_turn("sess-alice", "alice", "user", "secret")

    response = await main.list_chat_sessions(_request("bob", "sess-alice"))
    payload = _json_body(response)

    assert payload["sessions"] == []
    assert payload["current_session_id"] is None


@pytest.mark.asyncio
async def test_select_chat_session_denies_other_users_session():
    main.memory_store.log_turn("sess-alice", "alice", "user", "secret")

    response = await main.select_chat_session(
        main.SessionSelectRequest(session_id="sess-alice"),
        _request("bob"),
    )

    assert response.status_code == 404
    assert _json_body(response)["code"] == "SESSION_NOT_FOUND"


@pytest.mark.asyncio
async def test_delete_chat_session_denies_other_users_session():
    main.memory_store.log_turn("sess-alice", "alice", "user", "secret")

    response = await main.delete_chat_session("sess-alice", _request("bob"))

    assert response.status_code == 404
    assert _json_body(response)["code"] == "SESSION_NOT_FOUND"

    turns = main.memory_store.get_session_turns("sess-alice", "alice", 500)
    assert len(turns) == 1


@pytest.mark.asyncio
async def test_delete_chat_session_clears_checkpoint_thread_for_owned_session(monkeypatch):
    session_id = "sess-delete-owned"
    main.memory_store.log_turn(session_id, "alice", "user", "hello")
    cleared: list[str] = []

    async def _fake_clear_checkpoint_thread(target_session_id: str) -> None:
        cleared.append(target_session_id)

    monkeypatch.setattr(main, "_clear_checkpoint_thread", _fake_clear_checkpoint_thread)

    response = await main.delete_chat_session(session_id, _request("alice", session_id))
    payload = _json_body(response)

    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["session_id"] == session_id
    assert cleared == [session_id]


@pytest.mark.asyncio
async def test_get_chat_session_messages_reanchors_stale_foreign_cookie():
    main.memory_store.log_turn("sess-alice", "alice", "user", "secret alice context")

    response = await main.get_chat_session_messages(_request("bob", "sess-alice"))
    payload = _json_body(response)

    assert payload["session_id"] != "sess-alice"
    assert payload["messages"] == []
    new_session_id = payload["session_id"]
    new_turns = main.memory_store.get_session_turns(new_session_id, "bob", 500)
    assert len(new_turns) == 0


@pytest.mark.asyncio
async def test_get_chat_session_messages_works_for_existing_session():
    session_id = "sess-existing"
    main.memory_store.log_turn(session_id, "alice", "user", "hello")

    response = await main.get_chat_session_messages(_request("alice", session_id))
    payload = _json_body(response)

    assert payload["session_id"] == session_id
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"] == "hello"


@pytest.mark.asyncio
async def test_reset_chat_session_clears_messages():
    session_id = "sess-reset"
    main.memory_store.log_turn(session_id, "alice", "user", "hello")
    main.memory_store.log_turn(session_id, "alice", "assistant", "hi")

    await main.reset_chat_session(_request("alice", session_id))

    turns = main.memory_store.get_session_turns(session_id, "alice", 500)
    assert len(turns) == 0


@pytest.mark.asyncio
async def test_reset_chat_session_returns_ok():
    session_id = "sess-reset-2"
    main.memory_store.log_turn(session_id, "alice", "user", "reset me")

    response = await main.reset_chat_session(_request("alice", session_id))
    payload = _json_body(response)

    assert payload["ok"] is True
    assert payload["session_id"] == session_id

    turns = main.memory_store.get_session_turns(session_id, "alice", 500)
    assert len(turns) == 0


@pytest.mark.asyncio
async def test_reset_chat_session_clears_checkpoint_thread(monkeypatch):
    session_id = "sess-reset-checkpoint"
    main.memory_store.log_turn(session_id, "alice", "user", "reset me")
    cleared: list[str] = []

    async def _fake_clear_checkpoint_thread(target_session_id: str) -> None:
        cleared.append(target_session_id)

    monkeypatch.setattr(main, "_clear_checkpoint_thread", _fake_clear_checkpoint_thread)

    response = await main.reset_chat_session(_request("alice", session_id))

    assert response.status_code == 200
    assert cleared == [session_id]


def test_select_history_for_budget_respects_turn_cap(monkeypatch):
    import graph as graph_mod  # noqa: E402

    monkeypatch.setattr(graph_mod, "CHAT_MAX_TURNS", 2)
    monkeypatch.setattr(graph_mod, "CHAT_TOKEN_BUDGET", 10_000)

    messages = [
        {"role": "user" if idx % 2 == 0 else "assistant", "content": f"message {idx}"}
        for idx in range(8)
    ]

    selected, _, truncated, _ = graph_mod._select_history_for_budget(
        messages=messages,
        system="system",
        current_user_message="current prompt",
        summary_text="",
    )

    assert [item["content"] for item in selected] == [f"message {idx}" for idx in range(4, 8)]
    assert truncated is True


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
    class _FakeGraph:
        async def astream(self, _state, config=None, stream_mode=None):
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
            yield {"text": "hello"}
            yield {"text": " world"}

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


@pytest.mark.asyncio
async def test_chat_stream_omits_voice_metadata_for_text_source(monkeypatch):
    class _FakeGraph:
        async def astream(self, _state, config=None, stream_mode=None):
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
            yield {"text": "text only"}

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
        main.ChatRequest(message="hello", source="text"),
        _request("alice"),
    )
    events = await _read_sse_events(response)

    assert any(json.loads(event).get("text") == "text only" for event in events if event != "[DONE]")
    assert not any("voice" in json.loads(event) for event in events if event != "[DONE]")
    assert events[-1] == "[DONE]"


@pytest.mark.asyncio
async def test_chat_stream_includes_thinking_chunks_when_available(monkeypatch):
    class _FakeGraph:
        async def astream(self, _state, config=None, stream_mode=None):
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
            yield {"thinking": "let me think this through"}
            yield {"text": "final answer"}

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
        main.ChatRequest(message="hello", source="text"),
        _request("alice"),
    )
    events = await _read_sse_events(response)

    parsed = [json.loads(event) for event in events if event != "[DONE]"]
    assert any(item.get("thinking") == "let me think this through" for item in parsed)
    assert any(item.get("text") == "final answer" for item in parsed)
    assert events[-1] == "[DONE]"


@pytest.mark.asyncio
async def test_chat_music_fast_path_bypasses_router_and_dispatches_music_tool(monkeypatch):
    class _UnexpectedGraph:
        async def astream(self, *_args, **_kwargs):
            raise AssertionError("graph should not run for explicit music commands")

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

    monkeypatch.setattr(main.app.state, "assistant_graph", _UnexpectedGraph(), raising=False)
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
    class _UnexpectedGraph:
        async def astream(self, *_args, **_kwargs):
            raise AssertionError("graph should not run for explicit music commands")

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

    monkeypatch.setattr(main.app.state, "assistant_graph", _UnexpectedGraph(), raising=False)
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
async def test_chat_graph_stream_uses_session_id_as_checkpoint(monkeypatch):
    session_id = "sess-checkpoint"
    main.memory_store.log_turn(session_id, "alice", "user", "hello")
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
    session_id = "sess-graph-state"
    main.memory_store.log_turn(session_id, "alice", "user", "hello")

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
    session_id = "sess-foreign"
    main.memory_store.log_turn(session_id, "alice", "user", "hello")

    response = await main.get_graph_state(session_id, _request("bob", None))

    assert response.status_code == 404
    assert _json_body(response)["code"] == "SESSION_NOT_FOUND"


@pytest.mark.asyncio
async def test_health_reports_embedding_router_state():
    # Ensure no embedding router is set on app.state.
    if hasattr(main.app.state, "embedding_router"):
        delattr(main.app.state, "embedding_router")

    payload = await main.health()

    assert payload["embedding_router"] is False
    assert "status" in payload


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
