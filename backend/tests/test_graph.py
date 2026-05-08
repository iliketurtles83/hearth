from __future__ import annotations

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


_tmp_dir = tempfile.mkdtemp(prefix="assistant-graph-tests-")
os.environ["MEMORY_DB_PATH"] = os.path.join(_tmp_dir, "memory.db")
os.environ["CHROMA_PATH"] = os.path.join(_tmp_dir, "chroma")
os.environ["AUTH_DB_PATH"] = os.path.join(_tmp_dir, "auth.db")


import graph as assistant_graph  # noqa: E402


TEST_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "gemma3:4b")
TEST_CLOUD_MODEL = os.getenv("MODEL_CLOUD", "claude-sonnet-4-20250514")


class _FakeMemoryStore:
    def retrieve(self, _user_id: str, _query: str):
        return []


@pytest.fixture(autouse=True)
def _fake_planner(monkeypatch):
    async def _planner(prompt: str):
        return {
            "intent": "quick-local",
            "route": "local",
            "tool": None,
            "needs_memory": False,
            "confidence": 0.99,
            "reasoning": f"stubbed planner for {prompt}",
        }

    monkeypatch.setattr(assistant_graph, "_call_planner", _planner)


def _base_state() -> assistant_graph.AssistantState:
    return {
        "user_id": "alice",
        "session_id": "session-1",
        "message": "hello",
        "system": "You are a helpful assistant.",
        "source": "text",
        "history": [],
        "session_summary": "",
    }


def _deps_for_local_stream(chunks: list[str]) -> assistant_graph.AssistantGraphDependencies:
    async def _fake_router(_message: str):
        return SimpleNamespace(
            intent="quick-local",
            confidence=0.99,
            use_cloud=False,
            model=TEST_CHAT_MODEL,
            tool=None,
            planner_status="planner",
            reasoning_summary="short prompt",
            needs_memory=False,
        )

    async def _fake_stream_local(_request, model_name=None):
        assert model_name == TEST_CHAT_MODEL
        for chunk in chunks:
            yield chunk

    async def _fake_stream_cloud(_system: str, _messages: list[dict]):
        yield "cloud"

    async def _fake_tool_dispatch(_tool_name: str, _params: dict):
        raise AssertionError("tool dispatch should not run in local smoke tests")

    return assistant_graph.AssistantGraphDependencies(
        memory_store=_FakeMemoryStore(),
        router_route=_fake_router,
        stream_local=_fake_stream_local,
        stream_cloud=_fake_stream_cloud,
        tool_dispatch=_fake_tool_dispatch,
        chat_model=TEST_CHAT_MODEL,
        cloud_model=TEST_CLOUD_MODEL,
    )


@pytest.mark.asyncio
async def test_graph_ainvoke_returns_response_text_for_local_route():
    graph = assistant_graph.build_assistant_graph(_deps_for_local_stream(["hello", " world"]))

    result = await graph.ainvoke(_base_state())

    assert result["intent"] == "quick-local"
    assert result["route_type"] == "local"
    assert result["response_model"] == TEST_CHAT_MODEL
    assert result["response_text"] == "hello world"


@pytest.mark.asyncio
async def test_graph_astream_custom_emits_response_chunks_with_async_sqlite_checkpointer():
    checkpoint_path = os.path.join(_tmp_dir, "graph-checkpoints.sqlite")
    deps = _deps_for_local_stream(["hello", " world"])

    async with assistant_graph.create_assistant_graph(deps, checkpoint_path=checkpoint_path) as graph:
        chunks: list[dict] = []
        async for item in graph.astream(
            _base_state(),
            config=assistant_graph.checkpoint_config("stream-session"),
            stream_mode="custom",
        ):
            chunks.append(item)

    assert len(chunks) == 3
    assert "meta" in chunks[0]
    meta = chunks[0]["meta"]
    assert meta["intent"] == "quick-local"
    assert meta["model"] == TEST_CHAT_MODEL
    assert chunks[1:] == [{"text": "hello"}, {"text": " world"}]


@pytest.mark.asyncio
async def test_graph_checkpoint_resume_reloads_state_without_reexecution():
    checkpoint_path = os.path.join(_tmp_dir, "graph-checkpoints-resume.sqlite")
    calls = {"router": 0, "local": 0}

    async def _fake_router(_message: str):
        calls["router"] += 1
        return SimpleNamespace(
            intent="quick-local",
            confidence=0.99,
            use_cloud=False,
            model=TEST_CHAT_MODEL,
            tool=None,
            planner_status="planner",
            reasoning_summary="short prompt",
            needs_memory=False,
        )

    async def _fake_stream_local(_request, model_name=None):
        assert model_name == TEST_CHAT_MODEL
        calls["local"] += 1
        yield "checkpoint"
        yield " state"

    async def _fake_stream_cloud(_system: str, _messages: list[dict]):
        yield "cloud"

    async def _fake_tool_dispatch(_tool_name: str, _params: dict):
        raise AssertionError("tool dispatch should not run in checkpoint resume test")

    deps = assistant_graph.AssistantGraphDependencies(
        memory_store=_FakeMemoryStore(),
        router_route=_fake_router,
        stream_local=_fake_stream_local,
        stream_cloud=_fake_stream_cloud,
        tool_dispatch=_fake_tool_dispatch,
        chat_model=TEST_CHAT_MODEL,
        cloud_model=TEST_CLOUD_MODEL,
    )
    config = assistant_graph.checkpoint_config("resume-session")

    async with assistant_graph.create_assistant_graph(deps, checkpoint_path=checkpoint_path) as graph:
        result = await graph.ainvoke(_base_state(), config=config)

    assert result["response_text"] == "checkpoint state"
    assert calls == {"router": 0, "local": 1}

    async with assistant_graph.create_assistant_graph(deps, checkpoint_path=checkpoint_path) as graph_restarted:
        snapshot = await graph_restarted.aget_state(config)

    assert snapshot.values["response_text"] == "checkpoint state"
    assert snapshot.values["intent"] == "quick-local"
    # Snapshot read should not execute planner/LLM nodes again.
    assert calls == {"router": 0, "local": 1}


@pytest.mark.asyncio
async def test_graph_orphan_yes_after_write_prompt_returns_no_pending_write():
    async def _fake_router(_message: str):
        return SimpleNamespace(
            intent="quick-local",
            confidence=0.99,
            use_cloud=False,
            model=TEST_CHAT_MODEL,
            tool=None,
            planner_status="planner",
            reasoning_summary="short prompt",
            needs_memory=False,
        )

    async def _fake_stream_local(_request, model_name=None):
        yield "should not run"

    async def _fake_stream_cloud(_system: str, _messages: list[dict]):
        yield "should not run"

    async def _fake_tool_dispatch(_tool_name: str, _params: dict):
        raise AssertionError("tool dispatch should not run in confirm-write orphan test")

    deps = assistant_graph.AssistantGraphDependencies(
        memory_store=_FakeMemoryStore(),
        router_route=_fake_router,
        stream_local=_fake_stream_local,
        stream_cloud=_fake_stream_cloud,
        tool_dispatch=_fake_tool_dispatch,
        chat_model=TEST_CHAT_MODEL,
        cloud_model=TEST_CLOUD_MODEL,
    )

    graph = assistant_graph.build_assistant_graph(deps)
    state = _base_state()
    state["message"] = "yes"
    state["history"] = [
        {
            "role": "assistant",
            "content": "Type **yes** to confirm the write, or describe any changes you want first.",
        }
    ]
    state["pending_write"] = {}
    state["awaiting_confirmation"] = False

    result = await graph.ainvoke(state)
    assert result["intent"] == "confirm_write"
    assert result["response_text"] == "No pending write to execute."