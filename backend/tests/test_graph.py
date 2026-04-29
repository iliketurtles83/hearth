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


class _FakeMemoryStore:
    def retrieve(self, _user_id: str, _query: str):
        return []


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
            model="gemma3:4b",
            tool=None,
            planner_status="planner",
            reasoning_summary="short prompt",
            needs_memory=False,
        )

    async def _fake_stream_local(_request, model_name=None):
        assert model_name == "gemma3:4b"
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
        chat_model="gemma3:4b",
        cloud_model="claude-sonnet-4-20250514",
    )


@pytest.mark.asyncio
async def test_graph_ainvoke_returns_response_text_for_local_route():
    graph = assistant_graph.build_assistant_graph(_deps_for_local_stream(["hello", " world"]))

    result = await graph.ainvoke(_base_state())

    assert result["intent"] == "quick-local"
    assert result["route_type"] == "local"
    assert result["response_model"] == "gemma3:4b"
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
    assert meta["model"] == "gemma3:4b"
    assert chunks[1:] == [{"text": "hello"}, {"text": " world"}]