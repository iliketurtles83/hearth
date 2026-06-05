from __future__ import annotations

import os
import sys
import tempfile
import types
from types import SimpleNamespace

import numpy as np
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
from embedding_router import (  # noqa: E402
    ClassifierResult,
    DualClassifierResult,
    EmbeddingRouterSnapshotMismatchError,
)


TEST_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "gemma3:4b")
TEST_CLOUD_MODEL = os.getenv("MODEL_CLOUD", "claude-sonnet-4-20250514")


class _FakeMemoryStore:
    def retrieve(self, _user_id: str, _query: str, _limit: int | None = None):
        return []

    def get_session_turns(
        self,
        _session_id: str,
        _user_id: str,
        _limit: int = 500,
    ):
        return []

    def get_latest_session_summary(
        self,
        _session_id: str,
        _user_id: str,
    ) -> str:
        return ""

    def log_turn(
        self,
        _session_id: str,
        _user_id: str,
        _role: str,
        _content: str,
    ) -> None:
        return None

    def ingest_user_message(self, _user_id: str, _message: str, _source: str = "text"):
        return {"status": "none", "saved": [], "blocked": [], "needs_confirmation": []}

    def count_unconsolidated(self, _user_id: str) -> int:
        return 0

    def consolidate_pending(self, _user_id=None, _limit: int = 50):
        return {}


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


def _deps_for_local_stream(
    chunks: list[str], *, embedding_router=None
) -> assistant_graph.AssistantGraphDependencies:
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
        embedding_router=embedding_router,
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
        embedding_router=None,
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
async def test_graph_yes_after_old_write_prompt_routes_normally():
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
        yield "normal yes response"

    async def _fake_stream_cloud(_system: str, _messages: list[dict]):
        yield "cloud"

    async def _fake_tool_dispatch(_tool_name: str, _params: dict):
        raise AssertionError("tool dispatch should not run in yes-routing test")

    # Keep legacy text in history to ensure it does not trigger removed
    # confirm-write routing behavior.
    class _HistoryMemoryStore(_FakeMemoryStore):
        def get_session_turns(self, _session_id, _user_id, _limit=500):
            return [
                {
                    "role": "assistant",
                    "content": "Type **yes** to confirm the write, or describe any changes you want first.",
                }
            ]

    deps = assistant_graph.AssistantGraphDependencies(
        memory_store=_HistoryMemoryStore(),
        embedding_router=None,
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

    result = await graph.ainvoke(state)
    assert result["intent"] == "quick-local"
    assert result["response_text"] == "normal yes response"


@pytest.mark.asyncio
async def test_force_code_uses_chat_model_local_route():
    graph = assistant_graph.build_assistant_graph(_deps_for_local_stream(["code answer"]))
    state = _base_state()
    state["message"] = "Refactor this sorting function"
    state["force_code"] = True

    result = await graph.ainvoke(state)

    assert result["intent"] == "code-question"
    assert result["route_type"] == "local"
    assert result["response_model"] == TEST_CHAT_MODEL
    assert result["response_text"] == "code answer"


@pytest.mark.asyncio
async def test_code_question_uses_chat_model_when_router_selects_code_intent(monkeypatch):
    class _CodeEmbeddingRouter:
        def classify_embedding(self, _query_embedding):
            return DualClassifierResult(
                tool=ClassifierResult(label="code", score=0.93, gap=0.30),
                dialogue=ClassifierResult(label="local", score=0.52, gap=0.08),
                should_escalate=False,
            )

    async def _fake_embed_text(*_args, **_kwargs):
        return np.array([0.2, 0.1, 0.3], dtype=np.float32)

    monkeypatch.setattr(assistant_graph, "ollama_embed_text", _fake_embed_text)

    graph = assistant_graph.build_assistant_graph(
        _deps_for_local_stream(["code via local chat"], embedding_router=_CodeEmbeddingRouter())
    )
    state = _base_state()
    state["message"] = "How does this SQL migration work?"

    result = await graph.ainvoke(state)

    assert result["intent"] == "code-question"
    assert result["route_type"] == "local"
    assert result["response_model"] == TEST_CHAT_MODEL
    assert result["response_text"] == "code via local chat"


# ── Slice 1 — Heuristic gate tests ───────────────────────────────────────────

def _deps_for_weather_stream(*, embedding_router=None) -> assistant_graph.AssistantGraphDependencies:
    """Deps suitable for weather-routing tests: tool_dispatch returns a valid weather result."""
    from tools.base import ToolResult

    async def _fake_stream_local(_request, model_name=None):
        yield "It is cloudy."

    async def _fake_stream_cloud(_system: str, _messages: list[dict]):
        yield "cloud"

    async def _fake_tool_dispatch(tool_name: str, params: dict):
        return ToolResult(
            ok=True,
            data={
                "location": "London, UK",
                "temperature": 12.0,
                "feels_like": 9.0,
                "humidity": 75,
                "wind_speed": 15.0,
                "condition": "Cloudy",
                "units": {"temperature": "°C", "wind_speed": "km/h"},
                "clothing": "Wear a jacket.",
            },
        )

    return assistant_graph.AssistantGraphDependencies(
        memory_store=_FakeMemoryStore(),
        embedding_router=embedding_router,
        router_route=lambda _m: None,
        stream_local=_fake_stream_local,
        stream_cloud=_fake_stream_cloud,
        tool_dispatch=_fake_tool_dispatch,
        chat_model=TEST_CHAT_MODEL,
        cloud_model=TEST_CLOUD_MODEL,
    )


@pytest.mark.asyncio
async def test_heuristic_routing_for_weather_without_planner():
    """High-confidence weather query should route correctly with no planner path."""

    graph = assistant_graph.build_assistant_graph(_deps_for_weather_stream())
    state = _base_state()
    state["message"] = "weather in london"

    result = await graph.ainvoke(state)

    assert result["intent"] == "external-data-needed"
    assert result["tool"] == "weather"


@pytest.mark.asyncio
async def test_embedding_router_absence_uses_heuristic_fallback(monkeypatch):
    """Without an embedding router, routing falls back to heuristic immediately."""

    async def _fake_stream_local(_request, model_name=None):
        yield "fallback response"

    async def _fake_stream_cloud(_system: str, _messages: list[dict]):
        yield "cloud"

    async def _fake_tool_dispatch(_tool_name: str, _params: dict):
        raise AssertionError("tool dispatch should not run in fallback test")

    deps = assistant_graph.AssistantGraphDependencies(
        memory_store=_FakeMemoryStore(),
        embedding_router=None,
        router_route=lambda _m: None,
        stream_local=_fake_stream_local,
        stream_cloud=_fake_stream_cloud,
        tool_dispatch=_fake_tool_dispatch,
        chat_model=TEST_CHAT_MODEL,
        cloud_model=TEST_CLOUD_MODEL,
    )

    # "hello" should route through the heuristic fallback when embedding is unavailable.
    graph = assistant_graph.build_assistant_graph(deps)
    state = _base_state()
    state["message"] = "hello"

    result = await graph.ainvoke(state)

    assert result["intent"] == "quick-local"
    assert result["planner_status"] == "heuristic"


@pytest.mark.asyncio
async def test_embedding_unambiguous_weather_skips_planner(monkeypatch):
    class _FakeEmbedRouter:
        def classify_embedding(self, _query_embedding):
            return DualClassifierResult(
                tool=ClassifierResult(
                    label="weather",
                    score=0.92,
                    second_label="none",
                    second_score=0.41,
                    gap=0.51,
                    ambiguous=False,
                ),
                dialogue=ClassifierResult(
                    label="local",
                    score=0.74,
                    second_label="memory-augmented",
                    second_score=0.30,
                    gap=0.44,
                    ambiguous=False,
                ),
                should_escalate=False,
            )

    async def _fake_embed_text(*_args, **_kwargs):
        return np.asarray([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(assistant_graph, "ollama_embed_text", _fake_embed_text)

    graph = assistant_graph.build_assistant_graph(
        _deps_for_weather_stream(embedding_router=_FakeEmbedRouter())
    )
    state = _base_state()
    state["message"] = "weather in london"

    result = await graph.ainvoke(state)

    assert result["intent"] == "external-data-needed"
    assert result["tool"] == "weather"
    assert result["planner_status"] == "embedding"


@pytest.mark.asyncio
async def test_embedding_ambiguous_uses_heuristic_fallback(monkeypatch):

    class _FakeEmbedRouter:
        def classify_embedding(self, _query_embedding):
            return DualClassifierResult(
                tool=ClassifierResult(
                    label="none",
                    score=0.52,
                    second_label="weather",
                    second_score=0.50,
                    gap=0.02,
                    ambiguous=True,
                ),
                dialogue=ClassifierResult(
                    label="local",
                    score=0.62,
                    second_label="cloud",
                    second_score=0.61,
                    gap=0.01,
                    ambiguous=True,
                ),
                should_escalate=True,
            )

    async def _fake_embed_text(*_args, **_kwargs):
        return np.asarray([0.1, 0.9], dtype=np.float32)

    monkeypatch.setattr(assistant_graph, "ollama_embed_text", _fake_embed_text)

    graph = assistant_graph.build_assistant_graph(
        _deps_for_local_stream(["ok"], embedding_router=_FakeEmbedRouter())
    )
    state = _base_state()
    state["message"] = "hello"

    result = await graph.ainvoke(state)

    assert result["intent"] == "quick-local"
    assert result["planner_status"] == "embedding_ambiguous_fallback"


@pytest.mark.asyncio
async def test_embedding_ambiguous_does_not_call_planner_function(monkeypatch):
    class _FakeEmbedRouter:
        def classify_embedding(self, _query_embedding):
            return DualClassifierResult(
                tool=ClassifierResult(
                    label="none",
                    score=0.55,
                    second_label="weather",
                    second_score=0.53,
                    gap=0.02,
                    ambiguous=True,
                ),
                dialogue=ClassifierResult(
                    label="local",
                    score=0.59,
                    second_label="cloud",
                    second_score=0.58,
                    gap=0.01,
                    ambiguous=True,
                ),
                should_escalate=True,
            )

    async def _fake_embed_text(*_args, **_kwargs):
        return np.asarray([0.2, 0.8], dtype=np.float32)

    assert not hasattr(assistant_graph, "_call_planner")
    monkeypatch.setattr(assistant_graph, "ollama_embed_text", _fake_embed_text)

    graph = assistant_graph.build_assistant_graph(
        _deps_for_local_stream(["ok"], embedding_router=_FakeEmbedRouter())
    )
    state = _base_state()
    state["message"] = "hello"

    result = await graph.ainvoke(state)

    assert result["intent"] == "quick-local"
    assert result["planner_status"] == "embedding_ambiguous_fallback"


@pytest.mark.asyncio
async def test_embedding_snapshot_mismatch_falls_back_to_legacy_router(monkeypatch):
    class _FakeEmbedRouter:
        def classify_embedding(self, _query_embedding):
            raise EmbeddingRouterSnapshotMismatchError("snapshot=old runtime=new")

    async def _fake_embed_text(*_args, **_kwargs):
        return np.asarray([0.1, 0.9], dtype=np.float32)

    monkeypatch.setattr(assistant_graph, "ollama_embed_text", _fake_embed_text)

    graph = assistant_graph.build_assistant_graph(
        _deps_for_local_stream(["ok"], embedding_router=_FakeEmbedRouter())
    )
    state = _base_state()
    state["message"] = "hello"

    result = await graph.ainvoke(state)

    assert result["intent"] == "quick-local"
    assert result["planner_status"] == "heuristic"
