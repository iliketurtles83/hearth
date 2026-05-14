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


# ── Phase 13 — coding agent integration ───────────────────────────────────────

def _deps_for_code_write() -> assistant_graph.AssistantGraphDependencies:
    """Return minimal deps for coding-agent tests (stream_local not needed)."""

    async def _fake_stream_local(_request, model_name=None):
        yield "should not stream in coding agent gate tests"

    async def _fake_stream_cloud(_system: str, _messages: list[dict]):
        yield "cloud"

    async def _fake_tool_dispatch(_tool_name: str, _params: dict):
        raise AssertionError("tool dispatch should not run in coding agent tests")

    return assistant_graph.AssistantGraphDependencies(
        memory_store=_FakeMemoryStore(),
        router_route=lambda _m: None,  # unused — routing goes through _call_planner
        stream_local=_fake_stream_local,
        stream_cloud=_fake_stream_cloud,
        tool_dispatch=_fake_tool_dispatch,
        chat_model=TEST_CHAT_MODEL,
        cloud_model=TEST_CLOUD_MODEL,
    )


def _planner_returning(intent: str):
    """Return an async planner stub that emits the given intent."""
    async def _planner(prompt: str):
        return {
            "intent": intent,
            "route": "local",
            "tool": None,
            "needs_memory": False,
            "confidence": 0.9,
            "reasoning": f"test stub for {intent}",
        }
    return _planner


@pytest.mark.asyncio
async def test_code_write_routes_to_coding_agent_tool(monkeypatch):
    """code-write intent should trigger the confirmation gate, not the executor."""
    monkeypatch.setattr(assistant_graph, "_call_planner", _planner_returning("code-write"))

    graph = assistant_graph.build_assistant_graph(_deps_for_code_write())
    state = _base_state()
    state["message"] = "Write a Python function that sorts a list of integers."

    result = await graph.ainvoke(state)

    assert result["intent"] == "code-write"
    assert result["awaiting_agent_confirmation"] is True
    assert result["pending_code_task"] == state["message"]
    # response_text should contain a confirmation prompt
    assert "yes" in result["response_text"].lower()


@pytest.mark.asyncio
async def test_code_write_voice_prompt_is_short(monkeypatch):
    """Voice confirmation prompt should be ≤ 12-word preview, not full task."""
    monkeypatch.setattr(assistant_graph, "_call_planner", _planner_returning("code-write"))

    graph = assistant_graph.build_assistant_graph(_deps_for_code_write())
    long_task = "Write " + " ".join([f"word{i}" for i in range(30)])
    state = _base_state()
    state["message"] = long_task
    state["modality"] = "voice"

    result = await graph.ainvoke(state)

    # The preview should be truncated — full task (30+ extra words) must not appear verbatim
    assert result["pending_code_task"] == long_task
    assert long_task not in result["response_text"]
    assert "..." in result["response_text"]


@pytest.mark.asyncio
async def test_confirm_agent_task_routes_to_executor(monkeypatch):
    """'yes' with pending agent task should invoke coding_agent_executor."""
    import tools.coding_agent as _coding_agent_mod
    from tools.base import ToolResult

    async def _mock_agent_run(params: dict) -> ToolResult:
        assert params["task"] == "Add type hints to utils.py"
        return ToolResult(
            ok=True,
            data={"result": "Done.", "files_changed": ["utils.py"], "status": "success"},
        )

    monkeypatch.setattr(_coding_agent_mod, "run", _mock_agent_run)

    graph = assistant_graph.build_assistant_graph(_deps_for_code_write())
    state = _base_state()
    state["message"] = "yes"
    state["awaiting_agent_confirmation"] = True
    state["pending_code_task"] = "Add type hints to utils.py"

    result = await graph.ainvoke(state)

    assert result["intent"] == "confirm_agent_task"
    assert result["awaiting_agent_confirmation"] is False
    assert result["pending_code_task"] == ""
    assert "utils.py" in result["response_text"]


@pytest.mark.asyncio
async def test_orphan_yes_without_agent_pending_does_not_trigger_executor():
    """'yes' without awaiting_agent_confirmation must not enter coding_agent_executor."""

    async def _fake_stream_local(_request, model_name=None):
        yield "normal response"

    async def _fake_stream_cloud(_system: str, _messages: list[dict]):
        yield "cloud"

    async def _fake_tool_dispatch(_tool_name: str, _params: dict):
        raise AssertionError("tool dispatch should not run")

    deps = assistant_graph.AssistantGraphDependencies(
        memory_store=_FakeMemoryStore(),
        router_route=lambda _m: None,
        stream_local=_fake_stream_local,
        stream_cloud=_fake_stream_cloud,
        tool_dispatch=_fake_tool_dispatch,
        chat_model=TEST_CHAT_MODEL,
        cloud_model=TEST_CLOUD_MODEL,
    )
    graph = assistant_graph.build_assistant_graph(deps)
    state = _base_state()
    state["message"] = "yes"
    state["awaiting_agent_confirmation"] = False
    state["pending_code_task"] = ""

    result = await graph.ainvoke(state)

    # Should NOT be confirm_agent_task — routes normally
    assert result.get("intent") != "confirm_agent_task"
    assert result.get("awaiting_agent_confirmation") is not True


# ── Slice 1 — Heuristic gate tests ───────────────────────────────────────────

def _deps_for_weather_stream() -> assistant_graph.AssistantGraphDependencies:
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
        router_route=lambda _m: None,
        stream_local=_fake_stream_local,
        stream_cloud=_fake_stream_cloud,
        tool_dispatch=_fake_tool_dispatch,
        chat_model=TEST_CHAT_MODEL,
        cloud_model=TEST_CLOUD_MODEL,
    )


@pytest.mark.asyncio
async def test_heuristic_gate_skips_planner_for_weather(monkeypatch):
    """High-confidence weather query must bypass _call_planner entirely."""
    planner_calls: list[str] = []

    async def _spy_planner(prompt: str):
        planner_calls.append(prompt)
        # Return a deliberately wrong intent — if this runs, the assertion below fails.
        return {
            "intent": "quick-local",
            "route": "local",
            "tool": None,
            "needs_memory": False,
            "confidence": 0.99,
            "reasoning": "spy",
        }

    monkeypatch.setattr(assistant_graph, "_call_planner", _spy_planner)

    graph = assistant_graph.build_assistant_graph(_deps_for_weather_stream())
    state = _base_state()
    state["message"] = "weather in london"

    result = await graph.ainvoke(state)

    assert result["intent"] == "external-data-needed"
    assert planner_calls == [], "Planner must not be called when heuristic gate fires"


@pytest.mark.asyncio
async def test_planner_failure_uses_precomputed_heuristic(monkeypatch):
    """When _call_planner raises, the pre-computed heuristic is used without re-running classify_intent."""
    async def _failing_planner(prompt: str):
        raise RuntimeError("simulated planner timeout")

    monkeypatch.setattr(assistant_graph, "_call_planner", _failing_planner)

    async def _fake_stream_local(_request, model_name=None):
        yield "fallback response"

    async def _fake_stream_cloud(_system: str, _messages: list[dict]):
        yield "cloud"

    async def _fake_tool_dispatch(_tool_name: str, _params: dict):
        raise AssertionError("tool dispatch should not run in fallback test")

    deps = assistant_graph.AssistantGraphDependencies(
        memory_store=_FakeMemoryStore(),
        router_route=lambda _m: None,
        stream_local=_fake_stream_local,
        stream_cloud=_fake_stream_cloud,
        tool_dispatch=_fake_tool_dispatch,
        chat_model=TEST_CHAT_MODEL,
        cloud_model=TEST_CLOUD_MODEL,
    )

    # "hello" → quick-local (not in _HEURISTIC_GATE) so planner is attempted then fails.
    graph = assistant_graph.build_assistant_graph(deps)
    state = _base_state()
    state["message"] = "hello"

    result = await graph.ainvoke(state)

    assert result["intent"] == "quick-local"
    assert result["planner_status"] == "fallback"