"""Tests for LLM-native tool calling within the assistant graph."""

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

_tmp_dir = tempfile.mkdtemp(prefix="assistant-tool-tests-")
os.environ["MEMORY_DB_PATH"] = os.path.join(_tmp_dir, "memory.db")
os.environ["CHROMA_PATH"] = os.path.join(_tmp_dir, "chroma")
os.environ["AUTH_DB_PATH"] = os.path.join(_tmp_dir, "auth.db")

import graph as assistant_graph
from tools.base import ToolResult

TEST_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gemma-4")
TEST_CLOUD_MODEL = os.getenv("MODEL_CLOUD", "claude-sonnet-4-20250514")


class _FakeMemoryStore:
    def __init__(self):
        self._turn_count = 0

    def retrieve(self, _user_id: str, _query: str, _limit: int | None = None):
        return []

    def get_session_turns(self, _session_id: str, _user_id: str, _limit: int = 500):
        return []

    def get_latest_session_summary(self, _session_id: str, _user_id: str) -> str:
        return ""

    def log_turn(self, _session_id: str, _user_id: str, _role: str, _content: str) -> None:
        self._turn_count += 1

    def ingest_user_message(self, _user_id: str, _message: str, _source: str = "text"):
        return {"status": "none", "saved": [], "blocked": [], "needs_confirmation": []}

    def count_unconsolidated(self, _user_id: str) -> int:
        return 0

    def count_session_turns(self, _session_id: str, _user_id: str) -> int:
        return self._turn_count

    def save_summary(self, _user_id: str, _session_id: str, _summary: str) -> int:
        return 1

    def consolidate_pending(self, _user_id=None, _limit: int = 50):
        return {}


def _base_state() -> assistant_graph.AssistantState:
    return {
        "user_id": "alice",
        "session_id": "tool-session",
        "message": "What is the weather in Tallinn?",
        "system": "You are a helpful assistant.",
        "source": "text",
        "history": [],
        "session_summary": "",
    }


@pytest.mark.asyncio
async def test_llm_native_weather_tool_calling():
    """Graph responder executes tool call when the local model yields tool_calls."""
    dispatched = []

    async def _fake_stream_local(req, model_name=None):
        # Simulate local LLM returning tool call for weather
        yield {
            "tool_calls": [
                {
                    "index": 0,
                    "function": {
                        "name": "weather",
                        "arguments": '{"location": "Tallinn"}',
                    },
                }
            ]
        }

    async def _fake_stream_cloud(_s, _m):
        yield "cloud"

    async def _fake_tool_dispatch(tool_name: str, params: dict):
        dispatched.append((tool_name, params))
        return ToolResult(
            ok=True,
            data={
                "location": "Tallinn, Estonia",
                "temperature": 15.0,
                "feels_like": 14.0,
                "condition": "Partly Cloudy",
                "humidity": 65,
                "units": {"temperature": "°C", "wind_speed": "km/h"},
                "clothing": "Light jacket.",
            },
        )

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
    graph = assistant_graph.build_assistant_graph(deps)

    state = _base_state()
    state["history"] = [
        {"role": "user", "content": "How's the weather in Tallinn today?"},
        {"role": "assistant", "content": "It is currently 14°C and cloudy in Tallinn."},
    ]
    state["message"] = "What about tomorrow?"

    result = await graph.ainvoke(state)

    assert len(dispatched) == 1
    assert dispatched[0][0] == "weather"
    assert dispatched[0][1]["location"] == "Tallinn"
    assert "Tallinn" in result["response_text"]
    assert "15" in result["response_text"]


@pytest.mark.asyncio
async def test_llm_native_music_tool_calling():
    """Graph responder executes music tool call when the local model yields tool_calls."""
    dispatched = []

    async def _fake_stream_local(req, model_name=None):
        yield {
            "tool_calls": [
                {
                    "index": 0,
                    "function": {
                        "name": "music",
                        "arguments": '{"action": "play", "query": "jazz"}',
                    },
                }
            ]
        }

    async def _fake_stream_cloud(_s, _m):
        yield "cloud"

    async def _fake_tool_dispatch(tool_name: str, params: dict):
        dispatched.append((tool_name, params))
        return ToolResult(
            ok=True,
            data={
                "action": "play",
                "track": {"title": "Autumn Leaves", "artist": "Miles Davis"},
                "tracks": None,
            },
        )

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
    graph = assistant_graph.build_assistant_graph(deps)

    state = _base_state()
    state["message"] = "play something chill like jazz"
    result = await graph.ainvoke(state)

    assert len(dispatched) == 1
    assert dispatched[0][0] == "music"
    assert dispatched[0][1]["action"] == "play"
    assert dispatched[0][1]["query"] == "jazz"
    assert 'Now playing: "Autumn Leaves" by Miles Davis.' in result["response_text"]
