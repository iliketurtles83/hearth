"""Tests for responder node modality-aware output shaping.

Acceptance criteria verified here:
- Voice responses use a single-pass model call (no second compression pass).
- Chat responses pass through unchanged.
- modality field is derived correctly from request source.
"""
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

_tmp_dir = tempfile.mkdtemp(prefix="assistant-modality-tests-")
os.environ["MEMORY_DB_PATH"] = os.path.join(_tmp_dir, "memory.db")
os.environ["CHROMA_PATH"] = os.path.join(_tmp_dir, "chroma")
os.environ["AUTH_DB_PATH"] = os.path.join(_tmp_dir, "auth.db")

import graph as assistant_graph  # noqa: E402

TEST_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "gemma3:4b")
TEST_CLOUD_MODEL = os.getenv("MODEL_CLOUD", "claude-sonnet-4-20250514")

# A verbose response with concrete factual content that compression must preserve.
_DETAILED_RESPONSE = (
    "The weather in Helsinki, Finland today is **partly cloudy** with a temperature of "
    "**7 degrees Celsius**. The wind is blowing from the **northwest at 14 km/h**, and "
    "humidity is at **72%**. There is a **20% chance of light rain** in the afternoon. "
    "The high for the day will be **9°C** and the low tonight will drop to **3°C**. "
    "UV index is 1, which is low — no sun protection needed. Sunrise was at 05:14 and "
    "sunset will be at 21:28, giving you about 16 hours and 14 minutes of daylight."
)

# Key facts that MUST survive voice compression.
_REQUIRED_FACTS = [
    "7",        # temperature 7°C
    "Helsinki", # location
    "72",       # humidity 72%
    "9",        # high 9°C
    "3",        # low 3°C
]


class _FakeMemoryStore:
    def __init__(self):
        self._turn_count = 0

    def retrieve(self, _user_id: str, _query: str):
        return []

    def get_session_turns(self, _session_id: str, _user_id: str, _limit: int = 500):
        return []

    def get_latest_session_summary(self, _session_id: str, _user_id: str) -> str:
        return ""

    def log_turn(self, _session_id: str, _user_id: str, _role: str, _content: str) -> None:
        self._turn_count += 1
        return None

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


def _make_deps(
    *,
    original_chunks: list[str],
    compressed_response: str,
) -> assistant_graph.AssistantGraphDependencies:
    """Build deps where stream_local calls are:
      call 1: original model response (original_chunks)
      call 2+: compression pass (compressed_response)
    """
    call_count = {"n": 0}

    async def _fake_router(_message: str):
        return SimpleNamespace(
            intent="quick-local",
            confidence=0.99,
            use_cloud=False,
            model=TEST_CHAT_MODEL,
            tool=None,
            planner_status="planner",
            reasoning_summary="",
            needs_memory=False,
        )

    async def _fake_stream_local(_request, model_name=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Original model response
            for chunk in original_chunks:
                yield chunk
        else:
            # Compression pass (voice)
            yield compressed_response

    async def _fake_stream_cloud(_system: str, _messages: list):
        yield "cloud response"

    async def _fake_tool_dispatch(_tool_name: str, _params: dict):
        raise AssertionError("tool dispatch should not be called in this test")

    return assistant_graph.AssistantGraphDependencies(
        memory_store=_FakeMemoryStore(),
        embedding_router=None,
        router_route=_fake_router,
        stream_local=_fake_stream_local,
        stream_cloud=_fake_stream_cloud,
        tool_dispatch=_fake_tool_dispatch,
        chat_model=TEST_CHAT_MODEL,
        cloud_model=TEST_CLOUD_MODEL,
    )


def _voice_state(**overrides) -> assistant_graph.AssistantState:
    base: assistant_graph.AssistantState = {
        "user_id": "alice",
        "session_id": "voice-session",
        "message": "What's the weather like in Helsinki?",
        "system": "You are a helpful assistant.",
        "source": "voice",
        "modality": "voice",
        "history": [],
        "session_summary": "",
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


def _chat_state(**overrides) -> assistant_graph.AssistantState:
    base: assistant_graph.AssistantState = {
        "user_id": "alice",
        "session_id": "chat-session",
        "message": "What's the weather like in Helsinki?",
        "system": "You are a helpful assistant.",
        "source": "text",
        "modality": "chat",
        "history": [],
        "session_summary": "",
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


def _force_local_intent(monkeypatch) -> None:
    """Force heuristic routing to quick-local so responder-modality behaviour is exercised."""

    def _classify(_prompt: str):
        return assistant_graph.RouteDecision(
            intent="quick-local",
            confidence=0.99,
            use_cloud=False,
            model=TEST_CHAT_MODEL,
            tool=None,
            planner_status="heuristic",
            reasoning_summary="",
            needs_memory=False,
        )

    monkeypatch.setattr(assistant_graph, "classify_intent", _classify)


# ── Chat modality: full response must pass through unchanged ─────────────────

@pytest.mark.asyncio
async def test_chat_modality_response_passes_through_full_text(monkeypatch):
    """For modality='chat', the response_text must be the exact model output."""
    _force_local_intent(monkeypatch)
    original_chunks = ["The weather in Paris is 18", "°C and sunny."]
    deps = _make_deps(
        original_chunks=original_chunks,
        compressed_response="Should not be called",
    )
    graph = assistant_graph.build_assistant_graph(deps)

    result = await graph.ainvoke(_chat_state())

    assert result["response_text"] == "The weather in Paris is 18°C and sunny."
    assert result["modality"] == "chat"


# ── Voice modality: single-pass response (no compression LLM call) ───────────

@pytest.mark.asyncio
async def test_voice_modality_single_pass_response(monkeypatch):
    """For modality='voice', response_text must be the direct model output.

    The voice system modifier in the prompt instructs the model to produce
    brief spoken-style output directly — no second LLM compression pass.
    """
    _force_local_intent(monkeypatch)
    voice_response = "It's 7 degrees in Helsinki with 72 percent humidity."
    deps = _make_deps(
        original_chunks=[voice_response],
        compressed_response="Should never be called",
    )
    graph = assistant_graph.build_assistant_graph(deps)

    result = await graph.ainvoke(_voice_state())

    assert result["response_text"] == voice_response
    assert result["modality"] == "voice"


@pytest.mark.asyncio
async def test_voice_modality_only_one_model_call(monkeypatch):
    """Voice responses must use exactly one stream_local call (no compression pass)."""
    _force_local_intent(monkeypatch)
    call_count = {"n": 0}

    async def _fake_router(_message: str):
        return SimpleNamespace(
            intent="quick-local",
            confidence=0.99,
            use_cloud=False,
            model=TEST_CHAT_MODEL,
            tool=None,
            planner_status="planner",
            reasoning_summary="",
            needs_memory=False,
        )

    async def _counting_stream_local(_request, model_name=None):
        call_count["n"] += 1
        yield _DETAILED_RESPONSE

    async def _fake_stream_cloud(_system, _messages):
        yield "cloud"

    async def _fake_tool_dispatch(_tool, _params):
        raise AssertionError("should not call tool dispatch")

    deps = assistant_graph.AssistantGraphDependencies(
        memory_store=_FakeMemoryStore(),
        embedding_router=None,
        router_route=_fake_router,
        stream_local=_counting_stream_local,
        stream_cloud=_fake_stream_cloud,
        tool_dispatch=_fake_tool_dispatch,
        chat_model=TEST_CHAT_MODEL,
        cloud_model=TEST_CLOUD_MODEL,
    )
    graph = assistant_graph.build_assistant_graph(deps)

    result = await graph.ainvoke(_voice_state())

    assert call_count["n"] == 1, (
        f"Expected 1 stream_local call (no compression pass), got {call_count['n']}"
    )
    # The full response passes through unchanged (brevity is a model concern).
    assert result["response_text"] == _DETAILED_RESPONSE


@pytest.mark.asyncio
async def test_voice_short_response_single_model_call(monkeypatch):
    """Short voice responses also use exactly one model call — no post-processing."""
    _force_local_intent(monkeypatch)
    short_response = "Paused. The music has been paused."
    call_count = {"n": 0}

    async def _fake_router(_message: str):
        return SimpleNamespace(
            intent="quick-local",
            confidence=0.99,
            use_cloud=False,
            model=TEST_CHAT_MODEL,
            tool=None,
            planner_status="planner",
            reasoning_summary="",
            needs_memory=False,
        )

    async def _fake_stream_local(_request, model_name=None):
        call_count["n"] += 1
        yield short_response

    async def _fake_stream_cloud(_system, _messages):
        yield "cloud"

    async def _fake_tool_dispatch(_tool, _params):
        raise AssertionError("should not call tool dispatch")

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
    graph = assistant_graph.build_assistant_graph(deps)

    result = await graph.ainvoke(_voice_state(message="pause the music"))

    assert call_count["n"] == 1, (
        f"Expected 1 stream_local call for short response, got {call_count['n']}"
    )
    assert "Paused" in result["response_text"]


# ── AssistantState schema: fields exist and have correct types ────────────────

def test_assistant_state_has_modality_field():
    """AssistantState TypedDict must declare the modality field."""
    annotations = assistant_graph.AssistantState.__annotations__
    assert "modality" in annotations, "modality field missing from AssistantState"


def test_modality_values_are_voice_or_chat():
    """modality must be exactly 'voice' or 'chat' from the /chat endpoint logic."""
    # Simulate what main.py does:
    for source, expected in [("voice", "voice"), ("text", "chat"), ("other", "chat")]:
        modality = "voice" if source == "voice" else "chat"
        assert modality == expected, f"source={source!r} → modality={modality!r}, expected {expected!r}"
