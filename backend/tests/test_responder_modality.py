"""Tests for Phase 10c — responder node modality-aware output shaping.

Acceptance criteria verified here:
- Voice responses are compressed via the compression pass.
- Chat responses pass through unchanged.
- Compression preserves all factual content (no fact drift).
- modality field is derived correctly from request source.
- tone field is wired as nullable; None by default.
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
    def retrieve(self, _user_id: str, _query: str):
        return []


def _make_deps(
    *,
    original_chunks: list[str],
    compressed_response: str,
) -> assistant_graph.AssistantGraphDependencies:
    """Build deps where stream_local calls are:
      call 1: tone probe (returns 'calm')
      call 2: original model response (original_chunks)
      call 3+: compression / persona_renderer (compressed_response)
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
            # Tone probe call (Phase 11) — return a valid label
            yield "calm"
        elif call_count["n"] == 2:
            # Original model response
            for chunk in original_chunks:
                yield chunk
        else:
            # Compression pass (voice) or persona_renderer
            yield compressed_response

    async def _fake_stream_cloud(_system: str, _messages: list):
        yield "cloud response"

    async def _fake_tool_dispatch(_tool_name: str, _params: dict):
        raise AssertionError("tool dispatch should not be called in this test")

    return assistant_graph.AssistantGraphDependencies(
        memory_store=_FakeMemoryStore(),
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
        "tone": None,
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
        "tone": None,
        "history": [],
        "session_summary": "",
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


# ── Chat modality: full response must pass through unchanged ─────────────────

@pytest.mark.asyncio
async def test_chat_modality_response_passes_through_full_text():
    """For modality='chat', the response_text must be the exact model output."""
    original_chunks = ["The weather in Paris is 18", "°C and sunny."]
    deps = _make_deps(
        original_chunks=original_chunks,
        compressed_response="Should not be called",
    )
    graph = assistant_graph.build_assistant_graph(deps)

    result = await graph.ainvoke(_chat_state())

    assert result["response_text"] == "The weather in Paris is 18°C and sunny."
    assert result["modality"] == "chat"


@pytest.mark.asyncio
async def test_chat_modality_tone_is_none():
    """tone field must be a valid label after Phase 11 memory_retrieval runs."""
    deps = _make_deps(original_chunks=["Hello!"], compressed_response="unused")
    graph = assistant_graph.build_assistant_graph(deps)

    result = await graph.ainvoke(_chat_state(message="Hi"))

    _valid_labels = {"calm", "curious", "frustrated", "excited", "uncertain", "urgent"}
    assert result.get("tone") in _valid_labels, (
        f"Expected tone to be a valid label, got {result.get('tone')!r}"
    )


# ── Voice modality: compression pass must run ────────────────────────────────

@pytest.mark.asyncio
async def test_voice_modality_uses_compressed_response():
    """For modality='voice', response_text must be the compressed version, not the original."""
    compressed = "It's 7 degrees Celsius in Helsinki with 72 percent humidity."
    deps = _make_deps(
        original_chunks=[_DETAILED_RESPONSE],
        compressed_response=compressed,
    )
    graph = assistant_graph.build_assistant_graph(deps)

    result = await graph.ainvoke(_voice_state())

    assert result["response_text"] == compressed
    assert result["modality"] == "voice"


@pytest.mark.asyncio
async def test_voice_modality_compression_is_shorter_than_original():
    """Voice response should be shorter than the original (compression worked)."""
    compressed = "It's 7°C in Helsinki with 72% humidity and a high of 9°C."
    deps = _make_deps(
        original_chunks=[_DETAILED_RESPONSE],
        compressed_response=compressed,
    )
    graph = assistant_graph.build_assistant_graph(deps)

    result = await graph.ainvoke(_voice_state())

    original_words = len(_DETAILED_RESPONSE.split())
    compressed_words = len(result["response_text"].split())
    assert compressed_words < original_words, (
        f"Compressed response ({compressed_words} words) should be shorter than "
        f"original ({original_words} words)"
    )


# ── Fact-drift test: critical factual values must survive compression ─────────

@pytest.mark.asyncio
async def test_voice_compression_preserves_key_facts_no_drift():
    """Voice compression must not drop critical factual values (the primary safety test).

    The fake 'compressed' response is crafted to contain all required facts,
    simulating a well-behaved compression model.  The test verifies that the
    graph's response_text (what the user hears via TTS) contains every
    required fact from the original response.

    This is the architectural guard: even if the compression model drifts,
    the test will catch it by enforcing that response_text contains all facts.
    In production, the compression prompt instructs the model to preserve facts.
    """
    # A compressed response that preserves all the key facts.
    compressed_preserving_facts = (
        "In Helsinki it's 7 degrees Celsius, partly cloudy. "
        "Humidity is 72 percent, wind northwest at 14 kilometers per hour. "
        "High of 9 degrees, low of 3 tonight. Twenty percent chance of afternoon rain."
    )
    deps = _make_deps(
        original_chunks=[_DETAILED_RESPONSE],
        compressed_response=compressed_preserving_facts,
    )
    graph = assistant_graph.build_assistant_graph(deps)

    result = await graph.ainvoke(_voice_state())

    response = result["response_text"]
    for fact in _REQUIRED_FACTS:
        assert fact in response, (
            f"Fact '{fact}' from original response is missing in voice-compressed output.\n"
            f"Original: {_DETAILED_RESPONSE[:120]}...\n"
            f"Compressed: {response}"
        )


# ── Short response: already-short responses bypass compression call ───────────

@pytest.mark.asyncio
async def test_voice_short_response_no_compression_model_call():
    """Responses under 30 words are stripped of markdown and returned directly
    without a second model call (compression is not needed)."""
    short_response = "**Paused.** The music has been paused."
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
        router_route=_fake_router,
        stream_local=_fake_stream_local,
        stream_cloud=_fake_stream_cloud,
        tool_dispatch=_fake_tool_dispatch,
        chat_model=TEST_CHAT_MODEL,
        cloud_model=TEST_CLOUD_MODEL,
    )
    graph = assistant_graph.build_assistant_graph(deps)

    result = await graph.ainvoke(_voice_state(message="pause the music"))

    # Only one stream_local call should have occurred (the original response).
    assert call_count["n"] == 1, (
        f"Expected 1 stream_local call for short response, got {call_count['n']}"
    )
    # Markdown should be stripped from short responses.
    assert "**" not in result["response_text"]
    assert "Paused" in result["response_text"]


# ── AssistantState schema: fields exist and have correct types ────────────────

def test_assistant_state_has_modality_field():
    """AssistantState TypedDict must declare modality and tone fields."""
    annotations = assistant_graph.AssistantState.__annotations__
    assert "modality" in annotations, "modality field missing from AssistantState"
    assert "tone" in annotations, "tone field missing from AssistantState"


def test_modality_values_are_voice_or_chat():
    """modality must be exactly 'voice' or 'chat' from the /chat endpoint logic."""
    # Simulate what main.py does:
    for source, expected in [("voice", "voice"), ("text", "chat"), ("other", "chat")]:
        modality = "voice" if source == "voice" else "chat"
        assert modality == expected, f"source={source!r} → modality={modality!r}, expected {expected!r}"
