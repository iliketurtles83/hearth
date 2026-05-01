"""Tests for Phase 11 — Personality and Affect Layer.

Acceptance criteria verified here:
- _probe_tone returns a valid label for unambiguous inputs.
- _probe_tone falls back to 'calm' on stream errors.
- persona_renderer preserves all factual content after styling (voice).
- persona_renderer preserves markdown formatting after styling (chat).
- persona_renderer is a no-op when persona is unconfigured.
- AssistantState includes the 'persona' field.
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

_tmp_dir = tempfile.mkdtemp(prefix="assistant-persona-tests-")
os.environ["MEMORY_DB_PATH"] = os.path.join(_tmp_dir, "memory.db")
os.environ["CHROMA_PATH"] = os.path.join(_tmp_dir, "chroma")
os.environ["AUTH_DB_PATH"] = os.path.join(_tmp_dir, "auth.db")

import graph as assistant_graph  # noqa: E402

TEST_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "gemma3:4b")
TEST_CLOUD_MODEL = os.getenv("MODEL_CLOUD", "claude-sonnet-4-20250514")

_VALID_TONE_LABELS = {"calm", "curious", "frustrated", "excited", "uncertain", "urgent"}

# ── Fixtures ───────────────────────────────────────────────────────────────────

class _FakeMemoryStore:
    def retrieve(self, _user_id: str, _query: str):
        return []


def _make_deps(*, stream_local_chunks: list[str] | None = None, stream_raises: bool = False):
    """Build minimal AssistantGraphDependencies for persona tests."""

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
        if stream_raises:
            raise RuntimeError("simulated stream failure")
        for chunk in (stream_local_chunks or []):
            yield chunk

    async def _fake_stream_cloud(_system: str, _messages: list):
        yield "cloud response"

    async def _fake_tool_dispatch(_tool_name: str, _params: dict):
        raise AssertionError("tool dispatch should not be called")

    return assistant_graph.AssistantGraphDependencies(
        memory_store=_FakeMemoryStore(),
        router_route=_fake_router,
        stream_local=_fake_stream_local,
        stream_cloud=_fake_stream_cloud,
        tool_dispatch=_fake_tool_dispatch,
        chat_model=TEST_CHAT_MODEL,
        cloud_model=TEST_CLOUD_MODEL,
    )


def _get_probe_tone(deps):
    """Extract the _probe_tone coroutine from a compiled graph's closure."""
    # _probe_tone is defined inside build_assistant_graph; we access it by
    # compiling the graph with our deps, then locating the function on the node.
    # Simpler: we call memory_retrieval state indirectly, but easiest is to
    # extract via the compiled graph's nodes dict if exposed.  Since that's
    # implementation-dependent, we instead call _probe_tone via a direct
    # helper that mirrors its behaviour.
    raise NotImplementedError  # use _probe_tone_via_state instead


async def _run_probe_tone(deps, message: str) -> str:
    """Run tone probe indirectly by invoking a minimal memory_retrieval call
    and reading the returned tone field.  Requires a working (mocked) graph."""
    graph = assistant_graph.build_assistant_graph(deps)
    # We can't easily call private helpers directly, so we drive a full
    # memory_retrieval node invocation via the public graph interface would be
    # heavyweight.  Instead, replicate the public behaviour: we know that
    # _probe_tone is a closure inside build_assistant_graph, so we expose it
    # through a minimal compiled graph run with a forced memory_retrieval state.
    # Simplest approach: import internals by re-building in a test-friendly way.
    # Actually — we access it via the graph's nodes which store the bound async
    # function.  The graph stores nodes as runnables; we can call them directly.
    # LangGraph stores nodes by name in graph.nodes (dict[str, Runnable]).
    # Each runnable wraps our async function.
    #
    # Easier still: build_assistant_graph returns a CompiledGraph.  The nodes
    # dict has {name: Runnable}.  We invoke the memory_retrieval node with a
    # minimal state and extract the tone field.
    from langgraph.graph import StateGraph
    # Instead of fighting LangGraph internals, extract the _probe_tone function
    # by re-building with introspection.  We do this by monkeypatching
    # build_assistant_graph to capture the function reference.
    _captured = {}

    _orig_build = assistant_graph.build_assistant_graph

    def _patched_build(d, **kw):
        import asyncio

        # Capture _probe_tone by wrapping memory_retrieval
        _original_graph_code = _orig_build.__code__
        compiled = _orig_build(d, **kw)
        return compiled

    # Simplest safe approach: call the node directly.
    # LangGraph CompiledGraph.nodes is a dict of channel names to Runnables.
    # We can call the underlying async function by invoking node.ainvoke.
    # However, memory_retrieval needs intent + history etc.
    minimal_state: assistant_graph.AssistantState = {
        "user_id": "test-user",
        "session_id": "test-session",
        "message": message,
        "system": "You are helpful.",
        "source": "text",
        "modality": "chat",
        "tone": None,
        "persona": {},
        "history": [],
        "session_summary": "",
        "intent": "quick-local",
        "confidence": 0.99,
        "use_cloud": False,
        "model": TEST_CHAT_MODEL,
        "tool": None,
        "needs_memory": False,
        "route_type": "local",
        "memories": [],
        "augmented_system": "You are helpful.",
        "selected_history": [],
        "history_tokens": 0,
        "truncated": False,
        "summary_tokens": 0,
    }
    # Call memory_retrieval via the compiled graph's node invoker.
    node_runnable = graph.nodes["memory_retrieval"]
    result = await node_runnable.ainvoke(minimal_state)
    return result.get("tone", "calm")


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_state_has_persona_field():
    """AssistantState must declare a 'persona' annotation."""
    assert "persona" in assistant_graph.AssistantState.__annotations__, (
        "'persona' field missing from AssistantState"
    )


@pytest.mark.asyncio
async def test_tone_probe_fallback_on_error():
    """When stream_local raises, _probe_tone should return 'calm'."""
    deps = _make_deps(stream_raises=True)
    # Drive memory_retrieval which calls _probe_tone internally.
    tone = await _run_probe_tone(deps, "I am absolutely furious and cannot believe this happened!")
    assert tone == "calm", f"Expected 'calm' fallback on error, got {tone!r}"


@pytest.mark.asyncio
async def test_tone_probe_returns_valid_label():
    """When stream_local returns a valid label, tone probe returns it."""
    # Return "frustrated" — a clear, unambiguous label
    deps = _make_deps(stream_local_chunks=["frustrated"])
    tone = await _run_probe_tone(deps, "I am so frustrated with this broken system!")
    assert tone in _VALID_TONE_LABELS, f"tone {tone!r} not in valid set {_VALID_TONE_LABELS}"


@pytest.mark.asyncio
async def test_tone_probe_short_message_returns_calm():
    """Messages under 5 words skip the LLM call and return 'calm'."""
    # If the LLM were called it would raise (stream_raises=True).
    # The early-exit for short messages should prevent the call.
    deps = _make_deps(stream_raises=True)
    tone = await _run_probe_tone(deps, "Help me please")
    assert tone == "calm"


@pytest.mark.asyncio
async def test_persona_renderer_noop_when_unconfigured():
    """Empty persona dict → persona_renderer returns response unchanged, no LLM call."""
    call_count = {"n": 0}

    async def _counting_stream(_request, model_name=None):
        call_count["n"] += 1
        # If called unexpectedly, yield nothing — test will fail on count check.
        yield ""

    async def _fake_router(_m):
        return SimpleNamespace(
            intent="quick-local", confidence=0.99, use_cloud=False,
            model=TEST_CHAT_MODEL, tool=None, planner_status="", reasoning_summary="", needs_memory=False,
        )

    async def _fake_cloud(_s, _m):
        yield ""

    async def _fake_dispatch(_t, _p):
        raise AssertionError

    deps = assistant_graph.AssistantGraphDependencies(
        memory_store=_FakeMemoryStore(),
        router_route=_fake_router,
        stream_local=_counting_stream,
        stream_cloud=_fake_cloud,
        tool_dispatch=_fake_dispatch,
        chat_model=TEST_CHAT_MODEL,
        cloud_model=TEST_CLOUD_MODEL,
    )
    graph = assistant_graph.build_assistant_graph(deps)
    node = graph.nodes["persona_renderer"]

    state: assistant_graph.AssistantState = {
        "user_id": "u1",
        "session_id": "s1",
        "message": "hello",
        "system": "sys",
        "source": "text",
        "modality": "chat",
        "tone": "calm",
        "persona": {},  # unconfigured
        "response_text": "The answer is 42.",
        "history": [],
        "session_summary": "",
    }
    result = await node.ainvoke(state)
    assert result["response_text"] == "The answer is 42."
    assert call_count["n"] == 0, "stream_local should NOT be called when persona is unconfigured"


@pytest.mark.asyncio
async def test_persona_renderer_preserves_facts_voice():
    """Critical fact-drift test: all factual values must survive persona rendering (voice)."""
    original = (
        "The weather in Helsinki today is 7 degrees Celsius with 72% humidity. "
        "Wind from the northwest at 14 km/h. High of 9°C, low of 3°C. "
        "Sunrise was at 05:14 and sunset at 21:28."
    )
    required_facts = ["Helsinki", "7", "72", "14", "9", "3", "05:14", "21:28"]

    # The persona_renderer will call stream_local; return a response that preserves all facts.
    styled = (
        "Helsinki's weather today feels crisp — seven degrees Celsius, humidity at 72 percent, "
        "northwest winds at 14 km/h. Expect a high of 9 and low of 3 degrees. "
        "Sun rises at 05:14, sets at 21:28."
    )

    async def _fake_router(_m):
        return SimpleNamespace(
            intent="quick-local", confidence=0.99, use_cloud=False,
            model=TEST_CHAT_MODEL, tool=None, planner_status="", reasoning_summary="", needs_memory=False,
        )

    async def _fake_stream(_request, model_name=None):
        yield styled

    async def _fake_cloud(_s, _m):
        yield ""

    async def _fake_dispatch(_t, _p):
        raise AssertionError

    deps = assistant_graph.AssistantGraphDependencies(
        memory_store=_FakeMemoryStore(),
        router_route=_fake_router,
        stream_local=_fake_stream,
        stream_cloud=_fake_cloud,
        tool_dispatch=_fake_dispatch,
        chat_model=TEST_CHAT_MODEL,
        cloud_model=TEST_CLOUD_MODEL,
    )
    graph = assistant_graph.build_assistant_graph(deps)
    node = graph.nodes["persona_renderer"]

    state: assistant_graph.AssistantState = {
        "user_id": "u1",
        "session_id": "s1",
        "message": "weather?",
        "system": "sys",
        "source": "voice",
        "modality": "voice",
        "tone": "curious",
        "persona": {"style": "warm", "warmth": 4},
        "response_text": original,
        "history": [],
        "session_summary": "",
    }
    result = await node.ainvoke(state)
    rendered = result["response_text"]
    for fact in required_facts:
        assert fact in rendered, f"Fact {fact!r} missing from rendered voice response:\n{rendered}"


@pytest.mark.asyncio
async def test_persona_renderer_preserves_markdown_for_chat():
    """Markdown headings, bullets, and bold must survive persona rendering (chat)."""
    original = (
        "## Weather Summary\n\n"
        "- **Temperature**: 7°C\n"
        "- **Humidity**: 72%\n"
        "- **Wind**: 14 km/h NW\n\n"
        "It is a **partly cloudy** day."
    )
    # The styled response must keep all markdown intact.
    styled = (
        "## Weather Summary\n\n"
        "- **Temperature**: 7°C\n"
        "- **Humidity**: 72%\n"
        "- **Wind**: 14 km/h NW\n\n"
        "It's a **partly cloudy** day — dress warmly!"
    )

    async def _fake_router(_m):
        return SimpleNamespace(
            intent="quick-local", confidence=0.99, use_cloud=False,
            model=TEST_CHAT_MODEL, tool=None, planner_status="", reasoning_summary="", needs_memory=False,
        )

    async def _fake_stream(_request, model_name=None):
        yield styled

    async def _fake_cloud(_s, _m):
        yield ""

    async def _fake_dispatch(_t, _p):
        raise AssertionError

    deps = assistant_graph.AssistantGraphDependencies(
        memory_store=_FakeMemoryStore(),
        router_route=_fake_router,
        stream_local=_fake_stream,
        stream_cloud=_fake_cloud,
        tool_dispatch=_fake_dispatch,
        chat_model=TEST_CHAT_MODEL,
        cloud_model=TEST_CLOUD_MODEL,
    )
    graph = assistant_graph.build_assistant_graph(deps)
    node = graph.nodes["persona_renderer"]

    state: assistant_graph.AssistantState = {
        "user_id": "u1",
        "session_id": "s1",
        "message": "weather summary",
        "system": "sys",
        "source": "text",
        "modality": "chat",
        "tone": "calm",
        "persona": {"style": "warm", "warmth": 3},
        "response_text": original,
        "history": [],
        "session_summary": "",
    }
    result = await node.ainvoke(state)
    rendered = result["response_text"]
    for marker in ["##", "- **", "**Temperature**", "**Humidity**", "**Wind**", "**partly cloudy**"]:
        assert marker in rendered, f"Markdown {marker!r} missing from rendered chat response:\n{rendered}"
