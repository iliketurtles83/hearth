"""Focused unit tests for the pure routing/response helpers in graph.py.

These helpers were extracted from the intent_classifier / responder node
closures so the decision and response-assembly logic can be tested in
isolation, without driving the full LangGraph.
"""
from __future__ import annotations

import os
import sys
import tempfile
import types


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


_tmp_dir = tempfile.mkdtemp(prefix="assistant-graph-helper-tests-")
os.environ["MEMORY_DB_PATH"] = os.path.join(_tmp_dir, "memory.db")
os.environ["CHROMA_PATH"] = os.path.join(_tmp_dir, "chroma")
os.environ["AUTH_DB_PATH"] = os.path.join(_tmp_dir, "auth.db")


import graph as assistant_graph  # noqa: E402
from intents import RouteDecision  # noqa: E402


CHAT = "local-chat-model"
CLOUD = "cloud-model"
VISION = "vision-model"


# ── _last_assistant_message ─────────────────────────────────────────────────

def test_last_assistant_message_returns_most_recent_assistant_turn():
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello there"},
        {"role": "user", "content": "how are you"},
        {"role": "assistant", "content": "I am well"},
        {"role": "user", "content": "thanks"},
    ]
    assert assistant_graph._last_assistant_message(history) == "I am well"


def test_last_assistant_message_returns_empty_when_none_present():
    assert assistant_graph._last_assistant_message([]) == ""
    assert assistant_graph._last_assistant_message([{"role": "user", "content": "hi"}]) == ""


# ── _looks_like_write_followup ──────────────────────────────────────────────

def test_write_followup_requires_confirmation_word_and_write_marker():
    assert assistant_graph._looks_like_write_followup("yes", "Shall I write the file?") is True
    assert assistant_graph._looks_like_write_followup("go ahead", "confirm the edit") is True


def test_write_followup_false_when_word_or_marker_missing():
    assert assistant_graph._looks_like_write_followup("yes", "Here is your answer.") is False
    assert assistant_graph._looks_like_write_followup("maybe", "Shall I write the file?") is False


# ── _downgrade_to_code_question ─────────────────────────────────────────────

def test_downgrade_to_code_question_sets_followup_status():
    decision = RouteDecision(intent="quick-local", confidence=0.5, use_cloud=True, model=CLOUD, tool="weather")
    result = assistant_graph._downgrade_to_code_question(
        decision, chat_model=CHAT, looks_like_write_followup=True
    )
    assert result.intent == "code-question"
    assert result.use_cloud is False
    assert result.tool is None
    assert result.model == CHAT
    assert result.planner_status == "write_followup_downgraded_to_code_question"


def test_downgrade_to_code_question_sets_write_status():
    decision = RouteDecision(intent="external-data-needed", confidence=0.6, use_cloud=False, model=CHAT, tool="weather")
    result = assistant_graph._downgrade_to_code_question(
        decision, chat_model=CHAT, looks_like_write_followup=False
    )
    assert result.planner_status == "write_downgraded_to_code_question"


def test_downgrade_to_code_question_is_noop_when_already_code_question():
    original = RouteDecision(
        intent="code-question",
        confidence=0.9,
        use_cloud=True,
        model=CLOUD,
        tool="weather",
        planner_status="embedding",
    )
    result = assistant_graph._downgrade_to_code_question(
        original, chat_model=CHAT, looks_like_write_followup=True
    )
    assert result is original
    assert result.use_cloud is True
    assert result.model == CLOUD
    assert result.tool == "weather"
    assert result.planner_status == "embedding"


# ── _heuristic_decision ─────────────────────────────────────────────────────

def test_heuristic_decision_picks_model_and_marks_heuristic():
    decision = RouteDecision(intent="vision", confidence=0.7, use_cloud=False, model="")
    result = assistant_graph._heuristic_decision(
        decision, chat_model=CHAT, cloud_model=CLOUD, vision_model=VISION
    )
    assert result.model == VISION
    assert result.planner_status == "heuristic"


def test_heuristic_decision_prefers_cloud_model_when_use_cloud():
    decision = RouteDecision(intent="reasoning-heavy", confidence=0.9, use_cloud=True, model="")
    result = assistant_graph._heuristic_decision(
        decision, chat_model=CHAT, cloud_model=CLOUD, vision_model=VISION
    )
    assert result.model == CLOUD
    assert result.planner_status == "heuristic"


# ── forced / deterministic decisions ────────────────────────────────────────

def test_forced_code_decision_shape():
    decision = assistant_graph._forced_code_decision(CHAT)
    assert decision.intent == "code-question"
    assert decision.confidence == 1.0
    assert decision.use_cloud is False
    assert decision.model == CHAT
    assert decision.tool is None
    assert decision.planner_status == "forced"
    assert decision.needs_memory is True
    assert assistant_graph._route_type_for_decision(decision) == "local"


def test_deterministic_vision_decision_shape():
    decision = assistant_graph._deterministic_vision_decision(VISION)
    assert decision.intent == "vision"
    assert decision.confidence == 1.0
    assert decision.use_cloud is False
    assert decision.model == VISION
    assert decision.tool is None
    assert decision.planner_status == "deterministic"
    assert decision.needs_memory is False


# ── _route_type_for_decision ────────────────────────────────────────────────

def test_route_type_is_tool_when_tool_present():
    decision = RouteDecision(intent="external-data-needed", confidence=0.8, use_cloud=False, model=CHAT, tool="weather")
    assert assistant_graph._route_type_for_decision(decision) == "tool"


def test_route_type_is_cloud_when_use_cloud():
    decision = RouteDecision(intent="reasoning-heavy", confidence=0.9, use_cloud=True, model=CLOUD)
    assert assistant_graph._route_type_for_decision(decision) == "cloud"


def test_route_type_is_local_by_default():
    decision = RouteDecision(intent="quick-local", confidence=0.5, use_cloud=False, model=CHAT)
    assert assistant_graph._route_type_for_decision(decision) == "local"


# ── _decision_meta / _decision_state_update ─────────────────────────────────

def _sample_decision() -> RouteDecision:
    return RouteDecision(
        intent="quick-local",
        confidence=0.62,
        use_cloud=False,
        model=CHAT,
        tool=None,
        planner_status="heuristic",
        reasoning_summary="note",
        needs_memory=False,
    )


def test_decision_meta_contains_expected_fields():
    meta = assistant_graph._decision_meta(_sample_decision(), "local")
    assert meta == {
        "model": CHAT,
        "intent": "quick-local",
        "confidence": 0.62,
        "route_type": "local",
        "needs_memory": False,
        "tool": None,
        "planner_status": "heuristic",
        "reasoning_summary": "note",
    }


def test_decision_state_update_contains_expected_fields():
    update = assistant_graph._decision_state_update(_sample_decision(), "local")
    assert update == {
        "intent": "quick-local",
        "confidence": 0.62,
        "use_cloud": False,
        "model": CHAT,
        "tool": None,
        "planner_status": "heuristic",
        "reasoning_summary": "note",
        "needs_memory": False,
        "route_type": "local",
    }


# ── _is_weather_fastpath ────────────────────────────────────────────────────

def test_is_weather_fastpath_true_for_plain_lookup():
    assert assistant_graph._is_weather_fastpath("weather", "weather in london") is True


def test_is_weather_fastpath_false_for_reasoning_or_other_tool():
    assert assistant_graph._is_weather_fastpath("weather", "should i go outside given the weather") is False
    assert assistant_graph._is_weather_fastpath("music", "weather in london") is False


# ── _build_vision_cloud_messages ────────────────────────────────────────────

def test_build_vision_cloud_messages_multimodal_shape():
    messages = assistant_graph._build_vision_cloud_messages("imgdata", "image/png", "what is this?")
    assert len(messages) == 1
    message = messages[0]
    assert message["role"] == "user"
    image_part, text_part = message["content"]
    assert image_part == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "imgdata"},
    }
    assert text_part == {"type": "text", "text": "what is this?"}


# ── _pick_model_for_decision ────────────────────────────────────────────────

def test_pick_model_for_decision_precedence():
    assert (
        assistant_graph._pick_model_for_decision(
            "vision", use_cloud=True, chat_model=CHAT, cloud_model=CLOUD, vision_model=VISION
        )
        == CLOUD
    )
    assert (
        assistant_graph._pick_model_for_decision(
            "vision", use_cloud=False, chat_model=CHAT, cloud_model=CLOUD, vision_model=VISION
        )
        == VISION
    )
    assert (
        assistant_graph._pick_model_for_decision(
            "quick-local", use_cloud=False, chat_model=CHAT, cloud_model=CLOUD, vision_model=VISION
        )
        == CHAT
    )
