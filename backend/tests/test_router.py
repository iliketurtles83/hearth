"""
Routing tests for the heuristic classifier and compatibility wrapper.

Covers:
- Heuristic fallback intent / confidence / routing rules.
- Tool inference for weather and music requests.
- Code intent stays local and uses the coder model.
- code-question intent (explain/how-does) routes to code_tool.
- code-write intent (write/fix/implement) routes to external coding agent.
- Async route() remains a thin wrapper over classify_intent().
"""

import os

import pytest

os.environ.setdefault("MODEL_LOCAL", "llama3.2")
os.environ.setdefault("MODEL_CLOUD", "claude-sonnet-4-20250514")
os.environ.setdefault("ROUTE_CONFIDENCE_THRESHOLD", "0.55")

# Import after env is set up
import router as r


class TestHeuristicClassifier:
    def test_short_greeting_stays_local(self):
        d = r.classify_intent("Hello there")
        assert d.intent == "quick-local"
        assert not d.use_cloud
        assert d.model == r.LOCAL_MODEL
        assert d.planner_status == "heuristic"

    def test_reasoning_heavy_routes_to_cloud(self):
        d = r.classify_intent(
            "Compare and contrast the architectural trade-offs of event sourcing "
            "versus CQRS for a large-scale distributed system."
        )
        assert d.intent == "reasoning-heavy"
        assert d.use_cloud
        assert d.model == r.CLOUD_MODEL

    def test_reasoning_heavy_low_confidence_stays_local(self):
        d = r.classify_intent("explain the pros and cons")
        if d.intent == "reasoning-heavy":
            assert d.confidence < r.ROUTE_CONFIDENCE_THRESHOLD or not d.use_cloud

    def test_external_data_intent_weather(self):
        d = r.classify_intent("What is the weather like today?")
        assert d.intent == "external-data-needed"
        assert not d.use_cloud
        assert d.tool == "weather"

    def test_external_data_intent_music(self):
        d = r.classify_intent("Play music by Miles Davis")
        assert d.intent == "external-data-needed"
        assert not d.use_cloud
        assert d.tool == "music"

    def test_memory_intent(self):
        d = r.classify_intent("What is my name? You mentioned it earlier.")
        assert d.intent == "memory-needed"

    def test_code_write_intent(self):
        d = r.classify_intent("Write a Python function that parses a JSON file.")
        assert d.intent == "code-write"
        assert not d.use_cloud
        assert d.model == r.CODER_MODEL

    def test_code_question_intent(self):
        d = r.classify_intent("Can you explain how this function works?")
        assert d.intent == "code-question"
        assert not d.use_cloud
        assert d.model == r.CODER_MODEL

    def test_fix_bug_is_code_write(self):
        d = r.classify_intent("Fix the bug in the authentication module.")
        assert d.intent == "code-write"
        assert not d.use_cloud
        assert d.model == r.CODER_MODEL

    def test_explain_is_code_question(self):
        d = r.classify_intent("Walk me through how the memory retrieval node works.")
        assert d.intent == "code-question"
        assert d.model == r.CODER_MODEL

    def test_confidence_clamped(self):
        d = r.classify_intent("x" * 700 + " analyze this deeply")
        assert 0.0 <= d.confidence <= 1.0

    def test_coding_solution_is_code_question(self):
        d = r.classify_intent("What is a coding solution for creating an api call parser in python?")
        assert d.intent == "code-question"
        assert d.model == r.CODER_MODEL

    def test_how_do_i_implement_is_code_question(self):
        d = r.classify_intent("How do I implement a rate limiter in python?")
        assert d.intent in {"code-question", "code-write"}
        assert d.model == r.CODER_MODEL


class TestModelSelection:
    def test_code_write_returns_coder(self):
        assert r._pick_local_model("code-write") == r.CODER_MODEL

    def test_code_question_returns_coder(self):
        assert r._pick_local_model("code-question") == r.CODER_MODEL

    def test_non_code_returns_chat(self):
        for intent in ["quick-local", "reasoning-heavy", "external-data-needed", "memory-needed"]:
            assert r._pick_local_model(intent) == r.CHAT_MODEL

    def test_coder_differs_when_overridden(self):
        original = r.CODER_MODEL
        r.CODER_MODEL = "qwen2.5-coder:7b"
        assert r._pick_local_model("code-write") == "qwen2.5-coder:7b"
        assert r._pick_local_model("code-question") == "qwen2.5-coder:7b"
        assert r._pick_local_model("quick-local") != "qwen2.5-coder:7b"
        r.CODER_MODEL = original


class TestAsyncRouteWrapper:
    @pytest.mark.asyncio
    async def test_route_matches_classifier(self):
        classified = r.classify_intent("Hello there")
        routed = await r.route("Hello there")
        assert routed == classified
        assert routed.planner_status == "heuristic"

    @pytest.mark.asyncio
    async def test_route_handles_code_write_intent(self):
        routed = await r.route("Write a Python function to read a CSV file.")
        assert routed.intent == "code-write"
        assert routed.model == r.CODER_MODEL
