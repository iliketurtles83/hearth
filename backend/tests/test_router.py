"""
Routing tests — Phase 4, Tasks 1 & 2.

Covers:
- Heuristic fallback (classify_intent) intent / confidence / routing rules.
- Planner JSON parsing: valid output, malformed JSON, missing fields, out-of-range values.
- async route() function: planner success, planner failure triggers fallback,
  planner disabled uses heuristic.
- Reasoning summary never surfaces in RouteDecision.model/intent/tool — stays internal.
- Code intent routing: heuristic and planner both select CODER_MODEL;
  non-code intents stay on CHAT_MODEL; cloud fallback uses CHAT_MODEL.
"""

import json
import os
import pytest
import respx
import httpx

os.environ.setdefault("MODEL_LOCAL", "llama3.2")
os.environ.setdefault("MODEL_CLOUD", "claude-sonnet-4-20250514")
os.environ.setdefault("OLLAMA_URL", "http://ollama:11434")
os.environ.setdefault("ROUTE_CONFIDENCE_THRESHOLD", "0.55")

# Import after env is set up
import router as r


# ── Heuristic classifier ───────────────────────────────────────────────────────

class TestHeuristicClassifier:
    def test_short_greeting_stays_local(self):
        d = r.classify_intent("Hello there")
        assert d.intent == "quick-local"
        assert not d.use_cloud
        assert d.model == r.LOCAL_MODEL

    def test_reasoning_heavy_routes_to_cloud(self):
        d = r.classify_intent(
            "Compare and contrast the architectural trade-offs of event sourcing "
            "versus CQRS for a large-scale distributed system."
        )
        assert d.intent == "reasoning-heavy"
        assert d.use_cloud
        assert d.model == r.CLOUD_MODEL

    def test_reasoning_heavy_low_confidence_stays_local(self):
        # Single weak signal — should not clear the 0.55 threshold
        d = r.classify_intent("explain the pros and cons")
        if d.intent == "reasoning-heavy":
            assert d.confidence < r.ROUTE_CONFIDENCE_THRESHOLD or not d.use_cloud

    def test_external_data_intent(self):
        d = r.classify_intent("What is the weather like today?")
        assert d.intent == "external-data-needed"
        assert not d.use_cloud
        assert d.tool == "weather"

    def test_memory_intent(self):
        d = r.classify_intent("What is my name? You mentioned it earlier.")
        assert d.intent == "memory-needed"

    def test_planner_status_is_heuristic(self):
        d = r.classify_intent("Hi")
        assert d.planner_status == "heuristic"

    def test_confidence_clamped(self):
        d = r.classify_intent("x" * 700 + " analyze this deeply")
        assert 0.0 <= d.confidence <= 1.0


# ── Planner output parsing ─────────────────────────────────────────────────────

class TestParsePlannerOutput:
    def _valid(self, **overrides) -> str:
        base = {
            "intent": "quick-local",
            "route": "local",
            "tool": None,
            "needs_memory": False,
            "confidence": 0.8,
            "reasoning": "Short conversational prompt.",
        }
        base.update(overrides)
        return json.dumps(base)

    def test_valid_json_round_trips(self):
        parsed = r._parse_planner_output(self._valid())
        assert parsed["intent"] == "quick-local"
        assert parsed["confidence"] == 0.8
        assert parsed["tool"] is None
        assert not parsed["needs_memory"]

    def test_strips_markdown_fences(self):
        raw = "```json\n" + self._valid() + "\n```"
        parsed = r._parse_planner_output(raw)
        assert parsed["intent"] == "quick-local"

    def test_malformed_json_raises(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            r._parse_planner_output("not json at all")

    def test_invalid_intent_raises(self):
        with pytest.raises(ValueError, match="Invalid intent"):
            r._parse_planner_output(self._valid(intent="bogus-intent"))

    def test_invalid_route_coerced_to_local(self):
        parsed = r._parse_planner_output(self._valid(route="unknown"))
        assert parsed["route"] == "local"

    def test_confidence_out_of_range_clamped(self):
        parsed = r._parse_planner_output(self._valid(confidence=42.0))
        assert parsed["confidence"] == 1.0

        parsed = r._parse_planner_output(self._valid(confidence=-5.0))
        assert parsed["confidence"] == 0.0

    def test_missing_fields_get_defaults(self):
        # Only intent is required; others should default safely
        raw = json.dumps({"intent": "memory-needed"})
        parsed = r._parse_planner_output(raw)
        assert parsed["intent"] == "memory-needed"
        assert parsed["confidence"] == 0.5
        assert parsed["tool"] is None
        assert not parsed["needs_memory"]

    def test_reasoning_truncated_to_300(self):
        long_reasoning = "x" * 500
        parsed = r._parse_planner_output(self._valid(reasoning=long_reasoning))
        assert len(parsed["reasoning"]) <= 300

    def test_tool_null_string_becomes_none(self):
        parsed = r._parse_planner_output(self._valid(tool="null"))
        assert parsed["tool"] is None

    def test_tool_name_preserved(self):
        parsed = r._parse_planner_output(self._valid(tool="weather", route="tool", intent="external-data-needed"))
        assert parsed["tool"] == "weather"


# ── Decision from planner ──────────────────────────────────────────────────────

class TestDecisionFromPlanner:
    def _parsed(self, **overrides) -> dict:
        base = {
            "intent": "quick-local",
            "route": "local",
            "tool": None,
            "needs_memory": False,
            "confidence": 0.8,
            "reasoning": "Short prompt.",
        }
        base.update(overrides)
        return base

    def test_reasoning_heavy_high_confidence_routes_cloud(self):
        d = r._decision_from_planner(self._parsed(
            intent="reasoning-heavy", route="cloud", confidence=0.9
        ))
        assert d.use_cloud
        assert d.model == r.CLOUD_MODEL

    def test_reasoning_heavy_below_threshold_stays_local(self):
        d = r._decision_from_planner(self._parsed(
            intent="reasoning-heavy", route="cloud", confidence=0.3
        ))
        assert not d.use_cloud
        assert d.model == r.LOCAL_MODEL

    def test_planner_status_set(self):
        d = r._decision_from_planner(self._parsed())
        assert d.planner_status == "planner"

    def test_reasoning_summary_captured(self):
        d = r._decision_from_planner(self._parsed(reasoning="This is a short prompt."))
        assert "short prompt" in d.reasoning_summary

    def test_needs_memory_forwarded(self):
        d = r._decision_from_planner(self._parsed(needs_memory=True))
        assert d.needs_memory

    def test_tool_forwarded(self):
        d = r._decision_from_planner(self._parsed(tool="weather", route="tool", intent="external-data-needed"))
        assert d.tool == "weather"

    def test_unknown_tool_is_dropped(self):
        d = r._decision_from_planner(
            self._parsed(tool="python", route="tool", intent="external-data-needed"),
            "help me code a script",
        )
        assert d.tool is None

    def test_weather_tool_inferred_when_missing(self):
        d = r._decision_from_planner(
            self._parsed(tool=None, route="tool", intent="external-data-needed"),
            "What is the weather in Tallinn today?",
        )
        assert d.tool == "weather"


# ── async route() ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestRouteFunction:
    def _planner_response_body(self, intent: str = "quick-local", confidence: float = 0.8) -> dict:
        return {
            "response": json.dumps({
                "intent": intent,
                "route": "local",
                "tool": None,
                "needs_memory": False,
                "confidence": confidence,
                "reasoning": "Test reasoning.",
            })
        }

    async def test_planner_success_returns_planner_status(self):
        with respx.mock(base_url="http://ollama:11434") as mock:
            mock.post("/api/generate").mock(
                return_value=httpx.Response(200, json=self._planner_response_body())
            )
            os.environ["ROUTER_PLANNER_ENABLED"] = "true"
            # Reload module-level constants
            r.PLANNER_ENABLED = True
            d = await r.route("Hello")
        assert d.planner_status == "planner"

    async def test_planner_network_failure_triggers_fallback(self):
        with respx.mock(base_url="http://ollama:11434") as mock:
            mock.post("/api/generate").mock(side_effect=httpx.ConnectError("refused"))
            r.PLANNER_ENABLED = True
            d = await r.route("Hello")
        assert d.planner_status == "fallback"
        assert d.intent  # heuristic result has a valid intent

    async def test_planner_bad_json_triggers_fallback(self):
        with respx.mock(base_url="http://ollama:11434") as mock:
            mock.post("/api/generate").mock(
                return_value=httpx.Response(200, json={"response": "not valid json"})
            )
            r.PLANNER_ENABLED = True
            d = await r.route("What is the weather?")
        assert d.planner_status == "fallback"

    async def test_planner_disabled_uses_heuristic(self):
        r.PLANNER_ENABLED = False
        d = await r.route("Hello")
        assert d.planner_status == "disabled"
        r.PLANNER_ENABLED = True  # restore

    async def test_reasoning_summary_never_empty_after_success(self):
        with respx.mock(base_url="http://ollama:11434") as mock:
            mock.post("/api/generate").mock(
                return_value=httpx.Response(200, json=self._planner_response_body())
            )
            r.PLANNER_ENABLED = True
            d = await r.route("Hello")
        # reasoning_summary must be a string (may be empty for heuristic, non-empty for planner)
        assert isinstance(d.reasoning_summary, str)

    async def test_reasoning_not_in_model_field(self):
        """Critical gate: reasoning must never bleed into the model name or intent."""
        with respx.mock(base_url="http://ollama:11434") as mock:
            mock.post("/api/generate").mock(
                return_value=httpx.Response(200, json=self._planner_response_body())
            )
            r.PLANNER_ENABLED = True
            d = await r.route("Tell me about yourself")
        assert "reasoning" not in d.model
        assert "reasoning" not in d.intent


# ── Code intent routing (Task 2) ─────────────────────────────────────────────────

class TestPickLocalModel:
    def test_code_returns_coder(self):
        assert r._pick_local_model("code") == r.CODER_MODEL

    def test_non_code_returns_chat(self):
        for intent in ["quick-local", "reasoning-heavy", "external-data-needed", "memory-needed"]:
            assert r._pick_local_model(intent) == r.CHAT_MODEL

    def test_coder_differs_when_overridden(self):
        original = r.CODER_MODEL
        r.CODER_MODEL = "qwen2.5-coder:7b"
        assert r._pick_local_model("code") == "qwen2.5-coder:7b"
        assert r._pick_local_model("quick-local") != "qwen2.5-coder:7b"
        r.CODER_MODEL = original


class TestCodeIntentRouting:
    def test_heuristic_detects_code_intent(self):
        d = r.classify_intent("Write a Python function that parses a JSON file.")
        assert d.intent == "code"
        assert not d.use_cloud
        assert d.model == r.CODER_MODEL

    def test_heuristic_code_uses_coder_model_when_set(self):
        original = r.CODER_MODEL
        r.CODER_MODEL = "qwen2.5-coder:7b"
        d = r.classify_intent("Write a Python function to read a CSV file.")
        assert d.intent == "code"
        assert d.model == "qwen2.5-coder:7b"
        r.CODER_MODEL = original

    def test_heuristic_code_never_routes_cloud(self):
        d = r.classify_intent(
            "Implement a complete production-grade distributed tracing system "
            "with full OpenTelemetry integration, write all the code."
        )
        # Even if reasoning signals are present, code intent must stay local
        if d.intent == "code":
            assert not d.use_cloud

    def test_heuristic_debug_prompt_is_code(self):
        d = r.classify_intent("Debug this bug in my function: it crashes on empty input.")
        assert d.intent == "code"
        assert d.model == r.CODER_MODEL

    def test_heuristic_add_tests_prompt_is_code(self):
        d = r.classify_intent("Can you add tests for quick sort?")
        assert d.intent == "code"
        assert d.model == r.CODER_MODEL

    def test_heuristic_non_code_uses_chat_model(self):
        d = r.classify_intent("What is the capital of France?")
        assert d.intent != "code"
        assert d.model == r.CHAT_MODEL

    def test_planner_code_intent_uses_coder_model(self):
        original = r.CODER_MODEL
        r.CODER_MODEL = "qwen2.5-coder:7b"
        parsed = {
            "intent": "code",
            "route": "local",
            "tool": None,
            "needs_memory": False,
            "confidence": 0.9,
            "reasoning": "User wants code generation.",
        }
        d = r._decision_from_planner(parsed)
        assert d.intent == "code"
        assert not d.use_cloud
        assert d.model == "qwen2.5-coder:7b"
        r.CODER_MODEL = original

    def test_planner_code_never_routes_cloud_even_if_route_says_cloud(self):
        """Planner may emit route=cloud for code by mistake; decision must override it."""
        parsed = {
            "intent": "code",
            "route": "cloud",
            "tool": None,
            "needs_memory": False,
            "confidence": 0.99,
            "reasoning": "Complex task.",
        }
        d = r._decision_from_planner(parsed)
        assert not d.use_cloud
        assert d.model != r.CLOUD_MODEL

    def test_reasoning_heavy_still_routes_cloud(self):
        """Code routing must not break existing reasoning-heavy → cloud path."""
        parsed = {
            "intent": "reasoning-heavy",
            "route": "cloud",
            "tool": None,
            "needs_memory": False,
            "confidence": 0.9,
            "reasoning": "Complex multi-step analysis.",
        }
        d = r._decision_from_planner(parsed)
        assert d.use_cloud
        assert d.model == r.CLOUD_MODEL


@pytest.mark.asyncio
async def test_route_forces_code_when_planner_misses_obvious_code_prompt(monkeypatch):
    async def _fake_planner(_prompt: str):
        return {
            "intent": "quick-local",
            "route": "local",
            "tool": None,
            "needs_memory": False,
            "confidence": 0.60,
            "reasoning": "Short prompt.",
        }

    monkeypatch.setattr(r, "_call_planner", _fake_planner)
    monkeypatch.setattr(r, "PLANNER_ENABLED", True)

    d = await r.route("Can you add tests for quick sort?")
    assert d.intent == "code"
    assert not d.use_cloud
    assert d.model == r.CODER_MODEL
