"""
Intent-based model router (Phase 4).

Intent categories:
  quick-local          — factual, short, conversational
  reasoning-heavy      — multi-step planning, analysis, architecture
  code                 — code generation, debugging, explanation, file editing
  external-data-needed — weather, news, live data (tool path, not cloud)
  memory-needed        — references to prior facts or user preferences

Routing rule:
  reasoning-heavy with confidence >= ROUTE_CONFIDENCE_THRESHOLD → cloud
  everything else → local
  cloud unavailable → local with caller-visible fallback flag

Inner-monologue planner (Phase 4b):
  When ROUTER_PLANNER_ENABLED=true (default), a short reasoning pass asks the
  local model to emit a structured JSON routing decision before dispatch.
  The reasoning_summary is logged server-side only — never sent to the client.
  On any failure (timeout / network / bad JSON), falls back to heuristic classifier.
"""

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field

import httpx

log = logging.getLogger("assistant.router")

# ── Model config ───────────────────────────────────────────────────────────────
# Chat model:  OLLAMA_CHAT_MODEL → MODEL_LOCAL → "llama3.2"
# Coder model: OLLAMA_CODER_MODEL → OLLAMA_CHAT_MODEL → MODEL_LOCAL → "llama3.2"
# LOCAL_MODEL is a backward-compatible alias for CHAT_MODEL.
CHAT_MODEL: str = (
    os.getenv("OLLAMA_CHAT_MODEL")
    or os.getenv("MODEL_LOCAL")
    or "llama3.2"
)
CODER_MODEL: str = (
    os.getenv("OLLAMA_CODER_MODEL")
    or os.getenv("OLLAMA_CHAT_MODEL")
    or os.getenv("MODEL_LOCAL")
    or "llama3.2"
)
LOCAL_MODEL = CHAT_MODEL  # backward-compat alias
CLOUD_MODEL = os.getenv("MODEL_CLOUD", "claude-sonnet-4-20250514")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434").rstrip("/")

# ── Routing thresholds ─────────────────────────────────────────────────────────
# Minimum confidence score in "reasoning-heavy" required to route to cloud.
ROUTE_CONFIDENCE_THRESHOLD = float(os.getenv("ROUTE_CONFIDENCE_THRESHOLD", "0.55"))

# ── Planner controls ───────────────────────────────────────────────────────────
PLANNER_ENABLED: bool = os.getenv("ROUTER_PLANNER_ENABLED", "true").lower() == "true"
PLANNER_TIMEOUT_MS: int = int(os.getenv("ROUTER_PLANNER_TIMEOUT_MS", "4000"))
PLANNER_MAX_TOKENS: int = int(os.getenv("ROUTER_PLANNER_MAX_TOKENS", "200"))
PLANNER_TEMPERATURE: float = float(os.getenv("ROUTER_PLANNER_TEMPERATURE", "0.0"))

_VALID_INTENTS = frozenset(
    ["quick-local", "reasoning-heavy", "code", "external-data-needed", "memory-needed"]
)
_VALID_ROUTES = frozenset(["local", "cloud", "tool"])


@dataclass
class RouteDecision:
    intent: str           # quick-local | reasoning-heavy | code | external-data-needed | memory-needed
    confidence: float     # 0.0–1.0
    use_cloud: bool
    model: str
    tool: str | None = None          # tool name when route == "tool", else None
    needs_memory: bool = False       # planner signal: memory retrieval recommended
    planner_status: str = "heuristic"  # planner | fallback | heuristic | disabled
    reasoning_summary: str = ""      # server-log only, never sent to client


def _pick_local_model(intent: str) -> str:
    """Return the appropriate local Ollama model for a given intent.

    Code intents use CODER_MODEL; all others use CHAT_MODEL.
    """
    return CODER_MODEL if intent == "code" else CHAT_MODEL


# ── Pattern banks ──────────────────────────────────────────────────────────────

_REASONING_PATTERNS = [
    r"\barchitect\w*\b",
    r"\bdesign\b.{0,30}\bsystem\b",
    r"\b(pros\s+and\s+cons|trade.?offs?)\b",
    r"\bstep.by.step\b",
    r"\b(refactor|restructure)\b.{0,40}\b(entire|whole|codebase|project)\b",
    r"\bexplain\b.{0,30}\b(in\s+depth|in\s+detail|thoroughly|deeply)\b",
    r"\bwrite\s+(a\s+)?(full|complete|entire|comprehensive)\b",
    r"\b(analyze|analyse|analysis)\b",
    r"\b(compare|contrast)\b.{0,30}\b(approach|option|solution|method)\b",
    r"\b(implementation\s+plan|system\s+design|design\s+pattern)\b",
    r"\b(multi.?step|multi.?part|break\s+it\s+down)\b",
]

_REASONING_KEYWORDS = [
    "architecture", "refactor", "design pattern", "compare and contrast",
    "deep dive", "comprehensive plan", "write a full", "write a complete",
    "explain in detail", "step by step", "analyze this", "analyse this",
]

_EXTERNAL_DATA_KEYWORDS = [
    "weather", "forecast", "temperature", "rain", "snow", "humidity",
    "news", "headline", "current events", "latest news",
    "stock", "share price", "market today", "market now",
    "what time is it", "what date is it", "live score", "real-time",
]

_EXTERNAL_DATA_PATTERNS = [
    r"\b(weather|forecast|temperature|rain|snow|sunny|humidity)\b",
    r"\b(news|headlines|current\s+events|latest\s+news)\b",
    r"\b(stock|share\s+price|market)\b.{0,20}\b(today|now|current)\b",
    r"\bwhat\s+(time|date)\s+is\s+it\b",
    r"\b(live|real.?time)\b.{0,20}\b(data|score|result)\b",
]

_MEMORY_PATTERNS = [
    r"\b(remember|recall)\b.{0,30}\b(i|my|me)\b",
    r"\byou\s+(said|mentioned|told\s+me)\b",
    r"\b(earlier|last\s+time|previously|before)\b.{0,30}\b(said|mentioned|told)\b",
    r"\bmy\s+(name|preference|setting|favorite|favourit|location|city)\b",
    r"\blike\s+i\s+(said|mentioned|told)\b",
]

_MEMORY_KEYWORDS = [
    "remember", "recall", "you said", "you mentioned", "you told me",
    "earlier", "last time", "previously", "before",
    "my name", "my preference", "my favorite", "my favourite", "my city",
]

_CODE_PATTERNS = [
    r"\b(write|create|generate|implement|build)\b.{0,40}\b(function|class|method|script|endpoint|api|module|test)\b",
    r"\b(debug|fix)\b.{0,30}\b(bug|error|exception|crash|issue|problem)\b",
    r"\b(refactor|optimise|optimize|rewrite)\b.{0,30}\b(function|class|code|snippet)\b",
    r"\bwrite\s+(a\s+)?(python|javascript|typescript|sql|bash|shell|css|html)\b",
    r"\b(code|snippet|function|class|method)\b.{0,20}\bthat\b",
    r"\bhow\s+(do\s+I|to)\s+(implement|code|write|build|create)\b",
    r"\b(unit\s+test|integration\s+test|write\s+test)\b",
]

_CODE_KEYWORDS = [
    "write a function", "write a class", "write a script", "write a test",
    "implement a", "code this", "generate code", "write code",
    "debug this", "fix this bug", "fix the error", "fix the bug",
    "refactor this", "optimize this", "add a function",
    "create an endpoint", "write an api", "write a module",
]


def _score_patterns(text: str, patterns: list[str]) -> float:
    hits = sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))
    return min(1.0, hits * 0.30)


def _score_keywords(text: str, keywords: list[str]) -> float:
    hits = sum(1 for kw in keywords if kw in text)
    return min(1.0, hits * 0.25)


def classify_intent(prompt: str) -> RouteDecision:
    p = prompt.lower()

    scores: dict[str, float] = {
        "quick-local": 0.0,
        "reasoning-heavy": 0.0,
        "code": 0.0,
        "external-data-needed": 0.0,
        "memory-needed": 0.0,
    }

    # reasoning-heavy signals
    scores["reasoning-heavy"] += _score_patterns(p, _REASONING_PATTERNS)
    scores["reasoning-heavy"] += _score_keywords(p, _REASONING_KEYWORDS)
    if len(prompt) > 600:
        scores["reasoning-heavy"] = min(1.0, scores["reasoning-heavy"] + 0.30)
    elif len(prompt) > 300:
        scores["reasoning-heavy"] = min(1.0, scores["reasoning-heavy"] + 0.10)

    # external-data signals
    scores["external-data-needed"] += _score_patterns(p, _EXTERNAL_DATA_PATTERNS)
    scores["external-data-needed"] += _score_keywords(p, _EXTERNAL_DATA_KEYWORDS)

    # memory signals
    scores["memory-needed"] += _score_patterns(p, _MEMORY_PATTERNS)
    scores["memory-needed"] += _score_keywords(p, _MEMORY_KEYWORDS)

    # code signals — dampen quick-local baseline when strong
    scores["code"] += _score_patterns(p, _CODE_PATTERNS)
    scores["code"] += _score_keywords(p, _CODE_KEYWORDS)

    # quick-local baseline — short, conversational prompts
    if len(prompt) < 60:
        scores["quick-local"] += 0.60
    elif len(prompt) < 120:
        scores["quick-local"] += 0.35

    # If we already have strong external-data, memory, or code signals, dampen
    # quick-local baseline so intent-specific routes win short prompts.
    if scores["external-data-needed"] >= 0.25:
        scores["quick-local"] = max(0.0, scores["quick-local"] - 0.35)
    if scores["memory-needed"] >= 0.25:
        scores["quick-local"] = max(0.0, scores["quick-local"] - 0.35)
    if scores["code"] >= 0.25:
        scores["quick-local"] = max(0.0, scores["quick-local"] - 0.35)

    # Clamp all scores
    for k in scores:
        scores[k] = min(1.0, scores[k])

    intent = max(scores, key=lambda k: scores[k])
    confidence = scores[intent]

    # If all scores are zero, treat as quick-local with baseline confidence
    if confidence == 0.0:
        intent = "quick-local"
        confidence = 0.50

    # Routing: only send to cloud for reasoning-heavy with sufficient confidence.
    # Code intent always stays local and uses the coder model.
    use_cloud = (intent == "reasoning-heavy" and confidence >= ROUTE_CONFIDENCE_THRESHOLD)
    model = CLOUD_MODEL if use_cloud else _pick_local_model(intent)

    return RouteDecision(
        intent=intent,
        confidence=round(confidence, 3),
        use_cloud=use_cloud,
        model=model,
        planner_status="heuristic",
    )


# ── Inner-monologue planner ────────────────────────────────────────────────────

_PLANNER_SYSTEM = """\
You are a routing assistant. Analyse the user message and output a JSON object — \
nothing else, no prose, no markdown fences. Use exactly this schema:
{
  "intent": "<quick-local|reasoning-heavy|code|external-data-needed|memory-needed>",
  "route": "<local|cloud|tool>",
  "tool": "<weather|music|null>",
  "needs_memory": <true|false>,
  "confidence": <0.0–1.0>,
  "reasoning": "<one sentence>"
}

Rules:
- intent=code  → route=local (code almost never needs cloud)
- intent=external-data-needed → route=tool
- intent=reasoning-heavy AND confidence>=0.55 → route=cloud
- everything else → route=local
- needs_memory=true when the request references prior facts or user preferences
- confidence reflects how certain you are of the intent, not the answer quality
"""


def _parse_planner_output(raw: str) -> dict:
    """Extract and validate the JSON object from planner output.

    Raises ValueError on any validation failure so callers can fall back.
    """
    # Strip optional markdown fences the model may still emit
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(
            line for line in lines if not line.startswith("```")
        ).strip()

    # Find first { … } block
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in planner output: {raw!r}")
    data = json.loads(cleaned[start:end])

    intent = str(data.get("intent", "")).strip()
    if intent not in _VALID_INTENTS:
        raise ValueError(f"Invalid intent {intent!r}")

    route = str(data.get("route", "local")).strip()
    if route not in _VALID_ROUTES:
        route = "local"

    confidence = float(data.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))

    tool_raw = data.get("tool")
    tool = str(tool_raw).strip() if tool_raw and str(tool_raw).lower() != "null" else None

    needs_memory = bool(data.get("needs_memory", False))
    reasoning = str(data.get("reasoning", "")).strip()[:300]

    return {
        "intent": intent,
        "route": route,
        "tool": tool,
        "needs_memory": needs_memory,
        "confidence": round(confidence, 3),
        "reasoning": reasoning,
    }


async def _call_planner(prompt: str) -> dict:
    """Call local Ollama with the planner system prompt and return parsed JSON."""
    timeout = PLANNER_TIMEOUT_MS / 1000.0
    payload = {
        "model": LOCAL_MODEL,
        "prompt": f"User message: {prompt}",
        "system": _PLANNER_SYSTEM,
        "stream": False,
        "options": {
            "num_predict": PLANNER_MAX_TOKENS,
            "temperature": PLANNER_TEMPERATURE,
        },
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("response", "")
    return _parse_planner_output(raw)


def _decision_from_planner(parsed: dict) -> RouteDecision:
    """Convert parsed planner dict → RouteDecision."""
    intent = parsed["intent"]
    route = parsed["route"]
    confidence = parsed["confidence"]

    use_cloud = (route == "cloud" and intent == "reasoning-heavy"
                 and confidence >= ROUTE_CONFIDENCE_THRESHOLD)
    model = CLOUD_MODEL if use_cloud else _pick_local_model(intent)

    return RouteDecision(
        intent=intent,
        confidence=confidence,
        use_cloud=use_cloud,
        model=model,
        tool=parsed["tool"],
        needs_memory=parsed["needs_memory"],
        planner_status="planner",
        reasoning_summary=parsed["reasoning"],
    )


async def route(prompt: str) -> RouteDecision:
    """Primary entry point.

    Tries the inner-monologue planner when enabled; falls back to the
    deterministic heuristic classifier on any failure.
    The reasoning_summary in the returned decision is for server logs only
    and must never be sent to the client.
    """
    if PLANNER_ENABLED:
        try:
            parsed = await _call_planner(prompt)
            decision = _decision_from_planner(parsed)
            log.debug(
                "router.planner | intent=%s route=%s tool=%s needs_memory=%s "
                "confidence=%.3f reasoning=%s",
                decision.intent,
                "cloud" if decision.use_cloud else "local",
                decision.tool,
                decision.needs_memory,
                decision.confidence,
                decision.reasoning_summary,
            )
            return decision
        except Exception as exc:
            log.warning("router.planner_failed | reason=%s — falling back to heuristic", exc)

    decision = classify_intent(prompt)
    decision.planner_status = "disabled" if not PLANNER_ENABLED else "fallback"
    return decision
