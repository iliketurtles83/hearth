"""
Intent classifier for the assistant graph.

This file now exists as a compatibility shim around the graph's routing entry
point: a deterministic heuristic classifier plus a thin async wrapper for code
that still expects ``await route(prompt)``.

Intent categories:
    quick-local          — factual, short, conversational
    reasoning-heavy      — multi-step planning, analysis, architecture
    code                 — code generation, debugging, explanation, file editing
    external-data-needed — weather, news, live data, music playback commands
    memory-needed        — references to prior facts or user preferences
"""

import os
import re
from dataclasses import dataclass

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

# ── Routing threshold ──────────────────────────────────────────────────────────
# Minimum confidence score in "reasoning-heavy" required to route to cloud.
ROUTE_CONFIDENCE_THRESHOLD = float(os.getenv("ROUTE_CONFIDENCE_THRESHOLD", "0.55"))

_VALID_TOOL_NAMES = frozenset(["weather", "music"])


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

_WEATHER_PATTERNS = [
    r"\b(weather|forecast|temperature|rain|snow|sunny|humidity)\b",
]

_MUSIC_PATTERNS = [
    r"\b(play|queue|put\s+on)\b.{0,60}\b(song|track|music|album|artist|band)\b",
    r"\b(pause|resume|unpause)\b.{0,30}\b(music|song|track|playback)?\b",
    r"\b(next|skip)\b.{0,20}\b(track|song)?\b",
    r"\bstop\s+(the\s+)?(music|playback|song)\b",
    r"\b(now\s+playing|what'?s\s+playing|what\s+is\s+playing)\b",
    r"\b(what'?s|what\s+is)\s+in\s+the\s+queue\b",
]

_MUSIC_KEYWORDS = [
    "play", "queue", "pause", "stop", "resume", "next track", "next song",
    "skip", "now playing", "what's playing", "what is playing",
    "artist radio", "put on", "shuffle",
]


def _looks_like_weather_request(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in _WEATHER_PATTERNS)


def _looks_like_music_request(text: str) -> bool:
    """Return True when the prompt is almost certainly a music command.

    Uses keyword gate first (fast path), then regex patterns for structural matches.
    Intentionally conservative — broad "play" alone is not sufficient; it must
    be accompanied by a music-related keyword or structural pattern.
    """
    p = text.lower()
    # Fast keyword gate
    if not any(kw in p for kw in _MUSIC_KEYWORDS):
        return False
    # Require at least one structural pattern
    return any(re.search(pat, text, re.IGNORECASE) for pat in _MUSIC_PATTERNS)


def _normalize_external_tool(intent: str, prompt: str, tool_name: str | None) -> str | None:
    """Return a safe tool name for external-data intents, else None.

    - Only external-data intents may dispatch to tools.
    - Unknown planner tool names are ignored.
    - Weather requests infer the weather tool when planner omits a tool.
    - Music requests infer the music tool when planner omits a tool.
    """
    if intent != "external-data-needed":
        return None

    normalized_tool = (tool_name or "").strip().lower() or None
    if normalized_tool and normalized_tool not in _VALID_TOOL_NAMES:
        normalized_tool = None

    if normalized_tool:
        return normalized_tool

    if _looks_like_weather_request(prompt):
        return "weather"

    if _looks_like_music_request(prompt):
        return "music"

    return None


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
    r"\bwrite\s+me\s+code\b",
    r"\b(write|generate|create|implement)\b.{0,80}\b(in|using)\s+(python|javascript|typescript|sql|bash|shell|css|html)\b",
    r"\b(quicksort|quick\s*sort|merge\s*sort|binary\s*search)\b",
    r"\b(write|create|generate|implement|build)\b.{0,40}\b(function|class|method|script|endpoint|api|module|test)\b",
    r"\b(add|create|generate|write)\b.{0,40}\b(test|tests|test\s+case|test\s+cases|pytest|unittest)\b",
    r"\bcan\s+you\b.{0,20}\b(add|write|create|generate)\b.{0,40}\btests?\b",
    r"\b(debug|fix)\b.{0,30}\b(bug|error|exception|crash|issue|problem)\b",
    r"\b(refactor|optimise|optimize|rewrite)\b.{0,30}\b(function|class|code|snippet)\b",
    r"\bwrite\s+(a\s+)?(python|javascript|typescript|sql|bash|shell|css|html)\b",
    r"\b(code|snippet|function|class|method)\b.{0,20}\bthat\b",
    r"\bhow\s+(do\s+I|to)\s+(implement|code|write|build|create)\b",
    r"\b(unit\s+test|integration\s+test|write\s+test)\b",
]

_CODE_KEYWORDS = [
    "write me code",
    "in python",
    "write a function", "write a class", "write a script", "write a test",
    "implement a", "code this", "generate code", "write code",
    "debug this", "fix this bug", "fix the error", "fix the bug",
    "refactor this", "optimize this", "add a function",
    "add tests", "add test", "test this", "test this function",
    "test cases", "pytest", "unittest",
    "create an endpoint", "write an api", "write a module",
]


def _looks_like_code_request(text: str) -> bool:
    """Conservative gate for forcing code routing on obvious coding prompts."""
    p = text.lower()
    score = _score_patterns(p, _CODE_PATTERNS) + _score_keywords(p, _CODE_KEYWORDS)
    return score >= 0.25


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
    if _looks_like_weather_request(p):
        scores["external-data-needed"] = min(1.0, scores["external-data-needed"] + 0.30)
    if _looks_like_music_request(p):
        scores["external-data-needed"] = min(1.0, scores["external-data-needed"] + 0.55)

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

    tool = _normalize_external_tool(intent, prompt, None)

    return RouteDecision(
        intent=intent,
        confidence=round(confidence, 3),
        use_cloud=use_cloud,
        model=model,
        tool=tool,
        planner_status="heuristic",
    )

async def route(prompt: str) -> RouteDecision:
    """Compatibility wrapper for legacy async call sites."""
    return classify_intent(prompt)
