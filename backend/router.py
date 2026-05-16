"""
Intent classifier for the assistant graph.

This file now exists as a compatibility shim around the graph's routing entry
point: a deterministic heuristic classifier plus a thin async wrapper for code
that still expects ``await route(prompt)``.

Intent categories:
    quick-local          — factual, short, conversational
    reasoning-heavy      — multi-step planning, analysis, architecture
    code-question        — explanation, understanding, code review (answered by local code_tool)
    code-write           — code generation, file edits, debugging (dispatched to external coding agent)
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
# Phase 14 — vision model: defaults to CHAT_MODEL (e.g. gemma:e4b is multimodal)
VISION_MODEL: str = (
    os.getenv("OLLAMA_VISION_MODEL")
    or CHAT_MODEL
)
LOCAL_MODEL = CHAT_MODEL  # backward-compat alias
CLOUD_MODEL = os.getenv("MODEL_CLOUD", "claude-sonnet-4-20250514")

# ── Routing threshold ──────────────────────────────────────────────────────────
# Minimum confidence score in "reasoning-heavy" required to route to cloud.
ROUTE_CONFIDENCE_THRESHOLD = float(os.getenv("ROUTE_CONFIDENCE_THRESHOLD", "0.55"))

_VALID_TOOL_NAMES = frozenset(["weather", "music"])


@dataclass
class RouteDecision:
    intent: str           # quick-local | reasoning-heavy | code-question | code-write | external-data-needed | memory-needed | vision
    confidence: float     # 0.0–1.0
    use_cloud: bool
    model: str
    tool: str | None = None          # tool name when route == "tool", else None
    needs_memory: bool = False       # set by LLM planner only; classify_intent() never sets this;
                                     # graph._should_inject_memory() uses intent+term-overlap instead
    planner_status: str = "heuristic"  # planner | fallback | heuristic | disabled
    reasoning_summary: str = ""      # server-log only, never sent to client


def _is_code_intent(intent: str) -> bool:
    """Return True for any code-related intent (question or write)."""
    return intent in ("code-question", "code-write")


def _pick_local_model(intent: str) -> str:
    """Return the appropriate local Ollama model for a given intent.

    Both code-question and code-write use CODER_MODEL; vision uses VISION_MODEL;
    all others use CHAT_MODEL.
    """
    if _is_code_intent(intent):
        return CODER_MODEL
    if intent == "vision":
        return VISION_MODEL
    return CHAT_MODEL


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
    # (track|song) is required (not optional) to avoid matching bare "skip" or "next" in non-music context.
    r"\b(next|skip)\b.{0,20}\b(track|song)\b",
    r"\bstop\s+(the\s+)?(music|playback|song)\b",
    r"\b(now\s+playing|what'?s\s+playing|what\s+is\s+playing)\b",
    r"\b(what'?s|what\s+is)\s+in\s+the\s+queue\b",
    r"\bshuffle\b.{0,30}\b(music|songs?|tracks?|playlist|queue)\b",
]

_MUSIC_KEYWORDS = [
    "play", "queue", "pause", "stop", "resume", "next track", "next song",
    "skip", "now playing", "what's playing", "what is playing",
    "artist radio", "put on", "shuffle",
]


def _looks_like_weather_request(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in _WEATHER_PATTERNS)


# ── Vision patterns ──────────────────────────────────────────────────────────
# NOTE: when an image is attached, graph.py forces intent="vision" and bypasses
# classify_intent() entirely. These patterns only fire for text-only prompts
# where the user mentions an image without attaching one.

_VISION_PATTERNS = [
    r"\b(what|who|where|how|why|when)\b.{0,40}\b(image|picture|photo|screenshot|diagram|chart|graph)\b",
    r"\b(describe|analyze|analyse|explain|identify|read|ocr)\b.{0,40}\b(image|picture|photo|screenshot|diagram)\b",
    r"\bwhat('?s|\s+is)\b.{0,20}\b(in|on|this|the)\b.{0,20}\b(image|picture|photo)\b",
    r"\bthis\s+(image|picture|photo|screenshot|diagram)\b",
    r"\b(image|picture|photo|screenshot)\b.{0,40}\b(shows?|contain|display|depict)\b",
    r"\b(look|see|view)\b.{0,20}\b(at|this|the)\b.{0,30}\b(image|picture|photo)\b",
]

_VISION_KEYWORDS = [
    "in this image", "in the image", "in this photo", "in this picture",
    "what do you see", "what's in", "what is in",
    "describe this", "describe the image", "describe the photo",
    "analyze this image", "analyse this image", "look at this",
    "what does this image", "what does the image",
    "read this", "ocr this", "text in the image", "text in this image",
]


def _looks_like_vision_request(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in _VISION_PATTERNS)


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

# ── Code-write patterns (user wants code produced or files changed) ───────────

_CODE_WRITE_PATTERNS = [
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
    r"\b(add|remove|delete|rename|move|update|change)\b.{0,40}\b(function|class|method|file|module|endpoint|variable)\b",
]

_CODE_WRITE_KEYWORDS = [
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

# ── Code-question patterns (user wants explanation or understanding) ───────────

_CODE_QUESTION_PATTERNS = [
    r"\bexplain\b.{0,60}\b(function|class|method|code|script|algorithm|module|file|logic|pattern)\b",
    r"\bhow\s+does\b.{0,60}\b(work|function|run|operate)\b",
    r"\bwhat\s+(does|is)\b.{0,60}\b(function|class|method|code|this|doing|mean)\b",
    r"\bwalk\s+(me\s+)?through\b",
    r"\b(understand|explain)\b.{0,40}\b(code|function|class|module|script|algorithm)\b",
    r"\bwhat'?s?\s+the\s+difference\b",
    r"\bwhy\s+(does|is|do)\b.{0,40}\b(this|it|code)\b",
    r"\b(review|look\s+at|read|inspect)\b.{0,40}\b(code|file|function|class|script)\b",
    r"\bcan\s+you\s+(explain|describe|clarify|summarize|summarise)\b",
    r"\bhow\s+(does|do)\s+(it|this|that)\b",
    r"\b(coding|programming|code)\s+(solution|approach|example|snippet)\b",
]

_CODE_QUESTION_KEYWORDS = [
    "explain", "how does", "what does", "walk through", "walk me through",
    "how does this work", "what does this do", "what is this doing",
    "understand this code", "make sense of", "what's the difference",
    "why does this", "review this code", "look at this code",
    "can you explain", "can you describe", "can you clarify",
    "what's happening", "what is happening",
    "coding solution", "code snippet", "how do i",
]


def _looks_like_code_request(text: str) -> bool:
    """Conservative gate for detecting any code-related prompt (question or write)."""
    p = text.lower()
    write_score = _score_patterns(p, _CODE_WRITE_PATTERNS) + _score_keywords(p, _CODE_WRITE_KEYWORDS)
    question_score = _score_patterns(p, _CODE_QUESTION_PATTERNS) + _score_keywords(p, _CODE_QUESTION_KEYWORDS)
    return max(write_score, question_score) >= 0.25


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
        "code-question": 0.0,
        "code-write": 0.0,
        "external-data-needed": 0.0,
        "memory-needed": 0.0,
        "vision": 0.0,
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

    # code-write signals — user wants code produced or files changed
    scores["code-write"] += _score_patterns(p, _CODE_WRITE_PATTERNS)
    scores["code-write"] += _score_keywords(p, _CODE_WRITE_KEYWORDS)

    # code-question signals — user wants explanation or understanding
    scores["code-question"] += _score_patterns(p, _CODE_QUESTION_PATTERNS)
    scores["code-question"] += _score_keywords(p, _CODE_QUESTION_KEYWORDS)

    # quick-local baseline — short, conversational prompts
    if len(prompt) < 60:
        scores["quick-local"] += 0.60
    elif len(prompt) < 120:
        scores["quick-local"] += 0.35

    # vision signals — user explicitly mentions an image in text
    scores["vision"] += _score_patterns(p, _VISION_PATTERNS)
    scores["vision"] += _score_keywords(p, _VISION_KEYWORDS)
    if _looks_like_vision_request(p):
        scores["vision"] = min(1.0, scores["vision"] + 0.40)

    # If we already have strong external-data, memory, code, or vision signals,
    # dampen quick-local baseline so intent-specific routes win short prompts.
    if scores["external-data-needed"] >= 0.25:
        scores["quick-local"] = max(0.0, scores["quick-local"] - 0.35)
    if scores["memory-needed"] >= 0.25:
        scores["quick-local"] = max(0.0, scores["quick-local"] - 0.35)
    if scores["code-write"] >= 0.25 or scores["code-question"] >= 0.25:
        scores["quick-local"] = max(0.0, scores["quick-local"] - 0.35)
    if scores["vision"] >= 0.25:
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
    # Code intents always stay local and use the coder model.
    # Vision intents stay local first (cloud fallback is handled in the responder node).
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
