"""
Deterministic intent classifier and shared routing model constants.

Intent categories:
    quick-local          — factual, short, conversational
    reasoning-heavy      — multi-step planning, analysis, architecture
    code-question        — explanation, understanding, code review
    external-data-needed — weather/news/live-data/music commands
    memory-needed        — references to prior facts or user preferences
    vision               — image-related requests
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from routing_config import ROUTING_CONFIG


CHAT_MODEL: str = (
    os.getenv("OLLAMA_CHAT_MODEL")
    or os.getenv("MODEL_LOCAL")
    or "llama3.2"
)
VISION_MODEL: str = (
    os.getenv("OLLAMA_VISION_MODEL")
    or CHAT_MODEL
)
LOCAL_MODEL = CHAT_MODEL
CLOUD_MODEL = os.getenv("MODEL_CLOUD", "claude-sonnet-4-20250514")

ROUTE_CONFIDENCE_THRESHOLD = ROUTING_CONFIG.route_confidence_threshold
_VALID_TOOL_NAMES = frozenset(["weather", "music"])


@dataclass
class RouteDecision:
    intent: str
    confidence: float
    use_cloud: bool
    model: str
    tool: str | None = None
    needs_memory: bool = False
    planner_status: str = "heuristic"
    reasoning_summary: str = ""


def _is_code_intent(intent: str) -> bool:
    return intent == "code-question"


def _pick_local_model(intent: str) -> str:
    if intent == "vision":
        return VISION_MODEL
    return CHAT_MODEL


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
    p = text.lower()
    if not any(kw in p for kw in _MUSIC_KEYWORDS):
        return False
    return any(re.search(pat, text, re.IGNORECASE) for pat in _MUSIC_PATTERNS)


def _normalize_external_tool(intent: str, prompt: str, tool_name: str | None) -> str | None:
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

_CODE_ACTION_PATTERNS = [
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

_CODE_ACTION_KEYWORDS = [
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
    p = text.lower()
    write_score = _score_patterns(p, _CODE_ACTION_PATTERNS) + _score_keywords(p, _CODE_ACTION_KEYWORDS)
    question_score = _score_patterns(p, _CODE_QUESTION_PATTERNS) + _score_keywords(p, _CODE_QUESTION_KEYWORDS)
    return max(write_score, question_score) >= 0.25


def is_write_like_code_request(text: str) -> bool:
    p = text.lower()
    score = _score_patterns(p, _CODE_ACTION_PATTERNS) + _score_keywords(p, _CODE_ACTION_KEYWORDS)
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
        "code-question": 0.0,
        "external-data-needed": 0.0,
        "memory-needed": 0.0,
        "vision": 0.0,
    }

    scores["reasoning-heavy"] += _score_patterns(p, _REASONING_PATTERNS)
    scores["reasoning-heavy"] += _score_keywords(p, _REASONING_KEYWORDS)
    if len(prompt) > 600:
        scores["reasoning-heavy"] = min(1.0, scores["reasoning-heavy"] + 0.30)
    elif len(prompt) > 300:
        scores["reasoning-heavy"] = min(1.0, scores["reasoning-heavy"] + 0.10)

    scores["external-data-needed"] += _score_patterns(p, _EXTERNAL_DATA_PATTERNS)
    scores["external-data-needed"] += _score_keywords(p, _EXTERNAL_DATA_KEYWORDS)
    if _looks_like_weather_request(p):
        scores["external-data-needed"] = min(1.0, scores["external-data-needed"] + 0.30)
    if _looks_like_music_request(p):
        scores["external-data-needed"] = min(1.0, scores["external-data-needed"] + 0.55)

    scores["memory-needed"] += _score_patterns(p, _MEMORY_PATTERNS)
    scores["memory-needed"] += _score_keywords(p, _MEMORY_KEYWORDS)

    scores["code-question"] += _score_patterns(p, _CODE_QUESTION_PATTERNS)
    scores["code-question"] += _score_keywords(p, _CODE_QUESTION_KEYWORDS)
    scores["code-question"] += _score_patterns(p, _CODE_ACTION_PATTERNS)
    scores["code-question"] += _score_keywords(p, _CODE_ACTION_KEYWORDS)

    if len(prompt) < 60:
        scores["quick-local"] += 0.60
    elif len(prompt) < 120:
        scores["quick-local"] += 0.35

    scores["vision"] += _score_patterns(p, _VISION_PATTERNS)
    scores["vision"] += _score_keywords(p, _VISION_KEYWORDS)
    if _looks_like_vision_request(p):
        scores["vision"] = min(1.0, scores["vision"] + 0.40)

    if scores["external-data-needed"] >= 0.25:
        scores["quick-local"] = max(0.0, scores["quick-local"] - 0.35)
    if scores["memory-needed"] >= 0.25:
        scores["quick-local"] = max(0.0, scores["quick-local"] - 0.35)
    if scores["code-question"] >= 0.25:
        scores["quick-local"] = max(0.0, scores["quick-local"] - 0.35)
    if scores["vision"] >= 0.25:
        scores["quick-local"] = max(0.0, scores["quick-local"] - 0.35)

    for key in scores:
        scores[key] = min(1.0, scores[key])

    intent = max(scores, key=lambda k: scores[k])
    confidence = scores[intent]

    if confidence == 0.0:
        intent = "quick-local"
        confidence = 0.50

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
    """Compatibility async wrapper for callers that still await route()."""
    return classify_intent(prompt)
