from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, TypedDict

log = logging.getLogger("assistant.graph")

# Keep strong references to fire-and-forget background tasks so they are not
# garbage-collected before completion, and surface any exceptions they raise.
_background_tasks: set = set()


def _track_background_task(task) -> None:
    _background_tasks.add(task)

    def _on_done(t) -> None:
        _background_tasks.discard(t)
        try:
            exc = t.exception()
        except Exception:
            return
        if exc is not None:
            log.warning("Background task failed: %r", exc)

    task.add_done_callback(_on_done)


from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from intents import (
    CHAT_MODEL,
    CLOUD_MODEL,
    VISION_MODEL,
    RouteDecision,
    classify_intent,
    is_write_like_code_request,
)
from embedding_router import (
    EmbeddingRouterSnapshotMismatchError,
    openai_embed_text,
)
from routing_config import ROUTING_CONFIG
from tools.weather import format_weather_response, is_weather_reasoning
from tools.timer import format_timer_response
from tools.calculator import format_calculator_response
from tools.datetime_tool import format_datetime_response
from tools.schemas import HEARTH_TOOLS
from music_fastpath import format_music_response, is_direct_play_request, normalize_music_action

CHAT_TOKEN_BUDGET = ROUTING_CONFIG.chat_token_budget
CHAT_MAX_TURNS = ROUTING_CONFIG.chat_max_turns
OPENAI_BASE_URL = ROUTING_CONFIG.openai_base_url
OPENAI_EMBED_BASE_URL = ROUTING_CONFIG.openai_embed_base_url
ROUTER_EMBEDDING_ENABLED = ROUTING_CONFIG.router_embedding_enabled
ROUTER_EMBED_MODEL = ROUTING_CONFIG.router_embed_model
ROUTER_EMBED_TIMEOUT_MS = ROUTING_CONFIG.router_embed_timeout_ms


class AssistantState(TypedDict, total=False):
    user_id: str
    session_id: str
    message: str
    system: str
    source: str
    history: list[dict[str, Any]]
    session_summary: str
    selected_history: list[dict[str, Any]]
    history_tokens: int
    truncated: bool
    summary_tokens: int
    intent: str
    confidence: float
    use_cloud: bool
    model: str
    tool: str | None
    planner_status: str
    reasoning_summary: str
    needs_memory: bool
    route_type: str
    memories: list[dict[str, Any]]
    augmented_system: str
    local_prompt: str
    cloud_messages: list[dict[str, Any]]
    response_text: str
    response_model: str
    memory_result: dict[str, Any]
    force_code: bool                  # True for /code endpoint to bias toward code-question intent
    # Responder modality split
    modality: str                     # "voice" or "chat"; set by /chat endpoint from request source
    # Vision input
    image_base64: str | None          # raw base64 image (ephemeral, not persisted)
    image_mime: str | None            # "image/png" | "image/jpeg" | "image/webp"
    response_failed: bool             # responder produced only an error notice; turn is not persisted


@dataclass
class PromptRequest:
    message: str
    system: str
    tools: list[dict[str, Any]] | None = None


@dataclass
class AssistantGraphDependencies:
    memory_store: Any
    embedding_router: Any | None
    router_route: Callable[[str], Awaitable[Any]]
    stream_local: Callable[[PromptRequest, str], AsyncIterator[Any]]
    stream_cloud: Callable[[str, list[dict[str, Any]]], AsyncIterator[str]]
    tool_dispatch: Callable[[str, dict[str, Any]], Awaitable[Any]]
    chat_model: str
    cloud_model: str
    # Vision model callable — calls Ollama /api/chat with images
    stream_local_vision: Callable[[PromptRequest, str, str], AsyncIterator[str]] | None = None
    vision_model: str = ""            # OLLAMA_VISION_MODEL (defaults to chat_model)


def checkpoint_config(session_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": session_id, "checkpoint_ns": ""}}


def default_checkpoint_path() -> str:
    return os.getenv(
        "GRAPH_CHECKPOINT_DB_PATH",
        os.path.join(os.path.dirname(__file__), "graph_checkpoints.sqlite"),
    )


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _chunk_text(chunk: Any) -> str:
    if isinstance(chunk, dict):
        return str(chunk.get("text", "") or "")
    return str(chunk or "")


def _chunk_thinking(chunk: Any) -> str:
    if isinstance(chunk, dict):
        return str(chunk.get("thinking", "") or "")
    return ""


def _chunk_tool_calls(chunk: Any) -> list[dict[str, Any]]:
    if isinstance(chunk, dict) and "tool_calls" in chunk:
        val = chunk["tool_calls"]
        return val if isinstance(val, list) else []
    return []


def _select_history_for_budget(
    messages: list[dict[str, Any]],
    system: str,
    current_user_message: str,
    summary_text: str,
) -> tuple[list[dict[str, Any]], int, bool, int]:
    summary_tokens = _estimate_tokens(summary_text) if summary_text else 0
    history_budget = max(
        0,
        CHAT_TOKEN_BUDGET
        - _estimate_tokens(system)
        - _estimate_tokens(current_user_message)
        - summary_tokens
        - 32,
    )
    selected_reversed: list[dict[str, Any]] = []
    used_tokens = 0
    truncated = False
    max_messages = max(1, CHAT_MAX_TURNS * 2)
    candidates = messages[-max_messages:]

    for message in reversed(candidates):
        cost = _estimate_tokens(str(message.get("content", ""))) + 4
        if used_tokens + cost > history_budget:
            truncated = True
            continue
        selected_reversed.append(message)
        used_tokens += cost

    selected = list(reversed(selected_reversed))
    if len(messages) > len(selected):
        truncated = True
    return selected, used_tokens, truncated, summary_tokens


def _build_local_prompt(history: list[dict[str, Any]], current_user_message: str) -> str:
    if not history:
        return current_user_message

    role_map = {"user": "User", "assistant": "Assistant"}
    lines = ["Conversation so far:"]
    for message in history:
        role = role_map.get(message.get("role", ""), "User")
        lines.append(f"{role}: {message['content']}")
    lines.append("")
    lines.append(f"User: {current_user_message}")
    lines.append("Assistant:")
    return "\n".join(lines)


def _augment_system_with_session_summary(system: str, summary_text: str) -> str:
    if not summary_text:
        return system
    return "\n".join(
        [
            system,
            "",
            "Session summary of older messages (use as context for continuity):",
            summary_text,
        ]
    )


def _augment_system_with_memories(system: str, memory_hits: list[dict[str, Any]]) -> str:
    if not memory_hits:
        return system

    lines = [
        system,
        "",
        "Relevant user memory (apply only if directly helpful to this request):",
        "If a memory item is not clearly relevant, ignore it.",
    ]
    for hit in memory_hits[:5]:
        lines.append(f"- {hit['text']}")
    return "\n".join(lines)


def _should_inject_memory(
    decision_intent: str,
    memory_hits: list[dict[str, Any]],
    user_message: str,
    *,
    needs_memory: bool = False,
) -> bool:
    if not memory_hits:
        return False
    if needs_memory or decision_intent == "memory-needed":
        return True

    # Even for non-memory intents, inject if the query is about the user
    # and we have relevant memory hits. This catches cases where the
    # embedding router escalated to heuristic (which often classifies
    # user-queries as "quick-local" instead of "memory-needed").
    terms = [t for t in re.findall(r"[a-z0-9]+", user_message.lower()) if len(t) > 2][:10]
    if not terms:
        return False

    # Check if any memory hit has a meaningful score (>= 0.3)
    # and if the query terms overlap with the memory content.
    top_text = " ".join(str(h.get("text", "")).lower() for h in memory_hits[:5])
    overlap = sum(1 for t in terms if t in top_text)
    if overlap >= 2:
        return True

    # Also inject if the top hit has a high score, indicating strong semantic relevance.
    top_score = float(memory_hits[0].get("score", 0.0))
    if top_score >= 0.3:
        return True

    return False


def _tool_summary_prompt(user_message: str, tool_data: dict[str, Any]) -> str:
    tool_data_str = json.dumps(tool_data, ensure_ascii=False)
    return (
        "You are a system that reports tool execution results. "
        "Based on the following structured data, write a concise response.\n"
        f"User request: {user_message}\n"
        f"Data: {tool_data_str}\n"
        "Rules:\n"
        "- Do not ask follow-up questions.\n"
        "- Do not suggest alternatives.\n"
        "- If action is play/queue/control and tool succeeded, state what was done in one sentence.\n"
        "- Mention title/artist only from Data fields.\n"
        "- If data says nothing is playing, say that plainly.\n"
        "- Keep to 1-2 short sentences max."
    )


def _rolling_summary_prompt(turns: list[dict[str, Any]]) -> str:
    role_map = {"user": "User", "assistant": "Assistant"}
    lines = ["Recent conversation turns:"]
    for turn in turns:
        role = role_map.get(str(turn.get("role", "")).lower(), "User")
        content = str(turn.get("content", "") or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    lines.append("")
    lines.append("Session summary:")
    return "\n".join(lines)


def _similarity_to_confidence(score: float) -> float:
    # Cosine similarity range is [-1, 1]; remap to confidence range [0, 1].
    return max(0.0, min(1.0, (score + 1.0) / 2.0))


def _decision_from_embedding(
    tool_label: str,
    tool_score: float,
    tool_gap: float,
    dialogue_label: str,
    dialogue_score: float,
    heuristic: RouteDecision,
    *,
    chat_model: str,
    cloud_model: str,
    vision_model: str,
    reasoning_summary: str,
) -> RouteDecision:
    # When the tool classifier matches weather/music but the gap is small,
    # defer to the dialogue classifier. This prevents city names or
    # conversational phrases from accidentally triggering weather/music routing.
    TOOL_OVERRIDE_GAP = 0.10

    if tool_label in {"weather", "music", "code", "timer", "calculator", "datetime"} and tool_gap < TOOL_OVERRIDE_GAP:
        if dialogue_label == "memory-augmented":
            return RouteDecision(
                intent="memory-needed",
                confidence=round(_similarity_to_confidence(dialogue_score), 3),
                use_cloud=False,
                model=chat_model,
                tool=None,
                planner_status="embedding",
                reasoning_summary=reasoning_summary,
                needs_memory=True,
            )
        if dialogue_label == "cloud":
            return RouteDecision(
                intent="reasoning-heavy",
                confidence=round(_similarity_to_confidence(dialogue_score), 3),
                use_cloud=True,
                model=cloud_model,
                tool=None,
                planner_status="embedding",
                reasoning_summary=reasoning_summary,
                needs_memory=False,
            )

    if tool_label == "weather":
        return RouteDecision(
            intent="external-data-needed",
            confidence=round(_similarity_to_confidence(tool_score), 3),
            use_cloud=False,
            model=chat_model,
            tool=tool_label,
            planner_status="embedding",
            reasoning_summary=reasoning_summary,
            needs_memory=False,
        )

    if tool_label == "music":
        return RouteDecision(
            intent="quick-local",
            confidence=round(_similarity_to_confidence(tool_score), 3),
            use_cloud=False,
            model=chat_model,
            tool=None,
            planner_status="embedding",
            reasoning_summary=reasoning_summary,
            needs_memory=False,
        )

    if tool_label == "vision":
        return RouteDecision(
            intent="vision",
            confidence=round(_similarity_to_confidence(tool_score), 3),
            use_cloud=False,
            model=vision_model,
            tool=None,
            planner_status="embedding",
            reasoning_summary=reasoning_summary,
            needs_memory=False,
        )

    if tool_label == "code":
        return RouteDecision(
            intent="code-question",
            confidence=round(_similarity_to_confidence(tool_score), 3),
            use_cloud=False,
            model=chat_model,
            tool=None,
            planner_status="embedding",
            reasoning_summary=reasoning_summary,
            needs_memory=False,
        )

    if tool_label in ("timer", "calculator", "datetime"):
        return RouteDecision(
            intent="external-data-needed",
            confidence=round(_similarity_to_confidence(tool_score), 3),
            use_cloud=False,
            model=chat_model,
            tool=tool_label,
            planner_status="embedding",
            reasoning_summary=reasoning_summary,
            needs_memory=False,
        )

    if dialogue_label == "cloud":
        return RouteDecision(
            intent="reasoning-heavy",
            confidence=round(_similarity_to_confidence(dialogue_score), 3),
            use_cloud=True,
            model=cloud_model,
            tool=None,
            planner_status="embedding",
            reasoning_summary=reasoning_summary,
            needs_memory=False,
        )

    if dialogue_label == "memory-augmented":
        return RouteDecision(
            intent="memory-needed",
            confidence=round(_similarity_to_confidence(dialogue_score), 3),
            use_cloud=False,
            model=chat_model,
            tool=None,
            planner_status="embedding",
            reasoning_summary=reasoning_summary,
            needs_memory=True,
        )

    # Default conversational route.
    return RouteDecision(
        intent="quick-local",
        confidence=round(_similarity_to_confidence(dialogue_score), 3),
        use_cloud=False,
        model=chat_model,
        tool=None,
        planner_status="embedding",
        reasoning_summary=reasoning_summary,
        needs_memory=False,
    )


def _pick_model_for_decision(
    intent: str,
    *,
    use_cloud: bool,
    chat_model: str,
    cloud_model: str,
    vision_model: str,
) -> str:
    if use_cloud:
        return cloud_model
    if intent == "vision":
        return vision_model
    return chat_model


# ── Pure routing-decision helpers ────────────────────────────────────────────

_WRITE_FOLLOWUP_CONFIRMATIONS = frozenset({
    "yes",
    "y",
    "ok",
    "okay",
    "go ahead",
    "do it",
    "please do",
    "sounds good",
})

_WRITE_FOLLOWUP_MARKERS = (
    "write",
    "edit",
    "create",
    "implement",
    "modify",
    "patch",
    "file",
    "confirm",
)


def _last_assistant_message(history: list[dict[str, Any]]) -> str:
    for item in reversed(list(history)):
        if item.get("role") == "assistant":
            return str(item.get("content", ""))
    return ""


def _looks_like_write_followup(followup_message: str, last_assistant: str) -> bool:
    return (
        followup_message in _WRITE_FOLLOWUP_CONFIRMATIONS
        and any(marker in last_assistant for marker in _WRITE_FOLLOWUP_MARKERS)
    )


def _downgrade_to_code_question(
    decision: RouteDecision,
    *,
    chat_model: str,
    looks_like_write_followup: bool,
) -> RouteDecision:
    if decision.intent == "code-question":
        return decision
    decision.intent = "code-question"
    decision.use_cloud = False
    decision.tool = None
    decision.model = chat_model
    decision.planner_status = (
        "write_followup_downgraded_to_code_question"
        if looks_like_write_followup
        else "write_downgraded_to_code_question"
    )
    return decision


def _heuristic_decision(
    heuristic: RouteDecision,
    *,
    chat_model: str,
    cloud_model: str,
    vision_model: str,
) -> RouteDecision:
    if heuristic.tool == "music":
        heuristic.tool = None
        heuristic.intent = "quick-local"
    heuristic.model = _pick_model_for_decision(
        heuristic.intent,
        use_cloud=heuristic.use_cloud,
        chat_model=chat_model,
        cloud_model=cloud_model,
        vision_model=vision_model,
    )
    heuristic.planner_status = "heuristic"
    if heuristic.intent == "memory-needed":
        heuristic.needs_memory = True
    return heuristic


def _forced_code_decision(chat_model: str) -> RouteDecision:
    return RouteDecision(
        intent="code-question",
        confidence=1.0,
        use_cloud=False,
        model=chat_model,
        tool=None,
        planner_status="forced",
        reasoning_summary="",
        needs_memory=True,
    )


def _direct_music_decision(chat_model: str) -> RouteDecision:
    return RouteDecision(
        intent="external-data-needed",
        confidence=1.0,
        use_cloud=False,
        model=chat_model,
        tool="music",
        planner_status="music_direct",
        reasoning_summary="",
        needs_memory=False,
    )


def _deterministic_vision_decision(vision_model: str) -> RouteDecision:
    return RouteDecision(
        intent="vision",
        confidence=1.0,
        use_cloud=False,
        model=vision_model,
        tool=None,
        planner_status="deterministic",
        reasoning_summary="",
        needs_memory=False,
    )


def _route_type_for_decision(decision: RouteDecision) -> str:
    if getattr(decision, "tool", None):
        return "tool"
    return "cloud" if decision.use_cloud else "local"


def _decision_meta(decision: RouteDecision, route_type: str) -> dict[str, Any]:
    return {
        "model": decision.model,
        "intent": decision.intent,
        "confidence": decision.confidence,
        "route_type": route_type,
        "needs_memory": decision.needs_memory,
        "tool": decision.tool,
        "planner_status": decision.planner_status,
        "reasoning_summary": decision.reasoning_summary,
    }


def _decision_state_update(decision: RouteDecision, route_type: str) -> dict[str, Any]:
    return {
        "intent": decision.intent,
        "confidence": decision.confidence,
        "use_cloud": decision.use_cloud,
        "model": decision.model,
        "tool": decision.tool,
        "planner_status": decision.planner_status,
        "reasoning_summary": decision.reasoning_summary,
        "needs_memory": decision.needs_memory,
        "route_type": route_type,
    }


# ── Pure responder helpers ───────────────────────────────────────────────────

# Pattern to catch Gemma 4 tool-call leakage: <|tool_call|>call:tool_name{...}
# or raw <tool_call|> tokens that the model generates when it thinks it should
# call a tool but the system handles tool dispatch externally.
_TOOL_CALL_LEAK_PATTERN = re.compile(
    r"<\|tool_call\|>call:\w+[\s\S]*?(?:<tool_call\|>|<\|/tool_call\|>)",
    re.DOTALL,
)


def _strip_tool_call_leaks(text: str) -> str:
    """Remove any leaked tool-call tokens from the LLM response."""
    return _TOOL_CALL_LEAK_PATTERN.sub("", text).strip()

def _is_weather_fastpath(tool: str, message: str) -> bool:
    return tool == "weather" and not is_weather_reasoning(message)


def _build_vision_cloud_messages(image_b64: str, image_mime: str, text: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_mime,
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": text},
            ],
        }
    ]


def build_assistant_graph(
    deps: AssistantGraphDependencies,
    *,
    checkpointer: Any | None = None,
):
    graph = StateGraph(AssistantState)

    # ── Helpers ───────────────────────────────────────────────────────────────

    # ── Nodes ─────────────────────────────────────────────────────────────────

    async def intent_classifier(state: AssistantState) -> dict[str, Any]:
        writer = get_stream_writer()
        vision_model = deps.vision_model or deps.chat_model

        # Dedicated /code endpoint can force code routing deterministically.
        if state.get("force_code"):
            decision = _forced_code_decision(deps.chat_model)
            route_type = _route_type_for_decision(decision)
            writer({"meta": _decision_meta(decision, route_type)})
            return _decision_state_update(decision, route_type)

        # Image attachment is a structural signal; skip the classifier
        # entirely.  There is no ambiguous case: an attached image always means
        # "vision request".  The classifier still runs for imageless visual queries
        # (e.g. "describe this photo?" with no image) so keyword scoring is preserved.
        if state.get("image_base64"):
            decision = _deterministic_vision_decision(vision_model)
            route_type = "vision"
            writer({"meta": _decision_meta(decision, route_type)})
            return _decision_state_update(decision, route_type)

        # Self-contained "play <thing>" goes straight to the music tool: the
        # LLM too often claims playback in prose without calling it.
        if is_direct_play_request(state["message"]):
            decision = _direct_music_decision(deps.chat_model)
            route_type = _route_type_for_decision(decision)
            writer({"meta": _decision_meta(decision, route_type)})
            return _decision_state_update(decision, route_type)

        # Compute heuristic once; used as the deterministic fallback.
        heuristic = classify_intent(state["message"])
        decision: RouteDecision

        def _heuristic_fallback() -> RouteDecision:
            return _heuristic_decision(
                heuristic,
                chat_model=deps.chat_model,
                cloud_model=deps.cloud_model,
                vision_model=vision_model,
            )

        if ROUTER_EMBEDDING_ENABLED:
            embed_router = deps.embedding_router
            if embed_router is None:
                log.info("embedding_route.fallback | reason=router_unavailable")
                decision = _heuristic_fallback()
            else:
                try:
                    query_embedding = await openai_embed_text(
                        state["message"],
                        base_url=OPENAI_EMBED_BASE_URL,
                        model=ROUTER_EMBED_MODEL,
                        timeout_seconds=ROUTER_EMBED_TIMEOUT_MS / 1000.0,
                    )
                    embed_result = embed_router.classify_embedding(query_embedding)
                    log.info(
                        "embedding_route.classified | tool=%s tool_score=%.3f tool_gap=%.3f dialogue=%s "
                        "dialogue_score=%.3f dialogue_gap=%.3f escalate=%s",
                        embed_result.tool.label,
                        embed_result.tool.score,
                        embed_result.tool.gap,
                        embed_result.dialogue.label,
                        embed_result.dialogue.score,
                        embed_result.dialogue.gap,
                        embed_result.should_escalate,
                    )
                except EmbeddingRouterSnapshotMismatchError as exc:
                    log.warning("embedding_route.snapshot_mismatch | error=%s", exc)
                    decision = _heuristic_fallback()
                except Exception as exc:
                    log.warning("embedding_route.fallback | reason=embedding_failed error=%s", exc)
                    decision = _heuristic_fallback()
                else:
                    reasoning_summary = (
                        "embed"
                        f" tool={embed_result.tool.label}:{embed_result.tool.score:.3f}/gap={embed_result.tool.gap:.3f}"
                        f" dialogue={embed_result.dialogue.label}:{embed_result.dialogue.score:.3f}/gap={embed_result.dialogue.gap:.3f}"
                    )
                    tool_is_decisive = (
                        embed_result.tool.label in {"weather", "music", "code", "vision"}
                        and not embed_result.tool.ambiguous
                    )
                    dialogue_is_decisive = (
                        embed_result.tool.label == "none"
                        and not embed_result.dialogue.ambiguous
                    )
                    if embed_result.should_escalate and not (tool_is_decisive or dialogue_is_decisive):
                        log.info("embedding_route.ambiguous | action=heuristic")
                        decision = _heuristic_fallback()
                        decision.planner_status = "embedding_ambiguous_fallback"
                        if not decision.reasoning_summary:
                            decision.reasoning_summary = reasoning_summary
                        # Heuristic sets intent but not needs_memory.
                        # If the heuristic classified as memory-needed,
                        # inject memory even though the embedding router
                        # was ambiguous about the tool classification.
                        if decision.intent == "memory-needed":
                            decision.needs_memory = True
                    else:
                        decision = _decision_from_embedding(
                            embed_result.tool.label,
                            embed_result.tool.score,
                            embed_result.tool.gap,
                            embed_result.dialogue.label,
                            embed_result.dialogue.score,
                            heuristic,
                            chat_model=deps.chat_model,
                            cloud_model=deps.cloud_model,
                            vision_model=vision_model,
                            reasoning_summary=reasoning_summary,
                        )
        else:
            decision = _heuristic_fallback()

        looks_like_write_followup = _looks_like_write_followup(
            state["message"].strip().lower(),
            _last_assistant_message(state.get("history", [])).lower(),
        )

        if is_write_like_code_request(state["message"]) or looks_like_write_followup:
            decision = _downgrade_to_code_question(
                decision,
                chat_model=deps.chat_model,
                looks_like_write_followup=looks_like_write_followup,
            )

        route_type = _route_type_for_decision(decision)

        writer({"meta": _decision_meta(decision, route_type)})
        return _decision_state_update(decision, route_type)

    async def history_loader(state: AssistantState) -> dict[str, Any]:
        session_id = str(state.get("session_id", ""))
        user_id = str(state.get("user_id", ""))
        if not session_id or not user_id:
            return {"history": [], "session_summary": "", "response_failed": False}

        turns = await asyncio.to_thread(
            deps.memory_store.get_session_turns,
            session_id,
            user_id,
            CHAT_MAX_TURNS * 2,
        )
        session_summary = await asyncio.to_thread(
            deps.memory_store.get_latest_session_summary,
            session_id,
            user_id,
        )
        history = [
            {
                "role": str(turn.get("role", "")),
                "content": str(turn.get("content", "")),
            }
            for turn in turns
        ]
        # response_failed is checkpointed per thread; reset it so a failed turn doesn't leak forward.
        return {"history": history, "session_summary": str(session_summary or ""), "response_failed": False}

    async def memory_retrieval(state: AssistantState) -> dict[str, Any]:
        history = list(state.get("history", []))
        session_summary = str(state.get("session_summary", "") or "")
        selected_history, history_tokens, truncated, summary_tokens = _select_history_for_budget(
            messages=history,
            system=state["system"],
            current_user_message=state["message"],
            summary_text=session_summary,
        )

        system_with_summary = _augment_system_with_session_summary(state["system"], session_summary)

        # Skip expensive retrieval (list_items, ChromaDB, graph_recall) for
        # intents that don't need memory augmentation. History selection and
        # session summary still apply — they're cheap and needed for context.
        intent = state.get("intent", "")
        needs_memory = bool(state.get("needs_memory", False))
        _SKIP_RETRIEVAL_INTENTS = frozenset({"quick-local", "vision"})
        if intent in _SKIP_RETRIEVAL_INTENTS and not needs_memory:
            log.debug(
                "graph.memory_retrieval | skipped | intent=%s needs_memory=%s session=%s",
                intent, needs_memory, state.get("session_id", ""),
            )
            return {
                "selected_history": selected_history,
                "history_tokens": history_tokens,
                "truncated": truncated,
                "summary_tokens": summary_tokens,
                "memories": [],
                "augmented_system": system_with_summary,
            }

        memory_hits_all = await asyncio.to_thread(
            deps.memory_store.retrieve,
            state["user_id"],
            state["message"],
        )
        inject_memory = _should_inject_memory(
            state["intent"],
            memory_hits_all,
            state["message"],
            needs_memory=needs_memory,
        )
        memory_hits = memory_hits_all if inject_memory else []
        augmented_system = _augment_system_with_memories(system_with_summary, memory_hits)

        log.debug(
            "graph.memory_retrieval | collection=conversation_memory | hits=%d | session=%s",
            len(memory_hits),
            state.get("session_id", ""),
        )

        return {
            "selected_history": selected_history,
            "history_tokens": history_tokens,
            "truncated": truncated,
            "summary_tokens": summary_tokens,
            "memories": memory_hits,
            "augmented_system": augmented_system,
        }

    async def tool_router(state: AssistantState) -> dict[str, Any]:
        selected_history = list(state.get("selected_history", []))
        local_prompt = _build_local_prompt(selected_history, state["message"])
        cloud_messages = [
            {"role": item["role"], "content": item["content"]}
            for item in selected_history
        ]
        cloud_messages.append({"role": "user", "content": state["message"]})
        return {
            "local_prompt": local_prompt,
            "cloud_messages": cloud_messages,
        }

    # Voice modality modifier: appended to the system prompt so the model
    # generates brief, spoken-style output directly — no second LLM pass.
    _VOICE_SYSTEM_MODIFIER = (
        "\n\n[MODALITY: VOICE] This response will be spoken aloud. "
        "Follow the voice reply rules: write for the ear, short sentences, "
        "no markdown, no lists, no headers. Keep it brief — 1-3 sentences "
        "for simple answers. Punctuate naturally for pacing."
    )

    def _voice_system(system: str, modality: str) -> str:
        """Append voice modifier to system prompt when modality is voice."""
        if modality == "voice":
            return system + _VOICE_SYSTEM_MODIFIER
        return system

    async def _emit_response_chunks(
        stream: AsyncIterator[Any],
    ) -> tuple[str, list[dict[str, Any]]]:
        writer = get_stream_writer()
        response_text = ""
        tool_calls: list[dict[str, Any]] = []
        async for chunk in stream:
            thinking = _chunk_thinking(chunk)
            if thinking:
                writer({"thinking": thinking})

            text = _chunk_text(chunk)
            if text:
                writer({"text": text})
                response_text += text

            chunk_tools = _chunk_tool_calls(chunk)
            if chunk_tools:
                for tc in chunk_tools:
                    idx = int(tc.get("index", 0))
                    while len(tool_calls) <= idx:
                        tool_calls.append({"name": "", "arguments": ""})
                    fn = tc.get("function", {})
                    if "name" in fn and fn["name"]:
                        tool_calls[idx]["name"] += fn["name"]
                    if "arguments" in fn and fn["arguments"]:
                        tool_calls[idx]["arguments"] += fn["arguments"]
        return response_text, tool_calls

    async def responder(state: AssistantState) -> dict[str, Any]:
        writer = get_stream_writer()
        response_text = ""
        response_model = state.get("model", deps.chat_model)
        modality = state.get("modality", "chat")
        effective_system = _voice_system(
            state.get("augmented_system") or state.get("system") or "",
            modality,
        )

        async def _execute_llm_tool_calls(
            tool_calls: list[dict[str, Any]],
        ) -> str:
            call = tool_calls[0]
            t_name = call.get("name", "").strip()
            raw_args = call.get("arguments", "{}")
            try:
                t_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except Exception:
                t_args = {}
            if not isinstance(t_args, dict):
                t_args = {}

            log.info("graph.responder | llm_tool_call | tool=%s args=%s", t_name, t_args)
            writer({"tool": t_name})

            if t_name == "weather":
                t_params = {
                    "prompt": state["message"],
                    "location": t_args.get("location"),
                    "user_id": state["user_id"],
                    "memory": deps.memory_store,
                }
                tool_result = await deps.tool_dispatch("weather", t_params)
                if getattr(tool_result, "ok", False):
                    if _is_weather_fastpath("weather", state["message"]):
                        res = format_weather_response(getattr(tool_result, "data", {}))
                        writer({"text": res})
                        return res
                    else:
                        summary_request = PromptRequest(
                            message=_tool_summary_prompt(state["message"], getattr(tool_result, "data", {})),
                            system=effective_system,
                        )
                        res, _ = await _emit_response_chunks(
                            deps.stream_local(summary_request, model_name=deps.chat_model),
                        )
                        return res
                else:
                    err = getattr(tool_result, "error", "The weather service was unavailable.") or "The weather service was unavailable."
                    writer({"text": err})
                    return err

            elif t_name == "music":
                t_params = {
                    "prompt": state["message"],
                    "action": t_args.get("action", "play"),
                    "query": t_args.get("query"),
                    "album": t_args.get("album"),
                    "playlist": t_args.get("playlist"),
                    "user_id": state["user_id"],
                }
                # artist alone → artist radio; artist + query → "title by artist".
                if t_args.get("query") and t_args.get("artist"):
                    t_params["artist_filter"] = t_args.get("artist")
                else:
                    t_params["artist"] = t_args.get("artist")
                normalize_music_action(t_params)
                tool_result = await deps.tool_dispatch("music", t_params)
                res = format_music_response(tool_result, t_params)
                writer({"text": res})
                return res

            elif t_name == "timer":
                t_params = {
                    "prompt": state["message"],
                    "action": t_args.get("action"),
                    "duration_minutes": t_args.get("duration_minutes"),
                    "label": t_args.get("label"),
                    "timer_id": t_args.get("timer_id"),
                    "user_id": state["user_id"],
                }
                tool_result = await deps.tool_dispatch("timer", t_params)
                res = format_timer_response(tool_result, t_args.get("action", ""))
                writer({"text": res})
                return res

            elif t_name == "calculator":
                t_params = {
                    "prompt": state["message"],
                    "expression": t_args.get("expression", state["message"]),
                    "user_id": state["user_id"],
                }
                tool_result = await deps.tool_dispatch("calculator", t_params)
                res = format_calculator_response(tool_result)
                writer({"text": res})
                return res

            elif t_name == "datetime":
                t_params = {
                    "prompt": state["message"],
                    "query": t_args.get("query", state["message"]),
                    "timezone": t_args.get("timezone"),
                    "user_id": state["user_id"],
                }
                tool_result = await deps.tool_dispatch("datetime", t_params)
                res = format_datetime_response(tool_result)
                writer({"text": res})
                return res

            return "Tool call completed."

        # ── Vision path ───────────────────────────────────────────────────────
        if state.get("intent") == "vision" and state.get("image_base64"):
            image_b64 = state["image_base64"]
            image_mime = state.get("image_mime") or "image/png"
            vision_request = PromptRequest(
                message=state.get("local_prompt") or state["message"],
                system=effective_system,
            )
            local_vision_ok = False
            vision_failed = False
            if deps.stream_local_vision is not None:
                try:
                    response_text, _ = await _emit_response_chunks(
                        deps.stream_local_vision(vision_request, image_b64, image_mime),
                    )
                    response_model = deps.vision_model or deps.chat_model
                    local_vision_ok = True
                except Exception as exc:
                    log.warning(
                        "graph.responder | vision_local_failed=%s | trying_cloud",
                        exc,
                    )

            if not local_vision_ok:
                # Cloud fallback: Anthropic vision API (multimodal message format)
                response_model = deps.cloud_model
                vision_cloud_messages = _build_vision_cloud_messages(
                    image_b64, image_mime, vision_request.message
                )
                try:
                    response_text, _ = await _emit_response_chunks(
                        deps.stream_cloud(vision_request.system, vision_cloud_messages),
                    )
                except Exception as exc:
                    log.error("graph.responder | vision_cloud_failed=%s", exc)
                    response_text = (
                        "I can't process this image right now — the local vision model and "
                        "cloud fallback are both unavailable. "
                        "Run `llama-server` with the vision-capable model to enable local image understanding."
                    )
                    writer({"text": response_text})
                    vision_failed = True

            return {
                "response_text": response_text.strip(),
                "response_model": response_model,
                "response_failed": vision_failed,
            }
        # ── End vision path ──────────────────────────────────────────────────

        if state.get("tool"):
            writer({"tool": state["tool"]})
            tool_result = await deps.tool_dispatch(
                state["tool"],
                {"prompt": state["message"], "user_id": state["user_id"], "memory": deps.memory_store},
            )
            if getattr(tool_result, "ok", False):
                response_model = deps.chat_model
                # Weather fast-path: skip LLM for plain lookups.
                if _is_weather_fastpath(state["tool"], state["message"]):
                    response_text = format_weather_response(getattr(tool_result, "data", {}))
                    writer({"text": response_text})
                elif state["tool"] == "music":
                    response_text = format_music_response(tool_result, {"action": "play", "prompt": state["message"]})
                    writer({"text": response_text})
                elif state["tool"] == "timer":
                    response_text = format_timer_response(tool_result)
                    writer({"text": response_text})
                elif state["tool"] == "calculator":
                    response_text = format_calculator_response(tool_result)
                    writer({"text": response_text})
                elif state["tool"] == "datetime":
                    response_text = format_datetime_response(tool_result)
                    writer({"text": response_text})
                else:
                    summary_request = PromptRequest(
                        message=_tool_summary_prompt(state["message"], getattr(tool_result, "data", {})),
                        system=effective_system,
                    )
                    response_text, _ = await _emit_response_chunks(
                        deps.stream_local(summary_request, model_name=deps.chat_model),
                    )
            else:
                response_text = getattr(tool_result, "error", "The tool returned no data.") or "The tool returned no data."
                writer({"text": response_text})
        elif state.get("use_cloud"):
            response_model = deps.cloud_model
            try:
                response_text, _ = await _emit_response_chunks(
                    deps.stream_cloud(effective_system, state["cloud_messages"]),
                )
            except Exception:
                log.warning("graph.cloud_fallback | session_id=%s", state.get("session_id", ""))
                response_model = deps.chat_model
                writer({"notice": "Cloud unavailable \u2014 responding with local model"})
                writer({"model": deps.chat_model, "intent": state.get("intent", ""), "confidence": state.get("confidence", 0.0), "fallback": True})
                local_request = PromptRequest(message=state["local_prompt"], system=effective_system, tools=HEARTH_TOOLS)
                response_text, tool_calls = await _emit_response_chunks(
                    deps.stream_local(local_request, model_name=deps.chat_model),
                )
                if tool_calls:
                    response_text = await _execute_llm_tool_calls(tool_calls)
        else:
            local_request = PromptRequest(
                message=state["local_prompt"],
                system=effective_system,
                tools=HEARTH_TOOLS,
            )
            response_text, tool_calls = await _emit_response_chunks(
                deps.stream_local(local_request, model_name=state["model"]),
            )
            if tool_calls:
                response_text = await _execute_llm_tool_calls(tool_calls)

        # Clean up any leaked tool-call tokens from Gemma 4.
        response_text = _strip_tool_call_leaks(response_text)

        return {"response_text": response_text.strip(), "response_model": response_model}

    async def memory_writer(state: AssistantState) -> dict[str, Any]:
        writer = get_stream_writer()
        user_id = str(state.get("user_id", ""))
        session_id = str(state.get("session_id", ""))
        message = str(state.get("message", "") or "")
        response_text = str(state.get("response_text", "") or "").strip()

        if not user_id or not session_id:
            return {"memory_result": {}}

        # 1) Persist the turn in conversation_log. A failed or empty response skips the
        # whole turn: an orphan user message would show up as back-to-back "User:" lines
        # in the next prompt, and an error notice isn't conversation.
        if response_text and not state.get("response_failed"):
            await asyncio.to_thread(
                deps.memory_store.log_turn,
                session_id,
                user_id,
                "user",
                message,
            )
            await asyncio.to_thread(
                deps.memory_store.log_turn,
                session_id,
                user_id,
                "assistant",
                response_text,
            )
        else:
            log.warning(
                "graph.memory_writer | turn_not_persisted | session=%s failed=%s empty=%s",
                session_id,
                bool(state.get("response_failed")),
                not response_text,
            )

        # 2) Extract explicit/inline memory from the user message.
        raw_memory_result = await asyncio.to_thread(
            deps.memory_store.ingest_user_message,
            user_id,
            message,
            str(state.get("source", "text") or "text"),
        )
        memory_payload = {
            "status": raw_memory_result.get("status", "none"),
            "saved": len(raw_memory_result.get("saved", [])),
            "blocked": len(raw_memory_result.get("blocked", [])),
            "needs_confirmation": len(raw_memory_result.get("needs_confirmation", [])),
            "deleted": int(raw_memory_result.get("deleted", 0) or 0),
            "explicit": bool(raw_memory_result.get("explicit", False)),
            "hint": (
                "Memory needs confirmation. Say 'remember this' to store it."
                if raw_memory_result.get("status") == "needs-confirmation"
                else ""
            ),
        }

        if memory_payload["status"] != "none" or memory_payload["hint"]:
            writer({"memory": memory_payload})

        async def _rolling_summary_task(trigger_turns: int) -> None:
            try:
                turns = await asyncio.to_thread(
                    deps.memory_store.get_session_turns,
                    session_id,
                    user_id,
                    trigger_turns,
                )
                if not turns:
                    return
                summary_request = PromptRequest(
                    message=_rolling_summary_prompt(turns[-trigger_turns:]),
                    system=(
                        "Summarize the recent conversation turns for future context. "
                        "Keep it concise and factual. Include user preferences, commitments, "
                        "decisions, and unresolved follow-ups. Do not invent details."
                    ),
                )
                summary_text = ""
                async for chunk in deps.stream_local(summary_request, model_name=deps.chat_model):
                    summary_text += _chunk_text(chunk)
                summary_text = summary_text.strip()
                if not summary_text:
                    return
                await asyncio.to_thread(
                    deps.memory_store.save_summary,
                    user_id,
                    session_id,
                    summary_text,
                )
                log.debug(
                    "graph.memory_writer.summary_saved | session_id=%s user_id=%s turns=%d",
                    session_id,
                    user_id,
                    trigger_turns,
                )
            except Exception as exc:
                log.warning(
                    "graph.memory_writer.summary_failed | session_id=%s user_id=%s error=%s",
                    session_id,
                    user_id,
                    exc,
                )

        summary_trigger = int(os.getenv("MEMORY_SUMMARY_TRIGGER", "18"))
        if summary_trigger > 0:
            turn_count = await asyncio.to_thread(
                deps.memory_store.count_session_turns,
                session_id,
                user_id,
            )
            if turn_count and turn_count % summary_trigger == 0:
                _track_background_task(asyncio.create_task(_rolling_summary_task(summary_trigger)))

        # 3) Optional mid-conversation consolidation trigger.
        # Disabled by default (threshold=0): the heavy LLM pass now runs in the
        # interval "sleep" scheduler (memory_scheduler) instead of on the hot
        # path. Set MEMORY_CONSOLIDATION_THRESHOLD>0 to restore the old behavior.
        consolidation_threshold = int(os.getenv("MEMORY_CONSOLIDATION_THRESHOLD", "0"))
        if consolidation_threshold > 0:
            unconsolidated = await asyncio.to_thread(
                deps.memory_store.count_unconsolidated,
                user_id,
            )
            if unconsolidated >= consolidation_threshold:
                consolidation_batch = int(os.getenv("MEMORY_CONSOLIDATION_BATCH_SIZE", "50"))
                task = asyncio.create_task(
                    asyncio.to_thread(
                        deps.memory_store.consolidate_pending,
                        user_id,
                        consolidation_batch,
                    )
                )
                _track_background_task(task)

        # Clear ephemeral image data so the checkpointer doesn't persist
        # potentially megabytes of base64 to graph_checkpoints.sqlite.
        return {
            "memory_result": memory_payload,
            "image_base64": None,
            "image_mime": None,
        }

    # ── Edge routing helpers ───────────────────────────────────────────────────

    def _after_intent_classifier(state: AssistantState) -> str:
        return "memory_retrieval"

    def _after_tool_router(state: AssistantState) -> str:
        return "responder"

    # ── Wire the graph ─────────────────────────────────────────────────────────

    graph.add_node("history_loader", history_loader)
    graph.add_node("intent_classifier", intent_classifier)
    graph.add_node("memory_retrieval", memory_retrieval)
    graph.add_node("tool_router", tool_router)
    graph.add_node("responder", responder)
    graph.add_node("memory_writer", memory_writer)

    graph.add_edge(START, "history_loader")
    graph.add_edge("history_loader", "intent_classifier")
    graph.add_conditional_edges("intent_classifier", _after_intent_classifier, {
        "memory_retrieval": "memory_retrieval",
    })
    graph.add_edge("memory_retrieval", "tool_router")
    graph.add_conditional_edges("tool_router", _after_tool_router, {
        "responder": "responder",
    })
    # The coding-agent confirmation nodes were removed in the code-question-only
    # architecture, so responder is the sole response-producing terminal path.
    graph.add_edge("responder", "memory_writer")
    graph.add_edge("memory_writer", END)

    return graph.compile(checkpointer=checkpointer)


@asynccontextmanager
async def create_assistant_graph(
    deps: AssistantGraphDependencies,
    *,
    checkpoint_path: str | None = None,
):
    async with AsyncSqliteSaver.from_conn_string(checkpoint_path or default_checkpoint_path()) as checkpointer:
        yield build_assistant_graph(deps, checkpointer=checkpointer)
