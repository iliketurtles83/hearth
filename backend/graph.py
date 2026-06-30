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
    ollama_embed_text,
)
from routing_config import ROUTING_CONFIG
from tools.weather import format_weather_response, is_weather_reasoning

CHAT_TOKEN_BUDGET = ROUTING_CONFIG.chat_token_budget
CHAT_MAX_TURNS = ROUTING_CONFIG.chat_max_turns
OLLAMA_URL = ROUTING_CONFIG.ollama_url
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


@dataclass
class PromptRequest:
    message: str
    system: str


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


def _should_inject_memory(decision_intent: str, memory_hits: list[dict[str, Any]], user_message: str) -> bool:
    if not memory_hits:
        return False
    if decision_intent == "memory-needed":
        return True

    terms = [t for t in re.findall(r"[a-z0-9]+", user_message.lower()) if len(t) > 2][:10]
    if not terms:
        return False

    top_text = " ".join(str(h.get("text", "")).lower() for h in memory_hits[:3])
    overlap = sum(1 for t in terms if t in top_text)
    return overlap >= 2


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
    dialogue_label: str,
    dialogue_score: float,
    heuristic: RouteDecision,
    *,
    chat_model: str,
    cloud_model: str,
    vision_model: str,
    reasoning_summary: str,
) -> RouteDecision:
    if tool_label in {"weather", "music"}:
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

        def _last_assistant_message() -> str:
            for item in reversed(list(state.get("history", []))):
                if item.get("role") == "assistant":
                    return str(item.get("content", ""))
            return ""

        # Dedicated /code endpoint can force code routing deterministically.
        if state.get("force_code"):
            writer({
                "meta": {
                    "model": deps.chat_model,
                    "intent": "code-question",
                    "confidence": 1.0,
                    "route_type": "local",
                    "needs_memory": True,
                    "tool": None,
                    "planner_status": "forced",
                    "reasoning_summary": "",
                }
            })
            return {
                "intent": "code-question",
                "confidence": 1.0,
                "use_cloud": False,
                "model": deps.chat_model,
                "tool": None,
                "planner_status": "forced",
                "reasoning_summary": "",
                "needs_memory": True,
                "route_type": "local",
            }

        # Image attachment is a structural signal; skip the classifier
        # entirely.  There is no ambiguous case: an attached image always means
        # "vision request".  The classifier still runs for imageless visual queries
        # (e.g. "describe this photo?" with no image) so keyword scoring is preserved.
        if state.get("image_base64"):
            vision_model = deps.vision_model or deps.chat_model
            writer({
                "meta": {
                    "model": vision_model,
                    "intent": "vision",
                    "confidence": 1.0,
                    "route_type": "vision",
                    "needs_memory": False,
                    "tool": None,
                    "planner_status": "deterministic",
                    "reasoning_summary": "",
                }
            })
            return {
                "intent": "vision",
                "confidence": 1.0,
                "use_cloud": False,
                "model": vision_model,
                "tool": None,
                "planner_status": "deterministic",
                "reasoning_summary": "",
                "needs_memory": False,
                "route_type": "vision",
            }

        # Compute heuristic once; used as deterministic fallback.
        heuristic = classify_intent(state["message"])
        decision: RouteDecision

        def _heuristic_router_decision() -> RouteDecision:
            fallback = heuristic
            fallback.model = _pick_model_for_decision(
                fallback.intent,
                use_cloud=fallback.use_cloud,
                chat_model=deps.chat_model,
                cloud_model=deps.cloud_model,
                vision_model=deps.vision_model or deps.chat_model,
            )
            fallback.planner_status = "heuristic"
            return fallback

        if ROUTER_EMBEDDING_ENABLED:
            embed_router = deps.embedding_router
            if embed_router is None:
                log.info("embedding_route.fallback | reason=router_unavailable")
                decision = _heuristic_router_decision()
            else:
                try:
                    query_embedding = await ollama_embed_text(
                        state["message"],
                        base_url=OLLAMA_URL,
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
                    decision = _heuristic_router_decision()
                except Exception as exc:
                    log.warning("embedding_route.fallback | reason=embedding_failed error=%s", exc)
                    decision = _heuristic_router_decision()
                else:
                    reasoning_summary = (
                        "embed"
                        f" tool={embed_result.tool.label}:{embed_result.tool.score:.3f}/gap={embed_result.tool.gap:.3f}"
                        f" dialogue={embed_result.dialogue.label}:{embed_result.dialogue.score:.3f}/gap={embed_result.dialogue.gap:.3f}"
                    )
                    if embed_result.should_escalate:
                        log.info("embedding_route.ambiguous | action=heuristic")
                        decision = _heuristic_router_decision()
                        decision.planner_status = "embedding_ambiguous_fallback"
                        if not decision.reasoning_summary:
                            decision.reasoning_summary = reasoning_summary
                    else:
                        decision = _decision_from_embedding(
                            embed_result.tool.label,
                            embed_result.tool.score,
                            embed_result.dialogue.label,
                            embed_result.dialogue.score,
                            heuristic,
                            chat_model=deps.chat_model,
                            cloud_model=deps.cloud_model,
                            vision_model=deps.vision_model or deps.chat_model,
                            reasoning_summary=reasoning_summary,
                        )
        else:
            decision = _heuristic_router_decision()

        followup_message = state["message"].strip().lower()
        last_assistant = _last_assistant_message().lower()
        looks_like_write_followup = followup_message in {
            "yes",
            "y",
            "ok",
            "okay",
            "go ahead",
            "do it",
            "please do",
            "sounds good",
        } and any(
            marker in last_assistant
            for marker in [
                "write",
                "edit",
                "create",
                "implement",
                "modify",
                "patch",
                "file",
                "confirm",
            ]
        )

        if (is_write_like_code_request(state["message"]) or looks_like_write_followup) and decision.intent != "code-question":
            decision.intent = "code-question"
            decision.use_cloud = False
            decision.tool = None
            decision.model = deps.chat_model
            decision.planner_status = (
                "write_followup_downgraded_to_code_question"
                if looks_like_write_followup
                else "write_downgraded_to_code_question"
            )

        route_type = "tool" if getattr(decision, "tool", None) else ("cloud" if decision.use_cloud else "local")

        writer({
            "meta": {
                "model": decision.model,
                "intent": decision.intent,
                "confidence": decision.confidence,
                "route_type": route_type,
                "needs_memory": decision.needs_memory,
                "tool": decision.tool,
                "planner_status": decision.planner_status,
                "reasoning_summary": decision.reasoning_summary,
            }
        })
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

    async def history_loader(state: AssistantState) -> dict[str, Any]:
        session_id = str(state.get("session_id", ""))
        user_id = str(state.get("user_id", ""))
        if not session_id or not user_id:
            return {"history": [], "session_summary": ""}

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
        return {"history": history, "session_summary": str(session_summary or "")}

    async def memory_retrieval(state: AssistantState) -> dict[str, Any]:
        history = list(state.get("history", []))
        session_summary = str(state.get("session_summary", "") or "")
        selected_history, history_tokens, truncated, summary_tokens = _select_history_for_budget(
            messages=history,
            system=state["system"],
            current_user_message=state["message"],
            summary_text=session_summary,
        )
        memory_hits_all = await asyncio.to_thread(
            deps.memory_store.retrieve,
            state["user_id"],
            state["message"],
        )
        inject_memory = _should_inject_memory(state["intent"], memory_hits_all, state["message"])
        memory_hits = memory_hits_all if inject_memory else []
        system_with_summary = _augment_system_with_session_summary(state["system"], session_summary)
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

    _VOICE_COMPRESS_SYSTEM = (
        "You convert assistant responses to natural spoken English for audio output. "
        "Rules:\n"
        "1. Remove all markdown formatting (headers, bullets, bold, italic, code blocks).\n"
        "2. Preserve ALL factual content: numbers, names, dates, locations, measurements.\n"
        "3. Target 20-30% of original length. If already short (under 40 words), keep as-is.\n"
        "4. Use natural conversational phrasing, as if speaking aloud to someone.\n"
        "5. Do not add new information. Do not ask follow-up questions.\n"
        "6. Output the spoken version ONLY — no preamble, no labels."
    )

    async def _compress_response_for_voice(original: str, model_name: str) -> str:
        """Compress a full chat response into a brief spoken version."""
        if not original.strip():
            return original
        word_count = len(original.split())
        if word_count <= 30:
            # Already short — strip markdown and return.
            import re as _re
            clean = _re.sub(r"[`*_#>\[\]!]", "", original)
            clean = _re.sub(r"\s+", " ", clean).strip()
            return clean
        compress_request = PromptRequest(
            message=(
                f"Original response ({word_count} words):\n{original}\n\nSpoken version:"
            ),
            system=_VOICE_COMPRESS_SYSTEM,
        )
        compressed = ""
        async for chunk in deps.stream_local(compress_request, model_name=model_name):
            compressed += _chunk_text(chunk)
        result = compressed.strip()
        log.info(
            "graph.responder | voice_compress | original_words=%d compressed_words=%d",
            word_count,
            len(result.split()),
        )
        return result if result else original

    async def _emit_response_chunks(
        stream: AsyncIterator[Any],
        *,
        modality: str,
        compress_model: str,
    ) -> str:
        writer = get_stream_writer()
        if modality == "voice":
            collected = ""
            async for chunk in stream:
                collected += _chunk_text(chunk)
            return await _compress_response_for_voice(collected, compress_model)

        response_text = ""
        async for chunk in stream:
            thinking = _chunk_thinking(chunk)
            if thinking:
                writer({"thinking": thinking})

            text = _chunk_text(chunk)
            if text:
                writer({"text": text})
                response_text += text
        return response_text

    async def responder(state: AssistantState) -> dict[str, Any]:
        writer = get_stream_writer()
        response_text = ""
        response_model = state.get("model", deps.chat_model)
        modality = state.get("modality", "chat")

        # ── Vision path ───────────────────────────────────────────────────────
        if state.get("intent") == "vision" and state.get("image_base64"):
            image_b64 = state["image_base64"]
            image_mime = state.get("image_mime") or "image/png"
            vision_request = PromptRequest(
                message=state.get("local_prompt") or state["message"],
                system=state.get("augmented_system") or state.get("system") or "",
            )
            local_vision_ok = False
            if deps.stream_local_vision is not None:
                try:
                    response_text = await _emit_response_chunks(
                        deps.stream_local_vision(vision_request, image_b64, image_mime),
                        modality=modality,
                        compress_model=deps.chat_model,
                    )
                    if modality == "voice":
                        writer({"text": response_text})
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
                vision_cloud_messages = [
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
                            {"type": "text", "text": vision_request.message},
                        ],
                    }
                ]
                try:
                    response_text = await _emit_response_chunks(
                        deps.stream_cloud(vision_request.system, vision_cloud_messages),
                        modality=modality,
                        compress_model=deps.chat_model,
                    )
                    if modality == "voice":
                        writer({"text": response_text})
                except Exception as exc:
                    log.error("graph.responder | vision_cloud_failed=%s", exc)
                    response_text = (
                        "I can't process this image right now — the local vision model and "
                        "cloud fallback are both unavailable. "
                        "Run `ollama pull gemma:e4b` to enable local image understanding."
                    )
                    writer({"text": response_text})

            return {"response_text": response_text.strip(), "response_model": response_model}
        # ── End vision path ──────────────────────────────────────────────────

        if state.get("tool"):
            tool_result = await deps.tool_dispatch(
                state["tool"],
                {"prompt": state["message"], "user_id": state["user_id"], "memory": deps.memory_store},
            )
            if getattr(tool_result, "ok", False):
                response_model = deps.chat_model
                # Weather fast-path: skip LLM for plain lookups.
                if state["tool"] == "weather" and not is_weather_reasoning(state["message"]):
                    response_text = format_weather_response(getattr(tool_result, "data", {}))
                    writer({"text": response_text})
                else:
                    summary_request = PromptRequest(
                        message=_tool_summary_prompt(state["message"], getattr(tool_result, "data", {})),
                        system=state["augmented_system"],
                    )
                    response_text = await _emit_response_chunks(
                        deps.stream_local(summary_request, model_name=deps.chat_model),
                        modality=modality,
                        compress_model=deps.chat_model,
                    )
                    if modality == "voice":
                        writer({"text": response_text})
            else:
                response_text = getattr(tool_result, "error", "The tool returned no data.") or "The tool returned no data."
                writer({"text": response_text})
        elif state.get("use_cloud"):
            response_model = deps.cloud_model
            try:
                response_text = await _emit_response_chunks(
                    deps.stream_cloud(state["augmented_system"], state["cloud_messages"]),
                    modality=modality,
                    compress_model=deps.chat_model,
                )
                if modality == "voice":
                    writer({"text": response_text})
            except Exception:
                log.warning("graph.cloud_fallback | session_id=%s", state.get("session_id", ""))
                response_model = deps.chat_model
                writer({"notice": "Cloud unavailable \u2014 responding with local model"})
                writer({"model": deps.chat_model, "intent": state.get("intent", ""), "confidence": state.get("confidence", 0.0), "fallback": True})
                local_request = PromptRequest(message=state["local_prompt"], system=state["augmented_system"])
                response_text = await _emit_response_chunks(
                    deps.stream_local(local_request, model_name=deps.chat_model),
                    modality=modality,
                    compress_model=deps.chat_model,
                )
                if modality == "voice":
                    writer({"text": response_text})
        else:
            local_request = PromptRequest(message=state["local_prompt"], system=state["augmented_system"])
            response_text = await _emit_response_chunks(
                deps.stream_local(local_request, model_name=state["model"]),
                modality=modality,
                compress_model=deps.chat_model,
            )
            if modality == "voice":
                writer({"text": response_text})

        return {"response_text": response_text.strip(), "response_model": response_model}

    async def memory_writer(state: AssistantState) -> dict[str, Any]:
        writer = get_stream_writer()
        user_id = str(state.get("user_id", ""))
        session_id = str(state.get("session_id", ""))
        message = str(state.get("message", "") or "")
        response_text = str(state.get("response_text", "") or "").strip()

        if not user_id or not session_id:
            return {"memory_result": {}}

        # 1) Persist the turn in conversation_log.
        await asyncio.to_thread(
            deps.memory_store.log_turn,
            session_id,
            user_id,
            "user",
            message,
        )
        if response_text:
            await asyncio.to_thread(
                deps.memory_store.log_turn,
                session_id,
                user_id,
                "assistant",
                response_text,
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

        # 3) Threshold-based consolidation trigger.
        unconsolidated = await asyncio.to_thread(
            deps.memory_store.count_unconsolidated,
            user_id,
        )
        consolidation_threshold = int(os.getenv("MEMORY_CONSOLIDATION_THRESHOLD", "3"))
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

        return {"memory_result": memory_payload}

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
