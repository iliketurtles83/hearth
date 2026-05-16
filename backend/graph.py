from __future__ import annotations

import asyncio
import difflib
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, TypedDict

log = logging.getLogger("assistant.graph")

import httpx

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from router import CHAT_MODEL, CLOUD_MODEL, CODER_MODEL, VISION_MODEL, RouteDecision, classify_intent, _is_code_intent
from tools.weather import format_weather_response, is_weather_reasoning

CHAT_TOKEN_BUDGET = int(os.getenv("CHAT_TOKEN_BUDGET", "1500"))
CHAT_MAX_TURNS = int(os.getenv("CHAT_MAX_TURNS", "24"))
ROUTE_CONFIDENCE_THRESHOLD = float(os.getenv("ROUTE_CONFIDENCE_THRESHOLD", "0.55"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434").rstrip("/")
ROUTER_PLANNER_ENABLED = os.getenv("ROUTER_PLANNER_ENABLED", "true").lower() == "true"
ROUTER_PLANNER_TIMEOUT_MS = int(os.getenv("ROUTER_PLANNER_TIMEOUT_MS", "4000"))
ROUTER_PLANNER_MAX_TOKENS = int(os.getenv("ROUTER_PLANNER_MAX_TOKENS", "200"))
ROUTER_PLANNER_TEMPERATURE = float(os.getenv("ROUTER_PLANNER_TEMPERATURE", "0.0"))

# Minimum heuristic confidence required to skip the planner for deterministic intents.
# Intents not listed here always go through the planner (when enabled).
_HEURISTIC_GATE: dict[str, float] = {
    "external-data-needed": float(os.getenv("HEURISTIC_GATE_EXTERNAL", "0.50")),
    "code-question": float(os.getenv("HEURISTIC_GATE_CODE", "0.50")),
    "code-write": float(os.getenv("HEURISTIC_GATE_CODE", "0.50")),
}


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
    # Phase 10b — code tool
    active_files: list[str]          # files the coder has read/touched this turn
    code_context: str                 # tree-sitter snippets injected into coder prompt
    pending_write: dict[str, Any]    # {path, content, relative_path} awaiting confirmation
    awaiting_confirmation: bool       # True when a write diff is pending user approval
    force_code: bool                  # True for /code endpoint to bypass classifier
    # Phase 10c — responder modality split
    modality: str                     # "voice" or "chat"; set by /chat endpoint from request source
    # Phase 13 — coding agent tool
    pending_code_task: str            # code-write task description awaiting agent confirmation
    awaiting_agent_confirmation: bool # True when a code-write task needs user approval before dispatch
    # Phase 14 — vision input
    image_base64: str | None          # raw base64 image (ephemeral, not persisted)
    image_mime: str | None            # "image/png" | "image/jpeg" | "image/webp"


@dataclass
class PromptRequest:
    message: str
    system: str


@dataclass
class AssistantGraphDependencies:
    memory_store: Any
    router_route: Callable[[str], Awaitable[Any]]
    stream_local: Callable[[PromptRequest, str], AsyncIterator[str]]
    stream_cloud: Callable[[str, list[dict[str, Any]]], AsyncIterator[str]]
    tool_dispatch: Callable[[str, dict[str, Any]], Awaitable[Any]]
    chat_model: str
    cloud_model: str
    coder_model: str = ""             # Phase 10b: OLLAMA_CODER_MODEL for code_tool node
    chroma_path: str = ""             # Phase 10b: path to ChromaDB directory for code_context
    # Phase 14: vision model callable — calls Ollama /api/chat with images
    stream_local_vision: Callable[[PromptRequest, str, str], AsyncIterator[str]] | None = None
    vision_model: str = ""            # Phase 14: OLLAMA_VISION_MODEL (defaults to chat_model)


def checkpoint_config(session_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": session_id, "checkpoint_ns": ""}}


def default_checkpoint_path() -> str:
    return os.getenv(
        "GRAPH_CHECKPOINT_DB_PATH",
        os.path.join(os.path.dirname(__file__), "graph_checkpoints.sqlite"),
    )


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


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


_PLANNER_SYSTEM = """\
You are a routing assistant. Analyse the user message and output a JSON object — \
nothing else, no prose, no markdown fences. Use exactly this schema:
{
  "intent": "<quick-local|reasoning-heavy|code-question|code-write|external-data-needed|memory-needed>",
  "route": "<local|cloud|tool>",
  "tool": "<weather|music|null>",
  "needs_memory": <true|false>,
  "confidence": <0.0–1.0>,
  "reasoning": "<one sentence>"
}

Rules:
- intent=code-question or code-write  → route=local (code almost never needs cloud)
- intent=external-data-needed → route=tool
- intent=reasoning-heavy AND confidence>=0.55 → route=cloud
- everything else → route=local
- needs_memory=true when the request references prior facts or user preferences
- confidence reflects how certain you are of the intent, not the answer quality
"""


def _parse_planner_output(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(line for line in lines if not line.startswith("```"))
        cleaned = cleaned.strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in planner output")

    payload = json.loads(cleaned[start:end])
    if not isinstance(payload, dict):
        raise ValueError("Planner output must be a JSON object")

    return payload


def _decision_from_planner(
    parsed: dict[str, Any],
    prompt: str = "",
    *,
    chat_model: str,
    cloud_model: str,
    coder_model: str,
) -> RouteDecision:
    intent = str(parsed.get("intent", "")).strip()
    route = str(parsed.get("route", "local")).strip() or "local"
    tool = parsed.get("tool")
    confidence = float(parsed.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))
    needs_memory = bool(parsed.get("needs_memory", False))
    reasoning = str(parsed.get("reasoning", "")).strip()[:300]

    use_cloud = bool(route == "cloud" and intent == "reasoning-heavy" and confidence >= ROUTE_CONFIDENCE_THRESHOLD)

    if _is_code_intent(intent):
        model = coder_model
    elif use_cloud:
        model = cloud_model
    else:
        model = chat_model

    safe_tool = None
    if intent == "external-data-needed":
        candidate = str(tool).strip().lower() if tool is not None and str(tool).lower() != "null" else ""
        if candidate in {"weather", "music"}:
            safe_tool = candidate
        elif re.search(r"\b(weather|forecast|temperature|rain|snow|humidity)\b", prompt, re.IGNORECASE):
            safe_tool = "weather"
        elif re.search(r"\b(play|queue|pause|resume|skip|now\s+playing|what'?s\s+playing|what\s+is\s+playing)\b", prompt, re.IGNORECASE):
            safe_tool = "music"

    return RouteDecision(
        intent=intent,
        confidence=confidence,
        use_cloud=use_cloud,
        model=model,
        tool=safe_tool,
        planner_status="planner",
        reasoning_summary=reasoning,
        needs_memory=needs_memory,
    )


async def _call_planner(prompt: str) -> dict[str, Any]:
    payload = {
        "model": os.getenv("OLLAMA_CHAT_MODEL") or os.getenv("MODEL_LOCAL") or "llama3.2",
        "prompt": f"User message: {prompt}",
        "system": _PLANNER_SYSTEM,
        "format": "json",
        "stream": False,
        "options": {
            "num_predict": ROUTER_PLANNER_MAX_TOKENS,
            "temperature": ROUTER_PLANNER_TEMPERATURE,
        },
    }
    timeout = ROUTER_PLANNER_TIMEOUT_MS / 1000.0
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
        response.raise_for_status()
        data = response.json()
        raw = data.get("response", "")
    return _parse_planner_output(raw)


def code_context_retrieval(query: str, chroma_path: str, max_snippets: int = 5) -> str:
    """Fetch and format code-context snippets for prompt injection."""
    if not chroma_path:
        return ""
    try:
        from tools.code_indexer import query_code_context
    except Exception:
        return ""

    snippets = query_code_context(query, chroma_path, n=max_snippets)
    if not snippets:
        return ""

    formatted: list[str] = []
    for idx, snippet in enumerate(snippets, start=1):
        formatted.append(f"[code_context {idx}]\n{snippet}")
    return "\n\n---\n\n".join(formatted)


def build_assistant_graph(
    deps: AssistantGraphDependencies,
    *,
    checkpointer: Any | None = None,
):
    graph = StateGraph(AssistantState)

    # ── Helpers ───────────────────────────────────────────────────────────────

    _workspace_root: str = os.getenv("CODE_WORKSPACE_ROOT", "")

    def _resolve_workspace_path(relative_path: str) -> str:
        """Resolve a path within the workspace root.  Raises ValueError on traversal."""
        root = os.path.realpath(_workspace_root)
        candidate = os.path.realpath(os.path.join(root, relative_path))
        if not (candidate == root or candidate.startswith(root + os.sep)):
            raise ValueError(
                f"Path traversal blocked: {relative_path!r} resolves outside workspace root"
            )
        return candidate

    def _make_unified_diff(relative_path: str, original: str, proposed: str) -> str:
        lines = list(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                proposed.splitlines(keepends=True),
                fromfile=f"a/{relative_path}",
                tofile=f"b/{relative_path}",
                lineterm="\n",
            )
        )
        return "".join(lines) if lines else ""

    def _write_summary_for_voice(relative_path: str, diff_text: str) -> str:
        additions = 0
        deletions = 0
        for line in diff_text.splitlines():
            if line.startswith("+++") or line.startswith("---"):
                continue
            if line.startswith("+"):
                additions += 1
            elif line.startswith("-"):
                deletions += 1
        return (
            f"Planned write to {relative_path}. "
            f"About {additions} additions and {deletions} deletions. "
            "Say yes to apply, or tell me what to change."
        )

    _CONFIRM_PATTERN = re.compile(
        r"^\s*(yes|confirm|approve|go ahead|do it|write it|apply|proceed)\s*[.!]?\s*$",
        re.IGNORECASE,
    )

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
                    "model": deps.coder_model or deps.chat_model,
                    "intent": "code-question",
                    "confidence": 1.0,
                    "route_type": "code",
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
                "model": deps.coder_model or deps.chat_model,
                "tool": None,
                "planner_status": "forced",
                "reasoning_summary": "",
                "needs_memory": True,
                "route_type": "code",
            }

        # Short-circuit: if a write diff is pending and the user confirmed it,
        # skip normal LLM routing and go straight to write_executor.
        if state.get("awaiting_confirmation") and state.get("pending_write"):
            msg = str(state.get("message", "")).strip()
            if _CONFIRM_PATTERN.match(msg):
                log.info(
                    "graph.intent_classifier | confirm_write detected | session=%s",
                    state.get("session_id", ""),
                )
                writer({"meta": {"intent": "confirm_write", "route_type": "write_executor"}})
                return {
                    "intent": "confirm_write",
                    "route_type": "write_executor",
                    "confidence": 1.0,
                    "use_cloud": False,
                    "planner_status": "deterministic",
                    "reasoning_summary": "",
                    "needs_memory": False,
                }

        # Short-circuit: if a coding agent task is pending and the user confirmed it,
        # skip normal routing and go straight to coding_agent_executor.
        if state.get("awaiting_agent_confirmation") and state.get("pending_code_task"):
            msg = str(state.get("message", "")).strip()
            if _CONFIRM_PATTERN.match(msg):
                log.info(
                    "graph.intent_classifier | confirm_agent_task detected | session=%s",
                    state.get("session_id", ""),
                )
                writer({"meta": {"intent": "confirm_agent_task", "route_type": "coding_agent_executor"}})
                return {
                    "intent": "confirm_agent_task",
                    "route_type": "coding_agent_executor",
                    "confidence": 1.0,
                    "use_cloud": False,
                    "planner_status": "deterministic",
                    "reasoning_summary": "",
                    "needs_memory": False,
                }

        # Safety: if user sends a bare confirmation after a write-prompt but
        # pending_write was lost/cleared, route deterministically so the system
        # reports "No pending write" instead of hallucinating a write success.
        msg = str(state.get("message", "")).strip()
        if _CONFIRM_PATTERN.match(msg):
            last_assistant = _last_assistant_message().lower()
            looks_like_write_prompt = (
                "type **yes** to confirm the write" in last_assistant
                or "type yes to confirm the write" in last_assistant
                or "say yes to apply" in last_assistant
            )
            if looks_like_write_prompt and not state.get("pending_write"):
                log.info(
                    "graph.intent_classifier | confirm_without_pending | session=%s",
                    state.get("session_id", ""),
                )
                writer({"meta": {"intent": "confirm_write", "route_type": "write_executor"}})
                return {
                    "intent": "confirm_write",
                    "route_type": "write_executor",
                    "confidence": 1.0,
                    "use_cloud": False,
                    "planner_status": "deterministic",
                    "reasoning_summary": "",
                    "needs_memory": False,
                }

        # Phase 14: image attachment is a structural signal — skip the classifier
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

        # Compute heuristic unconditionally so it can (a) gate the planner for
        # high-confidence deterministic intents and (b) serve as a pre-computed
        # fallback on planner failure — classify_intent is never called twice.
        heuristic = classify_intent(state["message"])
        gate_threshold = _HEURISTIC_GATE.get(heuristic.intent)
        skip_planner = gate_threshold is not None and heuristic.confidence >= gate_threshold

        decision: RouteDecision
        if not ROUTER_PLANNER_ENABLED or skip_planner:
            decision = heuristic
            if not ROUTER_PLANNER_ENABLED:
                decision.planner_status = "disabled"
            else:
                log.debug(
                    "graph.intent_classifier | heuristic_gate | intent=%s confidence=%.2f",
                    heuristic.intent,
                    heuristic.confidence,
                )
                # planner_status already "heuristic" from classify_intent
        else:
            try:
                parsed = await _call_planner(state["message"])
                decision = _decision_from_planner(
                    parsed,
                    state["message"],
                    chat_model=deps.chat_model,
                    cloud_model=deps.cloud_model,
                    coder_model=deps.coder_model or deps.chat_model,
                )
            except Exception as exc:
                log.warning("graph.intent_classifier | planner_failed=%s | falling_back=heuristic", exc)
                decision = heuristic
                decision.planner_status = "fallback"

        route_type = "tool" if getattr(decision, "tool", None) else ("cloud" if decision.use_cloud else "local")
        if _is_code_intent(decision.intent):
            route_type = "code"

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
            deps.memory_store.retrieve, state["user_id"], state["message"]
        )
        inject_memory = _should_inject_memory(state["intent"], memory_hits_all, state["message"])
        memory_hits = memory_hits_all if inject_memory else []
        system_with_summary = _augment_system_with_session_summary(state["system"], session_summary)
        augmented_system = _augment_system_with_memories(system_with_summary, memory_hits)

        # Phase 10b/10d: route ChromaDB queries by intent.
        # - Non-code intents: deps.memory_store.retrieve() queries 'conversation_memory' only.
        # - Code intents: code_context_retrieval() queries 'code_context' only.
        # The two collections are strictly separate — see Phase 10d for migration details.
        code_context = ""
        if _is_code_intent(state.get("intent", "")):
            log.info(
                "graph.memory_retrieval | collection=code_context | session=%s",
                state.get("session_id", ""),
            )
            code_context = code_context_retrieval(state["message"], deps.chroma_path, max_snippets=5)
            if code_context:
                log.info(
                    "graph.memory_retrieval | code_context_injected=true | session=%s",
                    state.get("session_id", ""),
                )
        else:
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
            "code_context": code_context,
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

    async def code_tool(state: AssistantState) -> dict[str, Any]:
        """ReAct agent node for code generation and file operations.

        Uses OLLAMA_CODER_MODEL (qwen2.5-coder) via ChatOllama.
        File writes are intercepted: the diff is streamed to the user and
        pending_write is set in state; the actual write waits for confirmation.
        """
        writer = get_stream_writer()

        if not _workspace_root:
            msg = (
                "CODE_WORKSPACE_ROOT is not configured. "
                "Set it in .env and restart the backend."
            )
            writer({"text": msg})
            return {"response_text": msg, "response_model": deps.coder_model or deps.chat_model}

        # Mutable capture for write requests from within tool calls
        _pending: dict[str, Any] = {}
        touched_files: set[str] = set(state.get("active_files", []))

        # In code mode, a bare confirmation without a pending diff should not
        # re-enter the coder loop and potentially trigger unrelated writes.
        msg = str(state.get("message", "")).strip()
        if _CONFIRM_PATTERN.match(msg) and not state.get("pending_write"):
            no_pending_msg = (
                "No pending write to confirm. Describe the code change you want "
                "(for example: add pytest tests for bubble_sort in test_bubble_sort.py)."
            )
            writer({"text": no_pending_msg})
            return {
                "response_text": no_pending_msg,
                "response_model": deps.coder_model or deps.chat_model,
                "pending_write": {},
                "awaiting_confirmation": False,
                "active_files": sorted(touched_files),
            }

        # ── Custom tools ──────────────────────────────────────────────────────
        try:
            from langchain_core.tools import tool as lc_tool
        except ImportError:
            writer({"text": "langchain-core is not installed. Run: pip install langchain-core"})
            return {"response_text": "langchain-core missing", "response_model": ""}

        read_file_tool = None
        try:
            from langchain_community.tools import ReadFileTool, WriteFileTool  # type: ignore[import-untyped]
            read_file_tool = ReadFileTool(root_dir=_workspace_root)
            _ = WriteFileTool(root_dir=_workspace_root)
        except Exception as exc:
            log.warning("code_tool: ReadFileTool/WriteFileTool unavailable, using fallback (%s)", exc)

        @lc_tool
        def read_file(relative_path: str) -> str:
            """Read a file from the code workspace. Input: path relative to workspace root."""
            try:
                resolved = _resolve_workspace_path(relative_path)
            except ValueError as exc:
                return f"Error: {exc}"
            touched_files.add(relative_path)
            if read_file_tool is not None:
                try:
                    return str(read_file_tool.invoke(relative_path))
                except Exception as exc:
                    return f"Error reading file: {exc}"
            try:
                return Path(resolved).read_text(encoding="utf-8", errors="replace")
            except FileNotFoundError:
                return f"File not found: {relative_path}"
            except OSError as exc:
                return f"Error reading file: {exc}"

        @lc_tool
        def list_files(sub_path: str = "") -> str:
            """List files in the workspace (or a sub-directory). Input: optional sub-path."""
            try:
                base = _resolve_workspace_path(sub_path) if sub_path else _workspace_root
            except ValueError as exc:
                return f"Error: {exc}"
            result: list[str] = []
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in ("__pycache__", "node_modules", ".venv")]
                for fname in filenames:
                    p = Path(dirpath) / fname
                    try:
                        result.append(str(p.relative_to(_workspace_root)))
                    except ValueError:
                        result.append(str(p))
            return "\n".join(result) if result else "(empty)"

        @lc_tool
        def write_file(relative_path: str, content: str) -> str:
            """Propose writing content to a file.  Shows a diff; user must type 'yes' to confirm.
            Input: relative_path (str), content (str)."""
            try:
                resolved = _resolve_workspace_path(relative_path)
            except ValueError as exc:
                return f"Error: {exc}"
            touched_files.add(relative_path)
            try:
                original = Path(resolved).read_text(encoding="utf-8", errors="replace")
            except FileNotFoundError:
                original = ""
            diff = _make_unified_diff(relative_path, original, content)
            _pending["path"] = resolved
            _pending["content"] = content
            _pending["relative_path"] = relative_path
            if diff:
                if str(state.get("source", "text")).lower() == "voice":
                    return _write_summary_for_voice(relative_path, diff)
                return (
                    f"```diff\n{diff}\n```\n\n"
                    "Type **yes** to confirm the write, or describe any changes you want first."
                )
            return "No changes detected — the file already has that content."

        active_tools = [read_file, list_files, write_file]

        enable_shell = os.getenv("CODE_ENABLE_SHELL", "false").lower() == "true"
        if enable_shell:
            try:
                from langchain_community.tools import ShellTool  # type: ignore[import-untyped]
                active_tools.append(ShellTool())
                log.info("code_tool: ShellTool enabled (CODE_ENABLE_SHELL=true)")
            except ImportError:
                log.warning("code_tool: ShellTool requested but langchain-community not installed")

        enable_repl = os.getenv("CODE_ENABLE_REPL", "true").lower() == "true"
        if enable_repl:
            try:
                from langchain_community.tools import PythonREPLTool  # type: ignore[import-untyped]
                active_tools.append(PythonREPLTool())
            except ImportError:
                log.warning("code_tool: PythonREPLTool requested but langchain-community not installed")

        # ── Build the ReAct agent ─────────────────────────────────────────────
        try:
            from langchain_ollama import ChatOllama  # type: ignore[import-untyped]
            from langgraph.prebuilt import create_react_agent
        except ImportError as exc:
            msg = f"Missing dependency for code_tool: {exc}. Run: pip install langchain-ollama"
            writer({"text": msg})
            return {"response_text": msg, "response_model": ""}

        coder_model_name = deps.coder_model or deps.chat_model
        llm = ChatOllama(
            base_url=os.getenv("OLLAMA_URL", "http://ollama:11434"),
            model=coder_model_name,
            temperature=0,
        )

        # Build system prompt with code_context injection
        base_system = state.get("augmented_system", state.get("system", ""))
        code_context = state.get("code_context", "")
        workspace_note = f"\nYour workspace root is: {_workspace_root}\nAll file paths are relative to this root."
        if code_context:
            system_prompt = (
                f"{base_system}{workspace_note}\n\n"
                f"Relevant codebase context (tree-sitter summaries):\n{code_context}"
            )
        else:
            system_prompt = f"{base_system}{workspace_note}"

        agent = create_react_agent(llm, active_tools, prompt=system_prompt)

        response_text = ""
        handled_json_tool_call = False

        log.info(
            "graph.code_tool | model=%s | session=%s | tools=%s",
            coder_model_name,
            state.get("session_id", ""),
            [t.name for t in active_tools],
        )

        async def _run_agent_once(user_message: str) -> str:
            collected = ""
            try:
                async for event in agent.astream_events(
                    {"messages": [{"role": "user", "content": user_message}]},
                    version="v2",
                ):
                    event_type = event.get("event", "")
                    if event_type == "on_chat_model_stream":
                        chunk = event["data"].get("chunk")
                        if chunk and hasattr(chunk, "content") and chunk.content:
                            collected += chunk.content
                    elif event_type == "on_tool_start":
                        tool_name = event.get("name", "unknown")
                        log.info(
                            "graph.code_tool | tool_start | tool=%s | session=%s",
                            tool_name,
                            state.get("session_id", ""),
                        )
                    elif event_type == "on_tool_end":
                        tool_name = event.get("name", "unknown")
                        tool_output = event.get("data", {}).get("output", "")
                        log.info(
                            "graph.code_tool | tool_end | tool=%s | output_len=%d | session=%s",
                            tool_name,
                            len(str(tool_output)),
                            state.get("session_id", ""),
                        )
            except Exception as exc:
                log.error(
                    "graph.code_tool | stream error: %s | session=%s",
                    exc,
                    state.get("session_id", ""),
                    exc_info=True,
                )
                err_msg = f"Code tool error: {exc}"
                writer({"text": err_msg})
                return err_msg
            return collected

        def _extract_json_tool_payload(text: str) -> dict[str, Any] | None:
            candidate = text.strip()
            if not candidate:
                return None
            if not (candidate.startswith("{") and candidate.endswith("}")):
                first = candidate.find("{")
                last = candidate.rfind("}")
                if first != -1 and last != -1 and last > first:
                    candidate = candidate[first:last + 1]
            try:
                payload = json.loads(candidate)
            except Exception:
                return None
            if isinstance(payload, dict) and payload.get("name"):
                return payload
            return None

        def _execute_readonly_fallback_tool(payload: dict[str, Any]) -> str | None:
            tool_name = str(payload.get("name", ""))
            args = payload.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}

            if tool_name == "list_files":
                sub_path = args.get("sub_path", "") if isinstance(args, dict) else ""
                try:
                    base = _resolve_workspace_path(sub_path) if sub_path else _workspace_root
                except ValueError:
                    base = _workspace_root
                result_files: list[str] = []
                for dirpath, dirnames, filenames in os.walk(base):
                    dirnames[:] = [
                        d for d in dirnames
                        if not d.startswith(".") and d not in ("__pycache__", "node_modules", ".venv")
                    ]
                    for fname in filenames:
                        p = Path(dirpath) / fname
                        try:
                            rel = str(p.relative_to(_workspace_root))
                        except ValueError:
                            rel = str(p)
                        result_files.append(rel)
                        touched_files.add(rel)
                log.info(
                    "graph.code_tool | fallback_executed_list_files | count=%d | session=%s",
                    len(result_files),
                    state.get("session_id", ""),
                )
                return "\n".join(result_files) if result_files else "(empty)"

            if tool_name == "read_file":
                relative_path = args.get("relative_path", "") if isinstance(args, dict) else ""
                if not relative_path and isinstance(args, dict):
                    relative_path = args.get("file_path", "")
                if not relative_path:
                    return "Error: missing relative_path"
                try:
                    resolved = _resolve_workspace_path(relative_path)
                    touched_files.add(relative_path)
                    return Path(resolved).read_text(encoding="utf-8", errors="replace")
                except (FileNotFoundError, ValueError, OSError) as exc:
                    return f"Error: {exc}"

            return None

        response_text = await _run_agent_once(str(state.get("message", "")))

        # Fallback for models that emit JSON tool calls as plain text instead of
        # structured tool-call messages (observed with some Ollama coder models).
        # If we see a read-only tool call, execute it and run one follow-up turn
        # automatically so the model can proceed to a write_file call in the same turn.
        max_fallback_turns = 3
        fallback_turn = 0
        while not _pending and response_text.strip() and fallback_turn < max_fallback_turns:
            payload = _extract_json_tool_payload(response_text)
            if not payload:
                break

            tool_name = str(payload.get("name", ""))
            if tool_name == "write_file":
                args = payload.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                if isinstance(args, dict):
                    relative_path = str(args.get("relative_path", "") or "").strip()
                    content = args.get("content", "")
                    if relative_path and isinstance(content, str):
                        try:
                            resolved = _resolve_workspace_path(relative_path)
                            try:
                                original = Path(resolved).read_text(encoding="utf-8", errors="replace")
                            except FileNotFoundError:
                                original = ""

                            diff = _make_unified_diff(relative_path, original, content)
                            _pending["path"] = resolved
                            _pending["content"] = content
                            _pending["relative_path"] = relative_path
                            touched_files.add(relative_path)

                            if diff:
                                if str(state.get("source", "text")).lower() == "voice":
                                    confirm_msg = _write_summary_for_voice(relative_path, diff)
                                else:
                                    confirm_msg = (
                                        f"```diff\n{diff}\n```\n\n"
                                        "Type **yes** to confirm the write, or describe any changes you want first."
                                    )
                            else:
                                confirm_msg = "No changes detected — the file already has that content."

                            writer({"text": confirm_msg})
                            response_text = confirm_msg
                            handled_json_tool_call = True
                            log.info(
                                "graph.code_tool | fallback_tool_parse=write_file | path=%s | session=%s",
                                relative_path,
                                state.get("session_id", ""),
                            )
                        except ValueError as exc:
                            msg = f"Write blocked: {exc}"
                            writer({"text": msg})
                            response_text = msg
                            handled_json_tool_call = True
                break

            readonly_output = _execute_readonly_fallback_tool(payload)
            if readonly_output is None:
                log.warning(
                    "graph.code_tool | unknown_tool_in_fallback | tool=%s | session=%s",
                    tool_name,
                    state.get("session_id", ""),
                )
                break

            fallback_turn += 1
            followup_prompt = (
                "You emitted a tool call as JSON text instead of a structured tool call.\n"
                f"Tool name: {tool_name}\n"
                "Tool output:\n"
                f"{readonly_output}\n\n"
                f"Original user request: {state.get('message', '')}\n"
                "Continue solving the request now. If you need to create or edit files, "
                "emit a write_file JSON tool call with relative_path and content."
            )
            log.info(
                "graph.code_tool | fallback_followup_turn=%d | tool=%s | session=%s",
                fallback_turn,
                tool_name,
                state.get("session_id", ""),
            )
            response_text = await _run_agent_once(followup_prompt)

        if not handled_json_tool_call and response_text.strip() and not response_text.startswith("Code tool error:"):
            writer({"text": response_text})

        result: dict[str, Any] = {
            "response_text": response_text.strip(),
            "response_model": coder_model_name,
            "active_files": sorted(touched_files),
        }
        if _pending:
            result["pending_write"] = dict(_pending)
            result["awaiting_confirmation"] = True
            log.info(
                "graph.code_tool | pending_write=%s | session=%s",
                _pending.get("relative_path"),
                state.get("session_id", ""),
            )
        else:
            # Clear any stale pending_write from previous turn
            result["pending_write"] = {}
            result["awaiting_confirmation"] = False

        return result

    async def write_executor(state: AssistantState) -> dict[str, Any]:
        """Execute a confirmed file write from pending_write state."""
        writer = get_stream_writer()
        pending = state.get("pending_write") or {}
        touched_files: set[str] = set(state.get("active_files", []))

        if not pending or not pending.get("path") or not pending.get("content"):
            msg = "No pending write to execute."
            writer({"text": msg})
            return {
                "response_text": msg,
                "pending_write": {},
                "awaiting_confirmation": False,
            }

        resolved_path = pending["path"]
        relative_path = pending.get("relative_path", resolved_path)
        content = pending["content"]

        # Re-validate path (defence in depth — state could be replayed)
        try:
            _resolve_workspace_path(relative_path)
        except ValueError as exc:
            msg = f"Write blocked: {exc}"
            log.warning("graph.write_executor | blocked | %s", exc)
            writer({"text": msg})
            return {
                "response_text": msg,
                "pending_write": {},
                "awaiting_confirmation": False,
            }

        try:
            wrote_with_tool = False
            try:
                from langchain_community.tools import WriteFileTool  # type: ignore[import-untyped]
                wf = WriteFileTool(root_dir=_workspace_root)
                wf.invoke({"file_path": relative_path, "text": content})
                wrote_with_tool = True
            except Exception:
                wrote_with_tool = False

            if not wrote_with_tool:
                Path(resolved_path).parent.mkdir(parents=True, exist_ok=True)
                Path(resolved_path).write_text(content, encoding="utf-8")

            touched_files.add(relative_path)
            msg = f"Written: `{relative_path}`"
            log.info("graph.write_executor | written | path=%s | session=%s", relative_path, state.get("session_id", ""))
        except OSError as exc:
            msg = f"Write failed: {exc}"
            log.error("graph.write_executor | os_error | path=%s | %s", relative_path, exc)

        writer({"text": msg})
        return {
            "response_text": msg,
            "response_model": deps.chat_model,
            "pending_write": {},
            "awaiting_confirmation": False,
            "active_files": sorted(touched_files),
        }

    async def coding_agent_tool(state: AssistantState) -> dict[str, Any]:
        """Confirmation gate for code-write tasks dispatched to the external coding agent.

        Presents the planned task to the user and waits for explicit confirmation
        before dispatching to AgentAPI. Sets awaiting_agent_confirmation=True and
        stores the task in pending_code_task state.
        """
        writer = get_stream_writer()
        task = str(state.get("message", "")).strip()
        modality = state.get("modality", "chat")

        if modality == "voice":
            words = task.split()
            task_preview = " ".join(words[:12]) + ("..." if len(words) > 12 else "")
            confirm_text = (
                f"Got it. {task_preview}. "
                "Say yes to confirm, or tell me what to change."
            )
        else:
            confirm_text = (
                "I'll send this task to the external coding agent:\n\n"
                f"> {task}\n\n"
                "Type **yes** to confirm, or describe what you want changed."
            )

        writer({"text": confirm_text})
        log.info(
            "graph.coding_agent_tool | awaiting_confirmation | session=%s task_len=%d",
            state.get("session_id", ""),
            len(task),
        )
        return {
            "response_text": confirm_text,
            "response_model": deps.coder_model or deps.chat_model,
            "pending_code_task": task,
            "awaiting_agent_confirmation": True,
        }

    async def coding_agent_executor(state: AssistantState) -> dict[str, Any]:
        """Execute a confirmed code-write task via the external AgentAPI adapter.

        Reads pending_code_task and code_context from state, calls coding_agent.run(),
        shapes the result based on modality, and clears the confirmation flags.
        """
        writer = get_stream_writer()
        task = str(state.get("pending_code_task", "") or state.get("message", "")).strip()
        code_context = state.get("code_context", "") or ""
        session_id = state.get("session_id", "")
        modality = state.get("modality", "chat")

        _base: dict[str, Any] = {
            "pending_code_task": "",
            "awaiting_agent_confirmation": False,
            "response_model": deps.coder_model or deps.chat_model,
        }

        if not task:
            msg = "No pending coding task to execute."
            writer({"text": msg})
            return {**_base, "response_text": msg}

        log.info(
            "graph.coding_agent_executor | dispatching | session=%s task_len=%d context_len=%d",
            session_id,
            len(task),
            len(code_context),
        )

        try:
            from tools.coding_agent import run as _agent_run
        except ImportError:
            msg = "Coding agent tool is not available. Check backend/tools/coding_agent.py."
            writer({"text": msg})
            return {**_base, "response_text": msg}

        result = await _agent_run({
            "task": task,
            "context": code_context,
            "session_id": session_id,
        })

        if not result.ok:
            msg = result.error or "The coding agent returned an error."
            writer({"text": msg})
            return {**_base, "response_text": msg}

        result_text: str = result.data.get("result", "")
        files_changed: list[str] = result.data.get("files_changed", [])

        if modality == "voice":
            if files_changed:
                names = [f.split("/")[-1] for f in files_changed[:3]]
                files_summary = ", ".join(names)
                if len(files_changed) > 3:
                    files_summary += f" and {len(files_changed) - 3} more"
                spoken_base = f"Done. Modified {files_summary}."
            else:
                spoken_base = "Done. The coding agent completed the task."
            full_spoken = spoken_base + (f" {result_text}" if result_text else "")
            response_text = await _compress_response_for_voice(full_spoken, deps.chat_model)
            writer({"text": response_text})
        else:
            parts: list[str] = []
            if files_changed:
                files_list = "\n".join(f"- `{f}`" for f in files_changed)
                parts.append(f"**Files changed:**\n{files_list}")
            if result_text:
                parts.append(result_text)
            if not parts:
                parts.append("The coding agent completed the task.")
            response_text = "\n\n".join(parts)
            writer({"text": response_text})

        log.info(
            "graph.coding_agent_executor | done | files=%d session=%s",
            len(files_changed),
            session_id,
        )
        return {**_base, "response_text": response_text}

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
            compressed += chunk
        result = compressed.strip()
        log.info(
            "graph.responder | voice_compress | original_words=%d compressed_words=%d",
            word_count,
            len(result.split()),
        )
        return result if result else original

    async def _emit_response_chunks(
        stream: AsyncIterator[str],
        *,
        modality: str,
        compress_model: str,
    ) -> str:
        writer = get_stream_writer()
        if modality == "voice":
            collected = ""
            async for chunk in stream:
                collected += chunk
            return await _compress_response_for_voice(collected, compress_model)

        response_text = ""
        async for chunk in stream:
            writer({"text": chunk})
            response_text += chunk
        return response_text

    async def responder(state: AssistantState) -> dict[str, Any]:
        writer = get_stream_writer()
        response_text = ""
        response_model = state.get("model", deps.chat_model)
        modality = state.get("modality", "chat")

        # ── Phase 14: vision path ────────────────────────────────────────────
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

    # ── Edge routing helpers ───────────────────────────────────────────────────

    def _after_intent_classifier(state: AssistantState) -> str:
        if state.get("intent") == "confirm_write":
            return "write_executor"
        if state.get("intent") == "confirm_agent_task":
            return "coding_agent_executor"
        return "memory_retrieval"

    def _after_tool_router(state: AssistantState) -> str:
        if state.get("intent") == "code-write":
            return "coding_agent_tool"
        if state.get("intent") == "code-question":
            return "code_tool"
        return "responder"

    # ── Wire the graph ─────────────────────────────────────────────────────────

    graph.add_node("intent_classifier", intent_classifier)
    graph.add_node("memory_retrieval", memory_retrieval)
    graph.add_node("tool_router", tool_router)
    graph.add_node("code_tool", code_tool)
    graph.add_node("write_executor", write_executor)
    graph.add_node("coding_agent_tool", coding_agent_tool)
    graph.add_node("coding_agent_executor", coding_agent_executor)
    graph.add_node("responder", responder)

    graph.add_edge(START, "intent_classifier")
    graph.add_conditional_edges("intent_classifier", _after_intent_classifier, {
        "memory_retrieval": "memory_retrieval",
        "write_executor": "write_executor",
        "coding_agent_executor": "coding_agent_executor",
    })
    graph.add_edge("memory_retrieval", "tool_router")
    graph.add_conditional_edges("tool_router", _after_tool_router, {
        "coding_agent_tool": "coding_agent_tool",
        "code_tool": "code_tool",
        "responder": "responder",
    })
    graph.add_edge("code_tool", END)
    graph.add_edge("write_executor", END)
    graph.add_edge("coding_agent_tool", END)
    graph.add_edge("coding_agent_executor", END)
    graph.add_edge("responder", END)

    return graph.compile(checkpointer=checkpointer)


@asynccontextmanager
async def create_assistant_graph(
    deps: AssistantGraphDependencies,
    *,
    checkpoint_path: str | None = None,
):
    async with AsyncSqliteSaver.from_conn_string(checkpoint_path or default_checkpoint_path()) as checkpointer:
        yield build_assistant_graph(deps, checkpointer=checkpointer)