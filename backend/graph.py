from __future__ import annotations

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

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

CHAT_TOKEN_BUDGET = int(os.getenv("CHAT_TOKEN_BUDGET", "1500"))
CHAT_MAX_TURNS = int(os.getenv("CHAT_MAX_TURNS", "24"))


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
                lineterm="",
            )
        )
        return "".join(lines) if lines else ""

    _CONFIRM_PATTERN = re.compile(
        r"^\s*(yes|confirm|approve|go ahead|do it|write it|apply|proceed)\s*[.!]?\s*$",
        re.IGNORECASE,
    )

    # ── Nodes ─────────────────────────────────────────────────────────────────

    async def intent_classifier(state: AssistantState) -> dict[str, Any]:
        writer = get_stream_writer()

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

        decision = await deps.router_route(state["message"])
        route_type = "tool" if getattr(decision, "tool", None) else ("cloud" if decision.use_cloud else "local")
        if decision.intent == "code":
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
        memory_hits_all = deps.memory_store.retrieve(state["user_id"], state["message"])
        inject_memory = _should_inject_memory(state["intent"], memory_hits_all, state["message"])
        memory_hits = memory_hits_all if inject_memory else []
        system_with_summary = _augment_system_with_session_summary(state["system"], session_summary)
        augmented_system = _augment_system_with_memories(system_with_summary, memory_hits)

        # Phase 10b: inject code context for code intents
        code_context = ""
        if state.get("intent") == "code" and deps.chroma_path:
            from tools.code_indexer import query_code_context
            snippets = query_code_context(state["message"], deps.chroma_path)
            if snippets:
                code_context = "\n\n---\n".join(snippets)
                log.info(
                    "graph.memory_retrieval | code_context_snippets=%d | session=%s",
                    len(snippets),
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

        # ── Custom tools ──────────────────────────────────────────────────────
        try:
            from langchain_core.tools import tool as lc_tool
        except ImportError:
            writer({"text": "langchain-core is not installed. Run: pip install langchain-core"})
            return {"response_text": "langchain-core missing", "response_model": ""}

        @lc_tool
        def read_file(relative_path: str) -> str:
            """Read a file from the code workspace. Input: path relative to workspace root."""
            try:
                resolved = _resolve_workspace_path(relative_path)
            except ValueError as exc:
                return f"Error: {exc}"
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
            try:
                original = Path(resolved).read_text(encoding="utf-8", errors="replace")
            except FileNotFoundError:
                original = ""
            diff = _make_unified_diff(relative_path, original, content)
            _pending["path"] = resolved
            _pending["content"] = content
            _pending["relative_path"] = relative_path
            if diff:
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

        agent = create_react_agent(llm, active_tools, state_modifier=system_prompt)

        messages = [{"role": "user", "content": state["message"]}]
        response_text = ""

        log.info(
            "graph.code_tool | model=%s | session=%s | tools=%s",
            coder_model_name,
            state.get("session_id", ""),
            [t.name for t in active_tools],
        )

        try:
            async for event in agent.astream_events(
                {"messages": messages},
                version="v2",
            ):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        writer({"text": chunk.content})
                        response_text += chunk.content
        except Exception as exc:
            log.error("graph.code_tool | stream error: %s", exc, exc_info=True)
            err_msg = f"Code tool error: {exc}"
            writer({"text": err_msg})
            response_text = err_msg

        result: dict[str, Any] = {
            "response_text": response_text.strip(),
            "response_model": coder_model_name,
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
            Path(resolved_path).parent.mkdir(parents=True, exist_ok=True)
            Path(resolved_path).write_text(content, encoding="utf-8")
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
        }

    async def responder(state: AssistantState) -> dict[str, Any]:
        writer = get_stream_writer()
        response_text = ""
        response_model = state.get("model", deps.chat_model)

        if state.get("tool"):
            tool_result = await deps.tool_dispatch(
                state["tool"],
                {"prompt": state["message"], "user_id": state["user_id"], "memory": deps.memory_store},
            )
            if getattr(tool_result, "ok", False):
                response_model = deps.chat_model
                summary_request = PromptRequest(
                    message=_tool_summary_prompt(state["message"], getattr(tool_result, "data", {})),
                    system=state["augmented_system"],
                )
                async for chunk in deps.stream_local(summary_request, model_name=deps.chat_model):
                    writer({"text": chunk})
                    response_text += chunk
            else:
                response_text = getattr(tool_result, "error", "The tool returned no data.") or "The tool returned no data."
                writer({"text": response_text})
        elif state.get("use_cloud"):
            response_model = deps.cloud_model
            try:
                async for chunk in deps.stream_cloud(state["augmented_system"], state["cloud_messages"]):
                    writer({"text": chunk})
                    response_text += chunk
            except Exception:
                log.warning("graph.cloud_fallback | session_id=%s", state.get("session_id", ""))
                response_model = deps.chat_model
                writer({"notice": "Cloud unavailable \u2014 responding with local model"})
                writer({"model": deps.chat_model, "intent": state.get("intent", ""), "confidence": state.get("confidence", 0.0), "fallback": True})
                local_request = PromptRequest(message=state["local_prompt"], system=state["augmented_system"])
                async for chunk in deps.stream_local(local_request, model_name=deps.chat_model):
                    writer({"text": chunk})
                    response_text += chunk
        else:
            local_request = PromptRequest(message=state["local_prompt"], system=state["augmented_system"])
            async for chunk in deps.stream_local(local_request, model_name=state["model"]):
                writer({"text": chunk})
                response_text += chunk

        return {"response_text": response_text.strip(), "response_model": response_model}

    # ── Edge routing helpers ───────────────────────────────────────────────────

    def _after_intent_classifier(state: AssistantState) -> str:
        if state.get("intent") == "confirm_write":
            return "write_executor"
        return "memory_retrieval"

    def _after_tool_router(state: AssistantState) -> str:
        if state.get("intent") == "code":
            return "code_tool"
        return "responder"

    # ── Wire the graph ─────────────────────────────────────────────────────────

    graph.add_node("intent_classifier", intent_classifier)
    graph.add_node("memory_retrieval", memory_retrieval)
    graph.add_node("tool_router", tool_router)
    graph.add_node("code_tool", code_tool)
    graph.add_node("write_executor", write_executor)
    graph.add_node("responder", responder)

    graph.add_edge(START, "intent_classifier")
    graph.add_conditional_edges("intent_classifier", _after_intent_classifier, {
        "memory_retrieval": "memory_retrieval",
        "write_executor": "write_executor",
    })
    graph.add_edge("memory_retrieval", "tool_router")
    graph.add_conditional_edges("tool_router", _after_tool_router, {
        "code_tool": "code_tool",
        "responder": "responder",
    })
    graph.add_edge("code_tool", END)
    graph.add_edge("write_executor", END)
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