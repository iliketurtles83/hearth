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
    force_code: bool                  # True for /code endpoint to bypass classifier
    # Phase 10c — responder modality split
    modality: str                     # "voice" or "chat"; set by /chat endpoint from request source
    tone: str | None                  # populated by Phase 11 tone_probe; null until then
    # Phase 11 — personality and affect layer
    persona: dict[str, Any]           # {name, style, warmth, formality}; empty dict = unconfigured


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
                    "intent": "code",
                    "confidence": 1.0,
                    "route_type": "code",
                    "needs_memory": True,
                    "tool": None,
                    "planner_status": "forced",
                    "reasoning_summary": "",
                }
            })
            return {
                "intent": "code",
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

    _TONE_LABELS = frozenset({"calm", "curious", "frustrated", "excited", "uncertain", "urgent"})
    _TONE_PROBE_SYSTEM = (
        "Classify the emotional tone of the user's message with a single word. "
        "Choose ONLY from: calm, curious, frustrated, excited, uncertain, urgent. "
        "Reply with exactly one word and nothing else."
    )

    async def _probe_tone(message: str) -> str:
        """Return an affect label for *message*.  Fallback to 'calm' on any error."""
        if len(message.split()) < 5:
            return "calm"
        try:
            probe_request = PromptRequest(
                message=message,
                system=_TONE_PROBE_SYSTEM,
            )
            collected = ""
            async for chunk in deps.stream_local(probe_request, model_name=deps.chat_model):
                collected += chunk
                if len(collected) > 30:
                    break
            label = collected.strip().lower().split()[0] if collected.strip() else "calm"
            return label if label in _TONE_LABELS else "calm"
        except Exception as exc:  # noqa: BLE001
            log.debug("graph.tone_probe | fallback | %s", exc)
            return "calm"

    async def memory_retrieval(state: AssistantState) -> dict[str, Any]:
        history = list(state.get("history", []))
        session_summary = str(state.get("session_summary", "") or "")
        selected_history, history_tokens, truncated, summary_tokens = _select_history_for_budget(
            messages=history,
            system=state["system"],
            current_user_message=state["message"],
            summary_text=session_summary,
        )
        # Run memory retrieval and tone probe concurrently.
        memory_hits_all, tone = await asyncio.gather(
            asyncio.to_thread(deps.memory_store.retrieve, state["user_id"], state["message"]),
            _probe_tone(state["message"]),
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
        if state.get("intent") == "code":
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
            "tone": tone,
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

    async def responder(state: AssistantState) -> dict[str, Any]:
        writer = get_stream_writer()
        response_text = ""
        response_model = state.get("model", deps.chat_model)
        modality = state.get("modality", "chat")

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
                if modality == "voice":
                    collected = ""
                    async for chunk in deps.stream_local(summary_request, model_name=deps.chat_model):
                        collected += chunk
                    response_text = await _compress_response_for_voice(collected, deps.chat_model)
                    writer({"text": response_text})
                else:
                    async for chunk in deps.stream_local(summary_request, model_name=deps.chat_model):
                        writer({"text": chunk})
                        response_text += chunk
            else:
                response_text = getattr(tool_result, "error", "The tool returned no data.") or "The tool returned no data."
                writer({"text": response_text})
        elif state.get("use_cloud"):
            response_model = deps.cloud_model
            try:
                if modality == "voice":
                    collected = ""
                    async for chunk in deps.stream_cloud(state["augmented_system"], state["cloud_messages"]):
                        collected += chunk
                    response_text = await _compress_response_for_voice(collected, deps.chat_model)
                    writer({"text": response_text})
                else:
                    async for chunk in deps.stream_cloud(state["augmented_system"], state["cloud_messages"]):
                        writer({"text": chunk})
                        response_text += chunk
            except Exception:
                log.warning("graph.cloud_fallback | session_id=%s", state.get("session_id", ""))
                response_model = deps.chat_model
                writer({"notice": "Cloud unavailable \u2014 responding with local model"})
                writer({"model": deps.chat_model, "intent": state.get("intent", ""), "confidence": state.get("confidence", 0.0), "fallback": True})
                local_request = PromptRequest(message=state["local_prompt"], system=state["augmented_system"])
                if modality == "voice":
                    collected = ""
                    async for chunk in deps.stream_local(local_request, model_name=deps.chat_model):
                        collected += chunk
                    response_text = await _compress_response_for_voice(collected, deps.chat_model)
                    writer({"text": response_text})
                else:
                    async for chunk in deps.stream_local(local_request, model_name=deps.chat_model):
                        writer({"text": chunk})
                        response_text += chunk
        else:
            local_request = PromptRequest(message=state["local_prompt"], system=state["augmented_system"])
            if modality == "voice":
                collected = ""
                async for chunk in deps.stream_local(local_request, model_name=state["model"]):
                    collected += chunk
                response_text = await _compress_response_for_voice(collected, deps.chat_model)
                writer({"text": response_text})
            else:
                async for chunk in deps.stream_local(local_request, model_name=state["model"]):
                    writer({"text": chunk})
                    response_text += chunk

        return {"response_text": response_text.strip(), "response_model": response_model}

    _PERSONA_VOICE_SYSTEM = (
        "You are a style adapter for a voice assistant. "
        "Your job is to reword the assistant response to match the configured persona tone and style. "
        "Rules:\n"
        "1. Preserve ALL factual content: every number, name, date, measurement, and technical term must appear verbatim.\n"
        "2. Do not shorten or lengthen the response significantly.\n"
        "3. Do not add new information or ask follow-up questions.\n"
        "4. Output only the reworded response — no preamble or labels."
    )
    _PERSONA_CHAT_SYSTEM = (
        "You are a style adapter for a chat assistant. "
        "Your job is to lightly rephrase the assistant response to match the configured persona tone and style. "
        "Rules:\n"
        "1. Preserve ALL factual content: every number, name, date, measurement, and technical term must appear verbatim.\n"
        "2. Preserve all markdown formatting exactly: headers (##), bullets (-), bold (**), code blocks, etc.\n"
        "3. Do not change the length. Do not add new information. Do not ask follow-up questions.\n"
        "4. Output only the rephrased response — no preamble or labels."
    )

    async def persona_renderer(state: AssistantState) -> dict[str, Any]:
        """Apply persona styling to response_text.  No-op when persona is unconfigured."""
        response_text = str(state.get("response_text", ""))
        persona = state.get("persona") or {}
        tone = state.get("tone") or "calm"
        modality = state.get("modality", "chat")

        style = str(persona.get("style") or "").strip().lower()
        name = str(persona.get("name") or "").strip()

        # No-op: unconfigured or neutral+calm (no value in an LLM call)
        if not persona or (style in ("", "neutral") and tone == "calm"):
            return {"response_text": response_text}

        persona_desc_parts = []
        if name:
            persona_desc_parts.append(f"Persona name: {name}")
        if style and style != "neutral":
            persona_desc_parts.append(f"Style: {style}")
        warmth = persona.get("warmth")
        if warmth is not None:
            persona_desc_parts.append(f"Warmth level: {warmth}/5")
        formality = str(persona.get("formality") or "").strip()
        if formality:
            persona_desc_parts.append(f"Formality: {formality}")
        persona_desc_parts.append(f"User's detected tone: {tone}")
        persona_desc = "\n".join(persona_desc_parts)

        if modality == "voice":
            system_prompt = _PERSONA_VOICE_SYSTEM + f"\n\nPersona configuration:\n{persona_desc}"
        else:
            system_prompt = _PERSONA_CHAT_SYSTEM + f"\n\nPersona configuration:\n{persona_desc}"

        render_request = PromptRequest(
            message=f"Original response:\n{response_text}\n\nStyled response:",
            system=system_prompt,
        )
        styled = ""
        try:
            async for chunk in deps.stream_local(render_request, model_name=deps.chat_model):
                styled += chunk
            result = styled.strip() or response_text
        except Exception as exc:  # noqa: BLE001
            log.warning("graph.persona_renderer | error, returning original | %s", exc)
            result = response_text

        log.info(
            "graph.persona_renderer | modality=%s | tone=%s | style=%s | session=%s",
            modality, tone, style, state.get("session_id", ""),
        )
        return {"response_text": result}

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
    graph.add_node("persona_renderer", persona_renderer)

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
    graph.add_edge("responder", "persona_renderer")
    graph.add_edge("persona_renderer", END)

    return graph.compile(checkpointer=checkpointer)


@asynccontextmanager
async def create_assistant_graph(
    deps: AssistantGraphDependencies,
    *,
    checkpoint_path: str | None = None,
):
    async with AsyncSqliteSaver.from_conn_string(checkpoint_path or default_checkpoint_path()) as checkpointer:
        yield build_assistant_graph(deps, checkpointer=checkpointer)