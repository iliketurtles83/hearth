"""
Coding agent tool — AgentAPI HTTP adapter.

Sends tasks to an AgentAPI server (e.g. wrapping Aider) and returns structured results.
Hearth handles voice activation, intent routing, context injection, and confirmation.
The external agent handles all code generation, file editing, and agentic reasoning.

AgentAPI protocol:
  POST /message  {"message": str}               — send the task
  GET  /status   → {"status": "stable"|"running"} — poll for completion
  GET  /events   → SSE stream of agent output events

Starting the agent (local Ollama, no cloud cost):
  agentapi server -- aider --model ollama/qwen2.5-coder:7b

Normalised ToolResult.data schema:
{
    "result":        str,        # accumulated agent output text
    "files_changed": list[str],  # paths of files the agent modified
    "status":        str,        # "success"
}

Environment variables:
  CODING_AGENT_URL              Base URL of AgentAPI server (default: http://localhost:3284)
  CODING_AGENT_TIMEOUT_SECONDS  Max seconds to wait for agent to finish (default: 120)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys as _sys
from typing import Any

import httpx

from tools.base import ToolResult
import tools as _registry

log = logging.getLogger("assistant.tools.coding_agent")

CODING_AGENT_URL: str = os.environ.get("CODING_AGENT_URL", "http://localhost:3284")
CODING_AGENT_TIMEOUT_SECONDS: int = int(os.environ.get("CODING_AGENT_TIMEOUT_SECONDS", "120"))

_STATUS_POLL_INTERVAL: float = 2.0
_STATUS_POLL_TIMEOUT: float = 5.0
_POST_TIMEOUT: float = 10.0

_UNREACHABLE_MSG = (
    "The coding agent service is not running. "
    "Start it with: agentapi server -- aider --model ollama/qwen2.5-coder:7b"
)


async def _stream_events(
    url: str,
    result_parts: list[str],
    files_changed: list[str],
) -> None:
    """Stream GET /events and accumulate output into the provided lists.

    Runs as an asyncio Task; cancelled by the caller when status is stable.
    Designed to be resilient to connection errors — if the stream drops,
    log and exit silently (the caller decides success/failure via status polling).
    """
    events_url = f"{url}/events"
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "GET",
                events_url,
                timeout=httpx.Timeout(connect=5.0, read=None, write=5.0, pool=5.0),
            ) as response:
                async for raw_line in response.aiter_lines():
                    if not raw_line.startswith("data:"):
                        continue
                    payload = raw_line[5:].strip()
                    if not payload:
                        continue
                    try:
                        event: dict[str, Any] = json.loads(payload)
                    except json.JSONDecodeError:
                        log.debug("coding_agent._stream_events | non-JSON data line skipped")
                        continue

                    event_type = event.get("type", "")

                    if event_type == "text":
                        result_parts.append(event.get("content", ""))

                    elif event_type in ("local_change", "file_change", "write"):
                        path = event.get("file") or event.get("path", "")
                        if path and path not in files_changed:
                            files_changed.append(path)

    except asyncio.CancelledError:
        # Expected — caller cancels us when status polling completes.
        raise
    except Exception as exc:
        log.debug("coding_agent._stream_events | stream ended: %s", exc)


async def _poll_until_stable(
    client: httpx.AsyncClient,
    url: str,
    timeout_seconds: int,
) -> bool:
    """Poll GET /status until 'stable' or timeout.

    Returns True if the agent reached 'stable', False on timeout.
    """
    status_url = f"{url}/status"
    deadline = asyncio.get_event_loop().time() + timeout_seconds

    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(_STATUS_POLL_INTERVAL)
        try:
            resp = await client.get(status_url, timeout=_STATUS_POLL_TIMEOUT)
            resp.raise_for_status()
            if resp.json().get("status") == "stable":
                return True
        except httpx.HTTPError as exc:
            log.debug("coding_agent._poll_until_stable | poll failed: %s", exc)

    return False


async def run(params: dict) -> ToolResult:
    """Send a coding task to the AgentAPI server and return the structured result.

    Params:
        task (str):       The coding task description.
        context (str):    Optional code context snippets to inject (from ChromaDB).
        session_id (str): Optional session identifier for logging.

    Returns a ToolResult with data {"result", "files_changed", "status"} on success,
    or a failure ToolResult with a user-visible error message.
    """
    task: str = params.get("task", "").strip()
    context: str = params.get("context", "").strip()
    session_id: str = params.get("session_id", "")

    if not task:
        return ToolResult.failure("No task provided to coding agent.", retryable=False)

    url = CODING_AGENT_URL.rstrip("/")
    full_message = f"{task}\n\nContext:\n{context}" if context else task

    log.info(
        "coding_agent.run | session=%s url=%s task_len=%d context_len=%d",
        session_id,
        url,
        len(task),
        len(context),
    )

    result_parts: list[str] = []
    files_changed: list[str] = []

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(_POST_TIMEOUT)) as client:
            # Step 1: send the task to the agent.
            try:
                resp = await client.post(f"{url}/message", json={"message": full_message})
                resp.raise_for_status()
            except httpx.ConnectError:
                log.warning("coding_agent.run | unreachable: %s", url)
                return ToolResult.failure(_UNREACHABLE_MSG, retryable=True)
            except httpx.HTTPStatusError as exc:
                log.warning("coding_agent.run | POST /message failed: %s", exc)
                return ToolResult.failure(
                    f"Coding agent rejected the task (HTTP {exc.response.status_code}).",
                    retryable=False,
                )

            log.info("coding_agent.run | task dispatched, waiting for completion")

            # Step 2: concurrently stream events and poll for completion.
            events_task = asyncio.create_task(
                _stream_events(url, result_parts, files_changed)
            )
            try:
                stable = await _poll_until_stable(client, url, CODING_AGENT_TIMEOUT_SECONDS)
            finally:
                events_task.cancel()
                try:
                    await events_task
                except asyncio.CancelledError:
                    pass

    except httpx.ConnectError:
        log.warning("coding_agent.run | unreachable during polling: %s", url)
        return ToolResult.failure(_UNREACHABLE_MSG, retryable=True)

    if not stable:
        log.warning(
            "coding_agent.run | timed out after %ds | session=%s",
            CODING_AGENT_TIMEOUT_SECONDS,
            session_id,
        )
        return ToolResult.failure(
            f"Coding agent did not complete within {CODING_AGENT_TIMEOUT_SECONDS} seconds. "
            "The task may still be running.",
            retryable=True,
        )

    result_text = "".join(result_parts)
    log.info(
        "coding_agent.run | done | files_changed=%d result_len=%d session=%s",
        len(files_changed),
        len(result_text),
        session_id,
    )

    return ToolResult(
        ok=True,
        data={
            "result": result_text,
            "files_changed": files_changed,
            "status": "success",
        },
    )


# Self-register when the module is imported.
_registry.register("coding_agent", _sys.modules[__name__])
