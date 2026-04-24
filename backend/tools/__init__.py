"""
Tool module registry.

Adding a new tool:
  1. Create backend/tools/<name>.py implementing async def run(params: dict) -> ToolResult
  2. Call register("<name>", <module>) once, typically at module import time.
  3. No other files need to change — the dispatch() function handles routing.

Usage in main.py:
    from tools import dispatch
    result = await dispatch("weather", {"prompt": user_text, "memory": memory_store})
"""
import importlib
import logging
from types import ModuleType
from tools.base import ToolResult

log = logging.getLogger("assistant.tools")

_REGISTRY: dict[str, ModuleType] = {}


def register(name: str, module: ModuleType) -> None:
    """Register a tool module under a given name.

    The module must expose:  async def run(params: dict) -> ToolResult
    """
    if not hasattr(module, "run"):
        raise ValueError(f"Tool module '{name}' must expose a 'run' coroutine function")
    _REGISTRY[name] = module
    log.debug("tools.registry | registered tool=%s", name)


def get(name: str) -> ModuleType | None:
    """Return the registered module for *name*, or None if not registered."""
    return _REGISTRY.get(name)


async def dispatch(name: str, params: dict) -> ToolResult:
    """Look up *name* in the registry and call its run(params) coroutine.

    Returns a ToolResult.failure() if the tool is not registered or raises.
    """
    module = _REGISTRY.get(name)
    if module is None:
        log.warning("tools.dispatch | unknown tool=%s registered=%s", name, list(_REGISTRY))
        return ToolResult.failure(f"Tool '{name}' is not available.", retryable=False)
    try:
        result: ToolResult = await module.run(params)
        log.info(
            "tools.dispatch | tool=%s ok=%s retryable=%s",
            name, result.ok, result.retryable,
        )
        return result
    except Exception as exc:
        log.exception("tools.dispatch | tool=%s raised unexpectedly: %s", name, exc)
        return ToolResult.failure(
            f"Tool '{name}' encountered an unexpected error: {exc}",
            retryable=True,
        )


def _auto_register() -> None:
    """Import and register all built-in tool modules.

    Each module is responsible for calling register() on itself, but we also
    drive import here so that tools are available from first request without
    a lazy-import race condition.
    """
    _builtin_tools = ["weather"]  # extend as phases add new tools
    for tool_name in _builtin_tools:
        try:
            importlib.import_module(f"tools.{tool_name}")
        except ImportError as exc:
            log.warning("tools.auto_register | skipping tool=%s error=%s", tool_name, exc)


_auto_register()
