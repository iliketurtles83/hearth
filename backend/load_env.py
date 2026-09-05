"""Single entry point for loading the project ``.env`` file.

Several backend modules read ``os.getenv()`` at import time to bake configuration
into module-level constants. Those values are captured once, when the module is
first imported, so the environment must already be populated — otherwise they
silently fall back to hardcoded defaults. The env-reading modules are:

    routing_config, intents, embedding_router, memory, graph, auth,
    and the routes/* factory modules (e.g. auth_routes).

This module calls ``load_dotenv()`` exactly once and is the single,
well-defined entry point for that step. ``main.py`` imports it as its first
local import to guarantee the invariant:

    any module that reads env at import time must be imported (directly or
    transitively) AFTER ``load_env``.

A test (or script) that imports one of those modules standalone — without first
importing ``main`` or ``load_env`` — must set the relevant env vars itself.
"""
from dotenv import load_dotenv

load_dotenv()
