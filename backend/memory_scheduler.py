"""Interval-driven "sleep" consolidation scheduler.

Periodically wakes and runs a memory consolidation pass for users that have
pending episodic summaries and have been idle long enough. This keeps the heavy
LLM extraction off the conversational hot path: memory is distilled while the
assistant is effectively "asleep".

The loop is a plain asyncio task (no external scheduler dependency) and is
started and cancelled from the app lifespan.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Callable

from memory import MemoryStore

log = logging.getLogger("assistant.memory_scheduler")


def _sleep_enabled() -> bool:
    return os.getenv("MEMORY_SLEEP_ENABLED", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def read_sleep_config() -> dict[str, float]:
    return {
        "interval_seconds": max(1.0, float(os.getenv("MEMORY_SLEEP_INTERVAL_SECONDS", "900"))),
        "idle_seconds": max(0.0, float(os.getenv("MEMORY_SLEEP_IDLE_SECONDS", "1800"))),
        "batch_size": max(1, int(os.getenv("MEMORY_CONSOLIDATION_BATCH_SIZE", "50"))),
    }


async def memory_scheduler_loop(
    get_store: Callable[[], MemoryStore],
    interval_seconds: float,
    idle_seconds: float,
    batch_size: int,
) -> None:
    log.info(
        "memory_scheduler.start | interval_s=%.0f idle_s=%.0f batch=%d",
        interval_seconds,
        idle_seconds,
        batch_size,
    )
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            store = get_store()
            users = await asyncio.to_thread(store.list_users_with_pending)
            if not users:
                continue
            now = time.time()
            for user_id in users:
                last = await asyncio.to_thread(store.get_last_activity, user_id)
                idle_for = (now - last) if last is not None else None
                if idle_for is not None and idle_for < idle_seconds:
                    # Still active — don't burn GPU mid-conversation.
                    continue
                try:
                    result = await asyncio.to_thread(store.run_sleep_pass, user_id, batch_size)
                except Exception as exc:
                    log.warning(
                        "memory_scheduler.pass_failed | user_id=%s error=%s",
                        user_id,
                        exc,
                    )
                    continue
                log.info(
                    "memory_scheduler.sleep_pass | user_id=%s idle_s=%.0f processed=%d promoted=%d "
                    "triples=%d decayed=%d",
                    user_id,
                    idle_for if idle_for is not None else -1,
                    result.get("processed", 0),
                    result.get("promoted", 0),
                    result.get("triples", 0),
                    result.get("decayed", 0),
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("memory_scheduler.tick_failed | error=%s", exc)


def start_memory_scheduler(get_store: Callable[[], MemoryStore]) -> asyncio.Task | None:
    """Create the sleep-scheduler task, or None when disabled.

    Callers own the returned task and must cancel it on shutdown.
    """
    if not _sleep_enabled():
        log.info("memory_scheduler.disabled")
        return None
    config = read_sleep_config()
    return asyncio.create_task(
        memory_scheduler_loop(get_store, **config),
        name="memory-sleep-scheduler",
    )
