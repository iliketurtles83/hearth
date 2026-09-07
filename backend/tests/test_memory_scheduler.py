"""Tests for the interval "sleep" consolidation scheduler.

Covers config parsing, enable/disable gating, the idle gate (active users are
skipped, idle/unseen users are consolidated), and clean task cancellation.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from memory_scheduler import (  # noqa: E402
    _sleep_enabled,
    memory_scheduler_loop,
    read_sleep_config,
    start_memory_scheduler,
)


class _FakeStore:
    def __init__(self, last_activity, pending):
        self._last = dict(last_activity)
        self._pending = list(pending)
        self.pass_calls: list[str] = []

    def list_users_with_pending(self):
        return sorted(self._pending)

    def get_last_activity(self, user_id):
        return self._last.get(user_id)

    def run_sleep_pass(self, user_id, limit):
        self.pass_calls.append(user_id)
        return {"processed": 1, "promoted": 1, "blocked": 0, "triples": 0, "decayed": 0}


def _run_loop_ticks(store, idle_seconds=10.0, budget=0.05, interval=0.01):
    """Run the loop briefly (a few ticks) then cancel it cleanly."""

    async def _run():
        task = asyncio.create_task(
            memory_scheduler_loop(
                lambda: store,
                interval_seconds=interval,
                idle_seconds=idle_seconds,
                batch_size=50,
            )
        )
        await asyncio.sleep(budget)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_run())


def test_sleep_enabled_default_true(monkeypatch):
    monkeypatch.delenv("MEMORY_SLEEP_ENABLED", raising=False)
    assert _sleep_enabled() is True


def test_sleep_enabled_false(monkeypatch):
    monkeypatch.setenv("MEMORY_SLEEP_ENABLED", "false")
    assert _sleep_enabled() is False


def test_read_sleep_config_defaults(monkeypatch):
    for k in (
        "MEMORY_SLEEP_INTERVAL_SECONDS",
        "MEMORY_SLEEP_IDLE_SECONDS",
        "MEMORY_CONSOLIDATION_BATCH_SIZE",
    ):
        monkeypatch.delenv(k, raising=False)
    cfg = read_sleep_config()
    assert cfg["interval_seconds"] == 900
    assert cfg["idle_seconds"] == 1800
    assert cfg["batch_size"] == 50


def test_read_sleep_config_env(monkeypatch):
    monkeypatch.setenv("MEMORY_SLEEP_INTERVAL_SECONDS", "30")
    monkeypatch.setenv("MEMORY_SLEEP_IDLE_SECONDS", "600")
    monkeypatch.setenv("MEMORY_CONSOLIDATION_BATCH_SIZE", "7")
    cfg = read_sleep_config()
    assert cfg["interval_seconds"] == 30
    assert cfg["idle_seconds"] == 600
    assert cfg["batch_size"] == 7


def test_idle_gate_skips_active_users():
    now = time.time()
    store = _FakeStore(
        last_activity={"active": now, "idle": now - 1000},
        pending=["active", "idle"],
    )
    _run_loop_ticks(store, idle_seconds=10.0)
    assert "idle" in store.pass_calls
    assert "active" not in store.pass_calls


def test_unseen_user_is_treated_as_idle():
    store = _FakeStore(last_activity={}, pending=["newbie"])
    _run_loop_ticks(store, idle_seconds=10.0)
    assert "newbie" in store.pass_calls


def test_no_pending_no_calls():
    store = _FakeStore(last_activity={}, pending=[])
    _run_loop_ticks(store, idle_seconds=0.0)
    assert store.pass_calls == []


@pytest.mark.asyncio
async def test_start_memory_scheduler_disabled(monkeypatch):
    monkeypatch.setenv("MEMORY_SLEEP_ENABLED", "false")
    assert start_memory_scheduler(lambda: None) is None


@pytest.mark.asyncio
async def test_start_memory_scheduler_enabled_returns_task(monkeypatch):
    monkeypatch.setenv("MEMORY_SLEEP_ENABLED", "true")
    task = start_memory_scheduler(lambda: None)
    assert isinstance(task, asyncio.Task)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
