"""Tests for backend/tools/coding_agent.py — coding runtime HTTP adapter.

Coverage:
  1. Empty task → immediate failure (no HTTP calls made)
  2. ConnectError on POST /message → retryable ToolResult.failure
  3. HTTP 500 on POST /message → non-retryable ToolResult.failure
  4. Success path: POST ok + events accumulated + status stable → ToolResult.ok
  5. Timeout: POST ok but agent never reaches stable → failure with retryable=True
  6. Context string is concatenated into the outgoing message body
  7. files_changed deduplication (same path in multiple events)
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import respx

import tools.coding_agent as ca
from tools.base import ToolResult

_BASE_URL = "http://localhost:3284"
_MSG_URL = f"{_BASE_URL}/message"
_STATUS_URL = f"{_BASE_URL}/status"
_EVENTS_URL = f"{_BASE_URL}/events"


# ── helpers ────────────────────────────────────────────────────────────────────

def _stable_poll(client, url, timeout):
    """Fake _poll_until_stable that yields once then returns True."""
    async def _inner(*_, **__):
        await asyncio.sleep(0)
        return True
    return _inner(client, url, timeout)


def _timeout_poll(client, url, timeout):
    """Fake _poll_until_stable that always returns False (simulates timeout)."""
    async def _inner(*_, **__):
        await asyncio.sleep(0)
        return False
    return _inner(client, url, timeout)


def _events_with(texts=None, files=None):
    """Fake _stream_events that populates the provided lists."""
    texts = texts or []
    files = files or []

    async def _inner(url, result_parts, files_changed):
        for t in texts:
            result_parts.append(t)
        for f in files:
            if f not in files_changed:
                files_changed.append(f)

    return _inner


# ── 1. Empty task ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_empty_task_returns_failure():
    result = await ca.run({"task": "", "context": "", "session_id": "s1"})
    assert not result.ok
    assert not result.retryable
    assert "task" in result.error.lower()


@pytest.mark.asyncio
async def test_whitespace_only_task_returns_failure():
    result = await ca.run({"task": "   ", "context": "", "session_id": "s1"})
    assert not result.ok


# ── 2. ConnectError on POST → retryable ───────────────────────────────────────

@pytest.mark.asyncio
async def test_connect_error_returns_retryable_failure():
    with respx.mock() as mock:
        mock.post(_MSG_URL).mock(side_effect=httpx.ConnectError("refused"))
        result = await ca.run({"task": "Add tests", "context": "", "session_id": "s1"})

    assert not result.ok
    assert result.retryable
    assert "not available" in result.error.lower() or "coding runtime" in result.error.lower()


# ── 3. HTTP error on POST → non-retryable ─────────────────────────────────────

@pytest.mark.asyncio
async def test_http_500_on_post_returns_failure():
    with respx.mock() as mock:
        mock.post(_MSG_URL).mock(return_value=httpx.Response(500))
        result = await ca.run({"task": "Add tests", "context": "", "session_id": "s1"})

    assert not result.ok
    assert not result.retryable
    assert "500" in result.error


# ── 4. Success path ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_success_returns_ok_with_result_and_files(monkeypatch):
    monkeypatch.setattr(ca, "_poll_until_stable", lambda c, u, t: _stable_poll(c, u, t))
    monkeypatch.setattr(
        ca,
        "_stream_events",
        _events_with(texts=["Done. Added type hints."], files=["utils.py"]),
    )

    with respx.mock() as mock:
        mock.post(_MSG_URL).mock(return_value=httpx.Response(200, json={}))
        result = await ca.run({
            "task": "Add type hints to utils.py",
            "context": "",
            "session_id": "s1",
        })

    assert result.ok
    assert result.data["result"] == "Done. Added type hints."
    assert result.data["files_changed"] == ["utils.py"]
    assert result.data["status"] == "success"


@pytest.mark.asyncio
async def test_success_multiple_text_events_concatenated(monkeypatch):
    monkeypatch.setattr(ca, "_poll_until_stable", lambda c, u, t: _stable_poll(c, u, t))
    monkeypatch.setattr(
        ca,
        "_stream_events",
        _events_with(texts=["Part one. ", "Part two."], files=[]),
    )

    with respx.mock() as mock:
        mock.post(_MSG_URL).mock(return_value=httpx.Response(200, json={}))
        result = await ca.run({"task": "Do something", "context": "", "session_id": "s1"})

    assert result.ok
    assert result.data["result"] == "Part one. Part two."


# ── 5. Timeout ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_timeout_returns_failure(monkeypatch):
    monkeypatch.setattr(ca, "_poll_until_stable", lambda c, u, t: _timeout_poll(c, u, t))
    monkeypatch.setattr(ca, "_stream_events", _events_with())

    with respx.mock() as mock:
        mock.post(_MSG_URL).mock(return_value=httpx.Response(200, json={}))
        result = await ca.run({"task": "Long task", "context": "", "session_id": "s1"})

    assert not result.ok
    assert result.retryable
    assert "120 seconds" in result.error or "complete within" in result.error.lower()


# ── 6. Context is included in the outgoing message ───────────────────────────

@pytest.mark.asyncio
async def test_context_is_appended_to_message_body(monkeypatch):
    monkeypatch.setattr(ca, "_poll_until_stable", lambda c, u, t: _stable_poll(c, u, t))
    monkeypatch.setattr(ca, "_stream_events", _events_with(texts=["Done."]))

    sent_bodies: list[dict] = []

    def _capture(request, *_, **__):
        sent_bodies.append(json.loads(request.content))
        return httpx.Response(200, json={})

    with respx.mock() as mock:
        mock.post(_MSG_URL).mock(side_effect=_capture)
        await ca.run({
            "task": "Fix the bug",
            "context": "def foo(): pass",
            "session_id": "s1",
        })

    assert sent_bodies, "POST /message was not called"
    body = sent_bodies[0]["message"]
    assert "Fix the bug" in body
    assert "def foo(): pass" in body


@pytest.mark.asyncio
async def test_empty_context_sends_task_only(monkeypatch):
    monkeypatch.setattr(ca, "_poll_until_stable", lambda c, u, t: _stable_poll(c, u, t))
    monkeypatch.setattr(ca, "_stream_events", _events_with(texts=["Done."]))

    sent_bodies: list[dict] = []

    def _capture(request, *_, **__):
        sent_bodies.append(json.loads(request.content))
        return httpx.Response(200, json={})

    with respx.mock() as mock:
        mock.post(_MSG_URL).mock(side_effect=_capture)
        await ca.run({"task": "Do the thing", "context": "", "session_id": "s1"})

    assert sent_bodies[0]["message"] == "Do the thing"


# ── 7. SSE event parsing (unit-level) ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_events_parses_text_and_file_events():
    """_stream_events correctly accumulates text and file-change events from SSE lines."""
    sse_lines = [
        'data: {"type": "text", "content": "Hello"}',
        'data: {"type": "local_change", "file": "src/main.py"}',
        'data: {"type": "file_change", "path": "src/util.py"}',
        'data: {"type": "write", "file": "src/util.py"}',  # duplicate path → deduplicated
        'data: {"type": "unknown_event"}',                  # unknown → ignored
        'not-a-data-line',                                  # prefix mismatch → ignored
    ]

    result_parts: list[str] = []
    files_changed: list[str] = []

    class _FakeResponse:
        async def aiter_lines(self):
            for line in sse_lines:
                yield line

    class _FakeStream:
        async def __aenter__(self):
            return _FakeResponse()

        async def __aexit__(self, *_):
            pass

    class _FakeClient:
        def stream(self, method, url, **kwargs):
            return _FakeStream()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

    import unittest.mock as _mock
    with _mock.patch("httpx.AsyncClient", return_value=_FakeClient()):
        await ca._stream_events(_BASE_URL, result_parts, files_changed)

    assert result_parts == ["Hello"]
    assert files_changed == ["src/main.py", "src/util.py"]  # deduplicated
