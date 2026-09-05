from __future__ import annotations

import os
import sys
import tempfile
import types

import pytest


# main (transitively) imports musicpd. Fake it defensively so the test never
# opens a real MPD connection, mirroring test_chat_sessions.py.
if "musicpd" not in sys.modules:
    fake_musicpd = types.ModuleType("musicpd")

    class _FakeMPDClient:
        def connect(self, host: str, port: int) -> None:
            return None

        def disconnect(self) -> None:
            return None

    class _FakeConnectionError(Exception):
        pass

    fake_musicpd.MPDClient = _FakeMPDClient
    fake_musicpd.ConnectionError = _FakeConnectionError
    sys.modules["musicpd"] = fake_musicpd


# Isolate on-disk state so `import main` never touches the real DBs/chroma dir.
_tmp_dir = tempfile.mkdtemp(prefix="assistant-chat-warmup-tests-")
os.environ["MEMORY_DB_PATH"] = os.path.join(_tmp_dir, "memory.db")
os.environ["CHROMA_PATH"] = os.path.join(_tmp_dir, "chroma")
os.environ["AUTH_DB_PATH"] = os.path.join(_tmp_dir, "auth.db")


import main  # noqa: E402


class _SilentError(Exception):
    """Exception with an empty str() to exercise the repr() fallback."""

    def __str__(self) -> str:
        return ""


class _FakeResponse:
    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _make_fake_client(behaviour: str):
    calls = {"n": 0, "url": None, "payload": None}

    class _Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *exc) -> bool:
            return False

        async def post(self, url: str, json=None, **kwargs):
            calls["n"] += 1
            calls["url"] = url
            calls["payload"] = json
            if behaviour == "raise":
                raise _SilentError()
            return _FakeResponse(200)

    return _Client, calls


@pytest.mark.asyncio
async def test_warmup_chat_model_success(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    fake_client, calls = _make_fake_client("ok")
    monkeypatch.setattr(main.httpx, "AsyncClient", fake_client)
    monkeypatch.setattr(main, "CHAT_MODEL_WARMUP", True)

    with caplog.at_level("INFO", logger="assistant"):
        ok = await main.warmup_chat_model()

    assert ok is True
    assert calls["n"] == 1
    assert calls["url"] == f"{main.OLLAMA_URL}/api/generate"
    payload = calls["payload"]
    assert payload["model"] == main.CHAT_MODEL
    assert payload["keep_alive"] == main.CHAT_MODEL_KEEP_ALIVE
    assert payload["num_predict"] == 1
    assert payload["stream"] is False
    assert "chat_warmup.ready" in caplog.text


@pytest.mark.asyncio
async def test_warmup_chat_model_failure_logs_nonempty_error(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    fake_client, _ = _make_fake_client("raise")
    monkeypatch.setattr(main.httpx, "AsyncClient", fake_client)
    monkeypatch.setattr(main, "CHAT_MODEL_WARMUP", True)

    with caplog.at_level("WARNING", logger="assistant"):
        ok = await main.warmup_chat_model()

    assert ok is False
    failed = [r for r in caplog.records if "chat_warmup.failed" in r.getMessage()]
    assert failed, "expected a chat_warmup.failed log record"
    message = failed[0].getMessage()
    # The empty str(_SilentError()) must fall back to repr() so the cause is
    # diagnosable (never a bare "error=").
    assert "error=" in message
    assert message.rsplit("error=", 1)[1].strip() != ""


@pytest.mark.asyncio
async def test_warmup_chat_model_disabled_skips_call(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client, calls = _make_fake_client("ok")
    monkeypatch.setattr(main.httpx, "AsyncClient", fake_client)
    monkeypatch.setattr(main, "CHAT_MODEL_WARMUP", False)

    ok = await main.warmup_chat_model()

    assert ok is False
    assert calls["n"] == 0
