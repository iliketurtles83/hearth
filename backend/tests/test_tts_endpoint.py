from __future__ import annotations

import json
import os
import sys
import tempfile
import types

import pytest


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


if "memory" not in sys.modules:
    memory_stub = types.ModuleType("memory")

    class _FakeMemoryStore:
        def __init__(self, *args, **kwargs):
            return None

        def retrieve(self, *args, **kwargs):
            return []

        def ingest_user_message(self, *args, **kwargs):
            return {
                "status": "none",
                "saved": [],
                "blocked": [],
                "needs_confirmation": [],
                "deleted": 0,
                "explicit": False,
            }

        def get_preference(self, *args, **kwargs):
            return None

        def set_preference(self, *args, **kwargs):
            return None

    memory_stub.MemoryStore = _FakeMemoryStore
    sys.modules["memory"] = memory_stub


_tmp_dir = tempfile.mkdtemp(prefix="assistant-tts-endpoint-tests-")
os.environ["MEMORY_DB_PATH"] = os.path.join(_tmp_dir, "memory.db")
os.environ["CHROMA_PATH"] = os.path.join(_tmp_dir, "chroma")
os.environ["AUTH_DB_PATH"] = os.path.join(_tmp_dir, "auth.db")

import main  # noqa: E402


def _json_body(response) -> dict:
    return json.loads(response.body.decode("utf-8"))


@pytest.mark.asyncio
async def test_tts_endpoint_success(monkeypatch):
    async def _fake_synthesize(text: str, engine_name: str | None = None) -> bytes:
        assert text == "Hello from test"
        return b"RIFFtest-bytes"

    monkeypatch.setattr(main.tts, "synthesize", _fake_synthesize)

    response = await main.tts_synthesize(main.TTSRequest(text="Hello from test"))

    assert response.status_code == 200
    assert response.media_type == "audio/wav"
    assert response.body == b"RIFFtest-bytes"


@pytest.mark.asyncio
async def test_tts_endpoint_invalid_text_maps_to_400(monkeypatch):
    async def _fake_synthesize(text: str, engine_name: str | None = None) -> bytes:
        raise main.tts.TTSError("Text must not be empty", code="TTS_INVALID_TEXT", retryable=False)

    monkeypatch.setattr(main.tts, "synthesize", _fake_synthesize)

    response = await main.tts_synthesize(main.TTSRequest(text="   "))
    payload = _json_body(response)

    assert response.status_code == 400
    assert payload["code"] == "TTS_INVALID_TEXT"
    assert payload["retryable"] is False


@pytest.mark.asyncio
async def test_tts_endpoint_engine_unavailable_maps_to_503(monkeypatch):
    async def _fake_synthesize(text: str, engine_name: str | None = None) -> bytes:
        raise main.tts.TTSError("engine unavailable", code="TTS_ENGINE_UNAVAILABLE", retryable=False)

    monkeypatch.setattr(main.tts, "synthesize", _fake_synthesize)

    response = await main.tts_synthesize(main.TTSRequest(text="hello"))
    payload = _json_body(response)

    assert response.status_code == 503
    assert payload["code"] == "TTS_ENGINE_UNAVAILABLE"
    assert payload["retryable"] is False


@pytest.mark.asyncio
async def test_tts_endpoint_retryable_runtime_error_maps_to_502(monkeypatch):
    async def _fake_synthesize(text: str, engine_name: str | None = None) -> bytes:
        raise main.tts.TTSError("runtime failure", code="TTS_SYNTHESIS_FAILED", retryable=True)

    monkeypatch.setattr(main.tts, "synthesize", _fake_synthesize)

    response = await main.tts_synthesize(main.TTSRequest(text="hello"))
    payload = _json_body(response)

    assert response.status_code == 502
    assert payload["code"] == "TTS_SYNTHESIS_FAILED"
    assert payload["retryable"] is True
