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

    memory_stub.MemoryStore = _FakeMemoryStore
    sys.modules["memory"] = memory_stub


_tmp_dir = tempfile.mkdtemp(prefix="assistant-chat-voice-tests-")
os.environ["MEMORY_DB_PATH"] = os.path.join(_tmp_dir, "memory.db")
os.environ["CHROMA_PATH"] = os.path.join(_tmp_dir, "chroma")
os.environ["AUTH_DB_PATH"] = os.path.join(_tmp_dir, "auth.db")

import main  # noqa: E402


def test_normalize_chat_source_defaults_to_text():
    assert main._normalize_chat_source(None) == "text"
    assert main._normalize_chat_source("") == "text"
    assert main._normalize_chat_source("unknown") == "text"


def test_normalize_chat_source_accepts_voice_and_text():
    assert main._normalize_chat_source("voice") == "voice"
    assert main._normalize_chat_source("text") == "text"
    assert main._normalize_chat_source(" VOICE ") == "voice"


def test_voice_tts_metadata_for_voice_source():
    payload = main._voice_tts_metadata("voice")
    assert payload is not None
    assert payload["voice"]["source"] == "voice"
    assert payload["voice"]["tts_endpoint"] == "/tts"
    assert payload["voice"]["tts_ready"] is True


def test_voice_tts_metadata_for_text_source_is_none():
    assert main._voice_tts_metadata("text") is None


def test_transcribe_is_not_unprotected_path():
    assert "/transcribe" not in main._UNPROTECTED_PATHS


@pytest.mark.asyncio
async def test_transcribe_returns_503_when_model_not_loaded(monkeypatch):
    def _explode(*_args, **_kwargs):
        raise RuntimeError("simulated: model unavailable / offline")

    monkeypatch.setattr(main, "get_whisper_model", _explode)

    class _FakeUploadFile:
        async def read(self, _n=None):
            return b"\x00" * 16

    response = await main.transcribe(_FakeUploadFile())
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 503
    assert payload["code"] == "MODEL_NOT_LOADED"
    assert payload["retryable"] is False
    assert "download-whisper-model.sh" in payload["error"]


def test_resolve_whisper_model_source_honors_env_override(tmp_path, monkeypatch):
    target = str(tmp_path)
    monkeypatch.setattr(main, "WHISPER_MODEL_DIR", target)
    assert main._resolve_whisper_model_source() == target


def test_resolve_whisper_model_source_falls_back_to_size_when_local_absent(monkeypatch):
    monkeypatch.setattr(main, "WHISPER_MODEL_DIR", "")
    monkeypatch.setattr(main, "WHISPER_MODEL", "definitely-missing-model-xyz")
    assert main._resolve_whisper_model_source() == "definitely-missing-model-xyz"
