from __future__ import annotations

import os
import sys
import tempfile
import types


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
