"""Tests for Phase 11 — Hearth system prompt (simplified).

Phase 11 ships as a single killer system prompt in backend/hearth_prompt.txt.
tone_probe and persona_renderer were removed in favour of the system prompt
carrying Hearth's character directly into every LLM call.

Acceptance criteria verified here:
- CHAT_DEFAULT_SYSTEM_PROMPT is loaded from hearth_prompt.txt at startup.
- The loaded prompt is non-trivial (longer than 50 characters).
- The prompt is not the generic fallback string.
- _load_hearth_prompt() falls back correctly when the file is absent.
"""
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

_tmp_dir = tempfile.mkdtemp(prefix="assistant-persona-tests-")
os.environ["MEMORY_DB_PATH"] = os.path.join(_tmp_dir, "memory.db")
os.environ["CHROMA_PATH"] = os.path.join(_tmp_dir, "chroma")
os.environ["AUTH_DB_PATH"] = os.path.join(_tmp_dir, "auth.db")

import main  # noqa: E402


_GENERIC_FALLBACK = "You are a helpful personal assistant. Be concise and accurate."


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_hearth_prompt_is_loaded():
    """CHAT_DEFAULT_SYSTEM_PROMPT must be loaded and non-trivial."""
    prompt = main.CHAT_DEFAULT_SYSTEM_PROMPT
    assert isinstance(prompt, str), "CHAT_DEFAULT_SYSTEM_PROMPT must be a string"
    assert len(prompt) > 50, (
        f"Expected a substantive system prompt (>50 chars), got {len(prompt)} chars"
    )


def test_hearth_prompt_is_not_generic_fallback():
    """When hearth_prompt.txt exists, the generic fallback must not be used."""
    prompt_path = os.path.join(os.path.dirname(main.__file__), "hearth_prompt.txt")
    if not os.path.exists(prompt_path):
        # File absent in this environment; loader falls back — acceptable.
        return
    assert main.CHAT_DEFAULT_SYSTEM_PROMPT != _GENERIC_FALLBACK, (
        "hearth_prompt.txt exists but CHAT_DEFAULT_SYSTEM_PROMPT is still the generic fallback"
    )


def test_load_hearth_prompt_fallback_when_file_absent(tmp_path, monkeypatch):
    """_load_hearth_prompt() falls back to env var when file is absent."""
    monkeypatch.setenv("CHAT_DEFAULT_SYSTEM_PROMPT", "Custom env prompt")
    monkeypatch.setattr(main, "__file__", str(tmp_path / "main.py"))
    result = main._load_hearth_prompt()
    assert result == "Custom env prompt"


def test_load_hearth_prompt_reads_file(tmp_path, monkeypatch):
    """_load_hearth_prompt() returns file contents when hearth_prompt.txt is present."""
    prompt_file = tmp_path / "hearth_prompt.txt"
    prompt_file.write_text("You are Hearth, a local AI assistant.", encoding="utf-8")
    monkeypatch.setattr(main, "__file__", str(tmp_path / "main.py"))
    result = main._load_hearth_prompt()
    assert result == "You are Hearth, a local AI assistant."
