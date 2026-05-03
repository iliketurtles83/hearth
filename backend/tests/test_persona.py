"""Tests for Phase 11 — Hearth system prompt (simplified).

Phase 11 ships as a single killer system prompt in backend/hearth_prompt.txt.
tone_probe and persona_renderer were removed in favour of the system prompt
carrying Hearth's character directly into every LLM call.

Acceptance criteria verified here:
- CHAT_DEFAULT_SYSTEM_PROMPT is loaded from hearth_prompt.txt at startup.
- CODE_DEFAULT_SYSTEM_PROMPT is loaded from hearth_coder_prompt.txt at startup.
- Both loaded prompts are non-trivial (longer than 50 characters).
- Neither prompt is its generic fallback string when the file exists.
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


_GENERIC_CHAT_FALLBACK = "You are a helpful personal assistant. Be concise and accurate."
_GENERIC_CODE_FALLBACK = "You are a helpful coding assistant. Be concise and accurate."


# ── Chat prompt tests ─────────────────────────────────────────────────────────

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
        return
    assert main.CHAT_DEFAULT_SYSTEM_PROMPT != _GENERIC_CHAT_FALLBACK, (
        "hearth_prompt.txt exists but CHAT_DEFAULT_SYSTEM_PROMPT is still the generic fallback"
    )


def test_load_hearth_prompt_fallback_when_file_absent(tmp_path, monkeypatch):
    """_load_hearth_prompt() falls back to env var when the file is absent."""
    monkeypatch.setenv("MY_PROMPT_VAR", "Custom env prompt")
    monkeypatch.setattr(main, "__file__", str(tmp_path / "main.py"))
    result = main._load_hearth_prompt("missing_prompt.txt", "MY_PROMPT_VAR", "hardcoded fallback")
    assert result == "Custom env prompt"


def test_load_hearth_prompt_reads_file(tmp_path, monkeypatch):
    """_load_hearth_prompt() returns file contents when the file is present."""
    prompt_file = tmp_path / "hearth_prompt.txt"
    prompt_file.write_text("You are Hearth, a local AI assistant.", encoding="utf-8")
    monkeypatch.setattr(main, "__file__", str(tmp_path / "main.py"))
    result = main._load_hearth_prompt("hearth_prompt.txt", "CHAT_DEFAULT_SYSTEM_PROMPT", "fallback")
    assert result == "You are Hearth, a local AI assistant."


def test_load_hearth_prompt_uses_hardcoded_fallback_when_no_file_and_no_env(tmp_path, monkeypatch):
    """_load_hearth_prompt() uses hardcoded fallback when both file and env var are absent."""
    monkeypatch.delenv("MY_MISSING_VAR", raising=False)
    monkeypatch.setattr(main, "__file__", str(tmp_path / "main.py"))
    result = main._load_hearth_prompt("missing.txt", "MY_MISSING_VAR", "last resort")
    assert result == "last resort"


# ── Coder prompt tests ────────────────────────────────────────────────────────

def test_hearth_coder_prompt_is_loaded():
    """CODE_DEFAULT_SYSTEM_PROMPT must be loaded and non-trivial."""
    prompt = main.CODE_DEFAULT_SYSTEM_PROMPT
    assert isinstance(prompt, str), "CODE_DEFAULT_SYSTEM_PROMPT must be a string"
    assert len(prompt) > 50, (
        f"Expected a substantive coder prompt (>50 chars), got {len(prompt)} chars"
    )


def test_hearth_coder_prompt_is_not_generic_fallback():
    """When hearth_coder_prompt.txt exists, the generic coding fallback must not be used."""
    prompt_path = os.path.join(os.path.dirname(main.__file__), "hearth_coder_prompt.txt")
    if not os.path.exists(prompt_path):
        return
    assert main.CODE_DEFAULT_SYSTEM_PROMPT != _GENERIC_CODE_FALLBACK, (
        "hearth_coder_prompt.txt exists but CODE_DEFAULT_SYSTEM_PROMPT is still the generic fallback"
    )


def test_load_hearth_coder_prompt_reads_file(tmp_path, monkeypatch):
    """_load_hearth_prompt() reads hearth_coder_prompt.txt for the coder prompt."""
    prompt_file = tmp_path / "hearth_coder_prompt.txt"
    prompt_file.write_text("You are Hearth's coder.", encoding="utf-8")
    monkeypatch.setattr(main, "__file__", str(tmp_path / "main.py"))
    result = main._load_hearth_prompt("hearth_coder_prompt.txt", "CODE_DEFAULT_SYSTEM_PROMPT", "fallback")
    assert result == "You are Hearth's coder."


def test_chat_and_coder_prompts_are_independent():
    """CHAT_DEFAULT_SYSTEM_PROMPT and CODE_DEFAULT_SYSTEM_PROMPT must be different strings."""
    assert main.CHAT_DEFAULT_SYSTEM_PROMPT != main.CODE_DEFAULT_SYSTEM_PROMPT, (
        "Chat and coder prompts must be distinct — they serve different models and contexts"
    )
