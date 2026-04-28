from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import wave

import pytest

from tts import TTSError
from tts.engines import piper


def _write_silent_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 160)


def test_engine_requires_model_path(monkeypatch):
    monkeypatch.delenv("TTS_PIPER_MODEL", raising=False)
    monkeypatch.setattr(piper.shutil, "which", lambda _: "/usr/bin/piper")
    with pytest.raises(TTSError) as exc:
        piper.Engine()
    assert exc.value.code == "TTS_PIPER_MODEL_MISSING"


def test_engine_requires_existing_model(monkeypatch, tmp_path):
    model = tmp_path / "missing.onnx"
    monkeypatch.setenv("TTS_PIPER_MODEL", str(model))
    monkeypatch.setattr(piper.shutil, "which", lambda _: "/usr/bin/piper")
    with pytest.raises(TTSError) as exc:
        piper.Engine()
    assert exc.value.code == "TTS_PIPER_MODEL_NOT_FOUND"


def test_engine_requires_binary(monkeypatch, tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"x")
    monkeypatch.setenv("TTS_PIPER_MODEL", str(model))
    monkeypatch.setattr(piper.shutil, "which", lambda _: None)
    with pytest.raises(TTSError) as exc:
        piper.Engine()
    assert exc.value.code == "TTS_PIPER_BIN_NOT_FOUND"


def test_build_cmd_includes_rate_and_speaker(monkeypatch, tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"x")
    monkeypatch.setenv("TTS_PIPER_MODEL", str(model))
    monkeypatch.setenv("TTS_PIPER_RATE", "2.0")
    monkeypatch.setenv("TTS_PIPER_SPEAKER", "3")
    monkeypatch.setattr(piper.shutil, "which", lambda _: "/usr/bin/piper")

    engine = piper.Engine()
    cmd = engine._build_cmd(tmp_path / "out.wav")

    assert "--length_scale" in cmd
    idx = cmd.index("--length_scale")
    assert cmd[idx + 1] == "0.5000"
    assert "--speaker" in cmd


@pytest.mark.asyncio
async def test_synthesize_returns_wav_bytes(monkeypatch, tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"x")
    monkeypatch.setenv("TTS_PIPER_MODEL", str(model))
    monkeypatch.setattr(piper.shutil, "which", lambda _: "/usr/bin/piper")

    def fake_run(cmd, input, capture_output, text, check):
        out_path = Path(cmd[cmd.index("--output_file") + 1])
        _write_silent_wav(out_path)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(piper.subprocess, "run", fake_run)

    engine = piper.Engine()
    audio = await engine.synthesize("hello")

    assert isinstance(audio, bytes)
    assert len(audio) > 44
    assert audio[:4] == b"RIFF"


@pytest.mark.asyncio
async def test_synthesize_handles_cli_failure(monkeypatch, tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"x")
    monkeypatch.setenv("TTS_PIPER_MODEL", str(model))
    monkeypatch.setattr(piper.shutil, "which", lambda _: "/usr/bin/piper")

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=1, stderr="bad request")

    monkeypatch.setattr(piper.subprocess, "run", fake_run)

    engine = piper.Engine()
    with pytest.raises(TTSError) as exc:
        await engine.synthesize("hello")

    assert exc.value.code == "TTS_PIPER_CLI_FAILED"
