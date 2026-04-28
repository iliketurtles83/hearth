from __future__ import annotations

import io
import wave
from types import SimpleNamespace

import pytest

from tts import TTSError
from tts.engines import kokoro


def _wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(b"\x00\x00" * 32)
    return buf.getvalue()


def test_engine_fails_when_runtime_missing(monkeypatch):
    monkeypatch.setattr(kokoro.importlib, "import_module", lambda _: (_ for _ in ()).throw(ModuleNotFoundError("x")))

    with pytest.raises(TTSError) as exc:
        kokoro.Engine()

    assert exc.value.code == "TTS_KOKORO_UNAVAILABLE"


def test_engine_invalid_speed_config(monkeypatch):
    monkeypatch.setenv("TTS_KOKORO_SPEED", "nope")
    monkeypatch.setattr(kokoro.importlib, "import_module", lambda _: SimpleNamespace(Kokoro=lambda **_: object()))

    with pytest.raises(TTSError) as exc:
        kokoro.Engine()

    assert exc.value.code == "TTS_KOKORO_CONFIG_INVALID"


@pytest.mark.asyncio
async def test_synthesize_with_runtime_tuple_output(monkeypatch):
    class FakeRuntime:
        def create(self, text: str, voice: str, lang: str, speed: float):
            return [0.0, 0.25, -0.25], 22050

    monkeypatch.setattr(kokoro.importlib, "import_module", lambda _: SimpleNamespace(Kokoro=lambda **_: FakeRuntime()))

    engine = kokoro.Engine()
    audio = await engine.synthesize("hello")

    assert isinstance(audio, bytes)
    assert audio.startswith(b"RIFF")


@pytest.mark.asyncio
async def test_synthesize_passthrough_wav_bytes(monkeypatch):
    class FakeRuntime:
        def create(self, text: str, voice: str, lang: str, speed: float):
            return _wav_bytes(), 24000

    monkeypatch.setattr(kokoro.importlib, "import_module", lambda _: SimpleNamespace(Kokoro=lambda **_: FakeRuntime()))

    engine = kokoro.Engine()
    audio = await engine.synthesize("hello")

    assert audio.startswith(b"RIFF")


@pytest.mark.asyncio
async def test_synthesize_handles_runtime_exception(monkeypatch):
    class FakeRuntime:
        def create(self, text: str, voice: str, lang: str, speed: float):
            raise RuntimeError("boom")

    monkeypatch.setattr(kokoro.importlib, "import_module", lambda _: SimpleNamespace(Kokoro=lambda **_: FakeRuntime()))

    engine = kokoro.Engine()
    with pytest.raises(TTSError) as exc:
        await engine.synthesize("hello")

    assert exc.value.code == "TTS_KOKORO_FAILED"
    assert exc.value.retryable is True
