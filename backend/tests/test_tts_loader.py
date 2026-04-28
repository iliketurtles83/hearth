from __future__ import annotations

import types

import pytest

import tts


@pytest.fixture(autouse=True)
def clear_cache_between_tests() -> None:
    tts.clear_engine_cache()


class _GoodEngine:
    async def synthesize(self, text: str) -> bytes:
        return f"audio:{text}".encode("utf-8")


class _BadEngineNonAsync:
    def synthesize(self, text: str) -> bytes:
        return b"not-async"


def _module_with_engine_class(engine_cls):
    module = types.SimpleNamespace()
    module.Engine = engine_cls
    return module


def _module_with_factory(engine_cls):
    module = types.SimpleNamespace()

    def create_engine():
        return engine_cls()

    module.create_engine = create_engine
    return module


def test_get_engine_name_uses_env_default(monkeypatch):
    monkeypatch.delenv("TTS_ENGINE", raising=False)
    assert tts.get_engine_name() == "piper"


def test_invalid_engine_name_rejected():
    with pytest.raises(tts.TTSError) as exc:
        tts.get_engine("invalid")
    assert exc.value.code == "TTS_ENGINE_INVALID"


def test_get_engine_loads_from_factory(monkeypatch):
    module = _module_with_factory(_GoodEngine)

    def fake_import(name: str):
        assert name == "tts.engines.piper"
        return module

    monkeypatch.setattr(tts.importlib, "import_module", fake_import)
    engine = tts.get_engine("piper")
    assert isinstance(engine, _GoodEngine)


def test_get_engine_module_unavailable(monkeypatch):
    def fake_import(_: str):
        raise ModuleNotFoundError("no module")

    monkeypatch.setattr(tts.importlib, "import_module", fake_import)

    with pytest.raises(tts.TTSError) as exc:
        tts.get_engine("piper")
    assert exc.value.code == "TTS_ENGINE_UNAVAILABLE"


def test_get_engine_requires_async_interface(monkeypatch):
    module = _module_with_engine_class(_BadEngineNonAsync)

    monkeypatch.setattr(tts.importlib, "import_module", lambda _: module)

    with pytest.raises(tts.TTSError) as exc:
        tts.get_engine("piper")
    assert exc.value.code == "TTS_ENGINE_BAD_INTERFACE"


@pytest.mark.asyncio
async def test_synthesize_validates_empty_text(monkeypatch):
    module = _module_with_engine_class(_GoodEngine)
    monkeypatch.setattr(tts.importlib, "import_module", lambda _: module)

    with pytest.raises(tts.TTSError) as exc:
        await tts.synthesize("   ", engine_name="piper")
    assert exc.value.code == "TTS_INVALID_TEXT"


@pytest.mark.asyncio
async def test_synthesize_validates_max_chars(monkeypatch):
    module = _module_with_engine_class(_GoodEngine)
    monkeypatch.setattr(tts.importlib, "import_module", lambda _: module)
    monkeypatch.setenv("TTS_MAX_CHARS", "3")

    with pytest.raises(tts.TTSError) as exc:
        await tts.synthesize("hello", engine_name="piper")
    assert exc.value.code == "TTS_TEXT_TOO_LONG"


@pytest.mark.asyncio
async def test_synthesize_returns_bytes(monkeypatch):
    module = _module_with_engine_class(_GoodEngine)
    monkeypatch.setattr(tts.importlib, "import_module", lambda _: module)

    out = await tts.synthesize("ok", engine_name="piper")
    assert out == b"audio:ok"


@pytest.mark.asyncio
async def test_synthesize_wraps_engine_errors(monkeypatch):
    class _ExplodingEngine:
        async def synthesize(self, text: str) -> bytes:
            raise RuntimeError("boom")

    module = _module_with_engine_class(_ExplodingEngine)
    monkeypatch.setattr(tts.importlib, "import_module", lambda _: module)

    with pytest.raises(tts.TTSError) as exc:
        await tts.synthesize("ok", engine_name="piper")
    assert exc.value.code == "TTS_SYNTHESIS_FAILED"
    assert exc.value.retryable is True


def test_error_to_payload_for_tts_error():
    payload = tts.error_to_payload(tts.TTSError("bad", code="TTS_X", retryable=True))
    assert payload == {"error": "bad", "code": "TTS_X", "retryable": True}


def test_error_to_payload_for_generic_error():
    payload = tts.error_to_payload(RuntimeError("oops"))
    assert payload == {"error": "oops", "code": "TTS_UNKNOWN_ERROR", "retryable": False}
