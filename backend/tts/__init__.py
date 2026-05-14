from __future__ import annotations

# Phase 9 Slice 5 benchmark note (2026-04-28):
# Default engine remains "piper". On this stack, Piper has the lowest operational
# risk and most deterministic deployment path (CLI + explicit model file) while
# Kokoro stays optional and runtime-dependent. If assets are missing, the benchmark
# may report winner=none; default still stays Piper for deployment consistency.
# Re-run benchmarks with:
#   venv/bin/python backend/scripts/benchmark_tts.py
# after changing model assets, drivers, or TTS runtime packages.

import importlib
import inspect
import os
from dataclasses import dataclass
from typing import Protocol


class TTSEngine(Protocol):
    async def synthesize(self, text: str) -> bytes:
        ...


@dataclass(slots=True)
class TTSError(Exception):
    message: str
    code: str
    retryable: bool = False

    def __str__(self) -> str:
        return self.message


_ENGINE_MODULES: dict[str, str] = {
    "piper": "tts.engines.piper",
    "kokoro": "tts.engines.kokoro",
}

_ENGINE_CACHE: dict[str, TTSEngine] = {}


def get_engine_name(engine_name: str | None = None) -> str:
    raw = engine_name or os.getenv("TTS_ENGINE", "piper")
    return str(raw).strip().lower()


def _validate_engine_name(engine_name: str) -> None:
    if engine_name not in _ENGINE_MODULES:
        allowed = ", ".join(sorted(_ENGINE_MODULES))
        raise TTSError(
            message=f"Unsupported TTS_ENGINE '{engine_name}'. Allowed: {allowed}",
            code="TTS_ENGINE_INVALID",
            retryable=False,
        )


def _resolve_module_name(engine_name: str) -> str:
    _validate_engine_name(engine_name)
    return _ENGINE_MODULES[engine_name]


def _load_engine_instance(engine_name: str) -> TTSEngine:
    module_name = _resolve_module_name(engine_name)
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise TTSError(
            message=f"TTS engine '{engine_name}' is unavailable: {exc}",
            code="TTS_ENGINE_UNAVAILABLE",
            retryable=False,
        ) from exc

    create_engine = getattr(module, "create_engine", None)
    if callable(create_engine):
        try:
            engine = create_engine()
        except TTSError:
            raise
        except Exception as exc:
            raise TTSError(
                message=f"TTS engine '{engine_name}' failed to initialize: {exc}",
                code="TTS_ENGINE_INIT_FAILED",
                retryable=False,
            ) from exc
    else:
        engine_cls = getattr(module, "Engine", None)
        if engine_cls is None:
            raise TTSError(
                message=(
                    f"TTS engine module '{module_name}' must expose create_engine() "
                    "or Engine class"
                ),
                code="TTS_ENGINE_BAD_MODULE",
                retryable=False,
            )
        try:
            engine = engine_cls()
        except TTSError:
            raise
        except Exception as exc:
            raise TTSError(
                message=f"TTS engine '{engine_name}' failed to initialize: {exc}",
                code="TTS_ENGINE_INIT_FAILED",
                retryable=False,
            ) from exc

    synth = getattr(engine, "synthesize", None)
    if not callable(synth) or not inspect.iscoroutinefunction(synth):
        raise TTSError(
            message=f"TTS engine '{engine_name}' must provide async synthesize(text)",
            code="TTS_ENGINE_BAD_INTERFACE",
            retryable=False,
        )

    return engine


def get_engine(engine_name: str | None = None) -> TTSEngine:
    name = get_engine_name(engine_name)
    if name in _ENGINE_CACHE:
        return _ENGINE_CACHE[name]
    engine = _load_engine_instance(name)
    _ENGINE_CACHE[name] = engine
    return engine


def clear_engine_cache() -> None:
    _ENGINE_CACHE.clear()


def _validate_text(text: str) -> str:
    if text is None:
        raise TTSError("Missing text payload", code="TTS_INVALID_TEXT", retryable=False)
    stripped = text.strip()
    if not stripped:
        raise TTSError("Text must not be empty", code="TTS_INVALID_TEXT", retryable=False)

    max_chars = int(os.getenv("TTS_MAX_CHARS", "3000"))
    if len(stripped) > max_chars:
        raise TTSError(
            message=f"Text exceeds max length ({max_chars} chars)",
            code="TTS_TEXT_TOO_LONG",
            retryable=False,
        )

    return stripped


async def synthesize(text: str, engine_name: str | None = None) -> bytes:
    payload = _validate_text(text)
    from tts.normalise import normalise_for_speech  # local import to keep module lightweight
    payload = normalise_for_speech(payload) or payload
    engine = get_engine(engine_name)
    try:
        audio = await engine.synthesize(payload)
    except TTSError:
        raise
    except Exception as exc:
        raise TTSError(
            message=f"TTS synthesis failed: {exc}",
            code="TTS_SYNTHESIS_FAILED",
            retryable=True,
        ) from exc

    if not isinstance(audio, (bytes, bytearray)):
        raise TTSError(
            message="TTS engine returned non-binary audio payload",
            code="TTS_BAD_AUDIO_PAYLOAD",
            retryable=False,
        )

    return bytes(audio)


def error_to_payload(exc: Exception) -> dict[str, object]:
    if isinstance(exc, TTSError):
        return {"error": exc.message, "code": exc.code, "retryable": exc.retryable}
    return {
        "error": str(exc) or "Unknown TTS error",
        "code": "TTS_UNKNOWN_ERROR",
        "retryable": False,
    }
