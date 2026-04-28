from __future__ import annotations

import asyncio
import importlib
import inspect
import io
import os
import struct
import wave
from typing import Any

from tts import TTSError


class Engine:
    """Kokoro TTS engine backed by the optional `kokoro_onnx` runtime.

    This engine is capability-gated: if runtime deps or model assets are missing,
    initialization fails with a clear non-retryable TTSError.
    """

    def __init__(self) -> None:
        self.model_path = os.getenv("TTS_KOKORO_MODEL", "").strip() or None
        self.voices_path = os.getenv("TTS_KOKORO_VOICES", "").strip() or None
        self.voice = os.getenv("TTS_KOKORO_VOICE", "af_heart").strip() or "af_heart"
        self.language = os.getenv("TTS_KOKORO_LANG", "en-us").strip() or "en-us"
        self.speed = self._parse_float("TTS_KOKORO_SPEED", default=1.0)
        self.default_sample_rate = self._parse_int("TTS_KOKORO_SAMPLE_RATE", default=24000)

        self._runtime = self._load_runtime()

    async def synthesize(self, text: str) -> bytes:
        return await asyncio.to_thread(self._synthesize_sync, text)

    @staticmethod
    def _parse_float(name: str, default: float) -> float:
        raw = os.getenv(name, str(default)).strip()
        try:
            value = float(raw)
        except ValueError as exc:
            raise TTSError(
                message=f"Invalid {name}: {raw}",
                code="TTS_KOKORO_CONFIG_INVALID",
                retryable=False,
            ) from exc
        if value <= 0:
            raise TTSError(
                message=f"{name} must be > 0",
                code="TTS_KOKORO_CONFIG_INVALID",
                retryable=False,
            )
        return value

    @staticmethod
    def _parse_int(name: str, default: int) -> int:
        raw = os.getenv(name, str(default)).strip()
        try:
            value = int(raw)
        except ValueError as exc:
            raise TTSError(
                message=f"Invalid {name}: {raw}",
                code="TTS_KOKORO_CONFIG_INVALID",
                retryable=False,
            ) from exc
        if value <= 0:
            raise TTSError(
                message=f"{name} must be > 0",
                code="TTS_KOKORO_CONFIG_INVALID",
                retryable=False,
            )
        return value

    def _load_runtime(self) -> Any:
        try:
            module = importlib.import_module("kokoro_onnx")
        except Exception as exc:
            raise TTSError(
                message=(
                    "Kokoro runtime unavailable. Install kokoro_onnx and model assets "
                    "to enable TTS_ENGINE=kokoro"
                ),
                code="TTS_KOKORO_UNAVAILABLE",
                retryable=False,
            ) from exc

        runtime_cls = getattr(module, "Kokoro", None)
        if runtime_cls is None:
            raise TTSError(
                message="kokoro_onnx module missing Kokoro class",
                code="TTS_KOKORO_BAD_RUNTIME",
                retryable=False,
            )

        kwargs: dict[str, Any] = {}
        sig = inspect.signature(runtime_cls)
        params = set(sig.parameters.keys())
        if self.model_path and "model_path" in params:
            kwargs["model_path"] = self.model_path
        if self.voices_path and "voices_path" in params:
            kwargs["voices_path"] = self.voices_path

        try:
            return runtime_cls(**kwargs)
        except Exception as exc:
            raise TTSError(
                message=f"Kokoro runtime init failed: {exc}",
                code="TTS_KOKORO_INIT_FAILED",
                retryable=False,
            ) from exc

    def _call_runtime(self, text: str) -> tuple[Any, int]:
        method = None
        for method_name in ("create", "generate", "synthesize"):
            candidate = getattr(self._runtime, method_name, None)
            if callable(candidate):
                method = candidate
                break

        if method is None:
            raise TTSError(
                message="Kokoro runtime exposes no create/generate/synthesize method",
                code="TTS_KOKORO_BAD_RUNTIME",
                retryable=False,
            )

        kwargs: dict[str, Any] = {}
        args: list[Any] = []
        params = inspect.signature(method).parameters

        if "text" in params:
            kwargs["text"] = text
        elif "input" in params:
            kwargs["input"] = text
        else:
            args.append(text)

        if "voice" in params:
            kwargs["voice"] = self.voice
        if "speaker" in params:
            kwargs["speaker"] = self.voice
        if "lang" in params:
            kwargs["lang"] = self.language
        if "language" in params:
            kwargs["language"] = self.language
        if "speed" in params:
            kwargs["speed"] = self.speed
        if "rate" in params:
            kwargs["rate"] = self.speed

        result = method(*args, **kwargs)
        return self._extract_audio(result)

    def _extract_audio(self, result: Any) -> tuple[Any, int]:
        sample_rate = self.default_sample_rate
        audio = result

        if isinstance(result, tuple) and len(result) >= 2:
            audio = result[0]
            if isinstance(result[1], int) and result[1] > 0:
                sample_rate = result[1]
        elif isinstance(result, dict):
            audio = result.get("audio") or result.get("samples") or result.get("wav")
            sr = result.get("sample_rate") or result.get("rate") or result.get("sr")
            if isinstance(sr, int) and sr > 0:
                sample_rate = sr

        return audio, sample_rate

    @staticmethod
    def _coerce_pcm_float(samples: Any) -> list[float]:
        if samples is None:
            raise TTSError(
                message="Kokoro runtime returned empty audio",
                code="TTS_KOKORO_BAD_AUDIO",
                retryable=True,
            )

        if hasattr(samples, "tolist"):
            samples = samples.tolist()

        if not isinstance(samples, (list, tuple)):
            raise TTSError(
                message="Kokoro runtime returned unsupported audio container",
                code="TTS_KOKORO_BAD_AUDIO",
                retryable=True,
            )

        out: list[float] = []
        for raw in samples:
            value = float(raw)
            if abs(value) > 1.5:
                value = value / 32768.0
            value = max(-1.0, min(1.0, value))
            out.append(value)
        return out

    @staticmethod
    def _float_to_wav_bytes(samples: list[float], sample_rate: int) -> bytes:
        pcm = bytearray()
        for value in samples:
            iv = int(value * 32767.0)
            iv = max(-32768, min(32767, iv))
            pcm.extend(struct.pack("<h", iv))

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(bytes(pcm))
        return buf.getvalue()

    def _synthesize_sync(self, text: str) -> bytes:
        try:
            audio, sample_rate = self._call_runtime(text)
        except TTSError:
            raise
        except Exception as exc:
            raise TTSError(
                message=f"Kokoro synthesis failed: {exc}",
                code="TTS_KOKORO_FAILED",
                retryable=True,
            ) from exc

        if isinstance(audio, (bytes, bytearray)):
            data = bytes(audio)
            if data.startswith(b"RIFF"):
                return data

        samples = self._coerce_pcm_float(audio)
        return self._float_to_wav_bytes(samples, sample_rate)


def create_engine() -> Engine:
    return Engine()
