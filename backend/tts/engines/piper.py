from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from wave import open as wave_open

from tts import TTSError


class Engine:
    """Piper CLI-backed TTS engine.

    Environment variables:
    - TTS_PIPER_BIN: piper executable path (default: piper)
    - TTS_PIPER_MODEL: ONNX model path (required)
    - TTS_PIPER_SPEAKER: optional speaker id (int)
    - TTS_PIPER_RATE: speaking rate multiplier (default: 1.0)
    - TTS_PIPER_PITCH: pitch multiplier (default: 1.0)
    - TTS_PIPER_NOISE_SCALE: optional Piper noise scale
    - TTS_PIPER_NOISE_W: optional Piper noise_w
    - TTS_PIPER_SENTENCE_SILENCE: optional sentence silence seconds
    - TTS_FFMPEG_BIN: ffmpeg executable path for optional pitch shifting (default: ffmpeg)
    """

    def __init__(self) -> None:
        self.piper_bin = os.getenv("TTS_PIPER_BIN", "piper").strip() or "piper"
        self.model_path = os.getenv("TTS_PIPER_MODEL", "").strip()
        self.speaker = self._parse_optional_int("TTS_PIPER_SPEAKER")
        self.rate = self._parse_float("TTS_PIPER_RATE", default=1.0)
        self.pitch = self._parse_float("TTS_PIPER_PITCH", default=1.0)
        self.noise_scale = self._parse_optional_float("TTS_PIPER_NOISE_SCALE")
        self.noise_w = self._parse_optional_float("TTS_PIPER_NOISE_W")
        self.sentence_silence = self._parse_optional_float("TTS_PIPER_SENTENCE_SILENCE")
        self.ffmpeg_bin = os.getenv("TTS_FFMPEG_BIN", "ffmpeg").strip() or "ffmpeg"

        if not self.model_path:
            raise TTSError(
                message="TTS_PIPER_MODEL is required for the Piper engine",
                code="TTS_PIPER_MODEL_MISSING",
                retryable=False,
            )

        if not Path(self.model_path).is_file():
            raise TTSError(
                message=f"Piper model not found: {self.model_path}",
                code="TTS_PIPER_MODEL_NOT_FOUND",
                retryable=False,
            )

        if shutil.which(self.piper_bin) is None:
            raise TTSError(
                message=f"Piper binary not found: {self.piper_bin}",
                code="TTS_PIPER_BIN_NOT_FOUND",
                retryable=False,
            )

    @staticmethod
    def _parse_float(name: str, default: float) -> float:
        raw = os.getenv(name, str(default)).strip()
        try:
            value = float(raw)
        except ValueError as exc:
            raise TTSError(
                message=f"Invalid {name}: {raw}",
                code="TTS_PIPER_CONFIG_INVALID",
                retryable=False,
            ) from exc
        if value <= 0:
            raise TTSError(
                message=f"{name} must be > 0",
                code="TTS_PIPER_CONFIG_INVALID",
                retryable=False,
            )
        return value

    @staticmethod
    def _parse_optional_float(name: str) -> float | None:
        raw = os.getenv(name, "").strip()
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError as exc:
            raise TTSError(
                message=f"Invalid {name}: {raw}",
                code="TTS_PIPER_CONFIG_INVALID",
                retryable=False,
            ) from exc

    @staticmethod
    def _parse_optional_int(name: str) -> int | None:
        raw = os.getenv(name, "").strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError as exc:
            raise TTSError(
                message=f"Invalid {name}: {raw}",
                code="TTS_PIPER_CONFIG_INVALID",
                retryable=False,
            ) from exc

    def _build_cmd(self, output_path: Path) -> list[str]:
        cmd = [
            self.piper_bin,
            "--model",
            self.model_path,
            "--output_file",
            str(output_path),
            "--length_scale",
            f"{1.0 / self.rate:.4f}",
        ]

        if self.speaker is not None:
            cmd.extend(["--speaker", str(self.speaker)])
        if self.noise_scale is not None:
            cmd.extend(["--noise_scale", str(self.noise_scale)])
        if self.noise_w is not None:
            cmd.extend(["--noise_w", str(self.noise_w)])
        if self.sentence_silence is not None:
            cmd.extend(["--sentence_silence", str(self.sentence_silence)])
        return cmd

    def _run_piper(self, text: str) -> bytes:
        with tempfile.TemporaryDirectory(prefix="assistant-tts-") as td:
            output_path = Path(td) / "out.wav"
            cmd = self._build_cmd(output_path)
            try:
                proc = subprocess.run(
                    cmd,
                    input=text,
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except FileNotFoundError as exc:
                raise TTSError(
                    message=f"Piper binary not found: {self.piper_bin}",
                    code="TTS_PIPER_BIN_NOT_FOUND",
                    retryable=False,
                ) from exc

            if proc.returncode != 0:
                stderr = (proc.stderr or "").strip()
                msg = stderr or "Piper synthesis failed"
                raise TTSError(
                    message=f"Piper synthesis failed: {msg}",
                    code="TTS_PIPER_CLI_FAILED",
                    retryable=True,
                )

            if not output_path.exists():
                raise TTSError(
                    message="Piper did not produce an output wav file",
                    code="TTS_PIPER_NO_OUTPUT",
                    retryable=True,
                )

            audio = output_path.read_bytes()

        if self.pitch != 1.0:
            audio = self._apply_pitch_shift(audio, self.pitch)

        return audio

    def _apply_pitch_shift(self, wav_bytes: bytes, pitch: float) -> bytes:
        if shutil.which(self.ffmpeg_bin) is None:
            raise TTSError(
                message=(
                    "TTS_PIPER_PITCH requires ffmpeg in PATH "
                    f"(missing: {self.ffmpeg_bin})"
                ),
                code="TTS_PIPER_PITCH_UNSUPPORTED",
                retryable=False,
            )

        with tempfile.TemporaryDirectory(prefix="assistant-tts-pitch-") as td:
            src = Path(td) / "src.wav"
            dst = Path(td) / "dst.wav"
            src.write_bytes(wav_bytes)

            with wave_open(str(src), "rb") as wf:
                rate = wf.getframerate()

            filter_graph = f"asetrate={rate}*{pitch},aresample={rate}"
            proc = subprocess.run(
                [
                    self.ffmpeg_bin,
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(src),
                    "-af",
                    filter_graph,
                    str(dst),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                stderr = (proc.stderr or "").strip()
                raise TTSError(
                    message=f"Pitch processing failed: {stderr or 'ffmpeg error'}",
                    code="TTS_PIPER_PITCH_FAILED",
                    retryable=True,
                )

            return dst.read_bytes()

    async def synthesize(self, text: str) -> bytes:
        return await asyncio.to_thread(self._run_piper, text)
