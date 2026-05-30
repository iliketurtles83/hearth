from __future__ import annotations

from abc import ABC, abstractmethod
import os

from tts import TTSError


class BaseTTSEngine(ABC):
    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        raise NotImplementedError

    @staticmethod
    def _parse_float(
        name: str,
        *,
        default: float,
        error_code: str,
        require_positive: bool = True,
    ) -> float:
        raw = os.getenv(name, str(default)).strip()
        try:
            value = float(raw)
        except ValueError as exc:
            raise TTSError(
                message=f"Invalid {name}: {raw}",
                code=error_code,
                retryable=False,
            ) from exc

        if require_positive and value <= 0:
            raise TTSError(
                message=f"{name} must be > 0",
                code=error_code,
                retryable=False,
            )

        return value

    @staticmethod
    def _parse_int(
        name: str,
        *,
        default: int,
        error_code: str,
        require_positive: bool = True,
    ) -> int:
        raw = os.getenv(name, str(default)).strip()
        try:
            value = int(raw)
        except ValueError as exc:
            raise TTSError(
                message=f"Invalid {name}: {raw}",
                code=error_code,
                retryable=False,
            ) from exc

        if require_positive and value <= 0:
            raise TTSError(
                message=f"{name} must be > 0",
                code=error_code,
                retryable=False,
            )

        return value

    @staticmethod
    def _parse_optional_float(name: str, *, error_code: str) -> float | None:
        raw = os.getenv(name, "").strip()
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError as exc:
            raise TTSError(
                message=f"Invalid {name}: {raw}",
                code=error_code,
                retryable=False,
            ) from exc

    @staticmethod
    def _parse_optional_int(name: str, *, error_code: str) -> int | None:
        raw = os.getenv(name, "").strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError as exc:
            raise TTSError(
                message=f"Invalid {name}: {raw}",
                code=error_code,
                retryable=False,
            ) from exc
