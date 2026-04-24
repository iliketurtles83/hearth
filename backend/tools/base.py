"""
Shared base types for the tool module system.

Every tool module must implement:
    async def run(params: dict) -> ToolResult

params always contains at minimum:
    {"prompt": str, "memory": MemoryStore}

ToolResult.data must follow a normalized schema defined by the tool.
Raw provider-specific field names must never appear in ToolResult.data.
"""
from dataclasses import dataclass, field


@dataclass
class ToolResult:
    """Normalized return type for every tool module.

    Attributes:
        ok:        True if the tool call succeeded.
        data:      Normalized response payload (provider-agnostic field names).
                   Empty dict when ok=False.
        error:     Human-readable error message when ok=False. Empty string when ok=True.
        retryable: True when the failure is transient (network, timeout) and a
                   retry is likely to succeed.  False for permanent failures
                   (bad location, API key invalid, etc.).
    """

    ok: bool
    data: dict = field(default_factory=dict)
    error: str = ""
    retryable: bool = False

    @classmethod
    def failure(cls, error: str, *, retryable: bool = False) -> "ToolResult":
        """Convenience constructor for a failed result."""
        return cls(ok=False, data={}, error=error, retryable=retryable)
