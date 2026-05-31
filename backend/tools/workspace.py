"""Shared workspace path-safety and diff helpers.

Both the graph code-write flow (`graph.py`) and the explicit code-file API
(`routes/code_file_routes.py`) need identical path-traversal protection and
unified-diff formatting. Keeping a single implementation here prevents the two
surfaces from drifting apart, which would be a correctness and security hazard.
"""

from __future__ import annotations

import difflib
import os


class WorkspacePathError(ValueError):
    """Raised when a relative path resolves outside the workspace root."""


def resolve_workspace_path(root: str, relative_path: str) -> str:
    """Resolve ``relative_path`` within ``root``, blocking path traversal.

    Args:
        root: Workspace root directory.
        relative_path: User-supplied path relative to ``root``.

    Returns:
        The fully-resolved absolute path inside the workspace.

    Raises:
        WorkspacePathError: If the resolved path escapes ``root``.
    """
    real_root = os.path.realpath(root)
    candidate = os.path.realpath(os.path.join(real_root, relative_path))
    if not (candidate == real_root or candidate.startswith(real_root + os.sep)):
        raise WorkspacePathError(
            f"Path traversal blocked: {relative_path!r} resolves outside workspace root"
        )
    return candidate


def make_unified_diff(relative_path: str, original: str, proposed: str) -> str:
    """Return a unified diff between ``original`` and ``proposed`` content."""
    lines = list(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            proposed.splitlines(keepends=True),
            fromfile=f"a/{relative_path}",
            tofile=f"b/{relative_path}",
            lineterm="\n",
        )
    )
    return "".join(lines) if lines else ""
