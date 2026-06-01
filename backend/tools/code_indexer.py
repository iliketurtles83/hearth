"""Code context indexer for Phase 10b.

Extracts tree-sitter summaries (function signatures, class hierarchies, imports)
from workspace files and stores them in a dedicated ChromaDB 'code_context'
collection.  This collection is separate from 'assistant_memories' and is only
queried for code intents, never for general chat retrieval.

Public API
----------
index_workspace(workspace_root, chroma_path, index_paths=None, collection_name="code_context") -> int
    Walk the workspace, extract summaries, upsert into ChromaDB.

query_code_context(query, chroma_path, n=5, collection_name="code_context") -> list[str]
    Return the top-n relevant code snippets for a query string.

start_background_index(workspace_root, chroma_path, index_paths=None, collection_name="code_context") -> None
    Launch index_workspace in a daemon thread (non-blocking).

delete_collection(chroma_path, collection_name) -> bool
    Delete a Chroma collection if it exists. Returns True when a collection
    was removed, False otherwise.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from chromadb.api.types import Documents, EmbeddingFunction

log = logging.getLogger("assistant.code_indexer")

# ── Embedding ─────────────────────────────────────────────────────────────────
# Reuses the same deterministic hash-based approach as memory.py so the two
# collections are embedding-compatible without a heavy model dependency.


class _HashEmbeddingFunction(EmbeddingFunction):
    def __init__(self, dim: int = 192) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in re.findall(r"[a-z0-9_]+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:8], byteorder="big", signed=False) % self.dim
            vec[idx] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec.tolist()

    def __call__(self, input: Documents) -> list[list[float]]:
        return [self._embed_one(t) for t in input]


_embedder = _HashEmbeddingFunction()


def _get_collection(chroma_path: str, collection_name: str = "code_context") -> Any:
    client = chromadb.PersistentClient(path=chroma_path)
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=_embedder,
    )


def delete_collection(chroma_path: str, collection_name: str) -> bool:
    """Delete ``collection_name`` from Chroma if it exists.

    Returns True when the collection was deleted and False when it did not
    exist (or could not be listed).
    """
    client = chromadb.PersistentClient(path=chroma_path)
    try:
        existing = {c.name for c in client.list_collections()}
    except Exception:
        existing = set()
    if collection_name not in existing:
        return False
    client.delete_collection(collection_name)
    return True


# ── tree-sitter parser bootstrap ──────────────────────────────────────────────

_parsers: dict[str, Any] | None | bool = False  # False = not yet initialised


def _init_parsers() -> dict[str, Any] | None:
    try:
        from tree_sitter import Language, Parser  # type: ignore[import-untyped]
    except ImportError:
        log.warning("code_indexer: tree-sitter not installed — using regex fallback")
        return None

    result: dict[str, Any] = {}

    try:
        import tree_sitter_python as tspy  # type: ignore[import-untyped]
        result["python"] = Parser(Language(tspy.language()))
    except Exception as exc:
        log.warning("code_indexer: tree-sitter-python unavailable (%s)", exc)

    try:
        import tree_sitter_javascript as tsjs  # type: ignore[import-untyped]
        result["javascript"] = Parser(Language(tsjs.language()))
    except Exception as exc:
        log.warning("code_indexer: tree-sitter-javascript unavailable (%s)", exc)

    if result:
        log.info("code_indexer: tree-sitter parsers loaded: %s", list(result.keys()))
    return result or None


def _get_parsers() -> dict[str, Any] | None:
    global _parsers
    if _parsers is False:
        _parsers = _init_parsers()
    return _parsers if _parsers else None  # type: ignore[return-value]


# ── Extraction helpers ─────────────────────────────────────────────────────────


def _extract_python_tree_sitter(source: str, parser: Any) -> str:
    tree = parser.parse(source.encode("utf-8"))
    lines: list[str] = []

    def _sig(start_byte: int, end_byte: int) -> str:
        text = source[start_byte:end_byte]
        colon = text.find(":")
        return (text[:colon + 1] if colon >= 0 else text).split("\n")[0].strip()

    def _visit(node: Any, depth: int = 0) -> None:
        indent = "  " * depth
        t = node.type
        if t in ("import_statement", "import_from_statement"):
            first_line = source[node.start_byte:node.end_byte].split("\n")[0]
            lines.append(f"{indent}{first_line}")
        elif t == "function_definition":
            lines.append(f"{indent}{_sig(node.start_byte, node.end_byte)}")
            # Include leading docstring if present
            body = node.child_by_field_name("body")
            if body and body.child_count > 0:
                first = body.children[0]
                if first.type == "expression_statement" and first.child_count > 0:
                    child = first.children[0]
                    if child.type == "string":
                        doc = source[child.start_byte:child.end_byte]
                        lines.append(f"{indent}  # {doc[:80]}")
        elif t == "class_definition":
            lines.append(f"{indent}{_sig(node.start_byte, node.end_byte)}")
            for child in node.children:
                _visit(child, depth + 1)
        else:
            for child in node.children:
                _visit(child, depth)

    _visit(tree.root_node)
    return "\n".join(lines)


def _extract_python_regex(source: str) -> str:
    lines: list[str] = []
    for line in source.splitlines():
        s = line.strip()
        if (
            s.startswith("def ")
            or s.startswith("async def ")
            or s.startswith("class ")
            or s.startswith("import ")
            or s.startswith("from ")
        ):
            lines.append(s.split(":")[0])
    return "\n".join(lines)


def _extract_python_summary(source: str) -> str:
    parsers = _get_parsers()
    if parsers and "python" in parsers:
        try:
            return _extract_python_tree_sitter(source, parsers["python"])
        except Exception as exc:
            log.debug("code_indexer: tree-sitter parse failed (%s); falling back to regex", exc)
    return _extract_python_regex(source)


def _extract_js_summary(source: str) -> str:
    patterns = [
        r"^(export\s+)?(default\s+)?(async\s+)?function[\s*]",
        r"^(export\s+)?class\s+\w+",
        r"^(const|let|var)\s+\w+\s*=\s*(async\s+)?\(",
        r"^import\s+",
    ]
    lines: list[str] = []
    for line in source.splitlines():
        s = line.strip()
        if any(re.match(p, s) for p in patterns):
            lines.append(s[:120])
    return "\n".join(lines)


# ── File traversal ─────────────────────────────────────────────────────────────

_EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "javascript",
    ".jsx": "javascript",
    ".tsx": "javascript",
}

_IGNORE_DIRS: frozenset[str] = frozenset(
    [
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        ".mypy_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".eggs",
        "chroma",
        "models",
        "artifacts",
    ]
)

_MAX_FILE_BYTES = 256 * 1024  # skip files larger than 256 KB


def _summarize_file(path: Path, workspace_root: Path) -> str | None:
    ext = path.suffix.lower()
    lang = _EXTENSION_MAP.get(ext)
    if not lang:
        return None
    try:
        if path.stat().st_size > _MAX_FILE_BYTES:
            return None
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    summary = _extract_python_summary(source) if lang == "python" else _extract_js_summary(source)
    if not summary.strip():
        return None

    try:
        rel = str(path.relative_to(workspace_root))
    except ValueError:
        rel = str(path)

    return f"File: {rel}\n{summary}"


def _collect_files(workspace_root: Path, extra_paths: list[Path]) -> list[Path]:
    search_roots = extra_paths if extra_paths else [workspace_root]
    result: list[Path] = []
    for base in search_roots:
        if not base.exists():
            log.warning("code_indexer: index path does not exist: %s", base)
            continue
        for dirpath_str, dirnames, filenames in os.walk(base):
            dirnames[:] = [
                d for d in dirnames if d not in _IGNORE_DIRS and not d.startswith(".")
            ]
            for fname in filenames:
                p = Path(dirpath_str) / fname
                if p.suffix.lower() in _EXTENSION_MAP:
                    result.append(p)
    return result


# ── Public API ─────────────────────────────────────────────────────────────────


def index_workspace(
    workspace_root: str,
    chroma_path: str,
    index_paths: list[str] | None = None,
    collection_name: str = "code_context",
) -> int:
    """Walk the workspace and upsert code summaries into the code_context collection.

    Parameters
    ----------
    workspace_root:
        Absolute path to the code workspace root (CODE_WORKSPACE_ROOT).
    chroma_path:
        Absolute path to the ChromaDB persistence directory (CHROMA_PATH).
    index_paths:
        Optional list of sub-paths (relative to workspace_root) to restrict
        indexing to.  When None the entire workspace_root is indexed.

    Returns
    -------
    int
        Number of files successfully indexed.
    """
    root = Path(workspace_root)
    extra: list[Path] = [root / p for p in index_paths] if index_paths else []
    files = _collect_files(root, extra)

    collection = _get_collection(chroma_path, collection_name=collection_name)

    docs: list[str] = []
    ids: list[str] = []

    for fpath in files:
        summary = _summarize_file(fpath, root)
        if not summary:
            continue
        try:
            rel = str(fpath.relative_to(root))
        except ValueError:
            rel = str(fpath)
        docs.append(summary)
        ids.append(rel)

    if not docs:
        log.info("code_indexer: no indexable files found under %s", workspace_root)
        return 0

    batch_size = 100
    for i in range(0, len(docs), batch_size):
        collection.upsert(documents=docs[i : i + batch_size], ids=ids[i : i + batch_size])

    log.info(
        "code_indexer: indexed %d files into %s (root=%s)",
        len(docs),
        collection_name,
        workspace_root,
    )
    return len(docs)


def query_code_context(
    query: str,
    chroma_path: str,
    n: int = 5,
    collection_name: str = "code_context",
) -> list[str]:
    """Return the top-n relevant code context snippets for a query.

    Returns an empty list if the collection is empty or a query error occurs.
    """
    try:
        collection = _get_collection(chroma_path, collection_name=collection_name)
        count = collection.count()
        if count == 0:
            return []
        results = collection.query(query_texts=[query], n_results=min(n, count))
        return results.get("documents", [[]])[0]
    except Exception as exc:
        log.warning("code_indexer.query_code_context failed: %s", exc)
        return []


def start_background_index(
    workspace_root: str,
    chroma_path: str,
    index_paths: list[str] | None = None,
    collection_name: str = "code_context",
) -> None:
    """Launch index_workspace in a daemon thread so startup is non-blocking."""

    def _run() -> None:
        try:
            count = index_workspace(
                workspace_root,
                chroma_path,
                index_paths,
                collection_name=collection_name,
            )
            log.info("code_indexer: background index complete (%d files)", count)
        except Exception as exc:
            log.error("code_indexer: background index failed: %s", exc, exc_info=True)

    thread = threading.Thread(target=_run, daemon=True, name="code-indexer")
    thread.start()
