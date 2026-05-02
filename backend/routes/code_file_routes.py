from __future__ import annotations

import difflib
import os
import time
from pathlib import Path
from threading import Lock
from typing import Callable
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app_schemas import WriteRequest


def create_code_file_router(
    *,
    code_write_lock: Lock,
    pending_code_writes: dict[str, dict],
    log,
) -> APIRouter:
    router = APIRouter()

    def _get_code_root() -> str:
        root = os.getenv("CODE_WORKSPACE_ROOT", "")
        if not root or not os.path.isdir(root):
            raise HTTPException(status_code=503, detail="CODE_WORKSPACE_ROOT not configured or missing")
        return root

    def _safe_resolve(root: str, relative: str) -> str:
        real_root = os.path.realpath(root)
        candidate = os.path.realpath(os.path.join(real_root, relative))
        if not (candidate == real_root or candidate.startswith(real_root + os.sep)):
            raise HTTPException(status_code=400, detail="Path traversal is not allowed")
        return candidate

    def _make_unified_diff(relative_path: str, original: str, proposed: str) -> str:
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

    @router.get("/code/files", summary="List workspace files")
    async def list_code_files(sub_path: str = ""):
        root = _get_code_root()
        base = _safe_resolve(root, sub_path) if sub_path else os.path.realpath(root)
        skip_dirs = {"__pycache__", "node_modules", ".venv", ".git", "chroma", "models"}
        paths: list[str] = []
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]
            for fname in filenames:
                full = os.path.join(dirpath, fname)
                try:
                    paths.append(os.path.relpath(full, root))
                except ValueError:
                    paths.append(full)
        return JSONResponse({"files": sorted(paths)})

    @router.get("/code/files/{file_path:path}", summary="Read a workspace file")
    async def read_code_file(file_path: str):
        root = _get_code_root()
        resolved = _safe_resolve(root, file_path)
        if not os.path.isfile(resolved):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        try:
            content = Path(resolved).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return JSONResponse({"path": file_path, "content": content})

    @router.put("/code/files/{file_path:path}", summary="Write a workspace file (explicit API write)")
    async def write_code_file(file_path: str, body: WriteRequest, request: Request):
        root = _get_code_root()
        resolved = _safe_resolve(root, file_path)
        user_id = getattr(request.state, "user_id", "unknown")

        try:
            current = Path(resolved).read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            current = ""
        except OSError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        if not body.confirm:
            diff = _make_unified_diff(file_path, current, body.content)
            request_id = str(uuid4())
            with code_write_lock:
                pending_code_writes[request_id] = {
                    "user_id": user_id,
                    "file_path": file_path,
                    "resolved": resolved,
                    "content": body.content,
                    "created_at": time.time(),
                }
            summary = "No changes detected." if not diff else "Diff generated. Confirmation required before write."
            return JSONResponse(
                {
                    "status": "pending_confirmation",
                    "request_id": request_id,
                    "path": file_path,
                    "summary": summary,
                    "diff": diff,
                }
            )

        if not body.request_id:
            raise HTTPException(status_code=400, detail="request_id is required when confirm=true")

        with code_write_lock:
            pending = pending_code_writes.get(body.request_id)
            if not pending:
                raise HTTPException(status_code=404, detail="Pending write not found or expired")
            if str(pending.get("user_id", "")) != str(user_id):
                raise HTTPException(status_code=403, detail="Pending write belongs to another user")
            if pending.get("file_path") != file_path:
                raise HTTPException(status_code=400, detail="request_id does not match file_path")

            content_to_write = str(pending.get("content", ""))
            del pending_code_writes[body.request_id]

        try:
            Path(resolved).parent.mkdir(parents=True, exist_ok=True)
            Path(resolved).write_text(content_to_write, encoding="utf-8")
        except OSError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        log.info("code.write_file | path=%s | user=%s | confirmed=true", file_path, user_id)
        return JSONResponse({"status": "written", "written": file_path})

    return router
