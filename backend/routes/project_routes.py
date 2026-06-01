from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from projects import ProjectError
from tools.code_indexer import delete_collection, index_workspace
from tools.workspace import WorkspacePathError, resolve_workspace_path

log = logging.getLogger("assistant.projects")


class ProjectCreateRequest(BaseModel):
    name: str
    folder_name: str
    description: str = ""
    git_init: bool = False


def create_project_router(
    *,
    project_store,
    chroma_path: str,
) -> APIRouter:
    router = APIRouter()
    def _project_collection_name(project_id: str) -> str:
        return f"code_context_{project_id}"


    index_status_lock = threading.Lock()
    index_status: dict[str, dict] = {}

    def _resolve_project_folder(folder_name: str) -> str:
        try:
            return resolve_workspace_path(project_store.code_workspace_root, folder_name)
        except WorkspacePathError as exc:
            raise HTTPException(status_code=400, detail="Path traversal is not allowed") from exc

    def _list_project_files(folder_name: str) -> list[str]:
        base = _resolve_project_folder(folder_name)
        skip_dirs = {"__pycache__", "node_modules", ".venv", ".git", "chroma", "models"}
        paths: list[str] = []
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]
            for fname in filenames:
                full = os.path.join(dirpath, fname)
                try:
                    paths.append(os.path.relpath(full, base))
                except ValueError:
                    paths.append(full)
        return sorted(paths)

    def _get_owned_project(project_id: str, user_id: str) -> dict:
        project = project_store.get_project(project_id, user_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        return project

    def _set_index_status(project_id: str, patch: dict) -> None:
        with index_status_lock:
            current = index_status.get(
                project_id,
                {
                    "status": "idle",
                    "files_indexed": 0,
                    "chunks": 0,
                    "duration_s": 0.0,
                    "error": None,
                    "started_at": None,
                },
            )
            current.update(patch)
            index_status[project_id] = current

    def _start_index(project_id: str, folder_name: str) -> None:
        def _run() -> None:
            start = time.monotonic()
            _set_index_status(
                project_id,
                {
                    "status": "running",
                    "files_indexed": 0,
                    "chunks": 0,
                    "duration_s": 0.0,
                    "error": None,
                    "started_at": time.time(),
                },
            )
            try:
                count = index_workspace(
                    project_store.code_workspace_root,
                    chroma_path,
                    index_paths=[folder_name],
                    collection_name=_project_collection_name(project_id),
                )
                elapsed = time.monotonic() - start
                _set_index_status(
                    project_id,
                    {
                        "status": "done",
                        "files_indexed": int(count),
                        "chunks": int(count),
                        "duration_s": round(elapsed, 3),
                        "error": None,
                    },
                )
            except Exception as exc:
                elapsed = time.monotonic() - start
                _set_index_status(
                    project_id,
                    {
                        "status": "error",
                        "duration_s": round(elapsed, 3),
                        "error": str(exc),
                    },
                )

        thread = threading.Thread(target=_run, daemon=True, name=f"project-index-{project_id}")
        thread.start()

    @router.post("/projects")
    async def create_project(payload: ProjectCreateRequest, http_request: Request):
        user_id: str = http_request.state.user_id
        try:
            project = project_store.create_project(
                user_id=user_id,
                name=payload.name,
                folder_name=payload.folder_name,
                description=payload.description,
            )
        except ProjectError as exc:
            raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
        if payload.git_init:
            resolved_folder = _resolve_project_folder(str(project["folder_name"]))
            cmd = ["git", "init", str(resolved_folder)]
            try:
                proc = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                stdout = (proc.stdout or "").strip()
                stderr = (proc.stderr or "").strip()
                if proc.returncode != 0:
                    log.warning(
                        "projects.git_init_failed | user=%s project_id=%s path=%s returncode=%d stdout=%s stderr=%s",
                        user_id,
                        project.get("id"),
                        resolved_folder,
                        proc.returncode,
                        stdout,
                        stderr,
                    )
                else:
                    log.info(
                        "projects.git_init_ok | user=%s project_id=%s path=%s stdout=%s stderr=%s",
                        user_id,
                        project.get("id"),
                        resolved_folder,
                        stdout,
                        stderr,
                    )
            except OSError as exc:
                log.warning(
                    "projects.git_init_unavailable | user=%s project_id=%s path=%s error=%s",
                    user_id,
                    project.get("id"),
                    resolved_folder,
                    exc,
                )
            refreshed = project_store.get_project(str(project["id"]), user_id)
            if refreshed is not None:
                project = refreshed
        return JSONResponse(project, status_code=201)

    @router.get("/projects")
    async def list_projects(http_request: Request):
        user_id: str = http_request.state.user_id
        try:
            projects = project_store.list_projects(user_id)
        except ProjectError as exc:
            raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
        return JSONResponse({"projects": projects})

    @router.get("/projects/{project_id}")
    async def get_project(project_id: str, http_request: Request):
        user_id: str = http_request.state.user_id
        try:
            project = _get_owned_project(project_id, user_id)
        except ProjectError as exc:
            raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
        return JSONResponse(project)

    @router.delete("/projects/{project_id}")
    async def delete_project(project_id: str, http_request: Request):
        user_id: str = http_request.state.user_id
        _get_owned_project(project_id, user_id)
        deleted = project_store.delete_project(project_id, user_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Project not found")
        removed_collection = False
        try:
            removed_collection = delete_collection(
                chroma_path,
                _project_collection_name(project_id),
            )
        except Exception:
            removed_collection = False
        return JSONResponse(
            {
                "ok": True,
                "deleted": project_id,
                "collection": _project_collection_name(project_id),
                "collection_deleted": bool(removed_collection),
            }
        )


    @router.post("/projects/{project_id}/open")
    async def open_project(project_id: str, http_request: Request):
        user_id: str = http_request.state.user_id
        project = _get_owned_project(project_id, user_id)
        project_store.touch_opened(project_id, user_id)
        refreshed = _get_owned_project(project_id, user_id)
        files = _list_project_files(project["folder_name"])
        return JSONResponse({"project": refreshed, "files": files})

    @router.post("/projects/{project_id}/index")
    async def index_project(project_id: str, http_request: Request):
        user_id: str = http_request.state.user_id
        project = _get_owned_project(project_id, user_id)
        _start_index(project_id, str(project["folder_name"]))
        return JSONResponse(
            {
                "status": "started",
                "project_id": project_id,
                "folder": project["folder_name"],
                "collection": _project_collection_name(project_id),
            }
        )

    @router.get("/projects/{project_id}/index/status")
    async def index_status_project(project_id: str, http_request: Request):
        user_id: str = http_request.state.user_id
        _get_owned_project(project_id, user_id)
        with index_status_lock:
            status = index_status.get(
                project_id,
                {
                    "status": "idle",
                    "files_indexed": 0,
                    "chunks": 0,
                    "duration_s": 0.0,
                    "error": None,
                    "started_at": None,
                },
            )
        return JSONResponse(status)

    @router.get("/projects/{project_id}/files")
    async def project_files(project_id: str, http_request: Request):
        user_id: str = http_request.state.user_id
        project = _get_owned_project(project_id, user_id)
        files = _list_project_files(str(project["folder_name"]))
        return JSONResponse({"project_id": project_id, "files": files})

    return router
