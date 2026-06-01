from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from threading import Lock
from uuid import uuid4

from tools.workspace import WorkspacePathError, resolve_workspace_path


class ProjectError(Exception):
    def __init__(self, message: str, code: str, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


class ProjectStore:
    def __init__(self, *, db_path: str, code_workspace_root: str) -> None:
        self._db_path = db_path
        self._code_workspace_root_raw = code_workspace_root
        self._code_workspace_root = os.path.realpath(code_workspace_root)
        self._lock = Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    @property
    def code_workspace_root(self) -> str:
        return self._code_workspace_root

    def _init_db(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id          TEXT PRIMARY KEY,
                    user_id     TEXT NOT NULL,
                    name        TEXT NOT NULL,
                    folder_name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    created_at  REAL NOT NULL,
                    opened_at   REAL
                );

                CREATE INDEX IF NOT EXISTS idx_projects_user
                    ON projects(user_id, opened_at DESC);
                """
            )
            self._conn.commit()

    def _resolve_folder(self, folder_name: str) -> str:
        if not self._code_workspace_root_raw or not os.path.isdir(self._code_workspace_root):
            raise ProjectError(
                "CODE_WORKSPACE_ROOT not configured or missing",
                code="CODE_ROOT_UNAVAILABLE",
                status=503,
            )
        candidate = (folder_name or "").strip()
        if not candidate:
            raise ProjectError("folder_name is required", code="INVALID_FOLDER", status=400)
        try:
            return resolve_workspace_path(self._code_workspace_root, candidate)
        except WorkspacePathError as exc:
            raise ProjectError("Path traversal is not allowed", code="INVALID_FOLDER", status=400) from exc

    @staticmethod
    def _ensure_name(name: str) -> str:
        n = (name or "").strip()
        if not n:
            raise ProjectError("name is required", code="INVALID_NAME", status=400)
        return n

    def _row_to_project(self, row: sqlite3.Row) -> dict:
        folder_name = str(row["folder_name"])
        resolved = self._resolve_folder(folder_name)
        git = Path(resolved, ".git").is_dir()
        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "name": row["name"],
            "folder_name": folder_name,
            "description": row["description"],
            "created_at": row["created_at"],
            "opened_at": row["opened_at"],
            "git": git,
        }

    def create_project(
        self,
        user_id: str,
        name: str,
        folder_name: str,
        description: str = "",
    ) -> dict:
        if not user_id:
            raise ProjectError("user_id is required", code="INVALID_USER", status=400)

        safe_name = self._ensure_name(name)
        safe_folder_name = folder_name.strip()
        resolved_folder = self._resolve_folder(safe_folder_name)

        if Path(resolved_folder).exists():
            raise ProjectError("Project folder already exists", code="FOLDER_EXISTS", status=409)

        project_id = str(uuid4())
        now = time.time()

        with self._lock:
            cur = self._conn.cursor()
            cur.execute("BEGIN")
            try:
                cur.execute(
                    """
                    INSERT INTO projects (id, user_id, name, folder_name, description, created_at, opened_at)
                    VALUES (?, ?, ?, ?, ?, ?, NULL)
                    """,
                    (project_id, user_id, safe_name, safe_folder_name, description, now),
                )
                Path(resolved_folder).mkdir(parents=True, exist_ok=False)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

        project = self.get_project(project_id, user_id)
        if project is None:
            raise ProjectError("Failed to load created project", code="CREATE_FAILED", status=500)
        return project

    def list_projects(self, user_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, user_id, name, folder_name, description, created_at, opened_at
                FROM projects
                WHERE user_id = ?
                ORDER BY opened_at DESC, created_at DESC
                """,
                (user_id,),
            ).fetchall()
        return [self._row_to_project(r) for r in rows]

    def get_project(self, project_id: str, user_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT id, user_id, name, folder_name, description, created_at, opened_at
                FROM projects
                WHERE id = ? AND user_id = ?
                """,
                (project_id, user_id),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_project(row)

    def delete_project(self, project_id: str, user_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM projects WHERE id = ? AND user_id = ?",
                (project_id, user_id),
            )
            self._conn.commit()
        return cur.rowcount > 0

    def touch_opened(self, project_id: str, user_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE projects SET opened_at = ? WHERE id = ? AND user_id = ?",
                (time.time(), project_id, user_id),
            )
            self._conn.commit()
