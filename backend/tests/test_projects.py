import sqlite3
from pathlib import Path

import pytest

from projects import ProjectError, ProjectStore


@pytest.fixture
def project_store(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    db_path = tmp_path / "auth.db"
    return ProjectStore(db_path=str(db_path), code_workspace_root=str(workspace))


def _row_count(db_path: str, table: str) -> int:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return int(row[0]) if row else 0


def test_create_project_creates_db_row_and_folder(project_store):
    project = project_store.create_project(
        user_id="alice",
        name="My API",
        folder_name="my-api",
        description="sample",
    )

    assert project["name"] == "My API"
    assert project["folder_name"] == "my-api"
    assert project["git"] is False

    folder = Path(project_store._code_workspace_root) / "my-api"
    assert folder.is_dir()

    loaded = project_store.get_project(project["id"], "alice")
    assert loaded is not None
    assert loaded["id"] == project["id"]


def test_create_project_rejects_path_traversal(project_store):
    with pytest.raises(ProjectError) as exc_info:
        project_store.create_project(
            user_id="alice",
            name="Escape",
            folder_name="../outside",
        )

    assert exc_info.value.code == "INVALID_FOLDER"


def test_create_project_rolls_back_when_mkdir_fails(project_store):
    blocking = Path(project_store._code_workspace_root) / "taken"
    blocking.write_text("not a directory", encoding="utf-8")

    with pytest.raises(Exception):
        project_store.create_project(
            user_id="alice",
            name="Broken",
            folder_name="taken/subdir",
        )

    assert _row_count(project_store._db_path, "projects") == 0


def test_list_projects_orders_by_opened_time(project_store):
    first = project_store.create_project("alice", "First", "first")
    second = project_store.create_project("alice", "Second", "second")

    project_store.touch_opened(first["id"], "alice")

    rows = project_store.list_projects("alice")
    assert [r["id"] for r in rows] == [first["id"], second["id"]]


def test_get_project_returns_none_for_unknown(project_store):
    assert project_store.get_project("missing", "alice") is None


def test_delete_project_removes_row_only(project_store):
    project = project_store.create_project("alice", "Del", "del")
    folder = Path(project_store._code_workspace_root) / "del"

    assert project_store.delete_project(project["id"], "alice") is True
    assert project_store.get_project(project["id"], "alice") is None
    assert folder.is_dir()


def test_touch_opened_updates_timestamp(project_store):
    project = project_store.create_project("alice", "Touched", "touched")
    before = project_store.get_project(project["id"], "alice")
    assert before is not None
    assert before["opened_at"] is None

    project_store.touch_opened(project["id"], "alice")

    after = project_store.get_project(project["id"], "alice")
    assert after is not None
    assert isinstance(after["opened_at"], float)


def test_git_flag_reflects_dot_git_folder(project_store):
    project = project_store.create_project("alice", "Repo", "repo")
    repo_root = Path(project_store._code_workspace_root) / "repo"
    (repo_root / ".git").mkdir(parents=True, exist_ok=False)

    loaded = project_store.get_project(project["id"], "alice")
    assert loaded is not None
    assert loaded["git"] is True
