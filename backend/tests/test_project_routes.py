from __future__ import annotations

import pathlib
import sys
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def project_env(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "seed.py").write_text("print('seed')\n", encoding="utf-8")

    env = {
        "CODE_WORKSPACE_ROOT": str(workspace),
        "MEMORY_DB_PATH": str(tmp_path / "memory.db"),
        "CHROMA_PATH": str(tmp_path / "chroma"),
        "AUTH_DB_PATH": str(tmp_path / "auth.db"),
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    return workspace


def _import_main_module():
    import importlib
    import sys

    if "main" in sys.modules:
        return importlib.reload(sys.modules["main"])

    import main as main_mod  # type: ignore[import]

    return main_mod


@pytest.fixture()
def authed_client(project_env, monkeypatch):
    main_mod = _import_main_module()
    monkeypatch.setattr(main_mod.auth_service, "verify_token", lambda _token: "test-user")
    client = TestClient(main_mod.app, raise_server_exceptions=True)
    return client, main_mod


def test_projects_crud_endpoints(authed_client):
    client, _main_mod = authed_client
    headers = {"Authorization": "Bearer ok"}

    create_resp = client.post(
        "/projects",
        headers=headers,
        json={"name": "API", "folder_name": "api", "description": "demo"},
    )
    assert create_resp.status_code == 201
    project = create_resp.json()
    assert project["name"] == "API"
    project_id = project["id"]

    list_resp = client.get("/projects", headers=headers)
    assert list_resp.status_code == 200
    rows = list_resp.json().get("projects", [])
    assert any(p["id"] == project_id for p in rows)

    get_resp = client.get(f"/projects/{project_id}", headers=headers)
    assert get_resp.status_code == 200
    assert get_resp.json()["folder_name"] == "api"

    open_resp = client.post(f"/projects/{project_id}/open", headers=headers)
    assert open_resp.status_code == 200
    open_payload = open_resp.json()
    assert "project" in open_payload
    assert isinstance(open_payload.get("files"), list)

    files_resp = client.get(f"/projects/{project_id}/files", headers=headers)
    assert files_resp.status_code == 200
    assert files_resp.json()["project_id"] == project_id

    delete_resp = client.delete(f"/projects/{project_id}", headers=headers)
    assert delete_resp.status_code == 200
    assert delete_resp.json().get("ok") is True
    assert delete_resp.json().get("collection") == f"code_context_{project_id}"


def test_projects_index_endpoints(authed_client, monkeypatch):
    client, main_mod = authed_client
    headers = {"Authorization": "Bearer ok"}

    create_resp = client.post(
        "/projects",
        headers=headers,
        json={"name": "Idx", "folder_name": "idx", "description": ""},
    )
    assert create_resp.status_code == 201
    project_id = create_resp.json()["id"]

    called = {"count": 0}

    def _fake_index_workspace(_root, _chroma, index_paths=None, collection_name="code_context"):
        called["count"] += 1
        assert index_paths == ["idx"]
        assert collection_name == f"code_context_{project_id}"
        return 3

    import routes.project_routes as project_routes  # type: ignore[import]

    monkeypatch.setattr(project_routes, "index_workspace", _fake_index_workspace)

    start_resp = client.post(f"/projects/{project_id}/index", headers=headers)
    assert start_resp.status_code == 200
    assert start_resp.json().get("status") == "started"

    # Poll for background completion.
    deadline = time.time() + 2.0
    status_payload = None
    while time.time() < deadline:
        status_resp = client.get(f"/projects/{project_id}/index/status", headers=headers)
        assert status_resp.status_code == 200
        status_payload = status_resp.json()
        if status_payload.get("status") in ("done", "error"):
            break
        time.sleep(0.02)

    assert status_payload is not None
    assert status_payload.get("status") == "done"
    assert status_payload.get("files_indexed") == 3
    assert status_payload.get("chunks") == 3
    assert called["count"] == 1


def test_projects_delete_attempts_collection_cleanup(authed_client, monkeypatch):
    client, _main_mod = authed_client
    headers = {"Authorization": "Bearer ok"}

    create_resp = client.post(
        "/projects",
        headers=headers,
        json={"name": "Cleanup", "folder_name": "cleanup", "description": ""},
    )
    assert create_resp.status_code == 201
    project_id = create_resp.json()["id"]

    called = {"count": 0}

    def _fake_delete_collection(_chroma_path, collection_name):
        called["count"] += 1
        assert collection_name == f"code_context_{project_id}"
        return True

    import routes.project_routes as project_routes  # type: ignore[import]

    monkeypatch.setattr(project_routes, "delete_collection", _fake_delete_collection)

    delete_resp = client.delete(f"/projects/{project_id}", headers=headers)
    assert delete_resp.status_code == 200
    payload = delete_resp.json()
    assert payload.get("collection_deleted") is True
    assert called["count"] == 1


def test_projects_path_traversal_rejected(authed_client):
    client, _main_mod = authed_client
    headers = {"Authorization": "Bearer ok"}

    resp = client.post(
        "/projects",
        headers=headers,
        json={"name": "Bad", "folder_name": "../escape", "description": ""},
    )
    assert resp.status_code == 400


def test_projects_create_with_git_init_runs_strict_command(authed_client, monkeypatch):
    client, _main_mod = authed_client
    headers = {"Authorization": "Bearer ok"}

    called = {"cmd": None}

    def _fake_run(cmd, check, capture_output, text):
        called["cmd"] = cmd
        assert check is False
        assert capture_output is True
        assert text is True
        project_root = pathlib.Path(cmd[2])
        (project_root / ".git").mkdir(parents=True, exist_ok=True)

        class _Proc:
            returncode = 0
            stdout = "initialized"
            stderr = ""

        return _Proc()

    import routes.project_routes as project_routes  # type: ignore[import]

    monkeypatch.setattr(project_routes.subprocess, "run", _fake_run)

    resp = client.post(
        "/projects",
        headers=headers,
        json={"name": "GitProj", "folder_name": "git-proj", "description": "", "git_init": True},
    )
    assert resp.status_code == 201
    payload = resp.json()
    assert payload["git"] is True

    assert called["cmd"] is not None
    assert called["cmd"][0] == "git"
    assert called["cmd"][1] == "init"
    assert called["cmd"][2].endswith("git-proj")


def test_projects_create_with_git_init_failure_is_non_fatal(authed_client, monkeypatch):
    client, _main_mod = authed_client
    headers = {"Authorization": "Bearer ok"}

    def _fake_run(_cmd, check, capture_output, text):
        assert check is False
        assert capture_output is True
        assert text is True

        class _Proc:
            returncode = 1
            stdout = ""
            stderr = "simulated git error"

        return _Proc()

    import routes.project_routes as project_routes  # type: ignore[import]

    monkeypatch.setattr(project_routes.subprocess, "run", _fake_run)

    resp = client.post(
        "/projects",
        headers=headers,
        json={"name": "GitFail", "folder_name": "git-fail", "description": "", "git_init": True},
    )
    assert resp.status_code == 201
    payload = resp.json()
    assert payload["name"] == "GitFail"
    # Git init failed, but project creation still succeeds.
    assert payload["git"] is False


def test_projects_require_auth(project_env):
    main_mod = _import_main_module()
    client = TestClient(main_mod.app, raise_server_exceptions=False)

    resp = client.get("/projects")
    assert resp.status_code == 401
