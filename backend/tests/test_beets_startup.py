"""
Tests for the Beets startup bootstrap helpers in backend/main.py.

Covers:
  - _beets_db_has_items: returns False for missing/empty DB, True when populated
  - _bootstrap_beets_library_if_empty: skips when DB populated, MUSIC_ROOT missing,
    directory missing, or `beet` not in PATH; runs import when conditions are met
  - run_beets_update: config/PATH error shapes, update+import command sequence,
    failure and timeout handling
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
from unittest.mock import MagicMock, patch

import pytest

# Import the two helpers directly so we don't have to boot the full app.
import importlib
import sys
import types

# Minimal stubs for modules main.py imports at the top level.
def _stub_if_missing(name: str, attrs: dict | None = None) -> None:
    if name not in sys.modules:
        mod = types.ModuleType(name)
        if attrs:
            for k, v in attrs.items():
                setattr(mod, k, v)
        sys.modules[name] = mod


_stub_if_missing("faster_whisper")
_stub_if_missing("openwakeword")
_stub_if_missing("openwakeword.model")
_stub_if_missing("musicpd")
_stub_if_missing("numpy", {"ndarray": object, "float32": float})
_stub_if_missing("chromadb")
_stub_if_missing("chromadb.config")
_stub_if_missing("langgraph")
_stub_if_missing("langgraph.graph")
_stub_if_missing("kokoro_onnx")

# Patch the heavy imports before loading main.
# Delete stubs after loading to prevent shadowing real packages during test collection.
with (
    patch.dict(os.environ, {
        "ANTHROPIC_API_KEY": "test",
        "CORS_ORIGINS": "*",
        "SESSION_COOKIE_SECURE": "false",
    }),
    patch("builtins.__import__", side_effect=lambda name, *a, **kw: __builtins__["__import__"](name, *a, **kw) if isinstance(__builtins__, dict) else __import__(name, *a, **kw)),
):
    # We only need to pull in the two pure helper functions. Do that via
    # direct exec of a trimmed import rather than loading the full app.
    pass


def _make_db_with_items(path: str, n: int = 1) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, title TEXT, path TEXT)")
    for i in range(n):
        conn.execute("INSERT INTO items VALUES (?, ?, ?)", (i, f"Track {i}", f"/music/t{i}.mp3"))
    conn.commit()
    conn.close()


def _make_empty_db(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, title TEXT, path TEXT)")
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Direct unit tests for the two helpers, imported without loading the full app.
# ---------------------------------------------------------------------------

def _load_helpers():
    """Import _beets_db_has_items and _bootstrap_beets_library_if_empty
    without triggering FastAPI app startup."""
    import importlib.util, pathlib

    src = pathlib.Path(__file__).parent.parent / "main.py"
    spec = importlib.util.spec_from_file_location("_main_helpers", src)
    # We can't exec the full module cleanly, so extract the two functions by
    # compiling only the code that defines them via source slicing.
    lines = src.read_text().splitlines()
    # Find the function definitions and extract them.
    extracted: list[str] = [
        "import os, sqlite3, shutil, subprocess, logging, threading",
        "log = logging.getLogger('beets_startup_test')",
        "_BEETS_UPDATE_LOCK = threading.Lock()",
    ]
    targets = {"_beets_db_has_items", "_beet_base_cmd", "_bootstrap_beets_library_if_empty", "run_beets_update", "_run_beets_update", "_stamp_beets_mtimes"}
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if any(stripped.startswith(f"def {t}(") for t in targets):
            block = [line]
            i += 1
            while i < len(lines):
                next_line = lines[i]
                # Collect until next top-level definition or class.
                if next_line and next_line[0] not in (" ", "\t", "#", "\n") and next_line.strip():
                    break
                block.append(next_line)
                i += 1
            extracted.extend(block)
        else:
            i += 1

    ns: dict = {}
    exec(compile("\n".join(extracted), str(src), "exec"), ns)  # noqa: S102
    _load_helpers.ns = ns
    return (
        ns["_beets_db_has_items"],
        ns["_bootstrap_beets_library_if_empty"],
        ns["run_beets_update"],
    )


_beets_db_has_items, _bootstrap_beets_library_if_empty, run_beets_update = _load_helpers()

# Clear stubs that were only needed for loading main.py helpers.
# This prevents them from shadowing real packages during test collection.
for _stub_name in ["chromadb", "chromadb.config", "langgraph", "langgraph.graph", "numpy", "musicpd", "faster_whisper", "openwakeword", "openwakeword.model", "kokoro_onnx"]:
    sys.modules.pop(_stub_name, None)


# ── _beets_db_has_items ───────────────────────────────────────────────────────

def test_has_items_false_for_missing_file():
    assert _beets_db_has_items("/nonexistent/path/library.db") is False


def test_has_items_false_for_empty_string():
    assert _beets_db_has_items("") is False


def test_has_items_false_for_empty_table():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        _make_empty_db(db_path)
        assert _beets_db_has_items(db_path) is False
    finally:
        os.unlink(db_path)


def test_has_items_true_when_populated():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        _make_db_with_items(db_path, n=3)
        assert _beets_db_has_items(db_path) is True
    finally:
        os.unlink(db_path)


# ── _bootstrap_beets_library_if_empty ─────────────────────────────────────────

def test_bootstrap_skips_when_db_populated(tmp_path, monkeypatch):
    db = tmp_path / "library.db"
    _make_db_with_items(str(db), n=1)
    monkeypatch.setenv("BEETS_DB_PATH", str(db))
    monkeypatch.setenv("MUSIC_ROOT", str(tmp_path))

    with patch("subprocess.run") as mock_run:
        _bootstrap_beets_library_if_empty()
        mock_run.assert_not_called()


def test_bootstrap_skips_when_music_root_missing(tmp_path, monkeypatch):
    db = tmp_path / "library.db"
    monkeypatch.setenv("BEETS_DB_PATH", str(db))
    monkeypatch.delenv("MUSIC_ROOT", raising=False)

    with patch("subprocess.run") as mock_run:
        _bootstrap_beets_library_if_empty()
        mock_run.assert_not_called()


def test_bootstrap_skips_when_music_root_not_a_directory(tmp_path, monkeypatch):
    db = tmp_path / "library.db"
    monkeypatch.setenv("BEETS_DB_PATH", str(db))
    monkeypatch.setenv("MUSIC_ROOT", str(tmp_path / "nonexistent"))

    with patch("subprocess.run") as mock_run:
        _bootstrap_beets_library_if_empty()
        mock_run.assert_not_called()


def test_bootstrap_skips_when_beet_not_in_path(tmp_path, monkeypatch):
    db = tmp_path / "library.db"
    monkeypatch.setenv("BEETS_DB_PATH", str(db))
    monkeypatch.setenv("MUSIC_ROOT", str(tmp_path))

    with (
        patch("shutil.which", return_value=None),
        patch("subprocess.run") as mock_run,
    ):
        _bootstrap_beets_library_if_empty()
        mock_run.assert_not_called()


def test_bootstrap_runs_beet_import_when_db_empty(tmp_path, monkeypatch):
    db = tmp_path / "library.db"
    # Don't create the DB — it's missing → treat as empty.
    monkeypatch.setenv("BEETS_DB_PATH", str(db))
    monkeypatch.setenv("MUSIC_ROOT", str(tmp_path))

    fake_result = MagicMock()
    fake_result.returncode = 0

    with (
        patch("shutil.which", return_value="/usr/bin/beet"),
        patch("subprocess.run", return_value=fake_result) as mock_run,
    ):
        _bootstrap_beets_library_if_empty()

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "/usr/bin/beet"
    assert "-l" in cmd
    assert str(db) in cmd
    assert "import" in cmd
    assert "-A" in cmd
    assert str(tmp_path) in cmd


def test_bootstrap_logs_warning_on_import_failure(tmp_path, monkeypatch):
    db = tmp_path / "library.db"
    monkeypatch.setenv("BEETS_DB_PATH", str(db))
    monkeypatch.setenv("MUSIC_ROOT", str(tmp_path))

    err = subprocess_error = __import__("subprocess").CalledProcessError(
        1, ["beet"], stderr="import failed"
    )

    with (
        patch("shutil.which", return_value="/usr/bin/beet"),
        patch("subprocess.run", side_effect=err),
        patch("logging.Logger.warning") as mock_warn,
    ):
        # Should not raise — errors are caught and logged.
        _bootstrap_beets_library_if_empty()
        assert mock_warn.called


# ── run_beets_update ──────────────────────────────────────────────────────────

def test_beets_update_fails_when_music_root_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("BEETS_DB_PATH", str(tmp_path / "library.db"))
    monkeypatch.delenv("MUSIC_ROOT", raising=False)

    result = run_beets_update()

    assert result["ok"] is False
    assert result["code"] == "BEETS_MUSIC_ROOT_MISSING"


def test_beets_update_fails_when_music_root_not_a_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("BEETS_DB_PATH", str(tmp_path / "library.db"))
    monkeypatch.setenv("MUSIC_ROOT", str(tmp_path / "nonexistent"))

    result = run_beets_update()

    assert result["ok"] is False
    assert result["code"] == "BEETS_MUSIC_ROOT_MISSING"


def test_beets_update_fails_when_beet_not_in_path(tmp_path, monkeypatch):
    monkeypatch.setenv("BEETS_DB_PATH", str(tmp_path / "library.db"))
    monkeypatch.setenv("MUSIC_ROOT", str(tmp_path))

    with patch("shutil.which", return_value=None):
        result = run_beets_update()

    assert result["ok"] is False
    assert result["code"] == "BEETS_NOT_FOUND"


def test_beets_update_runs_update_then_import_no_autotag(tmp_path, monkeypatch):
    db = tmp_path / "library.db"
    monkeypatch.setenv("BEETS_DB_PATH", str(db))
    monkeypatch.setenv("MUSIC_ROOT", str(tmp_path))

    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = "15 items updated\n"
    fake_result.stderr = ""

    with (
        patch("shutil.which", return_value="/usr/bin/beet"),
        patch("subprocess.run", return_value=fake_result) as mock_run,
    ):
        result = run_beets_update()

    assert result["ok"] is True
    assert mock_run.call_count == 2
    first_cmd = mock_run.call_args_list[0][0][0]
    second_cmd = mock_run.call_args_list[1][0][0]
    assert first_cmd[:5] == ["/usr/bin/beet", "-l", str(db), "-d", str(tmp_path)]
    assert first_cmd[5:] == ["update"]
    assert second_cmd[:5] == ["/usr/bin/beet", "-l", str(db), "-d", str(tmp_path)]
    assert second_cmd[5:] == ["import", "-A", "-q", str(tmp_path)]


def test_beets_update_reports_failure_stderr_tail(tmp_path, monkeypatch):
    monkeypatch.setenv("BEETS_DB_PATH", str(tmp_path / "library.db"))
    monkeypatch.setenv("MUSIC_ROOT", str(tmp_path))

    err = __import__("subprocess").CalledProcessError(1, ["beet"], stderr="line one\nupdate failed")

    with (
        patch("shutil.which", return_value="/usr/bin/beet"),
        patch("subprocess.run", side_effect=err),
    ):
        result = run_beets_update()

    assert result["ok"] is False
    assert result["code"] == "BEETS_UPDATE_FAILED"
    assert result["error"] == "update failed"


def test_beets_update_reports_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("BEETS_DB_PATH", str(tmp_path / "library.db"))
    monkeypatch.setenv("MUSIC_ROOT", str(tmp_path))

    timeout_err = __import__("subprocess").TimeoutExpired(cmd=["beet"], timeout=1800)

    with (
        patch("shutil.which", return_value="/usr/bin/beet"),
        patch("subprocess.run", side_effect=timeout_err),
    ):
        result = run_beets_update()

    assert result["ok"] is False
    assert result["code"] == "BEETS_UPDATE_TIMEOUT"


def test_beets_update_rejects_concurrent_run(tmp_path, monkeypatch):
    monkeypatch.setenv("BEETS_DB_PATH", str(tmp_path / "library.db"))
    monkeypatch.setenv("MUSIC_ROOT", str(tmp_path))
    lock = _load_helpers.ns["_BEETS_UPDATE_LOCK"]

    assert lock.acquire(blocking=False)
    try:
        with patch("subprocess.run") as mock_run:
            result = run_beets_update()
    finally:
        lock.release()

    assert result["ok"] is False
    assert result["code"] == "BEETS_UPDATE_IN_PROGRESS"
    mock_run.assert_not_called()


def test_stamp_beets_mtimes_records_file_mtimes(tmp_path):
    db = tmp_path / "library.db"
    music = tmp_path / "music"
    (music / "album").mkdir(parents=True)
    track = music / "album" / "song.mp3"
    track.write_bytes(b"x")
    os.utime(track, (1_700_000_000, 1_700_000_000))

    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, path BLOB, mtime REAL)")
    conn.executemany(
        "INSERT INTO items (id, path, mtime) VALUES (?, ?, 0)",
        [(1, b"album/song.mp3"), (2, b"album/missing.mp3")],
    )
    conn.commit()
    conn.close()

    _load_helpers.ns["_stamp_beets_mtimes"](str(db), str(music))

    conn = sqlite3.connect(db)
    mtimes = dict(conn.execute("SELECT id, mtime FROM items").fetchall())
    conn.close()
    assert mtimes == {1: 1_700_000_000.0, 2: 0.0}
