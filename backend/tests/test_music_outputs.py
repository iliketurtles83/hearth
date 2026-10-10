"""
Tests for music output selection — Phase 1 Native Output Switching.

Covers:
  - Querying MPD outputs (_sync_get_outputs / action="outputs")
  - Selecting an output in exclusive mode (_sync_set_output / action="select_output")
  - Selecting an output in mirror / enable / disable / toggle modes
  - Error handling when MPD is unavailable
  - FastAPI endpoints /music/outputs and /music/outputs/select
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch
import pytest

import tools
import tools.music  # noqa: F401  (registers the tool)
from music.players import mpd as music_mpd


_SAMPLE_OUTPUTS = [
    {"outputid": "0", "outputname": "Host Speakers", "outputenabled": "1", "plugin": "pulse"},
    {"outputid": "1", "outputname": "Web Stream", "outputenabled": "0", "plugin": "httpd"},
]


class _MockMPDClient:
    def __init__(self, outputs=None):
        self._outputs = [dict(o) for o in (outputs or _SAMPLE_OUTPUTS)]
        self.timeout = 5

    def connect(self, host: str, port: int) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def outputs(self):
        return [dict(o) for o in self._outputs]

    def enableoutput(self, oid: int):
        for o in self._outputs:
            if o["outputid"] == str(oid):
                o["outputenabled"] = "1"

    def disableoutput(self, oid: int):
        for o in self._outputs:
            if o["outputid"] == str(oid):
                o["outputenabled"] = "0"


def test_sync_get_outputs():
    mock_client = _MockMPDClient()
    with patch.object(music_mpd, "_mpd_connect", return_value=mock_client):
        res = music_mpd._sync_get_outputs()
        assert len(res) == 2
        assert res[0] == {"id": "0", "name": "Host Speakers", "enabled": True, "plugin": "pulse"}
        assert res[1] == {"id": "1", "name": "Web Stream", "enabled": False, "plugin": "httpd"}


def test_sync_set_output_exclusive():
    mock_client = _MockMPDClient()
    with patch.object(music_mpd, "_mpd_connect", return_value=mock_client):
        # Switch exclusively to output 1
        res = music_mpd._sync_set_output("1", mode="exclusive")
        assert res[0]["enabled"] is False  # Host Speakers disabled
        assert res[1]["enabled"] is True   # Web Stream enabled

        # Switch exclusively back to output 0
        res = music_mpd._sync_set_output("0", mode="exclusive")
        assert res[0]["enabled"] is True   # Host Speakers enabled
        assert res[1]["enabled"] is False  # Web Stream disabled


def test_sync_set_output_mirror():
    mock_client = _MockMPDClient()
    with patch.object(music_mpd, "_mpd_connect", return_value=mock_client):
        # In mirror mode, both should be enabled
        res = music_mpd._sync_set_output("1", mode="mirror")
        assert res[0]["enabled"] is True
        assert res[1]["enabled"] is True


def test_sync_set_output_both():
    mock_client = _MockMPDClient(outputs=[
        {"outputid": "0", "outputname": "Host Speakers", "outputenabled": "0", "plugin": "pulse"},
        {"outputid": "1", "outputname": "Web Stream", "outputenabled": "0", "plugin": "httpd"},
    ])
    with patch.object(music_mpd, "_mpd_connect", return_value=mock_client):
        # In both/all mode, all outputs should be enabled
        res = music_mpd._sync_set_output("all", mode="both")
        assert res[0]["enabled"] is True
        assert res[1]["enabled"] is True


def test_sync_set_output_toggle_and_disable():
    mock_client = _MockMPDClient()
    with patch.object(music_mpd, "_mpd_connect", return_value=mock_client):
        # Toggle output 1 from disabled to enabled
        res = music_mpd._sync_set_output("1", mode="toggle")
        assert res[1]["enabled"] is True

        # Toggle output 1 from enabled to disabled
        res = music_mpd._sync_set_output("1", mode="toggle")
        assert res[1]["enabled"] is False

        # Explicit disable
        res = music_mpd._sync_set_output("0", mode="disable")
        assert res[0]["enabled"] is False


@pytest.mark.asyncio
async def test_tool_dispatch_outputs():
    mock_client = _MockMPDClient()
    with patch.object(music_mpd, "_mpd_connect", return_value=mock_client):
        result = await tools.dispatch("music", {"action": "outputs"})
        assert result.ok is True
        assert "outputs" in result.data
        assert len(result.data["outputs"]) == 2


@pytest.mark.asyncio
async def test_tool_dispatch_select_output():
    mock_client = _MockMPDClient()
    with patch.object(music_mpd, "_mpd_connect", return_value=mock_client):
        result = await tools.dispatch("music", {"action": "select_output", "output_id": "1", "mode": "exclusive"})
        assert result.ok is True
        assert result.data["action"] == "select_output"
        outputs = result.data["outputs"]
        assert outputs[0]["enabled"] is False
        assert outputs[1]["enabled"] is True


@pytest.mark.asyncio
async def test_tool_dispatch_select_output_missing_id():
    result = await tools.dispatch("music", {"action": "select_output"})
    assert result.ok is False
    assert "output_id is required" in result.error


def test_api_music_outputs_and_select():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from routes.memory_tool_routes import create_memory_tool_router

    app = FastAPI()
    router = create_memory_tool_router(
        get_memory_store=lambda: None,
        memory_consolidation_batch_size=10,
        error_response=lambda msg, code, ret, status_code: None,
        dispatch_tool=tools.dispatch,
        run_weather=lambda p: None,
        run_beets_update=lambda: {"ok": True},
    )
    app.include_router(router)
    client = TestClient(app)

    mock_client = _MockMPDClient()
    with patch.object(music_mpd, "_mpd_connect", return_value=mock_client):
        resp = client.get("/music/outputs")
        assert resp.status_code == 200
        data = resp.json()
        assert "outputs" in data
        assert len(data["outputs"]) == 2

        resp2 = client.post("/music/outputs/select", json={"output_id": "1", "mode": "exclusive"})
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["action"] == "select_output"
        assert data2["outputs"][1]["enabled"] is True
        assert data2["outputs"][0]["enabled"] is False

        resp3 = client.post("/music/outputs/select", json={"output_id": "all", "mode": "both"})
        assert resp3.status_code == 200
        data3 = resp3.json()
        assert data3["action"] == "select_output"
        assert data3["outputs"][0]["enabled"] is True
        assert data3["outputs"][1]["enabled"] is True



def test_api_music_timing_logs_client_marks(caplog):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from routes.memory_tool_routes import create_memory_tool_router

    app = FastAPI()
    app.include_router(
        create_memory_tool_router(
            get_memory_store=lambda: None,
            memory_consolidation_batch_size=10,
            error_response=lambda msg, code, ret, status_code: None,
            dispatch_tool=tools.dispatch,
            run_weather=lambda p: None,
            run_beets_update=lambda: {"ok": True},
        )
    )
    client = TestClient(app)

    with caplog.at_level("INFO", logger="assistant.music"):
        resp = client.post(
            "/music/timing",
            json={
                "mode": "fresh",
                "output": "phone",
                "marks": {"stream_playing": 6900, "reply_done": 1200, "Bad Key\n": 5},
                "drift_s": 0.25,
            },
        )
    assert resp.status_code == 200
    line = next(r.getMessage() for r in caplog.records if "music.client_timing" in r.getMessage())
    assert "mode=fresh output=phone drift_s=0.25" in line
    assert "marks_ms=[reply_done=1200 stream_playing=6900]" in line

    too_many = {f"k{chr(97 + i)}": i for i in range(20)}
    assert client.post("/music/timing", json={"mode": "x", "output": "y", "marks": too_many}).status_code == 422


def _music_router_client():
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient
    from routes.memory_tool_routes import create_memory_tool_router

    app = FastAPI()
    app.include_router(
        create_memory_tool_router(
            get_memory_store=lambda: None,
            memory_consolidation_batch_size=10,
            error_response=lambda msg, code, ret, status_code: JSONResponse({"error": msg, "code": code}, status_code=status_code),
            dispatch_tool=tools.dispatch,
            run_weather=lambda p: None,
            run_beets_update=lambda: {"ok": True},
        )
    )
    return TestClient(app)


@pytest.fixture
def music_root(tmp_path, monkeypatch):
    root = tmp_path / "music"
    (root / "rock" / "Band").mkdir(parents=True)
    (root / "rock" / "Band" / "01 - Song.mp3").write_bytes(b"ID3" + bytes(range(256)) * 4)
    (root / "rock" / "Band" / "cover.jpg").write_bytes(b"\xff\xd8")
    (root / "rock" / "Band" / "old.wma").write_bytes(b"wma")
    (tmp_path / "secret.mp3").write_bytes(b"outside")
    monkeypatch.setattr(music_mpd, "MUSIC_ROOT", str(root))
    return root


def test_resolve_audio_file_only_serves_audio_inside_music_root(music_root):
    full, media_type = music_mpd.resolve_audio_file("rock/Band/01 - Song.mp3")
    assert full == str(music_root / "rock" / "Band" / "01 - Song.mp3")
    assert media_type == "audio/mpeg"
    for bad in (
        "../secret.mp3",                # escapes the root
        "rock/../../secret.mp3",
        str(music_root / "rock" / "Band" / "01 - Song.mp3"),  # absolute
        "rock/Band/cover.jpg",          # not audio
        "rock/Band/old.wma",            # browsers can't play it
        "rock/Band/missing.mp3",
        "rock/Band",                    # directory
        "rock/Band/01 - Song.mp3\x00.jpg",
        "",
    ):
        assert music_mpd.resolve_audio_file(bad) is None, bad


def test_api_music_file_serves_ranges_and_rejects_outside_paths(music_root):
    client = _music_router_client()
    resp = client.get("/music/file", params={"path": "rock/Band/01 - Song.mp3"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    assert resp.content.startswith(b"ID3")

    # The browser seeks (and starts at MPD's position) with byte ranges.
    ranged = client.get("/music/file", params={"path": "rock/Band/01 - Song.mp3"}, headers={"Range": "bytes=3-6"})
    assert ranged.status_code == 206
    assert ranged.content == bytes([0, 1, 2, 3])

    assert client.get("/music/file", params={"path": "../secret.mp3"}).status_code == 404
    assert client.get("/music/file", params={"path": "rock/Band/cover.jpg"}).status_code == 404


def test_now_playing_reports_current_and_next_file():
    client = MagicMock()
    client.status.return_value = {"state": "play", "song": "0", "nextsong": "1", "elapsed": "12.5", "duration": "200.0", "volume": "70"}
    client.currentsong.return_value = {"title": "A", "artist": "X", "album": "Y", "file": "rock/a.mp3"}
    client.playlistinfo.return_value = [{"file": "rock/b.mp3"}]
    with patch.object(music_mpd, "_mpd_connect", return_value=client):
        data = music_mpd._sync_now_playing()
    assert data["track"]["file"] == "rock/a.mp3"
    assert data["next_file"] == "rock/b.mp3"
    client.playlistinfo.assert_called_once_with(1)

    client.status.return_value = {"state": "play", "song": "1", "elapsed": "1.0", "volume": "70"}
    with patch.object(music_mpd, "_mpd_connect", return_value=client):
        assert music_mpd._sync_now_playing()["next_file"] is None
