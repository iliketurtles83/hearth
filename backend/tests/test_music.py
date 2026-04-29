"""
Tests for backend/tools/music.py — Phase 8.

Covers:
  - Strawberry scan-lock handling (sqlite3.OperationalError → retryable ToolResult)
  - MPD reconnect-once policy (first connect fails, retry succeeds)
  - MPD total connection failure → retryable ToolResult
  - Search returns ranked results (ordered by playcount)
  - Artist radio: weighted-random selection seeded for determinism
  - Path rewrite (_url_to_mpd_path)
  - Standardized error shape from music endpoints (/music/control, /music/now_playing, etc.)
  - Auto-pick: top-ranked result is selected when multiple matches exist
  - Artist radio fallback when no LIKE results found
"""
from __future__ import annotations

import asyncio
import sqlite3
import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

# ── Stub out musicpd and tools registry before importing music.py ──────────────

# Create a fake musicpd module so importing music.py does not require the package.
_fake_musicpd = types.ModuleType("musicpd")


class _FakeMPDClient:
    timeout: int = 5
    _connected: bool = False

    def connect(self, host: str, port: int) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def status(self) -> dict:
        return {"state": "stop"}

    def currentsong(self) -> dict:
        return {}

    def playlistinfo(self):
        return []

    def clear(self) -> None:
        pass

    def add(self, path: str) -> None:
        pass

    def play(self, pos: int | None = None) -> None:
        pass

    def pause(self, val: int = 1) -> None:
        pass

    def next(self) -> None:
        pass

    def stop(self) -> None:
        pass


class _FakeConnectionError(Exception):
    pass


class _FakeCommandError(Exception):
    pass


_fake_musicpd.MPDClient = _FakeMPDClient
_fake_musicpd.ConnectionError = _FakeConnectionError
_fake_musicpd.CommandError = _FakeCommandError
sys.modules["musicpd"] = _fake_musicpd

# Create a minimal tools registry stub.
_fake_tools = types.ModuleType("tools")
import os
# Make the fake 'tools' module package-like so submodule imports (tools.music)
# can be resolved against the real `tools` directory on disk when running tests.
_fake_tools.__path__ = [os.path.join(os.getcwd(), "tools")]
_fake_tools_registry: dict[str, Any] = {}


def _fake_register(name: str, module: Any) -> None:
    _fake_tools_registry[name] = module


_fake_tools.register = _fake_register
_fake_tools.ToolResult = None  # will be overridden after import

# Stub tools.base so ToolResult can be imported.
_fake_base = types.ModuleType("tools.base")


class ToolResult:
    def __init__(self, ok: bool, data: Any = None, error: str = "", retryable: bool = False):
        self.ok = ok
        self.data = data
        self.error = error
        self.retryable = retryable

    @classmethod
    def failure(cls, error: str, retryable: bool = True) -> "ToolResult":
        return cls(ok=False, error=error, retryable=retryable)


_fake_base.ToolResult = ToolResult
sys.modules["tools"] = _fake_tools
sys.modules["tools.base"] = _fake_base

# Now import the module under test.
import importlib
import os

os.environ.setdefault("STRAWBERRY_DB_PATH", "/nonexistent/test.db")
os.environ.setdefault("MPD_HOST", "localhost")
os.environ.setdefault("MPD_PORT", "6600")
os.environ.setdefault("MUSIC_PATH_HOST", "/media/jack/buffer/audio")

import tools.music as music  # noqa: E402  (must come after stubs)

# Patch ToolResult references in the music module.
music.ToolResult = ToolResult


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_rows(records: list[dict]) -> list[sqlite3.Row]:
    """Create lightweight fake sqlite3.Row objects from dicts."""
    class FakeRow(dict):
        def __getitem__(self, key):
            return super().__getitem__(key)
    return [FakeRow(r) for r in records]


# ── Path rewrite tests ─────────────────────────────────────────────────────────

def test_url_to_mpd_path_standard():
    url = "file:///media/jack/buffer/audio/rock/artist/song.mp3"
    assert music._url_to_mpd_path(url) == "rock/artist/song.mp3"


def test_url_to_mpd_path_percent_encoded():
    url = "file:///media/jack/buffer/audio/rock/The%20Band/My%20Song.mp3"
    assert music._url_to_mpd_path(url) == "rock/The Band/My Song.mp3"


def test_url_to_mpd_path_no_prefix_match():
    """Paths that don't match MUSIC_PATH_HOST are returned as-is after stripping file://."""
    url = "file:///other/path/song.mp3"
    result = music._url_to_mpd_path(url)
    assert result == "other/path/song.mp3"


def test_url_to_mpd_path_no_file_prefix():
    """Non-file:// URLs fall through the decode branch."""
    url = "/media/jack/buffer/audio/song.mp3"
    result = music._url_to_mpd_path(url)
    assert result == "song.mp3"


# ── Strawberry scan-lock handling ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_db_locked_returns_retryable():
    """sqlite3.OperationalError during search → retryable ToolResult.failure."""
    with patch.object(music, "_sync_search", side_effect=sqlite3.OperationalError("database is locked")):
        result = await music.run({"action": "search", "query": "test", "prompt": "test"})
    assert not result.ok
    assert result.retryable
    assert "locked" in result.error.lower()


@pytest.mark.asyncio
async def test_play_db_locked_returns_retryable():
    """sqlite3.OperationalError during play search → retryable ToolResult.failure."""
    with patch.object(music, "_sync_search", side_effect=sqlite3.OperationalError("database is locked")):
        result = await music.run({"action": "play", "query": "song", "prompt": "play song"})
    assert not result.ok
    assert result.retryable


# ── MPD reconnect-once policy ──────────────────────────────────────────────────

def test_mpd_connect_retries_once_on_failure():
    """If first connect raises ConnectionError, _mpd_connect retries once."""
    call_count = 0

    class FlakyClient(_FakeMPDClient):
        def connect(self, host, port):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _FakeConnectionError("refused")
            # second attempt succeeds
            self._connected = True

    with patch.object(music.musicpd, "MPDClient", FlakyClient):
        client = music._mpd_connect()
    assert call_count == 2
    assert client._connected


def test_mpd_connect_raises_after_two_failures():
    """If both connection attempts fail, _mpd_connect propagates the exception."""
    class AlwaysFailClient(_FakeMPDClient):
        def connect(self, host, port):
            raise _FakeConnectionError("always refused")

    with patch.object(music.musicpd, "MPDClient", AlwaysFailClient):
        with pytest.raises(_FakeConnectionError):
            music._mpd_connect()


@pytest.mark.asyncio
async def test_play_mpd_total_failure_returns_retryable():
    """MPD connection failure during play → retryable ToolResult."""
    fake_track = {"id": 1, "title": "T", "artist": "A", "album": "B", "url": "file:///media/jack/buffer/audio/t.mp3", "score": 0.9}
    with (
        patch.object(music, "_sync_search", return_value=[fake_track]),
        patch.object(music, "_sync_play", side_effect=ConnectionRefusedError("mpd down")),
    ):
        result = await music.run({"action": "play", "query": "T", "prompt": "play T"})
    assert not result.ok
    assert result.retryable


# ── Search ranking ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_returns_ranked_results():
    """Search results are returned in playcount-descending order."""
    ranked = [
        {"id": 1, "title": "Popular Song", "artist": "Artist", "album": "Album", "url": "file:///media/jack/buffer/audio/a.mp3", "score": 0.9},
        {"id": 2, "title": "Obscure Song", "artist": "Artist", "album": "Album", "url": "file:///media/jack/buffer/audio/b.mp3", "score": 0.76},
    ]
    with patch.object(music, "_sync_search", return_value=ranked):
        result = await music.run({"action": "search", "query": "artist", "prompt": "artist"})
    assert result.ok
    assert result.data["results"][0]["id"] == 1
    assert result.data["total"] == 2


# ── Artist radio ───────────────────────────────────────────────────────────────

def test_artist_radio_seeded_determinism():
    """Same seed always returns the same track list."""
    songs = [
        {"id": i, "title": f"Song {i}", "artist": "TestBand", "album": "A",
         "url": f"file:///media/jack/buffer/audio/s{i}.mp3", "score": 0.0, "playcount": i * 10}
        for i in range(1, 20)
    ]
    with patch.object(music, "_sync_artist_songs", return_value=songs):
        result_a = music.artist_radio("TestBand", n=5, seed=42)
        result_b = music.artist_radio("TestBand", n=5, seed=42)
    assert result_a == result_b
    assert len(result_a) == 5


def test_artist_radio_no_duplicates():
    """Artist radio never returns the same song twice."""
    songs = [
        {"id": i, "title": f"Song {i}", "artist": "Band", "album": "A",
         "url": f"file:///media/jack/buffer/audio/s{i}.mp3", "score": 0.0, "playcount": 1}
        for i in range(1, 8)
    ]
    with patch.object(music, "_sync_artist_songs", return_value=songs):
        result = music.artist_radio("Band", n=6, seed=99)
    ids = [t["id"] for t in result]
    assert len(ids) == len(set(ids)), "Duplicate tracks returned by artist_radio"


def test_playlist_pick_count_adaptive_bounds(monkeypatch):
    monkeypatch.setattr(music, "MUSIC_PLAYLIST_MIN_N", 6)
    monkeypatch.setattr(music, "MUSIC_PLAYLIST_MAX_N", 12)

    assert music._playlist_pick_count(0) == 0
    assert music._playlist_pick_count(4) == 4
    assert music._playlist_pick_count(20) == 10
    assert music._playlist_pick_count(100) == 12
    assert music._playlist_pick_count(100, requested_n=5) == 5


def test_artist_radio_default_target_is_adaptive(monkeypatch):
    monkeypatch.setattr(music, "MUSIC_PLAYLIST_MIN_N", 12)
    monkeypatch.setattr(music, "MUSIC_PLAYLIST_MAX_N", 24)
    songs = [
        {"id": i, "title": f"Song {i}", "artist": "Band", "album": "A",
         "url": f"file:///media/jack/buffer/audio/s{i}.mp3", "score": 0.0, "playcount": i}
        for i in range(1, 41)
    ]
    with patch.object(music, "_sync_artist_songs", return_value=songs):
        result = music.artist_radio("Band", seed=7)
    # pool=40 -> adaptive pick count = min(max_n=24, max(min_n=12, pool//2=20)) => 20
    assert len(result) == 20


def test_artist_radio_empty_when_no_artist():
    """Returns empty list when artist is not in the library."""
    with patch.object(music, "_sync_artist_songs", return_value=[]):
        result = music.artist_radio("NonExistentArtist")
    assert result == []


def test_resolve_genre_query_matches_taxonomy_term(monkeypatch):
    monkeypatch.setattr(music, "MUSIC_GENRE_TREE_PATH", "/tmp/genres.txt")
    music._load_genre_terms.cache_clear()

    fake_tree = "Rock\n  Progressive Rock\n# comment\n"
    with patch("builtins.open", mock_open(read_data=fake_tree)), patch.object(music.os.path, "isfile", return_value=True):
        matched = music._resolve_genre_query("play progressive rock")

    assert matched == "progressive rock"


@pytest.mark.asyncio
async def test_play_prefers_genre_first_when_query_matches_known_genre(monkeypatch):
    tracks = [
        {"id": 1, "title": "Track 1", "artist": "Band", "album": "A", "url": "file:///media/jack/buffer/audio/t1.mp3", "score": 0.0, "playcount": 10},
        {"id": 2, "title": "Track 2", "artist": "Band", "album": "A", "url": "file:///media/jack/buffer/audio/t2.mp3", "score": 0.0, "playcount": 8},
    ]
    monkeypatch.setattr(music, "_resolve_genre_query", lambda _q: "metal")

    with (
        patch.object(music, "genre_radio", return_value=tracks) as genre_radio_mock,
        patch.object(music, "_sync_play_tracks", return_value=None),
        patch.object(music, "_sync_search", side_effect=AssertionError("_sync_search should not run on genre-first path")),
    ):
        result = await music.run({"action": "play", "query": "metal", "prompt": "play metal"})

    assert result.ok
    assert result.data["action"] == "play"
    assert result.data["tracks"][0]["id"] == 1
    genre_radio_mock.assert_called_once_with("metal")


@pytest.mark.asyncio
async def test_play_michael_jackson_uses_artist_heuristic_when_not_genre(monkeypatch):
    search_results = [
        {"id": 10, "title": "Billie Jean", "artist": "Michael Jackson", "album": "Thriller", "url": "file:///media/jack/buffer/audio/bj.mp3", "score": 0.9},
        {"id": 11, "title": "Beat It", "artist": "Michael Jackson", "album": "Thriller", "url": "file:///media/jack/buffer/audio/bi.mp3", "score": 0.8},
    ]
    radio_tracks = [
        {"id": 10, "title": "Billie Jean", "artist": "Michael Jackson", "album": "Thriller", "url": "file:///media/jack/buffer/audio/bj.mp3", "score": 0.0, "playcount": 100},
    ]
    monkeypatch.setattr(music, "_resolve_genre_query", lambda _q: None)

    with (
        patch.object(music, "_sync_search", return_value=search_results),
        patch.object(music, "artist_radio", return_value=radio_tracks) as artist_radio_mock,
        patch.object(music, "_sync_play_tracks", return_value=None),
    ):
        result = await music.run({"action": "play", "query": "michael jackson", "prompt": "play michael jackson"})

    assert result.ok
    assert result.data["tracks"][0]["artist"] == "Michael Jackson"
    artist_radio_mock.assert_called_once_with("michael jackson")


def test_sync_play_tracks_skips_missing_mpd_paths():
    """Artist radio should skip stale Strawberry rows instead of aborting playback."""

    class ClientWithMissingPath(_FakeMPDClient):
        def __init__(self):
            self.added: list[str] = []
            self.play_called = False

        def add(self, path: str) -> None:
            if path == "missing/song.mp3":
                raise _FakeCommandError("[50@0] {add} No such directory")
            self.added.append(path)

        def play(self, pos: int | None = None) -> None:
            self.play_called = True

    client = ClientWithMissingPath()
    tracks = [
        {"url": "file:///media/jack/buffer/audio/ok/song1.mp3"},
        {"url": "file:///media/jack/buffer/audio/missing/song.mp3"},
        {"url": "file:///media/jack/buffer/audio/ok/song2.mp3"},
    ]

    with patch.object(music, "_mpd_connect", return_value=client):
        music._sync_play_tracks(tracks)

    assert client.added == ["ok/song1.mp3", "ok/song2.mp3"]
    assert client.play_called is True


@pytest.mark.asyncio
async def test_play_falls_back_to_artist_radio():
    """When LIKE search returns nothing, fall back to artist radio."""
    tracks = [
        {"id": 1, "title": "Song", "artist": "Band", "album": "A",
         "url": "file:///media/jack/buffer/audio/s.mp3", "score": 0.0, "playcount": 1}
    ]
    with (
        patch.object(music, "_sync_search", return_value=[]),
        patch.object(music, "artist_radio", return_value=tracks),
        patch.object(music, "_sync_play_tracks", return_value=None),
    ):
        result = await music.run({"action": "play", "query": "Band", "prompt": "play Band"})
    assert result.ok
    assert result.data["action"] == "play"


# ── Auto-pick: top-ranked result ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_auto_pick_selects_first_result():
    """Multiple search results → auto-pick first (highest ranked)."""
    results = [
        {"id": 10, "title": "Hit Song", "artist": "A", "album": "B", "url": "file:///media/jack/buffer/audio/hit.mp3", "score": 0.9},
        {"id": 11, "title": "Other Song", "artist": "A", "album": "B", "url": "file:///media/jack/buffer/audio/other.mp3", "score": 0.76},
    ]
    with (
        patch.object(music, "_sync_search", return_value=results),
        patch.object(music, "_sync_play", return_value=None),
    ):
        result = await music.run({"action": "play", "query": "hit", "prompt": "play hit"})
    assert result.ok
    assert result.data["track"]["id"] == 10
    assert result.data["picked_from"] == 2


# ── Endpoint error shape ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_control_invalid_action():
    """Invalid control action returns retryable=False error."""
    result = await music.run({"action": "control", "control": "INVALID", "prompt": ""})
    assert not result.ok
    assert not result.retryable


@pytest.mark.asyncio
async def test_now_playing_mpd_down():
    """MPD ConnectionError during now_playing → retryable error."""
    with patch.object(music, "_sync_now_playing", side_effect=ConnectionRefusedError("down")):
        result = await music.run({"action": "now_playing", "prompt": ""})
    assert not result.ok
    assert result.retryable


@pytest.mark.asyncio
async def test_queue_view_mpd_down():
    """MPD ConnectionError during queue_view → retryable error."""
    with patch.object(music, "_sync_queue_view", side_effect=ConnectionRefusedError("down")):
        result = await music.run({"action": "queue_view", "prompt": ""})
    assert not result.ok
    assert result.retryable


@pytest.mark.asyncio
async def test_control_set_volume_applies_and_returns_level():
    with patch.object(music, "_sync_set_volume", return_value=37):
        result = await music.run({"action": "control", "control": "set_volume", "volume": 37, "prompt": ""})
    assert result.ok
    assert result.data["action"] == "set_volume"
    assert result.data["volume"] == 37


@pytest.mark.asyncio
async def test_control_set_volume_requires_value():
    result = await music.run({"action": "control", "control": "set_volume", "prompt": ""})
    assert not result.ok
    assert not result.retryable


def test_sync_now_playing_includes_pos_and_volume():
    class StatusClient(_FakeMPDClient):
        def status(self):
            return {"state": "play", "song": "3", "volume": "64", "elapsed": "12.1", "duration": "240"}

        def currentsong(self):
            return {"title": "Track", "artist": "Band", "album": "Album"}

    client = StatusClient()
    with patch.object(music, "_mpd_connect", return_value=client):
        data = music._sync_now_playing()

    assert data["pos"] == 3
    assert data["volume"] == 64


# ── Phase 8b: direct song_id resolution ───────────────────────────────────────

@pytest.mark.asyncio
async def test_song_id_resolution():
    """Direct song_id lookup bypasses search and plays by rowid."""
    track = {"id": 42, "title": "Direct Track", "artist": "A", "album": "B", "url": "file:///media/jack/buffer/audio/d.mp3", "score": 1.0}
    with (
        patch.object(music, "_sync_get_by_id", return_value=track),
        patch.object(music, "_sync_play", return_value=None),
    ):
        result = await music.run({"action": "play", "song_id": 42, "prompt": ""})
    assert result.ok
    assert result.data["track"]["id"] == 42
    assert result.data["confidence"] == 1.0


@pytest.mark.asyncio
async def test_song_id_not_found():
    """Missing rowid → non-retryable error."""
    with patch.object(music, "_sync_get_by_id", return_value=None):
        result = await music.run({"action": "play", "song_id": 9999, "prompt": ""})
    assert not result.ok
    assert not result.retryable
