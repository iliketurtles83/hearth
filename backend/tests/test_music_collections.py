"""
Tests for album, compilation and stored-playlist ("mixtape") playback.

Uses a real temporary Beets-shaped SQLite DB and a fake MPD client so the
resolver order (playlist → genre → album → track/artist) is exercised end to end.
"""
from __future__ import annotations

import asyncio
import random
import sqlite3
from typing import Any
from unittest.mock import patch

import pytest

import tools.music as music
from music_fastpath import format_music_response, normalize_music_action


_ITEMS = [
    # (title, artist, albumartist, album, album_id, disc, track, genre, comp)
    ("Freddie Freeloader", "Miles Davis", "Miles Davis", "Kind of Blue", 1, 1, 2, "Jazz", 0),
    ("So What", "Miles Davis", "Miles Davis", "Kind of Blue", 1, 1, 1, "Jazz", 0),
    ("Blue in Green", "Miles Davis", "Miles Davis", "Kind of Blue", 1, 1, 3, "Jazz", 0),
    ("Solar", "Miles Davis", "Miles Davis", "Walkin'", 2, 1, 1, "Jazz", 0),
    ("Say It Ain't So", "Weezer", "Weezer", "Weezer", 3, 1, 7, "Rock", 0),
    ("Buddy Holly", "Weezer", "Weezer", "Weezer", 3, 1, 4, "Rock", 0),
    ("El Scorcho", "Weezer", "Weezer", "Pinkerton", 4, 1, 2, "Rock", 0),
    # Compilation with unreliable albumartist, like real asis imports.
    ("Hey Ya!", "OutKast", "!!!", "The Pitchfork 500", 5, 1, 2, "Pop", 1),
    ("Heroes", "David Bowie", "!!!", "The Pitchfork 500", 5, 1, 1, "Rock", 1),
    # Two albums that differ only by a leading "The".
    ("Hit One", "Various", "Various", "The Hits", 6, 1, 1, "Pop", 1),
    ("Hit Two", "Various", "Various", "The Hits", 6, 1, 2, "Pop", 1),
    ("Lone Hit", "Somebody", "Somebody", "Hits", 7, 1, 1, "Pop", 0),
]


class _FakeMPD:
    def __init__(self, playlists: dict[str, list[dict]] | None = None):
        self.calls: list[tuple] = []
        self.playlists = playlists or {}

    def __getattr__(self, name):
        def _rec(*args):
            self.calls.append((name, *args))
        return _rec

    def disconnect(self):
        pass

    def status(self):
        return {"state": "play", "elapsed": "0.0", "duration": "100.0", "playlistlength": "3"}

    def listplaylists(self):
        return [{"playlist": n} for n in self.playlists]

    def listplaylistinfo(self, name):
        return self.playlists[name]

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]


@pytest.fixture
def beets_db(tmp_path, monkeypatch):
    db = tmp_path / "library.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, title TEXT, artist TEXT, albumartist TEXT,"
        " album TEXT, album_id INTEGER, disc INTEGER, track INTEGER, genre TEXT, comp INTEGER,"
        " path BLOB, rating REAL)"
    )
    for i, (title, artist, aa, album, aid, disc, track, genre, comp) in enumerate(_ITEMS, start=1):
        conn.execute(
            "INSERT INTO items VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (i, title, artist, aa, album, aid, disc, track, genre, comp,
             f"{music.MUSIC_ROOT}/{artist}/{title}.mp3".encode(), 0.0),
        )
    conn.commit()
    conn.close()
    monkeypatch.setattr(music, "BEETS_DB_PATH", str(db))
    music._beets_columns.cache_clear()
    yield db
    music._beets_columns.cache_clear()


def _run(params: dict[str, Any], mpd: _FakeMPD):
    with patch.object(music, "_mpd_connect", return_value=mpd):
        return asyncio.run(music.run(params))


def _added_titles(mpd: _FakeMPD) -> list[str]:
    return [c[1].rsplit("/", 1)[-1].removesuffix(".mp3") for c in mpd.calls if c[0] == "add"]


# ── Parsing ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("play the album Kind of Blue", {"kind": "album", "name": "Kind of Blue", "artist": None, "full": False}),
        ("play the whole album OK Computer", {"kind": "album", "name": "OK Computer", "artist": None, "full": True}),
        ("play the Kind of Blue album", {"kind": "album", "name": "Kind of Blue", "artist": None, "full": False}),
        ("play the whole Pitchfork 500", {"kind": "album", "name": "Pitchfork 500", "artist": None, "full": True}),
        ("play my chill mixtape", {"kind": "playlist", "name": "chill", "weak": False}),
        ("play the playlist called road trip", {"kind": "playlist", "name": "road trip", "weak": False}),
        ("play my new mix tape", {"kind": "playlist", "name": "new", "weak": False}),
        ("play a jazz mix", {"kind": "playlist", "name": "jazz", "weak": True}),
        ("play Radiohead", None),
        ("play some jazz", None),
        ("play that record", None),
    ],
)
def test_parse_collection_request(prompt, expected):
    assert music._parse_collection_request(music._extract_search_query(prompt)) == expected


def test_parse_album_splits_trailing_artist():
    assert music._parse_collection_request("the record Blue by Joni Mitchell") == {
        "kind": "album", "name": "Blue", "artist": "Joni Mitchell", "full": False,
    }


# ── Albums ─────────────────────────────────────────────────────────────────────

def test_explicit_album_plays_whole_album_in_track_order(beets_db):
    mpd = _FakeMPD()
    result = _run({"prompt": "play the album Kind of Blue"}, mpd)

    assert result.ok
    assert _added_titles(mpd) == ["So What", "Freddie Freeloader", "Blue in Green"]
    assert mpd.names()[0] == "clear"
    assert result.data["album"] == "Kind of Blue"
    assert format_music_response(result, {"action": "play"}) == (
        'Now playing the album "Kind of Blue" by Miles Davis (3 tracks).'
    )


def test_album_by_artist_prefers_album_over_artist_radio(beets_db):
    mpd = _FakeMPD()
    result = _run({"prompt": "play Kind of Blue by Miles Davis"}, mpd)

    assert result.ok
    assert result.data["album"] == "Kind of Blue"
    assert len(result.data["tracks"]) == 3


def test_bare_album_name_plays_album(beets_db):
    mpd = _FakeMPD()
    result = _run({"prompt": "play Pinkerton"}, mpd)

    assert result.ok
    assert result.data["album"] == "Pinkerton"


def test_self_titled_album_name_still_means_artist(beets_db):
    mpd = _FakeMPD()
    result = _run({"prompt": "play Weezer"}, mpd)

    assert result.ok
    assert "album" not in result.data
    assert {t["artist"] for t in result.data["tracks"]} == {"Weezer"}
    assert len(result.data["tracks"]) == 3


def test_compilation_plays_in_order_without_single_artist(beets_db):
    mpd = _FakeMPD()
    result = _run({"prompt": "play the Pitchfork 500"}, mpd)

    assert result.ok
    assert _added_titles(mpd) == ["Heroes", "Hey Ya!"]
    assert result.data["album_artist"] is None
    assert format_music_response(result, {"action": "play"}) == (
        'Now playing the album "The Pitchfork 500" (2 tracks).'
    )


@pytest.mark.parametrize(("name", "album", "count"), [("The Hits", "The Hits", 2), ("Hits", "Hits", 1)])
def test_album_lookup_prefers_literal_name_over_the_variant(beets_db, name, album, count):
    resolved, tracks = music._sync_album_tracks(name)

    assert (resolved, len(tracks)) == (album, count)


def test_sample_large_compilation_caps_and_keeps_album_order(monkeypatch):
    monkeypatch.setattr(music, "MUSIC_PLAYLIST_MAX_N", 24)
    tracks = [{"title": f"t{i:03}", "artist": f"a{i}"} for i in range(500)]

    sample = music._sample_large_compilation(tracks, random.Random(7))

    assert len(sample) == 24
    assert sample == sorted(sample, key=lambda t: t["title"])


def test_sample_large_compilation_never_trims_single_artist_album(monkeypatch):
    monkeypatch.setattr(music, "MUSIC_PLAYLIST_MAX_N", 24)
    tracks = [{"title": f"t{i}", "artist": "The Beatles"} for i in range(30)]

    assert music._sample_large_compilation(tracks, random.Random(7)) == tracks


def test_huge_compilation_is_sampled_unless_whole_requested(beets_db, monkeypatch):
    monkeypatch.setattr(music, "MUSIC_PLAYLIST_MAX_N", 1)

    sampled = _run({"prompt": "play the Pitchfork 500"}, _FakeMPD())
    assert sampled.ok
    assert len(sampled.data["tracks"]) == 1
    assert format_music_response(sampled, {"action": "play"}) == (
        'Now playing 1 of 2 tracks from "The Pitchfork 500".'
    )

    whole = _run({"prompt": "play the whole Pitchfork 500"}, _FakeMPD())
    assert len(whole.data["tracks"]) == 2

    llm_whole = _run(
        {"action": "play", "album": "The Pitchfork 500", "prompt": "play the entire pitchfork 500"},
        _FakeMPD(),
    )
    assert len(llm_whole.data["tracks"]) == 2

    # Single-artist albums over the limit still play in full.
    album = _run({"prompt": "play the album Kind of Blue"}, _FakeMPD())
    assert len(album.data["tracks"]) == 3


def test_queue_album_appends_without_clearing(beets_db):
    mpd = _FakeMPD()
    result = _run({"prompt": "queue the album Kind of Blue"}, mpd)

    assert result.ok
    assert result.data["action"] == "queue"
    assert "clear" not in mpd.names() and "play" not in mpd.names()
    assert len(_added_titles(mpd)) == 3


def test_explicit_album_miss_reports_not_found(beets_db):
    result = _run({"prompt": "play the album Nevermind"}, _FakeMPD())

    assert not result.ok
    assert "No album matching 'Nevermind'" in result.error


def test_llm_album_param_with_artist_is_not_artist_radio(beets_db):
    mpd = _FakeMPD()
    result = _run(
        {"action": "play", "album": "Kind of Blue", "artist": "Miles Davis", "prompt": "put on kind of blue"},
        mpd,
    )

    assert result.ok
    assert result.data["album"] == "Kind of Blue"


# ── Stored playlists ───────────────────────────────────────────────────────────

_PLAYLISTS = {
    "chill": [
        {"file": "a/one.mp3", "title": "One", "artist": "A"},
        {"file": "b/two.mp3", "title": "Two", "artist": "B"},
    ],
    "new": [{"file": "c/three.mp3"}],
}


def test_mixtape_loads_stored_playlist(beets_db):
    mpd = _FakeMPD(_PLAYLISTS)
    result = _run({"prompt": "play my chill mixtape"}, mpd)

    assert result.ok
    assert mpd.names() == ["clear", "load", "play"]
    assert ("load", "chill") in mpd.calls
    assert format_music_response(result, {"action": "play"}) == 'Now playing playlist "chill" (2 tracks).'


def test_playlist_name_is_fuzzy_matched(beets_db):
    mpd = _FakeMPD(_PLAYLISTS)
    result = _run({"prompt": "play the chil playlist"}, mpd)

    assert result.ok
    assert result.data["playlist"] == "chill"


def test_bare_playlist_name_plays_playlist(beets_db):
    mpd = _FakeMPD(_PLAYLISTS)
    result = _run({"prompt": "play chill"}, mpd)

    assert result.ok
    assert result.data["playlist"] == "chill"


def test_queue_playlist_does_not_clear(beets_db):
    mpd = _FakeMPD(_PLAYLISTS)
    result = _run({"prompt": "queue my new playlist"}, mpd)

    assert result.ok
    assert mpd.names() == ["load"]
    assert result.data["track"]["title"] == "three.mp3"
    assert format_music_response(result, {"action": "queue"}) == 'Queued playlist "new" (1 track).'


def test_unknown_playlist_lists_available(beets_db):
    result = _run({"prompt": "play my workout playlist"}, _FakeMPD(_PLAYLISTS))

    assert not result.ok
    assert result.error == "No playlist called 'workout'. Available playlists: chill, new."


def test_weak_mix_without_playlist_falls_back_to_genre(beets_db):
    mpd = _FakeMPD(_PLAYLISTS)
    result = _run({"prompt": "play a jazz mix"}, mpd)

    assert result.ok
    assert result.data["genre"] == "jazz"


# ── LLM tool-schema control actions ────────────────────────────────────────────

def test_schema_control_action_is_normalized_to_control(beets_db):
    mpd = _FakeMPD()
    params = {"action": "pause", "prompt": "could you pause it please"}
    result = _run(params, mpd)

    assert result.ok
    assert mpd.calls == [("pause", 1)]
    assert format_music_response(result, normalize_music_action({"action": "pause"})) == "Paused."
