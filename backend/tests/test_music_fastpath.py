from __future__ import annotations

from music_fastpath import format_music_response, parse_music_command
from tools.base import ToolResult


def test_parse_volume_command():
    cmd = parse_music_command("set volume to 77%")
    assert cmd is not None
    assert cmd["action"] == "control"
    assert cmd["control"] == "set_volume"
    assert cmd["volume"] == 77


def test_parse_queue_decade_command():
    cmd = parse_music_command("queue 90s")
    assert cmd is not None
    assert cmd["action"] == "queue"
    assert cmd["year_range"] == (1990, 1999)


def test_parse_queue_artist_command():
    cmd = parse_music_command("queue some Nightwish songs")
    assert cmd is not None
    assert cmd["action"] == "queue"
    assert cmd["artist"] == "nightwish"


def test_format_set_volume_response():
    msg = format_music_response(
        ToolResult(ok=True, data={"action": "set_volume", "ok": True, "volume": 35}),
        {"action": "control", "control": "set_volume", "volume": 35},
    )
    assert msg == "Volume set to 35%."


def test_format_play_multi_track_genre_response():
    msg = format_music_response(
        ToolResult(
            ok=True,
            data={
                "action": "play",
                "track": {"title": "Any", "artist": "Oratory"},
                "tracks": [
                    {"title": "Any", "artist": "Oratory"},
                    {"title": "Other", "artist": "Nightwish"},
                ],
                "genre": "heavy metal",
            },
        ),
        {"action": "play", "query": "heavy metal"},
    )
    assert msg == "Now playing: 2 Heavy Metal tracks."


def test_parse_shuffle_control_commands():
    for text in ("shuffle", "shuffle music", "shuffle the music", "shuffle my playlist", "shuffle the queue", "shuffle tracks"):
        cmd = parse_music_command(text)
        assert cmd is not None, f"Failed for {text}"
        assert cmd["action"] == "control"
        assert cmd["control"] == "shuffle"


def test_parse_shuffle_entity_command():
    cmd = parse_music_command("shuffle Radiohead")
    assert cmd is not None
    assert cmd["action"] == "play"
    assert cmd["query"] == "radiohead"


def test_parse_start_playing_prefix():
    cmd = parse_music_command("start playing Bohemian Rhapsody")
    assert cmd is not None
    assert cmd["action"] == "play"
    assert cmd["query"] == "bohemian rhapsody"


def test_parse_add_to_the_queue():
    cmd = parse_music_command("add to the queue Creep by Radiohead")
    assert cmd is not None
    assert cmd["action"] == "queue"
    assert cmd["query"] == "creep"
    assert cmd["artist_filter"] == "radiohead"


def test_format_shuffle_response():
    msg = format_music_response(
        ToolResult(ok=True, data={"action": "shuffle", "ok": True}),
        {"action": "control", "control": "shuffle"},
    )
    assert msg == "Queue shuffled."


def test_allow_vague_flag():
    # allow_vague=False returns None for fastpath
    assert parse_music_command("play something chill", allow_vague=False) is None
    # allow_vague=True returns parsed params for tools/music
    parsed = parse_music_command("play something chill", allow_vague=True)
    assert parsed is not None
    assert parsed["action"] == "play"
    assert parsed["query"] == "something chill"
