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
