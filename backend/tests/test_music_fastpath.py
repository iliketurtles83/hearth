from __future__ import annotations

from music_fastpath import format_music_response, is_direct_play_request, parse_music_command
from tools.base import ToolResult


def test_direct_play_request_matches_self_contained_requests():
    for text in (
        "Play metal",
        "Play heavy metal",
        "play death metal!",
        "Play Radiohead",
        "queue some Nightwish songs",
        "play Creep by Radiohead",
        "please play the album Master of Puppets",
        "play my chill mixtape",
        "Play some jazz please",
    ):
        assert is_direct_play_request(text), f"Expected direct play for {text!r}"


def test_direct_play_request_leaves_context_and_non_music_to_llm():
    for text in (
        "play",
        "play it again",
        "play that song again",
        "play more of this",
        "play something chill",
        "play something like Radiohead",
        "play another one",
        "play a game with me",
        "play chess",
        "play devil's advocate",
        "can you play metal",
        "what should I play next",
    ):
        assert not is_direct_play_request(text), f"Expected LLM path for {text!r}"


def test_parse_volume_command():
    cmd = parse_music_command("set volume to 77%")
    assert cmd is not None
    assert cmd["action"] == "control"
    assert cmd["control"] == "set_volume"
    assert cmd["volume"] == 77


def test_parse_playback_requests_bypass_fastpath():
    """Natural language playback requests must return None to fall through to the LLM."""
    for text in (
        "queue 90s",
        "queue some Nightwish songs",
        "shuffle Radiohead",
        "start playing Bohemian Rhapsody",
        "add to the queue Creep by Radiohead",
        "play something chill",
        "play a game with me",
        "play Radiohead",
    ):
        assert parse_music_command(text) is None, f"Expected None for {text!r}"


def test_parse_control_commands():
    """Literal machine playback controls are parsed deterministically."""
    for text, expected in (
        ("pause", "pause"),
        ("pause the music", "pause"),
        ("resume", "resume"),
        ("unpause", "resume"),
        ("next", "next"),
        ("skip track", "next"),
        ("stop", "stop"),
        ("shuffle", "shuffle"),
    ):
        cmd = parse_music_command(text)
        assert cmd is not None, f"Failed for {text}"
        assert cmd["action"] == "control"
        assert cmd["control"] == expected


def test_parse_now_playing_and_queue_view():
    assert parse_music_command("what's playing") == {"action": "now_playing"}
    assert parse_music_command("now playing") == {"action": "now_playing"}
    assert parse_music_command("show the queue") == {"action": "queue_view"}
    assert parse_music_command("what is in the playlist") == {"action": "queue_view"}
