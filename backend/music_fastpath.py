import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tools.base import ToolResult

# Maps control verbs to canonical MPD actions.
MUSIC_CTRL: dict[str, str] = {
    "pause": "pause",
    "stop": "stop",
    "resume": "resume",
    "continue": "resume",
    "unpause": "resume",
    "next": "next",
    "skip": "next",
}

# Target phrases that are too vague to resolve without LLM help.
MUSIC_VAGUE: frozenset[str] = frozenset({
    "something", "anything", "a song", "some songs", "any song",
    "a track", "some tracks", "any track", "some music", "any music",
    "a random song", "a random track", "a tune", "some tunes",
})


def parse_music_command(prompt: str) -> dict | None:
    """Deterministically parse a high-confidence music command.

    Returns a params dict ready for tools.dispatch("music", ...) if the
    prompt maps to a known music action with enough specificity.
    """
    pl = prompt.strip().lower().rstrip(".,!?")

    ctrl_m = re.match(
        r"^(pause|stop|resume|continue|unpause|next|skip)"
        r"(?:\s+(?:the\s+)?(?:music|song|track|playback|it))?$",
        pl,
    )
    if ctrl_m:
        action = MUSIC_CTRL.get(ctrl_m.group(1))
        if action:
            return {"action": "control", "control": action}

    vol_m = re.match(
        r"^(?:set\s+)?(?:the\s+)?volume(?:\s+to)?\s+(\d{1,3})%?$",
        pl,
    )
    if vol_m:
        return {
            "action": "control",
            "control": "set_volume",
            "volume": max(0, min(100, int(vol_m.group(1)))),
        }

    if re.search(
        r"\b(what'?s|what is)\s+(currently\s+)?(playing|on)\b"
        r"|\bnow\s+playing\b|\bcurrent\s+(song|track)\b",
        pl,
    ):
        return {"action": "now_playing"}

    if re.search(
        r"\b(what'?s|what is)\s+(in\s+)?(the\s+)?(queue|playlist)\b"
        r"|\bshow\s+(me\s+)?(the\s+)?(queue|playlist)\b",
        pl,
    ):
        return {"action": "queue_view"}

    play_m = re.match(
        r"^(play(?:back)?|queue|add\s+to\s+(?:the\s+)?queue|put\s+on)\s+(.+)$",
        pl,
    )
    if not play_m:
        return None

    verb = play_m.group(1)
    action = "queue" if re.match(r"queue|add\s+to", verb) else "play"
    target = play_m.group(2).strip().strip(".,!?\"'")

    if target in MUSIC_VAGUE:
        return None
    if re.match(
        r"^something\s+(like|similar\s+to|that\s+sounds?\s+like)",
        target,
        re.IGNORECASE,
    ):
        return None
    if re.match(
        r"^(something|anything)\s*(chill|relaxing|upbeat|heavy|fast|slow|random|good)?$",
        target,
        re.IGNORECASE,
    ):
        return None

    decade_m = re.match(r"^(?:some\s+)?(\d{2})s(?:\s+.*)?$", target, re.IGNORECASE)
    if decade_m:
        d = int(decade_m.group(1))
        yr = (2000 + d) if d < 30 else (1900 + d)
        return {"action": action, "year_range": (yr, yr + 9)}

    year_m = re.match(
        r"^(?:(?:music|songs?|tracks?)\s+from\s+)?(\d{4})$", target, re.IGNORECASE
    )
    if year_m:
        yr = int(year_m.group(1))
        return {"action": action, "year_range": (yr, yr)}

    by_m = re.match(r"^(?P<title>.+?)\s+by\s+(?P<artist>.+)$", target, re.IGNORECASE)
    if by_m:
        title = by_m.group("title").strip().strip("\"'")
        artist = by_m.group("artist").strip().strip("\"'")
        if title.lower() in MUSIC_VAGUE:
            return {"action": action, "artist": artist}
        return {"action": action, "query": title, "artist_filter": artist}

    artist_song_m = re.match(
        r"^(?:a|some|any)\s+(?:random\s+)?(?P<artist>.+?)\s+(?:song|track|music)s?$",
        target,
        re.IGNORECASE,
    )
    if artist_song_m:
        return {"action": action, "artist": artist_song_m.group("artist").strip()}

    return {"action": action, "query": target}


def format_music_response(tool_result: "ToolResult", music_cmd: dict) -> str:
    """Format a music ToolResult as a brief plain-text sentence (no LLM needed)."""
    if not tool_result.ok:
        return tool_result.error or "Music command failed."

    data = tool_result.data or {}
    req_action = music_cmd.get("action", "")

    if req_action in ("play", "queue"):
        data_action = data.get("action", req_action)
        track = data.get("track")
        tracks = data.get("tracks")
        verb = "Queued" if data_action == "queue" else "Now playing"
        if tracks and len(tracks) > 1:
            genre = data.get("genre")
            if isinstance(genre, str) and genre.strip():
                genre_label = genre.strip().title()
                return f"{verb}: {len(tracks)} {genre_label} tracks."
            artist = tracks[0].get("artist", "unknown artist")
            return f"{verb}: {len(tracks)} tracks by {artist}."
        if track:
            title = track.get("title", "unknown track")
            artist = track.get("artist", "unknown artist")
            return f'{verb}: "{title}" by {artist}.'
        return "Playback started."

    if req_action == "control":
        ctrl = music_cmd.get("control", "")
        if ctrl == "set_volume":
            vol = data.get("volume")
            return f"Volume set to {vol}%." if vol is not None else "Volume updated."
        return {
            "pause": "Paused.",
            "resume": "Resumed.",
            "stop": "Stopped.",
            "next": "Skipping to next track.",
        }.get(ctrl, "Done.")

    if req_action == "now_playing":
        state = data.get("state", "stop")
        track = data.get("track")
        if state == "stop" or not track:
            return "Nothing is playing."
        title = track.get("title", "unknown")
        artist = track.get("artist", "unknown")
        verb = "Paused" if state == "pause" else "Now playing"
        return f'{verb}: "{title}" by {artist}.'

    if req_action == "queue_view":
        queue = data.get("queue", [])
        n = len(queue)
        if n == 0:
            return "The queue is empty."
        items = ", ".join(
            f'"{t.get("title", "?")}" by {t.get("artist", "?")}' for t in queue[:5]
        )
        suffix = f" +{n - 5} more" if n > 5 else ""
        return f"Queue ({n} tracks): {items}{suffix}."

    return "Done."
