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
    "shuffle": "shuffle",
}

# Control commands the LLM tool schema may pass directly as `action`.
CONTROL_ACTIONS = frozenset({*MUSIC_CTRL.values(), "previous", "clear"})


def normalize_music_action(params: dict) -> dict:
    """Map schema-level control actions (action="pause") onto action="control".

    The music tool schema lists pause/resume/next/... as actions, but the tool
    and response formatter expect {"action": "control", "control": <cmd>}.
    """
    action = params.get("action")
    if action in CONTROL_ACTIONS:
        params["control"] = params.get("control") or action
        params["action"] = "control"
    return params

def parse_music_command(prompt: str, allow_vague: bool = False) -> dict | None:
    """Deterministically parse literal machine playback controls.

    Matches literal playback controls (pause, resume, next, skip, stop, shuffle),
    volume adjustments, and current queue/now-playing status.
    Natural language playback requests (e.g. 'play ...') are intentionally omitted
    so they fall through to LLM native tool-calling with context.
    """
    pl = prompt.strip().lower().rstrip(".,!?")

    ctrl_m = re.match(
        r"^(pause|stop|resume|continue|unpause|next|skip|shuffle)"
        r"(?:\s+(?:the\s+|my\s+)?(?:music|songs?|tracks?|playback|playlist|queue|it))?$",
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
        # "playing" arm is open-ended; "on" arm requires end-of-string so that
        # "what's on the picture?" / "what's on TV?" don't trigger now_playing.
        r"\b(what'?s|what is)\s+(currently\s+)?playing\b"
        r"|\b(what'?s|what is)\s+on\s*$"
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

    return None


def format_music_response(tool_result: "ToolResult", music_cmd: dict) -> str:
    """Format a music ToolResult as a brief plain-text sentence (no LLM needed)."""
    if not tool_result.ok:
        err = tool_result.error or "Music command failed."
        if "not playing" in err.lower():
            return "Nothing is currently playing."
        return err

    data = tool_result.data or {}
    req_action = music_cmd.get("action", "")

    if req_action in ("play", "queue"):
        data_action = data.get("action", req_action)
        track = data.get("track")
        tracks = data.get("tracks")
        verb = "Queued" if data_action == "queue" else "Now playing"
        count = len(tracks) if tracks else 0
        noun = "track" if count == 1 else "tracks"
        playlist = data.get("playlist")
        if isinstance(playlist, str) and playlist:
            return f'{verb} playlist "{playlist}" ({count} {noun}).'
        album = data.get("album")
        if isinstance(album, str) and album and count:
            by = f' by {data["album_artist"]}' if data.get("album_artist") else ""
            return f'{verb} the album "{album}"{by} ({count} {noun}).'
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
            "shuffle": "Queue shuffled.",
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
