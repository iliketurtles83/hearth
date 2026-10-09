"""MPD client: connection, playback control, queue, stored playlists, outputs.

Beets stores filesystem paths; _url_to_mpd_path strips MUSIC_ROOT so MPD gets
paths relative to its music_directory.
"""
from __future__ import annotations

import difflib
import logging
import os
from typing import Any

import musicpd

from music import library

log = logging.getLogger("assistant.tools.music")

MPD_HOST: str = os.getenv("MPD_HOST", "mpd")
MPD_PORT: int = int(os.getenv("MPD_PORT", "6600"))
MPD_TIMEOUT: int = int(os.getenv("MPD_TIMEOUT", "5"))
MUSIC_ROOT: str = os.getenv("MUSIC_ROOT", "/music")
MUSIC_PATH_HOST: str = os.getenv("MUSIC_PATH_HOST", os.getenv("MUSIC_PATH", ""))


def _url_to_mpd_path(path_value: Any) -> str:
    """Convert a Beets path to an MPD-relative path."""
    path = library._decode_beets_path(path_value).replace("\\", "/")

    roots: list[str] = []
    for candidate in (MUSIC_ROOT, MUSIC_PATH_HOST):
        root = (candidate or "").replace("\\", "/").rstrip("/")
        if root and root not in roots:
            roots.append(root)

    # Prefer stripping the longest matching configured root first.
    for root in sorted(roots, key=len, reverse=True):
        if path == root or path.startswith(f"{root}/"):
            path = path[len(root):]
            break

    return path.lstrip("/")


# ── MPD helpers (synchronous, called via asyncio.to_thread) ───────────────────

def _mpd_connect() -> musicpd.MPDClient:
    """Connect to MPD with one retry on failure."""
    client = musicpd.MPDClient()
    client.timeout = MPD_TIMEOUT
    try:
        client.connect(MPD_HOST, MPD_PORT)
        return client
    except Exception:
        log.warning("mpd.connect_retry | host=%s port=%s", MPD_HOST, MPD_PORT)
        client2 = musicpd.MPDClient()
        client2.timeout = MPD_TIMEOUT
        client2.connect(MPD_HOST, MPD_PORT)  # raises on second failure
        return client2


def _with_mpd(fn):
    """Open a fresh MPD connection, run fn(client), then disconnect cleanly."""
    client = _mpd_connect()
    try:
        return fn(client)
    finally:
        try:
            client.disconnect()
        except Exception:
            pass


def _sync_play(url: str) -> dict[str, Any]:
    """Clear queue, add track, and start playing. Returns MPD playback status."""
    path = _url_to_mpd_path(url)
    log.debug("mpd.play | path=%s", path)

    def _fn(c: musicpd.MPDClient) -> dict[str, Any]:
        c.clear()
        added = _add_mpd_paths(c, [path])
        if not added:
            raise FileNotFoundError(f"Track not available in MPD library: {path}")
        c.play()
        # Confirm MPD transitioned to play and capture timing for UI sync.
        status = c.status()
        return {
            "state": status.get("state", "stop"),
            "elapsed": float(status.get("elapsed", 0)),
            "duration": float(status["duration"]) if "duration" in status else None,
        }

    return _with_mpd(_fn)


def _sync_queue(url: str) -> None:
    """Append a track to the current MPD queue."""
    path = _url_to_mpd_path(url)
    log.debug("mpd.queue | path=%s", path)

    def _fn(c: musicpd.MPDClient) -> None:
        added = _add_mpd_paths(c, [path])
        if not added:
            raise FileNotFoundError(f"Track not available in MPD library: {path}")

    _with_mpd(_fn)


def _sync_play_pos(pos: int) -> None:
    """Jump to a specific position in the current MPD queue (0-indexed)."""
    log.debug("mpd.play_pos | pos=%d", pos)
    _with_mpd(lambda c: c.play(pos))


def _sync_control(action: str) -> None:
    """Execute a control command: pause / resume / next / previous / stop / shuffle / clear."""
    def _fn(c: musicpd.MPDClient) -> None:
        if action == "pause":
            c.pause(1)
        elif action == "resume":
            c.pause(0)
        elif action == "next":
            c.next()
        elif action == "previous":
            c.previous()
        elif action == "stop":
            c.stop()
        elif action == "shuffle":
            status = c.status()
            length = int(status.get("playlistlength", 0))
            if length == 0:
                raise ValueError("The queue is empty.")
            c.shuffle()
        elif action == "clear":
            c.clear()
        else:
            raise ValueError(f"Unknown control action: {action!r}")
    _with_mpd(_fn)


def _sync_set_volume(volume: int) -> int:
    """Set MPD volume (0-100) and return the applied value."""
    level = max(0, min(100, int(volume)))

    def _fn(c: musicpd.MPDClient) -> int:
        c.setvol(level)
        status = c.status()
        try:
            return int(status.get("volume", level))
        except Exception:
            return level

    return _with_mpd(_fn)


def _sync_now_playing() -> dict[str, Any]:
    """Return current playback state."""
    def _fn(c: musicpd.MPDClient) -> dict[str, Any]:
        status = c.status()
        state = status.get("state", "stop")
        current_pos: int | None = None
        try:
            if "song" in status:
                current_pos = int(status.get("song"))
        except Exception:
            current_pos = None
        try:
            volume = int(status.get("volume", 0))
        except Exception:
            volume = 0
        track = None
        if state in ("play", "pause"):
            try:
                current = c.currentsong()
                track = {
                    "title": current.get("title", ""),
                    "artist": current.get("artist", ""),
                    "album": current.get("album", ""),
                }
            except Exception:
                pass
        return {
            "playing": state == "play",
            "state": state,
            "track": track,
            "elapsed": float(status["elapsed"]) if "elapsed" in status else None,
            "duration": float(status["duration"]) if "duration" in status else None,
            "pos": current_pos,
            "volume": volume,
        }
    return _with_mpd(_fn)


def _sync_queue_view() -> dict[str, Any]:
    """Return the current MPD playlist."""
    def _fn(c: musicpd.MPDClient) -> dict[str, Any]:
        playlist = c.playlistinfo()
        if not isinstance(playlist, list):
            playlist = []
        items = [
            {
                "pos": int(entry.get("pos", 0)),
                "title": entry.get("title", ""),
                "artist": entry.get("artist", ""),
                "album": entry.get("album", ""),
            }
            for entry in playlist
        ]
        return {"queue": items, "length": len(items)}
    return _with_mpd(_fn)


def _sync_play_tracks(tracks: list[dict[str, Any]]) -> dict[str, Any]:
    """Clear queue, add all tracks, and start playing. Returns MPD playback status."""
    paths = [_url_to_mpd_path(t["url"]) for t in tracks]

    def _fn(c: musicpd.MPDClient) -> dict[str, Any]:
        c.clear()
        added_paths = _add_mpd_paths(c, paths)
        if not added_paths:
            raise FileNotFoundError("No selected tracks are available in the MPD library")
        c.play()
        status = c.status()
        added_set = set(added_paths)
        queued_tracks = [t for t, p in zip(tracks, paths) if p in added_set]
        return {
            "state": status.get("state", "stop"),
            "elapsed": float(status.get("elapsed", 0)),
            "duration": float(status["duration"]) if "duration" in status else None,
            "queued_tracks": queued_tracks,
        }

    return _with_mpd(_fn)


def _sync_queue_tracks(tracks: list[dict[str, Any]]) -> dict[str, Any]:
    """Append all tracks to the current MPD queue without changing playback."""
    paths = [_url_to_mpd_path(t["url"]) for t in tracks]

    def _fn(c: musicpd.MPDClient) -> dict[str, Any]:
        added_paths = _add_mpd_paths(c, paths)
        if not added_paths:
            raise FileNotFoundError("No selected tracks are available in the MPD library")
        added_set = set(added_paths)
        queued_tracks = [t for t, p in zip(tracks, paths) if p in added_set]
        return {"queued_tracks": queued_tracks}

    return _with_mpd(_fn)


def _sync_list_playlists() -> list[str]:
    """Return the names of MPD stored playlists (playlist_directory .m3u files)."""
    def _fn(c: musicpd.MPDClient) -> list[str]:
        return [str(p["playlist"]) for p in c.listplaylists() if p.get("playlist")]
    return _with_mpd(_fn)


def _match_playlist(name: str, available: list[str], fuzzy: bool) -> str | None:
    """Pick the stored playlist `name` refers to, case-insensitively.

    Exact match always counts. With fuzzy=True (the user said "playlist" /
    "mixtape"), also accept containment and close spellings.
    """
    want = name.strip().lower()
    if not want:
        return None
    by_lower = {p.lower(): p for p in available}
    if want in by_lower:
        return by_lower[want]
    if not fuzzy:
        return None
    contains = [p for p in available if want in p.lower() or p.lower() in want]
    if len(contains) == 1:
        return contains[0]
    close = difflib.get_close_matches(want, list(by_lower), n=1, cutoff=0.75)
    return by_lower[close[0]] if close else None


def _sync_load_playlist(name: str, replace: bool = True) -> dict[str, Any]:
    """Load a stored playlist in order. replace=True clears the queue and plays."""
    def _fn(c: musicpd.MPDClient) -> dict[str, Any]:
        entries = c.listplaylistinfo(name)
        if not entries:
            raise FileNotFoundError(f"Playlist '{name}' is empty")
        tracks = [
            {
                "title": e.get("title") or os.path.basename(str(e.get("file", ""))),
                "artist": e.get("artist", ""),
                "album": e.get("album", ""),
            }
            for e in entries
        ]
        if replace:
            c.clear()
        c.load(name)
        if not replace:
            return {"queued_tracks": tracks}
        c.play()
        status = c.status()
        return {
            "state": status.get("state", "stop"),
            "elapsed": float(status.get("elapsed", 0)),
            "duration": float(status["duration"]) if "duration" in status else None,
            "queued_tracks": tracks,
        }

    return _with_mpd(_fn)


def _sync_get_outputs() -> list[dict[str, Any]]:
    """Return all MPD audio outputs and their current status."""
    def _fn(c: musicpd.MPDClient) -> list[dict[str, Any]]:
        raw = c.outputs()
        return [
            {
                "id": str(o.get("outputid", "")),
                "name": o.get("outputname", f"Output {o.get('outputid')}"),
                "enabled": str(o.get("outputenabled", "0")) == "1",
                "plugin": o.get("plugin", ""),
            }
            for o in raw
        ]
    return _with_mpd(_fn)


def _sync_set_output(output_id: str, mode: str = "exclusive") -> list[dict[str, Any]]:
    """Set MPD output state (exclusive, mirror/both, enable, disable, toggle)."""
    def _fn(c: musicpd.MPDClient) -> list[dict[str, Any]]:
        outputs = c.outputs()
        target_id = str(output_id)
        if target_id in ("all", "both") or mode in ("all", "both"):
            for out in outputs:
                oid = out.get("outputid")
                if oid is not None:
                    c.enableoutput(int(oid))
            try:
                status = c.status()
                if "error" in status:
                    c.clearerror()
                if status.get("state") == "pause" and status.get("songid"):
                    c.play()
            except Exception:
                pass
        elif mode == "exclusive":
            # Enable target output FIRST to avoid having 0 active outputs while playing,
            # which causes MPD to pause with "Failed to open audio output".
            c.enableoutput(int(target_id))
            for out in outputs:
                oid = str(out.get("outputid", ""))
                if oid != target_id:
                    c.disableoutput(int(oid))
            try:
                status = c.status()
                if "error" in status:
                    c.clearerror()
                if status.get("state") == "pause" and status.get("songid"):
                    c.play()
            except Exception:
                pass
        elif mode in ("enable", "mirror"):
            c.enableoutput(int(target_id))
        elif mode == "disable":
            c.disableoutput(int(target_id))
        elif mode == "toggle":
            for out in outputs:
                oid = str(out.get("outputid", ""))
                if oid == target_id:
                    if str(out.get("outputenabled", "0")) == "1":
                        c.disableoutput(int(oid))
                    else:
                        c.enableoutput(int(oid))
        return [
            {
                "id": str(o.get("outputid", "")),
                "name": o.get("outputname", f"Output {o.get('outputid')}"),
                "enabled": str(o.get("outputenabled", "0")) == "1",
                "plugin": o.get("plugin", ""),
            }
            for o in c.outputs()
        ]
    return _with_mpd(_fn)


# ── MPD error classification ───────────────────────────────────────────────────

def _is_mpd_connection_error(exc: Exception) -> bool:
    connection_error = getattr(musicpd, "ConnectionError", None)
    if connection_error is not None and isinstance(exc, connection_error):
        return True
    # Different test modules may install independent fake ConnectionError classes.
    if exc.__class__.__name__ == "ConnectionError":
        return True
    return isinstance(exc, (ConnectionRefusedError, BrokenPipeError, OSError))


def _is_mpd_missing_path_error(exc: Exception) -> bool:
    command_error = getattr(musicpd, "CommandError", None)
    if command_error is not None and isinstance(exc, command_error):
        return "No such directory" in str(exc)
    # Test stubs may not expose CommandError; keep classification behavior via message.
    return "No such directory" in str(exc)


def _add_mpd_paths(client: musicpd.MPDClient, paths: list[str]) -> list[str]:
    added: list[str] = []
    for path in paths:
        try:
            client.add(path)
            added.append(path)
        except Exception as exc:
            if _is_mpd_missing_path_error(exc):
                log.warning("mpd.add_skip_missing | path=%s", path)
                continue
            raise
    return added
