"""
Music playback tool — Strawberry library search + MPD playback (Phase 8).

# ── Phase 8b Recommendation Engine Contract (locked pre-implementation) ────────
# The recommendation engine HTTP response MUST follow this primary shape:
#   { "song_id": int, "provider": str, "confidence": float }
# Optional fields:
#   { "score": float, "explanation": str }
#
# "song_id" here refers to Strawberry songs.rowid (integer primary key).
# Resolution: SELECT rowid, title, artist, album, url FROM songs WHERE rowid = ?
# → trivial join, zero ambiguity, no fuzzy matching.
#
# Metadata-only responses ({ "title": str, "artist": str }) are NOT the primary
# contract. Metadata lookups are available via a separate non-primary path only.
#
# 8b fallback (rec engine unavailable): call artist_radio() from this module.
# artist_radio() is implemented in Phase 8 core — 8b has nothing to fall back to
# without it.
# ─────────────────────────────────────────────────────────────────────────────────

Strawberry DB (Phase 8 schema notes):
  - Primary key: rowid (implicit SQLite integer rowid — no id column)
  - File path:   url column (file:// URI, URL-encoded, e.g.
                 file:///media/jack/buffer/audio/artist/song.mp3)
  - No songs_fts FTS table — use LIKE queries only
  - Opened read-only with timeout=5, check_same_thread=False, URI mode
  - All reads wrapped in try/except sqlite3.OperationalError for scan locks

MPD client (python-musicpd):
  - Per-request fresh connection (thread-safe, works with asyncio.to_thread)
  - Reconnect-once policy: if connect fails, retry once before raising
  - Connection errors bubble up as retryable ToolResult

Path rewrite:
  Strawberry stores host-side absolute file:// URIs.
  The backend URL-decodes them, strips MUSIC_PATH_HOST, and passes the
  resulting MPD-relative path (relative to MPD music_directory) to MPD.

  Example:
    Strawberry url: file:///media/jack/buffer/audio/rock/artist/song.mp3
    MUSIC_PATH_HOST: /media/jack/buffer/audio
    MPD-relative:   rock/artist/song.mp3
    MPD resolves:   /music/rock/artist/song.mp3  (music_directory=/music)

Environment variables:
  STRAWBERRY_DB_PATH       path to Strawberry sqlite db inside container
                           (default: /strawberry/strawberry.db)
  MPD_HOST                 MPD hostname (default: mpd)
  MPD_PORT                 MPD port (default: 6600)
  MPD_TIMEOUT              connection timeout seconds (default: 5)
  MUSIC_PATH_HOST          host-side path prefix stored in Strawberry URLs
                           (default: /media/jack/buffer/audio)
  MUSIC_PATH_CONTAINER     unused in backend; MPD resolves relative to its
                           music_directory (default: /music, for docs only)
  MUSIC_SEARCH_LIMIT       max LIKE search results (default: 20)
  MUSIC_ARTIST_RADIO_N     tracks queued for artist radio (default: 10)

Normalized ToolResult.data schemas:

  search:
    { "query": str, "results": [TrackDict+score], "total": int }

  play / queue (single):
    { "action": "play"|"queue", "track": TrackDict, "tracks": null,
      "confidence": float, "picked_from": int }

  play / queue (artist radio / multi):
    { "action": "play", "track": TrackDict, "tracks": [TrackDict],
      "confidence": float, "picked_from": int }

  control:
    { "action": str, "ok": true }

  now_playing:
    { "playing": bool, "state": "play"|"pause"|"stop",
      "track": {"title": str, "artist": str, "album": str} | null,
      "elapsed": float|null, "duration": float|null }

  queue_view:
    { "queue": [{"pos": int, "title": str, "artist": str, "album": str}],
      "length": int }

  TrackDict:
    { "id": int, "title": str, "artist": str, "album": str, "url": str,
      "score": float }
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import sqlite3
import time
from typing import Any
from urllib.parse import unquote

import musicpd

import tools as _registry
from tools.base import ToolResult

log = logging.getLogger("assistant.tools.music")

# ── Configuration ──────────────────────────────────────────────────────────────

STRAWBERRY_DB_PATH: str = os.getenv(
    "STRAWBERRY_DB_PATH",
    "/strawberry/strawberry.db",
)
MPD_HOST: str = os.getenv("MPD_HOST", "mpd")
MPD_PORT: int = int(os.getenv("MPD_PORT", "6600"))
MPD_TIMEOUT: int = int(os.getenv("MPD_TIMEOUT", "5"))
# The path prefix that Strawberry stores in its file:// URLs on the host.
MUSIC_PATH_HOST: str = os.getenv("MUSIC_PATH_HOST", "/media/jack/buffer/audio")
MUSIC_SEARCH_LIMIT: int = int(os.getenv("MUSIC_SEARCH_LIMIT", "20"))
MUSIC_ARTIST_RADIO_N: int = int(os.getenv("MUSIC_ARTIST_RADIO_N", "10"))


# ── Strawberry DB helpers ──────────────────────────────────────────────────────

def _open_strawberry() -> sqlite3.Connection:
    """Open Strawberry DB read-only with a short timeout for scan-lock resilience."""
    uri = f"file:{STRAWBERRY_DB_PATH}?mode=ro"
    conn = sqlite3.connect(uri, timeout=5, check_same_thread=False, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _url_to_mpd_path(url: str) -> str:
    """Convert a Strawberry file:// URL to an MPD-relative path.

    Steps:
      1. Strip 'file://' prefix (7 chars).
      2. URL-decode percent-encoded characters (%20 → space, etc.).
      3. Strip MUSIC_PATH_HOST prefix to get music-directory-relative path.
      4. Strip leading slash — MPD add() takes relative paths.

    Example:
      file:///media/jack/buffer/audio/rock/artist/song.mp3
      → /media/jack/buffer/audio/rock/artist/song.mp3  (after unquote)
      → rock/artist/song.mp3  (after stripping MUSIC_PATH_HOST)
    """
    if url.startswith("file://"):
        path = unquote(url[7:])
    else:
        path = unquote(url)

    if MUSIC_PATH_HOST and path.startswith(MUSIC_PATH_HOST):
        path = path[len(MUSIC_PATH_HOST):]

    return path.lstrip("/")


def _row_to_track(row: sqlite3.Row, score: float = 0.0) -> dict[str, Any]:
    return {
        "id": row["rowid"],
        "title": row["title"] or "",
        "artist": row["artist"] or "",
        "album": row["album"] or "",
        "url": row["url"] or "",
        "score": round(score, 3),
    }


# ── Synchronous DB operations (run via asyncio.to_thread) ─────────────────────

def _sync_search(query: str) -> list[dict[str, Any]]:
    """Search Strawberry songs with LIKE on title/artist/album, ranked by playcount.

    Raises sqlite3.OperationalError if the DB is locked during a Strawberry scan.
    """
    pattern = f"%{query}%"
    conn = _open_strawberry()
    try:
        cur = conn.execute(
            """
            SELECT rowid, title, artist, album, url, playcount
            FROM songs
            WHERE title LIKE ? OR artist LIKE ? OR album LIKE ?
            ORDER BY playcount DESC, title ASC
            LIMIT ?
            """,
            (pattern, pattern, pattern, MUSIC_SEARCH_LIMIT),
        )
        rows = cur.fetchall()
        results = []
        total = len(rows)
        for i, row in enumerate(rows):
            # Score: 0.9 for first result, declining to 0.5 at the tail.
            score = 0.9 - (i / max(total, 1)) * 0.4
            results.append(_row_to_track(row, score))
        return results
    finally:
        conn.close()


def _sync_get_by_id(song_id: int) -> dict[str, Any] | None:
    """Look up a song by rowid. Used by Phase 8b rec engine resolution."""
    conn = _open_strawberry()
    try:
        cur = conn.execute(
            "SELECT rowid, title, artist, album, url FROM songs WHERE rowid = ?",
            (song_id,),
        )
        row = cur.fetchone()
        return _row_to_track(row, score=1.0) if row else None
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()


def _sync_artist_songs(artist: str) -> list[dict[str, Any]]:
    """Return all songs matching artist LIKE pattern, with playcount attached."""
    conn = _open_strawberry()
    try:
        cur = conn.execute(
            """
            SELECT rowid, title, artist, album, url, playcount
            FROM songs
            WHERE artist LIKE ?
            ORDER BY playcount DESC, title ASC
            """,
            (f"%{artist}%",),
        )
        rows = cur.fetchall()
        return [
            {**_row_to_track(row, 0.0), "playcount": int(row["playcount"] or 0)}
            for row in rows
        ]
    finally:
        conn.close()


# ── Artist radio ───────────────────────────────────────────────────────────────

def artist_radio(
    artist: str,
    n: int = MUSIC_ARTIST_RADIO_N,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Return N songs by artist, weighted-randomly by playcount.

    Seeded for determinism in tests; defaults to current second.
    This is the Phase 8b fallback entry point — called directly when the rec
    engine is unavailable. 8b has nothing to fall back to without this function.

    Returns an empty list if the artist is not found (callers handle gracefully).
    Raises sqlite3.OperationalError if the DB is scan-locked.
    """
    songs = _sync_artist_songs(artist)
    if not songs:
        return []

    weights = [max(s["playcount"], 1) for s in songs]
    rng = random.Random(seed if seed is not None else int(time.time()))

    # Weighted sample with de-duplication.
    seen: set[int] = set()
    chosen: list[dict[str, Any]] = []
    attempts = 0
    while len(chosen) < min(n, len(songs)) and attempts < n * 3:
        attempts += 1
        pick = rng.choices(songs, weights=weights, k=1)[0]
        if pick["id"] not in seen:
            seen.add(pick["id"])
            chosen.append(pick)

    return chosen


# ── MPD helpers (synchronous, called via asyncio.to_thread) ───────────────────

def _mpd_connect() -> musicpd.MPDClient:
    """Connect to MPD with one retry on failure."""
    client = musicpd.MPDClient()
    client.timeout = MPD_TIMEOUT
    try:
        client.connect(MPD_HOST, MPD_PORT)
        return client
    except (musicpd.ConnectionError, ConnectionRefusedError, OSError):
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


def _sync_play(url: str) -> None:
    """Clear queue, add track, and start playing."""
    path = _url_to_mpd_path(url)
    log.debug("mpd.play | path=%s", path)

    def _fn(c: musicpd.MPDClient) -> None:
        c.clear()
        c.add(path)
        c.play()

    _with_mpd(_fn)


def _sync_queue(url: str) -> None:
    """Append a track to the current MPD queue."""
    path = _url_to_mpd_path(url)
    log.debug("mpd.queue | path=%s", path)
    _with_mpd(lambda c: c.add(path))


def _sync_control(action: str) -> None:
    """Execute a control command: pause / resume / next / stop."""
    def _fn(c: musicpd.MPDClient) -> None:
        if action == "pause":
            c.pause(1)
        elif action == "resume":
            c.pause(0)
        elif action == "next":
            c.next()
        elif action == "stop":
            c.stop()
        else:
            raise ValueError(f"Unknown control action: {action!r}")
    _with_mpd(_fn)


def _sync_now_playing() -> dict[str, Any]:
    """Return current playback state."""
    def _fn(c: musicpd.MPDClient) -> dict[str, Any]:
        status = c.status()
        state = status.get("state", "stop")
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


def _sync_play_tracks(tracks: list[dict[str, Any]]) -> None:
    """Clear queue, add all tracks, and start playing."""
    paths = [_url_to_mpd_path(t["url"]) for t in tracks]

    def _fn(c: musicpd.MPDClient) -> None:
        c.clear()
        for path in paths:
            c.add(path)
        c.play()

    _with_mpd(_fn)


# ── Intent parsing ─────────────────────────────────────────────────────────────

_CONTROL_MAP: dict[str, str] = {
    "pause": "pause",
    "stop": "stop",
    "resume": "resume",
    "continue": "resume",
    "unpause": "resume",
    "next": "next",
    "skip": "next",
}


def _extract_control_action(prompt: str) -> str | None:
    p = prompt.lower()
    for kw, action in _CONTROL_MAP.items():
        if re.search(rf"\b{re.escape(kw)}\b", p):
            return action
    return None


def _extract_search_query(prompt: str) -> str:
    """Strip common command prefixes and return a bare search string."""
    cleaned = re.sub(
        r"^(play(back)?|queue|put\s+on|start\s+playing|add\s+to\s+queue)\s+",
        "",
        prompt.strip(),
        flags=re.IGNORECASE,
    )
    return cleaned.strip().strip("\"'")


# ── MPD error classification ───────────────────────────────────────────────────

def _is_mpd_connection_error(exc: Exception) -> bool:
    return isinstance(exc, (musicpd.ConnectionError, ConnectionRefusedError, BrokenPipeError, OSError))


# ── Tool entry point ───────────────────────────────────────────────────────────

async def run(params: dict[str, Any]) -> ToolResult:
    """Entry point called by tools.dispatch().

    params:
      prompt   (str)       — user message (always present)
      action   (str|None)  — explicit override: search|play|queue|control|
                             now_playing|queue_view
      query    (str|None)  — explicit search/play query
      control  (str|None)  — explicit control command: pause|resume|next|stop
      song_id  (int|None)  — Phase 8b: direct rowid lookup, bypass search
      artist   (str|None)  — Phase 8b: trigger artist_radio() directly
    """
    prompt: str = params.get("prompt", "")
    action: str | None = params.get("action")
    query: str | None = params.get("query")
    control: str | None = params.get("control")
    song_id: int | None = params.get("song_id")
    artist_param: str | None = params.get("artist")

    log.info("music.run | prompt=%r action=%s", prompt[:80], action)

    # ── Infer action from prompt when not explicit ─────────────────────────────
    if action is None:
        p = prompt.lower()
        if any(kw in p for kw in ("now playing", "what's playing", "what is playing")):
            action = "now_playing"
        elif re.search(r"\b(what'?s|what is)\s+(in\s+)?(the\s+)?(queue|playlist)\b", p):
            action = "queue_view"
        elif any(kw in p for kw in ("pause", "stop", "resume", "unpause", "continue")):
            action = "control"
        elif re.search(r"\b(next|skip)\s*(track|song)?\b", p):
            action = "control"
        else:
            action = "play"

    # ── Now playing ───────────────────────────────────────────────────────────
    if action == "now_playing":
        try:
            data = await asyncio.to_thread(_sync_now_playing)
            return ToolResult(ok=True, data=data)
        except (musicpd.ConnectionError, ConnectionRefusedError, OSError) as exc:
            log.warning("music.now_playing | mpd_error=%s", exc)
            return ToolResult.failure("Could not reach MPD — is it running?", retryable=True)
        except Exception as exc:
            log.error("music.now_playing | unexpected=%s", exc)
            return ToolResult.failure(str(exc), retryable=False)

    # ── Queue view ────────────────────────────────────────────────────────────
    if action == "queue_view":
        try:
            data = await asyncio.to_thread(_sync_queue_view)
            return ToolResult(ok=True, data=data)
        except (musicpd.ConnectionError, ConnectionRefusedError, OSError) as exc:
            log.warning("music.queue_view | mpd_error=%s", exc)
            return ToolResult.failure("Could not reach MPD — is it running?", retryable=True)
        except Exception as exc:
            log.error("music.queue_view | unexpected=%s", exc)
            return ToolResult.failure(str(exc), retryable=False)

    # ── Playback control ──────────────────────────────────────────────────────
    if action == "control":
        cmd = control or _extract_control_action(prompt)
        if not cmd:
            return ToolResult.failure("No control action recognized.", retryable=False)
        try:
            await asyncio.to_thread(_sync_control, cmd)
            return ToolResult(ok=True, data={"action": cmd, "ok": True})
        except (musicpd.ConnectionError, ConnectionRefusedError, OSError) as exc:
            log.warning("music.control | action=%s mpd_error=%s", cmd, exc)
            return ToolResult.failure("Could not reach MPD — is it running?", retryable=True)
        except Exception as exc:
            log.error("music.control | action=%s unexpected=%s", cmd, exc)
            return ToolResult.failure(str(exc), retryable=False)

    # ── Phase 8b: direct song_id resolution (rec engine path) ─────────────────
    if song_id is not None:
        try:
            track = await asyncio.to_thread(_sync_get_by_id, song_id)
        except sqlite3.OperationalError:
            return ToolResult.failure(
                "Music library is temporarily locked (scan in progress). Please retry.",
                retryable=True,
            )
        if not track:
            return ToolResult.failure(f"Song ID {song_id} not found in library.", retryable=False)
        try:
            if action == "queue":
                await asyncio.to_thread(_sync_queue, track["url"])
                return ToolResult(ok=True, data={"action": "queue", "track": track, "tracks": None, "confidence": 1.0, "picked_from": 1})
            else:
                await asyncio.to_thread(_sync_play, track["url"])
                return ToolResult(ok=True, data={"action": "play", "track": track, "tracks": None, "confidence": 1.0, "picked_from": 1})
        except (musicpd.ConnectionError, ConnectionRefusedError, OSError) as exc:
            log.warning("music.song_id_play | mpd_error=%s", exc)
            return ToolResult.failure("Could not reach MPD — is it running?", retryable=True)

    # ── Phase 8b: artist radio (called directly from 8b fallback) ─────────────
    if artist_param:
        try:
            tracks = await asyncio.to_thread(artist_radio, artist_param)
        except sqlite3.OperationalError:
            return ToolResult.failure(
                "Music library is temporarily locked (scan in progress). Please retry.",
                retryable=True,
            )
        if not tracks:
            return ToolResult.failure(
                f"No songs found for artist '{artist_param}'.", retryable=False
            )
        try:
            await asyncio.to_thread(_sync_play_tracks, tracks)
        except (musicpd.ConnectionError, ConnectionRefusedError, OSError) as exc:
            log.warning("music.artist_radio | mpd_error=%s", exc)
            return ToolResult.failure("Could not reach MPD — is it running?", retryable=True)
        return ToolResult(ok=True, data={
            "action": "play",
            "track": tracks[0],
            "tracks": tracks,
            "confidence": 1.0,
            "picked_from": len(tracks),
        })

    # ── Search (explicit action="search") ─────────────────────────────────────
    if action == "search":
        q = query or _extract_search_query(prompt)
        if not q:
            return ToolResult.failure("No search query provided.", retryable=False)
        try:
            results = await asyncio.to_thread(_sync_search, q)
        except sqlite3.OperationalError:
            return ToolResult.failure(
                "Music library is temporarily locked (scan in progress). Please retry.",
                retryable=True,
            )
        return ToolResult(ok=True, data={"query": q, "results": results, "total": len(results)})

    # ── Play / Queue (default paths) ──────────────────────────────────────────
    q = query or _extract_search_query(prompt)
    if not q:
        return ToolResult.failure("No track or artist specified.", retryable=False)

    try:
        results = await asyncio.to_thread(_sync_search, q)
    except sqlite3.OperationalError:
        return ToolResult.failure(
            "Music library is temporarily locked (scan in progress). Please retry.",
            retryable=True,
        )

    if not results:
        # No exact/LIKE match — try artist radio as automatic fallback.
        log.info("music.play | no_results query=%r trying artist radio", q)
        try:
            tracks = await asyncio.to_thread(artist_radio, q)
        except sqlite3.OperationalError:
            return ToolResult.failure(
                "Music library is temporarily locked (scan in progress). Please retry.",
                retryable=True,
            )
        if not tracks:
            return ToolResult.failure(
                f"No tracks or artists found matching '{q}'.", retryable=False
            )
        try:
            await asyncio.to_thread(_sync_play_tracks, tracks)
        except (musicpd.ConnectionError, ConnectionRefusedError, OSError) as exc:
            log.warning("music.artist_radio_fallback | mpd_error=%s", exc)
            return ToolResult.failure("Could not reach MPD — is it running?", retryable=True)
        return ToolResult(ok=True, data={
            "action": "play",
            "track": tracks[0],
            "tracks": tracks,
            "confidence": 0.7,
            "picked_from": len(tracks),
        })

    # Auto-pick: top-ranked LIKE result.  Confidence logged server-side.
    top = results[0]
    confidence = top.get("score", 0.9)
    log.info(
        "music.auto_pick | query=%r picked=%r artist=%r confidence=%.3f candidates=%d",
        q, top["title"], top["artist"], confidence, len(results),
    )

    try:
        if action == "queue":
            await asyncio.to_thread(_sync_queue, top["url"])
            return ToolResult(ok=True, data={
                "action": "queue", "track": top, "tracks": None,
                "confidence": confidence, "picked_from": len(results),
            })
        else:
            await asyncio.to_thread(_sync_play, top["url"])
            return ToolResult(ok=True, data={
                "action": "play", "track": top, "tracks": None,
                "confidence": confidence, "picked_from": len(results),
            })
    except (musicpd.ConnectionError, ConnectionRefusedError, OSError) as exc:
        log.warning("music.play | mpd_error=%s", exc)
        return ToolResult.failure("Could not reach MPD — is it running?", retryable=True)
    except Exception as exc:
        log.error("music.play | unexpected=%s", exc)
        return ToolResult.failure(str(exc), retryable=False)


# Self-register when the module is imported.
import sys as _sys
_registry.register("music", _sys.modules[__name__])
