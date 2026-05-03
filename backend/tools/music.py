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
from functools import lru_cache
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
MUSIC_PLAYLIST_MIN_N: int = int(os.getenv("MUSIC_PLAYLIST_MIN_N", "12"))
MUSIC_PLAYLIST_MAX_N: int = int(os.getenv("MUSIC_PLAYLIST_MAX_N", "24"))
MUSIC_GENRE_TREE_PATH: str = os.getenv(
    "MUSIC_GENRE_TREE_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "genres.txt"),
)


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


def _sync_search_by_title_artist(title: str, artist: str) -> list[dict[str, Any]]:
    """Search Strawberry songs by title AND artist (compound LIKE filter).

    More precise than the general _sync_search when the user says
    "<title> by <artist>" — filters both dimensions in a single query.
    Raises sqlite3.OperationalError if the DB is scan-locked.
    """
    conn = _open_strawberry()
    try:
        cur = conn.execute(
            """
            SELECT rowid, title, artist, album, url, playcount
            FROM songs
            WHERE title LIKE ? AND artist LIKE ?
            ORDER BY playcount DESC, title ASC
            LIMIT ?
            """,
            (f"%{title}%", f"%{artist}%", MUSIC_SEARCH_LIMIT),
        )
        rows = cur.fetchall()
        total = len(rows)
        return [
            _row_to_track(row, 0.9 - (i / max(total, 1)) * 0.4)
            for i, row in enumerate(rows)
        ]
    finally:
        conn.close()


def _sync_search_by_year_range(year_start: int, year_end: int) -> list[dict[str, Any]]:
    """Return songs whose year column falls within [year_start, year_end].

    Ordered by playcount so popular tracks are surfaced first.
    Returns all matches (no LIMIT) — callers sample from the full pool.
    Raises sqlite3.OperationalError if the DB is scan-locked.
    """
    conn = _open_strawberry()
    try:
        cur = conn.execute(
            """
            SELECT rowid, title, artist, album, url, playcount
            FROM songs
            WHERE year BETWEEN ? AND ?
            ORDER BY playcount DESC, title ASC
            """,
            (year_start, year_end),
        )
        rows = cur.fetchall()
        return [_row_to_track(row, 0.9) for row in rows]
    finally:
        conn.close()


def _sync_genre_songs(genre: str) -> list[dict[str, Any]]:
    """Return all songs matching genre LIKE pattern, with playcount attached.

    Some Strawberry schemas omit a genre column. In that case, return an empty
    list so resolver logic can fall back to artist/search paths.
    """
    conn = _open_strawberry()
    try:
        cur = conn.execute(
            """
            SELECT rowid, title, artist, album, url, playcount
            FROM songs
            WHERE genre LIKE ?
            ORDER BY playcount DESC, title ASC
            """,
            (f"%{genre}%",),
        )
        rows = cur.fetchall()
        return [
            {**_row_to_track(row, 0.0), "playcount": int(row["playcount"] or 0)}
            for row in rows
        ]
    except sqlite3.OperationalError as exc:
        if "no such column" in str(exc).lower() and "genre" in str(exc).lower():
            log.info("music.genre_column_missing | fallback_to_artist_search")
            return []
        raise
    finally:
        conn.close()


# ── Artist radio ───────────────────────────────────────────────────────────────

def _playlist_pick_count(pool_size: int, requested_n: int | None = None) -> int:
    """Resolve how many tracks to queue for multi-track playback.

    - Explicit requested_n is respected, clamped to pool size.
    - Default behavior is adaptive with env bounds for richer queues.
    """
    if pool_size <= 0:
        return 0
    if requested_n is not None:
        return max(1, min(int(requested_n), pool_size))

    min_n = max(1, min(MUSIC_PLAYLIST_MIN_N, MUSIC_PLAYLIST_MAX_N))
    max_n = max(min_n, MUSIC_PLAYLIST_MAX_N)
    adaptive = max(1, pool_size // 2)
    return min(pool_size, max(min_n, min(max_n, adaptive)))

def artist_radio(
    artist: str,
    n: int | None = None,
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
    pick_count = _playlist_pick_count(len(songs), requested_n=n)

    # Weighted sample with de-duplication.
    seen: set[int] = set()
    chosen: list[dict[str, Any]] = []
    attempts = 0
    while len(chosen) < pick_count and attempts < max(3, pick_count * 3):
        attempts += 1
        pick = rng.choices(songs, weights=weights, k=1)[0]
        if pick["id"] not in seen:
            seen.add(pick["id"])
            chosen.append(pick)

    return chosen


def genre_radio(
    genre: str,
    n: int | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Return N songs by genre, weighted-randomly by playcount."""
    songs = _sync_genre_songs(genre)
    if not songs:
        return []

    weights = [max(s["playcount"], 1) for s in songs]
    rng = random.Random(seed if seed is not None else int(time.time()))
    pick_count = _playlist_pick_count(len(songs), requested_n=n)

    seen: set[int] = set()
    chosen: list[dict[str, Any]] = []
    attempts = 0
    while len(chosen) < pick_count and attempts < max(3, pick_count * 3):
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


def _sync_play(url: str) -> None:
    """Clear queue, add track, and start playing."""
    path = _url_to_mpd_path(url)
    log.debug("mpd.play | path=%s", path)

    def _fn(c: musicpd.MPDClient) -> None:
        c.clear()
        added = _add_mpd_paths(c, [path])
        if added == 0:
            raise FileNotFoundError(f"Track not available in MPD library: {path}")
        c.play()

    _with_mpd(_fn)


def _sync_queue(url: str) -> None:
    """Append a track to the current MPD queue."""
    path = _url_to_mpd_path(url)
    log.debug("mpd.queue | path=%s", path)

    def _fn(c: musicpd.MPDClient) -> None:
        added = _add_mpd_paths(c, [path])
        if added == 0:
            raise FileNotFoundError(f"Track not available in MPD library: {path}")

    _with_mpd(_fn)


def _sync_play_pos(pos: int) -> None:
    """Jump to a specific position in the current MPD queue (0-indexed)."""
    log.debug("mpd.play_pos | pos=%d", pos)
    _with_mpd(lambda c: c.play(pos))


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


def _sync_play_tracks(tracks: list[dict[str, Any]]) -> None:
    """Clear queue, add all tracks, and start playing."""
    paths = [_url_to_mpd_path(t["url"]) for t in tracks]

    def _fn(c: musicpd.MPDClient) -> None:
        c.clear()
        added = _add_mpd_paths(c, paths)
        if added == 0:
            raise FileNotFoundError("No selected tracks are available in the MPD library")
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


_GENERIC_TITLE = frozenset({
    "a song", "some songs", "any song", "a track", "some tracks", "any track",
    "some music", "any music", "a random song", "a random track",
    "something", "anything", "a tune", "some tunes",
})

_GENERIC_TITLE_RE = re.compile(
    r"^(?:a|some|any)(?:thing)?\s+(?:random\s+)?(?:song|track|music)s?$",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def _load_genre_terms() -> tuple[str, ...]:
    """Load canonical genre terms from the taxonomy file."""
    if not os.path.isfile(MUSIC_GENRE_TREE_PATH):
        return ()

    terms: list[str] = []
    with open(MUSIC_GENRE_TREE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            terms.append(s.lower())

    # Longer tokens first so "progressive rock" wins over "rock".
    return tuple(sorted(set(terms), key=len, reverse=True))


def _normalize_music_query(text: str) -> str:
    return re.sub(
        r"\s+(?:music|songs?|tracks?|bands?|artists?)$", "", text, flags=re.IGNORECASE
    ).strip().lower()


def _resolve_genre_query(query: str) -> str | None:
    """Return a canonical genre term if query maps to the taxonomy."""
    q_norm = _normalize_music_query(query)
    if not q_norm:
        return None
    terms = _load_genre_terms()
    if not terms:
        return None

    if q_norm in terms:
        return q_norm

    # Token-boundary contains match for cases like "classic rock".
    for term in terms:
        if re.search(rf"\b{re.escape(term)}\b", q_norm):
            return term
    return None


def _extract_search_query(prompt: str) -> str:
    """Strip common command prefixes and return a bare search string."""
    cleaned = re.sub(
        r"^(play(back)?|queue|put\s+on|start\s+playing|add\s+to\s+queue)\s+",
        "",
        prompt.strip(),
        flags=re.IGNORECASE,
    )
    # Remove polite fillers and trailing punctuation.
    cleaned = re.sub(r"^(the\s+song\s+)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip().strip("\"' .,!?")

    # "a/some/any [random] <artist> song/track/music[s]" → return artist name.
    # e.g. "a random Nightwish song" → "Nightwish"
    #      "some Metallica tracks"   → "Metallica"
    artist_song_m = re.match(
        r"^(?:a|some|any)\s+(?:random\s+)?(?P<artist>.+?)\s+(?:song|track|music)s?$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if artist_song_m:
        artist = artist_song_m.group("artist").strip()
        if artist:
            return artist

    # "<title> by <artist>" — extract title, but if the title is a generic
    # placeholder ("a song", "something", etc.) return the artist instead.
    by_match = re.match(
        r"^(?P<title>.+?)\s+by\s+(?P<artist>.+)$", cleaned, flags=re.IGNORECASE
    )
    if by_match:
        title = by_match.group("title").strip().strip("\"' .,!?")
        artist = by_match.group("artist").strip().strip("\"' .,!?")
        if title.lower() in _GENERIC_TITLE or _GENERIC_TITLE_RE.match(title):
            return artist
        if title:
            return title

    return cleaned


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


def _add_mpd_paths(client: musicpd.MPDClient, paths: list[str]) -> int:
    added = 0
    for path in paths:
        try:
            client.add(path)
            added += 1
        except Exception as exc:
            if _is_mpd_missing_path_error(exc):
                log.warning("mpd.add_skip_missing | path=%s", path)
                continue
            raise
    return added


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
    # Compound search params set by the deterministic pre-router.
    artist_filter: str | None = params.get("artist_filter")  # paired with query
    year_range: tuple | list | None = params.get("year_range")  # (start, end)

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
        if cmd == "set_volume":
            raw_volume = params.get("volume")
            if raw_volume is None:
                return ToolResult.failure("Volume value is required.", retryable=False)
            try:
                applied = await asyncio.to_thread(_sync_set_volume, int(raw_volume))
            except (musicpd.ConnectionError, ConnectionRefusedError, OSError) as exc:
                log.warning("music.set_volume | requested=%s mpd_error=%s", raw_volume, exc)
                return ToolResult.failure("Could not reach MPD — is it running?", retryable=True)
            except Exception as exc:
                log.error("music.set_volume | requested=%s unexpected=%s", raw_volume, exc)
                return ToolResult.failure(str(exc), retryable=False)
            return ToolResult(ok=True, data={"action": "set_volume", "ok": True, "volume": applied})
        # play_pos: jump to a queue position (sent by frontend queue click).
        if cmd == "play_pos":
            pos = int(params.get("pos", 0))
            try:
                await asyncio.to_thread(_sync_play_pos, pos)
                return ToolResult(ok=True, data={"action": "play_pos", "pos": pos, "ok": True})
            except (musicpd.ConnectionError, ConnectionRefusedError, OSError) as exc:
                log.warning("music.play_pos | pos=%d mpd_error=%s", pos, exc)
                return ToolResult.failure("Could not reach MPD — is it running?", retryable=True)
            except Exception as exc:
                log.error("music.play_pos | pos=%d unexpected=%s", pos, exc)
                return ToolResult.failure(str(exc), retryable=False)
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

    # ── Compound title + artist search ("title by artist") ────────────────────
    if artist_filter and query:
        try:
            results = await asyncio.to_thread(_sync_search_by_title_artist, query, artist_filter)
        except sqlite3.OperationalError:
            return ToolResult.failure(
                "Music library is temporarily locked (scan in progress). Please retry.",
                retryable=True,
            )
        if results:
            top = results[0]
            confidence = top.get("score", 0.9)
            log.info(
                "music.title_artist_pick | title=%r artist_filter=%r picked=%r confidence=%.3f candidates=%d",
                query, artist_filter, top["title"], confidence, len(results),
            )
            try:
                if action == "queue":
                    await asyncio.to_thread(_sync_queue, top["url"])
                    return ToolResult(ok=True, data={"action": "queue", "track": top, "tracks": None, "confidence": confidence, "picked_from": len(results)})
                else:
                    await asyncio.to_thread(_sync_play, top["url"])
                    return ToolResult(ok=True, data={"action": "play", "track": top, "tracks": None, "confidence": confidence, "picked_from": len(results)})
            except (musicpd.ConnectionError, ConnectionRefusedError, OSError) as exc:
                log.warning("music.title_artist_play | mpd_error=%s", exc)
                return ToolResult.failure("Could not reach MPD — is it running?", retryable=True)
        # No title+artist match — fall back to artist radio.
        log.info("music.title_artist_miss | title=%r artist_filter=%r trying artist radio", query, artist_filter)
        try:
            tracks = await asyncio.to_thread(artist_radio, artist_filter)
        except sqlite3.OperationalError:
            return ToolResult.failure(
                "Music library is temporarily locked (scan in progress). Please retry.",
                retryable=True,
            )
        if tracks:
            try:
                await asyncio.to_thread(_sync_play_tracks, tracks)
            except (musicpd.ConnectionError, ConnectionRefusedError, OSError) as exc:
                log.warning("music.title_artist_radio_fallback | mpd_error=%s", exc)
                return ToolResult.failure("Could not reach MPD — is it running?", retryable=True)
            return ToolResult(ok=True, data={"action": "play", "track": tracks[0], "tracks": tracks, "confidence": 0.7, "picked_from": len(tracks)})
        return ToolResult.failure(
            f"No tracks found for '{query}' by '{artist_filter}'.", retryable=False
        )

    # ── Year / decade range search ─────────────────────────────────────────────
    if year_range is not None:
        yr_start, yr_end = int(year_range[0]), int(year_range[1])
        try:
            all_year_tracks = await asyncio.to_thread(_sync_search_by_year_range, yr_start, yr_end)
        except sqlite3.OperationalError:
            return ToolResult.failure(
                "Music library is temporarily locked (scan in progress). Please retry.",
                retryable=True,
            )
        if not all_year_tracks:
            period = f"{yr_start}s" if yr_start != yr_end else str(yr_start)
            return ToolResult.failure(f"No tracks found from {period}.", retryable=False)
        rng = random.Random(int(time.time()))
        n_pick = _playlist_pick_count(len(all_year_tracks), requested_n=None)
        tracks = rng.sample(all_year_tracks, n_pick)
        log.info(
            "music.year_range | yr_start=%d yr_end=%d pool=%d picking=%d",
            yr_start, yr_end, len(all_year_tracks), n_pick,
        )
        try:
            await asyncio.to_thread(_sync_play_tracks, tracks)
        except (musicpd.ConnectionError, ConnectionRefusedError, OSError) as exc:
            log.warning("music.year_range_play | mpd_error=%s", exc)
            return ToolResult.failure("Could not reach MPD — is it running?", retryable=True)
        return ToolResult(ok=True, data={
            "action": "play",
            "track": tracks[0],
            "tracks": tracks,
            "confidence": 0.9,
            "picked_from": len(all_year_tracks),
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

    # Genre-first resolver path for ambiguous playback requests.
    genre_match = _resolve_genre_query(q)
    if genre_match and action != "queue":
        log.info("music.play | genre_first query=%r genre=%r", q, genre_match)
        try:
            tracks = await asyncio.to_thread(genre_radio, genre_match)
        except sqlite3.OperationalError:
            return ToolResult.failure(
                "Music library is temporarily locked (scan in progress). Please retry.",
                retryable=True,
            )
        if tracks:
            try:
                await asyncio.to_thread(_sync_play_tracks, tracks)
            except (musicpd.ConnectionError, ConnectionRefusedError, OSError) as exc:
                log.warning("music.genre_radio | mpd_error=%s", exc)
                return ToolResult.failure("Could not reach MPD — is it running?", retryable=True)
            return ToolResult(ok=True, data={
                "action": "play",
                "track": tracks[0],
                "tracks": tracks,
                "confidence": 0.9,
                "picked_from": len(tracks),
            })

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

    # Artist-radio heuristic: if the query looks like an artist/genre name rather
    # than a song title, use artist_radio() to queue multiple tracks instead of
    # single-picking the top LIKE result.
    #
    # Criteria (both must be true):
    #   1. The normalised query doesn't appear in any of the top-5 result titles
    #      (i.e. no title match → the LIKE hit was on artist/album, not title).
    #   2. The normalised query does appear in the top result's artist field
    #      (confirms it's an artist search, not e.g. an album search).
    #
    # Strip common genre suffixes ("music", "songs", "tracks") before comparing
    # so "classical music" matches artist "Classical".
    _q_norm = _normalize_music_query(q)
    _title_miss = not any(_q_norm in r["title"].lower() for r in results[:5])
    _artist_hit = _q_norm in results[0]["artist"].lower()
    if _title_miss and _artist_hit and action != "queue":
        log.info(
            "music.play | artist_radio_heuristic query=%r artist=%r candidates=%d",
            q, results[0]["artist"], len(results),
        )
        try:
            tracks = await asyncio.to_thread(artist_radio, q)
        except sqlite3.OperationalError:
            return ToolResult.failure(
                "Music library is temporarily locked (scan in progress). Please retry.",
                retryable=True,
            )
        if tracks:
            try:
                await asyncio.to_thread(_sync_play_tracks, tracks)
            except (musicpd.ConnectionError, ConnectionRefusedError, OSError) as exc:
                log.warning("music.artist_radio_heuristic | mpd_error=%s", exc)
                return ToolResult.failure("Could not reach MPD — is it running?", retryable=True)
            return ToolResult(ok=True, data={
                "action": "play",
                "track": tracks[0],
                "tracks": tracks,
                "confidence": 0.85,
                "picked_from": len(tracks),
            })
        # artist_radio returned nothing (shouldn't happen if LIKE found results,
        # but fall through to single-pick as safety net).

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
