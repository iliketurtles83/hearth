"""
Music: Beets library search + MPD playback, exposed to Hearth as the "music" tool.

Layout:
  commands.py     deterministic control parsing, play-request detection, reply text
  library.py      read-only Beets queries
  resolve.py      request text -> search query / genre / album / playlist
  radio.py        track sampling for artist/genre radio
  players/mpd.py  MPD connection and playback
  tool.py         run(params) -> ToolResult; registered by tools/music.py

Track metadata is cleaned upstream (cratedigger); this package only reads it.

# ── Phase 8b Recommendation Engine Contract (locked pre-implementation) ────────
# The recommendation engine HTTP response MUST follow this primary shape:
#   { "song_id": int, "provider": str, "confidence": float }
# Optional fields:
#   { "score": float, "explanation": str }
#
# "song_id" here refers to Beets items.id.
# Resolution: SELECT id, title, artist, album, path FROM items WHERE id = ?
# → trivial join, zero ambiguity, no fuzzy matching.
#
# Metadata-only responses ({ "title": str, "artist": str }) are NOT the primary
# contract. Metadata lookups are available via a separate non-primary path only.
#
# 8b fallback (rec engine unavailable): call artist_radio() from this module.
# artist_radio() is implemented in Phase 8 core — 8b has nothing to fall back to
# without it.
# ─────────────────────────────────────────────────────────────────────────────────

Beets DB schema notes:
    - Primary key: items.id
    - File path:   items.path (filesystem path; often absolute)
  - Opened read-only with timeout=5, check_same_thread=False, URI mode
    - All reads wrapped in try/except sqlite3.OperationalError for transient locks

MPD client (python-musicpd):
  - Per-request fresh connection (thread-safe, works with asyncio.to_thread)
  - Reconnect-once policy: if connect fails, retry once before raising
  - Connection errors bubble up as retryable ToolResult

Path rewrite:
    Beets stores filesystem paths. The backend strips MUSIC_ROOT and passes the
    resulting MPD-relative path (relative to MPD music_directory) to MPD.

  Example:
    MPD-relative:   rock/artist/song.mp3
    MPD resolves:   /music/rock/artist/song.mp3  (music_directory=/music)

Environment variables:
    BEETS_DB_PATH            path to Beets sqlite db
                                                     (default: ~/.config/beets/library.db)
  MPD_HOST                 MPD hostname (default: mpd)
  MPD_PORT                 MPD port (default: 6600)
  MPD_TIMEOUT              connection timeout seconds (default: 5)
    MUSIC_ROOT               filesystem music root used to derive MPD-relative
                                                     paths (default: /music)
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
