"""Track sampling for artist and genre radio and playlist sizing."""
from __future__ import annotations

import logging
import os
import random
import time
from typing import Any

from music import library

log = logging.getLogger("assistant.tools.music")

MUSIC_ARTIST_RADIO_N: int = int(os.getenv("MUSIC_ARTIST_RADIO_N", "10"))
MUSIC_PLAYLIST_MIN_N: int = int(os.getenv("MUSIC_PLAYLIST_MIN_N", "12"))
MUSIC_PLAYLIST_MAX_N: int = int(os.getenv("MUSIC_PLAYLIST_MAX_N", "24"))


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


def _weighted_unique_sample(
    items: list[dict[str, Any]],
    weights: list[float],
    pick_count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Weighted sample without replacement.

    This guarantees up to ``pick_count`` unique selections (bounded by pool size)
    while still biasing picks by weight.
    """
    if pick_count <= 0 or not items:
        return []

    pool = list(items)
    pool_weights = [max(float(w), 0.01) for w in weights]
    target = min(int(pick_count), len(pool))
    chosen: list[dict[str, Any]] = []

    while pool and len(chosen) < target:
        idx = rng.choices(range(len(pool)), weights=pool_weights, k=1)[0]
        chosen.append(pool.pop(idx))
        pool_weights.pop(idx)

    return chosen

def artist_radio(
    artist: str,
    n: int | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Return N songs by artist, weighted-randomly by rating.

    Seeded for determinism in tests; defaults to current second.
    This is the Phase 8b fallback entry point — called directly when the rec
    engine is unavailable. 8b has nothing to fall back to without this function.

    Returns an empty list if the artist is not found (callers handle gracefully).
    Raises sqlite3.OperationalError if the DB is temporarily locked.
    """
    songs = library._sync_artist_songs(artist)
    if not songs:
        return []

    weights = [max(float(s.get("rating", 0.0)), 0.01) for s in songs]
    rng = random.Random(seed if seed is not None else int(time.time()))
    pick_count = _playlist_pick_count(len(songs), requested_n=n)
    chosen = _weighted_unique_sample(songs, weights, pick_count, rng)
    log.info(
        "artist_radio | artist=%r pool=%d picking=%d selected=%d",
        artist,
        len(songs),
        pick_count,
        len(chosen),
    )

    return chosen


def genre_radio(
    genre: str,
    n: int | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Return N songs by genre, weighted-randomly by rating."""
    songs = library._sync_genre_songs(genre)
    if not songs:
        return []

    weights = [max(float(s.get("rating", 0.0)), 0.01) for s in songs]
    rng = random.Random(seed if seed is not None else int(time.time()))
    pick_count = _playlist_pick_count(len(songs), requested_n=n)
    chosen = _weighted_unique_sample(songs, weights, pick_count, rng)
    log.info(
        "genre_radio | genre=%r pool=%d picking=%d selected=%d",
        genre,
        len(songs),
        pick_count,
        len(chosen),
    )

    return chosen


def _sample_large_compilation(
    tracks: list[dict[str, Any]], rng: random.Random
) -> list[dict[str, Any]]:
    """Cap multi-artist albums over MUSIC_PLAYLIST_MAX_N to a random sample.

    The sample keeps album order, so chronological lists (e.g. "The Pitchfork
    500") still play oldest → newest. Single-artist albums are never sampled:
    a long double album is still one work.
    """
    limit = max(1, MUSIC_PLAYLIST_MAX_N)
    artists = {t["artist"] for t in tracks if t["artist"]}
    if len(tracks) <= limit or len(artists) <= 1:
        return tracks
    keep = sorted(rng.sample(range(len(tracks)), limit))
    return [tracks[i] for i in keep]
