"""Read-only Beets library queries: search, artist/genre/album/year lookups."""
from __future__ import annotations

from functools import lru_cache
import logging
import os
import re
import sqlite3
from typing import Any

log = logging.getLogger("assistant.tools.music")

BEETS_DB_PATH: str = os.getenv(
    "BEETS_DB_PATH",
    os.path.join(os.path.expanduser("~"), ".config", "beets", "library.db"),
)
MUSIC_SEARCH_LIMIT: int = int(os.getenv("MUSIC_SEARCH_LIMIT", "20"))


# ── Beets DB helpers ───────────────────────────────────────────────────────────

def _open_beets() -> sqlite3.Connection:
    """Open Beets DB read-only with a short timeout for lock resilience."""
    uri = f"file:{BEETS_DB_PATH}?mode=ro"
    conn = sqlite3.connect(uri, timeout=5, check_same_thread=False, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _decode_beets_path(value: Any) -> str:
    """Decode a Beets items.path value that may be bytes or text."""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="surrogateescape")
        except Exception:
            return value.decode("latin-1", errors="ignore")
    return str(value or "")


@lru_cache(maxsize=1)
def _beets_columns() -> set[str]:
    """Return available columns from Beets items table."""
    conn = _open_beets()
    try:
        rows = conn.execute("PRAGMA table_info(items)").fetchall()
        return {str(r["name"]).lower() for r in rows}
    finally:
        conn.close()


def _genre_expr() -> str | None:
    """Return the Beets genre-like column expression to query.

    Beets schemas can expose either ``genre`` or ``genres``. Prefer ``genre``
    when present for compatibility, otherwise use ``genres``.
    """
    columns = _beets_columns()
    if "genre" in columns:
        return "genre"
    if "genres" in columns:
        return "genres"
    return None


def _row_to_track(row: sqlite3.Row, score: float = 0.0) -> dict[str, Any]:
    return {
        "id": row["id"],
        "title": row["title"] or "",
        "artist": row["artist"] or "",
        "album": row["album"] or "",
        "url": _decode_beets_path(row["path"]),
        "score": round(score, 3),
    }


# ── Synchronous DB operations (run via asyncio.to_thread) ─────────────────────

def _sync_search(query: str) -> list[dict[str, Any]]:
    """Search Beets items with LIKE on title/artist/album, ranked by rating.

    An exact title match ranks first, so "Sunshine Superman" picks the title
    track rather than the alphabetically first song on the same-named album.

    Raises sqlite3.OperationalError if the DB is temporarily locked.
    """
    pattern = f"%{query}%"
    conn = _open_beets()
    try:
        if "rating" in _beets_columns():
            cur = conn.execute(
                """
                SELECT id, title, artist, album, path, COALESCE(rating, 0.0) AS rating
                FROM items
                WHERE title LIKE ? OR artist LIKE ? OR album LIKE ?
                ORDER BY lower(title) = lower(?) DESC, rating DESC, title ASC
                LIMIT ?
                """,
                (pattern, pattern, pattern, query.strip(), MUSIC_SEARCH_LIMIT),
            )
        else:
            cur = conn.execute(
                """
                SELECT id, title, artist, album, path, 0.0 AS rating
                FROM items
                WHERE title LIKE ? OR artist LIKE ? OR album LIKE ?
                ORDER BY lower(title) = lower(?) DESC, rating DESC, title ASC
                LIMIT ?
                """,
                (pattern, pattern, pattern, query.strip(), MUSIC_SEARCH_LIMIT),
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
    """Look up a song by id. Used by Phase 8b rec engine resolution."""
    conn = _open_beets()
    try:
        cur = conn.execute(
            "SELECT id, title, artist, album, path FROM items WHERE id = ?",
            (song_id,),
        )
        row = cur.fetchone()
        return _row_to_track(row, score=1.0) if row else None
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()


def _sync_artist_songs(artist: str) -> list[dict[str, Any]]:
    """Return all songs matching artist LIKE pattern, with rating attached."""
    conn = _open_beets()
    try:
        if "rating" in _beets_columns():
            cur = conn.execute(
                """
                SELECT id, title, artist, album, path, COALESCE(rating, 0.0) AS rating
                FROM items
                WHERE artist LIKE ?
                ORDER BY rating DESC, title ASC
                """,
                (f"%{artist}%",),
            )
        else:
            cur = conn.execute(
                """
                SELECT id, title, artist, album, path, 0.0 AS rating
                FROM items
                WHERE artist LIKE ?
                ORDER BY rating DESC, title ASC
                """,
                (f"%{artist}%",),
            )
        rows = cur.fetchall()
        return [
            {**_row_to_track(row, 0.0), "rating": float(row["rating"] or 0.0)}
            for row in rows
        ]
    finally:
        conn.close()


def _sync_search_by_title_artist(title: str, artist: str) -> list[dict[str, Any]]:
    """Search Beets items by title AND artist (compound LIKE filter).

    More precise than the general _sync_search when the user says
    "<title> by <artist>" — filters both dimensions in a single query.
    Raises sqlite3.OperationalError if the DB is temporarily locked.
    """
    conn = _open_beets()
    try:
        if "rating" in _beets_columns():
            cur = conn.execute(
                """
                SELECT id, title, artist, album, path, COALESCE(rating, 0.0) AS rating
                FROM items
                WHERE title LIKE ? AND artist LIKE ?
                ORDER BY rating DESC, title ASC
                LIMIT ?
                """,
                (f"%{title}%", f"%{artist}%", MUSIC_SEARCH_LIMIT),
            )
        else:
            cur = conn.execute(
                """
                SELECT id, title, artist, album, path, 0.0 AS rating
                FROM items
                WHERE title LIKE ? AND artist LIKE ?
                ORDER BY rating DESC, title ASC
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

    Ordered by rating so higher-ranked tracks are surfaced first.
    Returns all matches (no LIMIT) — callers sample from the full pool.
    Raises sqlite3.OperationalError if the DB is temporarily locked.
    """
    columns = _beets_columns()
    if "year" not in columns:
        return []

    conn = _open_beets()
    try:
        if "rating" in _beets_columns():
            cur = conn.execute(
                """
                SELECT id, title, artist, album, path, COALESCE(rating, 0.0) AS rating
                FROM items
                WHERE year BETWEEN ? AND ?
                ORDER BY rating DESC, title ASC
                """,
                (year_start, year_end),
            )
        else:
            cur = conn.execute(
                """
                SELECT id, title, artist, album, path, 0.0 AS rating
                FROM items
                WHERE year BETWEEN ? AND ?
                ORDER BY rating DESC, title ASC
                """,
                (year_start, year_end),
            )
        rows = cur.fetchall()
        return [_row_to_track(row, 0.9) for row in rows]
    finally:
        conn.close()


# "&", "and", "n" and "'n'" between two words are one connector in genre names:
# "drum and bass", "drum'n'bass", "Drum 'n Bass" -> "drum & bass". Both sides need
# 2+ letters so "r&b" stays intact.
_GENRE_CONNECTOR_RE = re.compile(
    r"(?<=[a-z]{2})\s*(?:&|'n'|'n\b|\bn'|\band\b|\bn\b)\s*(?=[a-z]{2})"
)
_GENRE_ALIAS_RE = re.compile(r"(?<![\w&'])(?:dnb|d&b|d'n'b)(?![\w&'])")


def _canonical_genre(text: str) -> str:
    """Lower-case a genre name and unify spelling variants ("dnb", "rock n roll")."""
    s = _GENRE_ALIAS_RE.sub("drum & bass", text.lower())
    return re.sub(r"\s+", " ", _GENRE_CONNECTOR_RE.sub(" & ", s)).strip()


def _sync_genre_songs(genre: str) -> list[dict[str, Any]]:
    """Return all songs matching genre LIKE pattern, with rating attached.

    Some Beets schemas may omit a genre column. In that case, return an empty
    list so resolver logic can fall back to artist/search paths.

    Spelling variants match the same set: "drum and bass", "dnb" and a
    "Drum 'n Bass" tag are all "drum & bass".
    """
    genre = _canonical_genre(genre)
    # LIKE can't see spelling variants, so "drum & bass" queries "%drum%bass%"
    # and rows are filtered on their canonical tag below.
    pattern = f"%{genre.replace(' & ', '%')}%"
    genre_expr = _genre_expr()
    if not genre_expr:
        log.info("music.genre_column_missing | fallback_to_artist_search")
        return []

    conn = _open_beets()
    try:
        has_rating = "rating" in _beets_columns()
        if genre_expr == "genre":
            if has_rating:
                cur = conn.execute(
                    """
                    SELECT id, title, artist, album, path, COALESCE(rating, 0.0) AS rating, genre AS genre_tag
                    FROM items
                    WHERE genre LIKE ?
                    ORDER BY rating DESC, title ASC
                    """,
                    (pattern,),
                )
            else:
                cur = conn.execute(
                    """
                    SELECT id, title, artist, album, path, 0.0 AS rating, genre AS genre_tag
                    FROM items
                    WHERE genre LIKE ?
                    ORDER BY rating DESC, title ASC
                    """,
                    (pattern,),
                )
        else:
            if has_rating:
                cur = conn.execute(
                    """
                    SELECT id, title, artist, album, path, COALESCE(rating, 0.0) AS rating, genres AS genre_tag
                    FROM items
                    WHERE genres LIKE ?
                    ORDER BY rating DESC, title ASC
                    """,
                    (pattern,),
                )
            else:
                cur = conn.execute(
                    """
                    SELECT id, title, artist, album, path, 0.0 AS rating, genres AS genre_tag
                    FROM items
                    WHERE genres LIKE ?
                    ORDER BY rating DESC, title ASC
                    """,
                    (pattern,),
                )
        rows = cur.fetchall()
        if " & " in genre:
            rows = [r for r in rows if genre in _canonical_genre(r["genre_tag"] or "")]
        return [
            {**_row_to_track(row, 0.0), "rating": float(row["rating"] or 0.0)}
            for row in rows
        ]
    finally:
        conn.close()


def _album_name_variants(name: str) -> list[str]:
    """Lower-cased album names to try, with and without a leading "the"."""
    base = name.strip().lower()
    if base.startswith("the "):
        return [base, base[4:].strip()]
    return [base, f"the {base}"]


def _sync_album_tracks(
    name: str,
    artist: str | None = None,
    exact: bool = True,
) -> tuple[str, list[dict[str, Any]]]:
    """Resolve one album and return (album name, tracks in disc/track order).

    exact=True matches the album name case-insensitively (± a leading "the");
    exact=False also accepts substring matches. Ranking: the literal name, then
    the "the"-toggled name, then the shortest name, then the most tracks. An optional artist narrows by track or album artist.
    Returns ("", []) when nothing matches.
    Raises sqlite3.OperationalError if the DB is temporarily locked.
    """
    columns = _beets_columns()
    if "album" not in columns or not name.strip():
        return "", []
    group_col = "album_id" if "album_id" in columns else "album"
    variants = _album_name_variants(name)
    placeholders = ", ".join("?" for _ in variants)

    if exact:
        where = f"lower(album) IN ({placeholders})"
        args: list[Any] = list(variants)
    else:
        where = f"(lower(album) IN ({placeholders}) OR album LIKE ?)"
        args = [*variants, f"%{name.strip()}%"]
    if artist:
        artist_cols = ["artist"] + (["albumartist"] if "albumartist" in columns else [])
        where += " AND (" + " OR ".join(f"{c} LIKE ?" for c in artist_cols) + ")"
        args += [f"%{artist.strip()}%"] * len(artist_cols)

    conn = _open_beets()
    try:
        # Column names come from the fixed set above; values are parameterised.
        pick = conn.execute(
            f"""
            SELECT {group_col} AS album_key, album, COUNT(*) AS n,
                   MAX(CASE WHEN lower(album) = ? THEN 2
                            WHEN lower(album) IN ({placeholders}) THEN 1 ELSE 0 END) AS is_exact
            FROM items
            WHERE {where} AND album IS NOT NULL AND album != ''
            GROUP BY {group_col}
            ORDER BY is_exact DESC, length(album) ASC, n DESC
            LIMIT 1
            """,  # nosec B608
            (variants[0], *variants, *args),
        ).fetchone()
        if not pick:
            return "", []

        order_cols = [c for c in ("disc", "track") if c in columns] + ["title"]
        rows = conn.execute(
            f"""
            SELECT id, title, artist, album, path
            FROM items
            WHERE {group_col} = ?
            ORDER BY {", ".join(order_cols)}
            """,  # nosec B608
            (pick["album_key"],),
        ).fetchall()
        return pick["album"] or "", [_row_to_track(row, 0.95) for row in rows]
    finally:
        conn.close()


def _sync_is_title_or_artist(name: str, artist: str | None = None) -> bool:
    """True if `name` is exactly a track title (optionally by `artist`) or an artist.

    Guards implicit album matches: "play Weezer" means the artist, not the
    self-titled album, and an exact song title beats a same-named album.
    """
    conn = _open_beets()
    try:
        n = name.strip().lower()
        if artist:
            row = conn.execute(
                "SELECT 1 FROM items WHERE lower(title) = ? AND artist LIKE ? LIMIT 1",
                (n, f"%{artist.strip()}%"),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT 1 FROM items WHERE lower(title) = ? OR lower(artist) = ? LIMIT 1",
                (n, n),
            ).fetchone()
        return row is not None
    finally:
        conn.close()
