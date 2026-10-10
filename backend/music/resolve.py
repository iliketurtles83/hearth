"""Turn request text into a query: search terms, genres, albums/playlists, controls."""
from __future__ import annotations

from functools import lru_cache
import logging
import os
import re
from typing import Any

from music import library

log = logging.getLogger("assistant.tools.music")

MUSIC_GENRE_TREE_PATH: str = os.getenv(
    "MUSIC_GENRE_TREE_PATH",
    os.path.join(os.path.dirname(__file__), "genres.txt"),
)


# ── Intent parsing ─────────────────────────────────────────────────────────────

_CONTROL_MAP: dict[str, str] = {
    "pause": "pause",
    "stop": "stop",
    "resume": "resume",
    "continue": "resume",
    "unpause": "resume",
    "next": "next",
    "skip": "next",
    "previous": "previous",
    "prev": "previous",
    "shuffle": "shuffle",
    "clear": "clear",
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
        log.warning("music.genre_terms_missing | path=%s", MUSIC_GENRE_TREE_PATH)
        return ()

    terms: list[str] = []
    with open(MUSIC_GENRE_TREE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            terms.append(library._canonical_genre(s))

    # Longer tokens first so "progressive rock" wins over "rock".
    resolved = tuple(sorted(set(terms), key=len, reverse=True))
    log.info("music.genre_terms_loaded | path=%s terms=%d", MUSIC_GENRE_TREE_PATH, len(resolved))
    return resolved


def _normalize_music_query(text: str) -> str:
    return re.sub(
        r"\s+(?:music|songs?|tracks?|bands?|artists?)$", "", text, flags=re.IGNORECASE
    ).strip().lower()


def _resolve_genre_query(query: str) -> str | None:
    """Return a canonical genre term if query maps to the taxonomy."""
    q_norm = library._canonical_genre(_normalize_music_query(query))
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
        r"^(?:(?:i\s+(?:want|would\s+like|d\s+like)\s+(?:you\s+)?to|can\s+you|could\s+you|please)\s+)?(play(?:back)?|start\s+playing|queue|add\s+to\s+(?:the\s+)?queue|put\s+on|shuffle)\s+",
        "",
        prompt.strip(),
        flags=re.IGNORECASE,
    )
    # Remove polite fillers, pronouns, and articles.
    cleaned = re.sub(r"^(?:(?:for\s+)?me\s+)?(?:the\s+song\s+|some\s+|a\s+|any\s+)?", "", cleaned, flags=re.IGNORECASE)
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


_ALBUM_WORD = r"(?:album|record|lp)"
_WHOLE_WORD = r"(?:whole|full|entire|complete)"
_PLAYLIST_WORD = r"(?:mix\s*tape|playlist)"

_ALBUM_PREFIX_RE = re.compile(
    rf"^(?:(?:the|my|that)\s+)?(?P<whole>{_WHOLE_WORD}\s+)?{_ALBUM_WORD}\s+(?:called\s+|named\s+)?(?P<name>.+)$",
    re.IGNORECASE,
)
_ALBUM_SUFFIX_RE = re.compile(
    rf"^(?:(?:the|my|that)\s+)?(?P<whole>{_WHOLE_WORD}\s+)?(?P<name>.+?)\s+{_ALBUM_WORD}$",
    re.IGNORECASE,
)
_ALBUM_WHOLE_RE = re.compile(rf"^(?:the\s+)?(?P<whole>{_WHOLE_WORD})\s+(?P<name>.+)$", re.IGNORECASE)
_PLAYLIST_PREFIX_RE = re.compile(
    rf"^(?:(?:the|my)\s+)?{_PLAYLIST_WORD}\s+(?:called\s+|named\s+)?(?P<name>.+)$",
    re.IGNORECASE,
)
_PLAYLIST_SUFFIX_RE = re.compile(
    rf"^(?:(?:the|my)\s+)?(?P<name>.+?)\s+{_PLAYLIST_WORD}$", re.IGNORECASE
)
# "mix" is a weak cue ("a jazz mix" is genre radio), so only the suffix form counts.
_MIX_SUFFIX_RE = re.compile(r"^(?:(?:the|my)\s+)?(?P<name>.+?)\s+mix$", re.IGNORECASE)
_FILLER_NAMES = frozenset({"the", "my", "that", "this", "a", "an", "whole", "full"})
_BY_ARTIST_RE = re.compile(r"^(?P<name>.+?)\s+by\s+(?P<artist>.+)$", re.IGNORECASE)


def _parse_collection_request(query: str) -> dict[str, Any] | None:
    """Detect an explicit album or stored-playlist request in a cleaned query.

    Returns {"kind": "album", "name", "artist", "full"} (full = "the whole/
    entire …", which opts out of sampling huge compilations) or
    {"kind": "playlist", "name", "weak"} (weak = only the word "mix" was used),
    or None for ordinary track/artist/genre queries.
    """
    q = query.strip().strip("\"' .,!?")
    if not q:
        return None

    for rx in (_PLAYLIST_PREFIX_RE, _PLAYLIST_SUFFIX_RE):
        m = rx.match(q)
        if m and m.group("name").strip("\"' ").lower() not in _FILLER_NAMES:
            return {"kind": "playlist", "name": m.group("name").strip("\"' "), "weak": False}

    for rx in (_ALBUM_PREFIX_RE, _ALBUM_SUFFIX_RE, _ALBUM_WHOLE_RE):
        m = rx.match(q)
        if m and m.group("name").strip("\"' ").lower() not in _FILLER_NAMES:
            name = m.group("name").strip("\"' ")
            artist = None
            by_m = _BY_ARTIST_RE.match(name)
            if by_m:
                name, artist = by_m.group("name").strip("\"' "), by_m.group("artist").strip("\"' ")
            return {"kind": "album", "name": name, "artist": artist, "full": bool(m.group("whole"))}

    m = _MIX_SUFFIX_RE.match(q)
    if m and m.group("name").strip("\"' ").lower() not in _FILLER_NAMES:
        return {"kind": "playlist", "name": m.group("name").strip("\"' "), "weak": True}
    return None


def _strip_owner(text: str) -> str:
    return re.sub(r"^(?:my|the)\s+", "", text.strip(), flags=re.IGNORECASE)
