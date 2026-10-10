"""The music tool: run(params) -> ToolResult, dispatched as tools.dispatch("music", ...)."""
from __future__ import annotations

import asyncio
import logging
import random
import re
import sqlite3
import time
from typing import Any

import musicpd

from music import library
from music import radio
from music import resolve
from music.commands import normalize_music_action, parse_music_command
from music.players import mpd
from tools.base import ToolResult

log = logging.getLogger("assistant.tools.music")

# ── Album / stored playlist playback ───────────────────────────────────────────

async def _start_ordered_tracks(
    tracks: list[dict[str, Any]],
    action: str | None,
    extra: dict[str, Any],
    log_tag: str,
) -> ToolResult:
    """Play (replace queue) or queue `tracks` in the given order."""
    try:
        if action == "queue":
            q_result = await asyncio.to_thread(mpd._sync_queue_tracks, tracks)
            queued = q_result.get("queued_tracks", tracks)
            mpd_status: dict[str, Any] = {}
        else:
            mpd_status = await asyncio.to_thread(mpd._sync_play_tracks, tracks)
            queued = mpd_status.pop("queued_tracks", tracks)
    except FileNotFoundError as exc:
        log.warning("music.%s | library_miss=%s", log_tag, exc)
        return ToolResult.failure("No matching tracks are available in the Beets library.", retryable=False)
    except (musicpd.ConnectionError, ConnectionRefusedError, OSError) as exc:
        log.warning("music.%s | mpd_error=%s", log_tag, exc)
        return ToolResult.failure("Could not reach MPD — is it running?", retryable=True)
    return ToolResult(ok=True, data={
        "action": "queue" if action == "queue" else "play",
        "track": queued[0],
        "tracks": queued,
        "confidence": 0.95,
        "picked_from": len(queued),
        **extra,
        **mpd_status,
    })


async def _play_album(
    name: str,
    artist: str | None,
    action: str | None,
    exact: bool,
    full: bool = False,
    seed: int | None = None,
) -> ToolResult | None:
    """Play/queue an album in disc/track order; None if no album matches.

    Huge multi-artist compilations are sampled down to MUSIC_PLAYLIST_MAX_N
    tracks unless full=True ("play the whole …").
    Raises sqlite3.OperationalError if the DB is temporarily locked.
    """
    album_name, tracks = await asyncio.to_thread(library._sync_album_tracks, name, artist, exact)
    if not tracks:
        return None
    total = len(tracks)
    artists = {t["artist"] for t in tracks if t["artist"]}
    if not full:
        rng = random.Random(seed if seed is not None else int(time.time()))
        tracks = radio._sample_large_compilation(tracks, rng)
    log.info(
        "music.album | query=%r artist=%r exact=%s full=%s album=%r tracks=%d/%d artists=%d",
        name, artist, exact, full, album_name, len(tracks), total, len(artists),
    )
    return await _start_ordered_tracks(
        tracks,
        action,
        {
            "album": album_name,
            "album_artist": next(iter(artists)) if len(artists) == 1 else None,
            "album_total": total,
        },
        "album",
    )


async def _play_playlist(name: str, action: str | None) -> ToolResult:
    """Play/queue an MPD stored playlist by its exact (canonical) name."""
    try:
        result = await asyncio.to_thread(mpd._sync_load_playlist, name, action != "queue")
    except FileNotFoundError:
        return ToolResult.failure(f"The playlist '{name}' is empty.", retryable=False)
    except (musicpd.ConnectionError, ConnectionRefusedError, OSError) as exc:
        log.warning("music.playlist | name=%r mpd_error=%s", name, exc)
        return ToolResult.failure("Could not reach MPD — is it running?", retryable=True)
    queued = result.pop("queued_tracks")
    log.info("music.playlist | name=%r action=%s tracks=%d", name, action, len(queued))
    return ToolResult(ok=True, data={
        "action": "queue" if action == "queue" else "play",
        "track": queued[0],
        "tracks": queued,
        "playlist": name,
        "confidence": 1.0,
        "picked_from": len(queued),
        **result,
    })


async def _find_playlist(name: str, fuzzy: bool) -> tuple[str | None, list[str]]:
    """Return (matched stored playlist, all playlist names).

    MPD errors propagate for explicit requests; implicit probes catch them.
    """
    available = await asyncio.to_thread(mpd._sync_list_playlists)
    return mpd._match_playlist(name, available, fuzzy), available


# ── Tool entry point ───────────────────────────────────────────────────────────

async def run(params: dict[str, Any]) -> ToolResult:
    """Entry point called by tools.dispatch().

    params:
      prompt   (str)       — user message (always present)
      action   (str|None)  — explicit override: search|play|queue|control|
                             now_playing|queue_view
      query    (str|None)  — explicit search/play query
      control  (str|None)  — explicit control command: pause|resume|next|stop
      song_id  (int|None)  — Phase 8b: direct id lookup, bypass search
      artist   (str|None)  — Phase 8b: trigger artist_radio() directly
      album    (str|None)  — play/queue a whole album in track order
      playlist (str|None)  — play/queue an MPD stored playlist ("mixtape")
    """
    normalize_music_action(params)
    prompt: str = params.get("prompt", "")
    action: str | None = params.get("action")
    query: str | None = params.get("query")
    control: str | None = params.get("control")
    song_id: int | None = params.get("song_id")
    artist_param: str | None = params.get("artist")
    # Compound search params set by the deterministic pre-router.
    artist_filter: str | None = params.get("artist_filter")  # paired with query
    year_range: tuple | list | None = params.get("year_range")  # (start, end)
    album_param: str | None = params.get("album")
    playlist_param: str | None = params.get("playlist")

    log.info("music.run | prompt=%r action=%s", prompt[:80], action)

    # ── Infer action from prompt when not explicit ─────────────────────────────
    if action is None:
        parsed = parse_music_command(prompt, allow_vague=True)
        if parsed:
            action = parsed.get("action")
            control = control or parsed.get("control")
            query = query or parsed.get("query")
            year_range = year_range or parsed.get("year_range")
            artist_filter = artist_filter or parsed.get("artist_filter")
            artist_param = artist_param or parsed.get("artist")
            if "volume" in parsed:
                params["volume"] = parsed["volume"]
        else:
            if re.match(r"^(?:queue|add\s+to\s+(?:the\s+)?queue)\b", prompt.strip(), re.IGNORECASE):
                action = "queue"
            else:
                action = "play"
            by_m = re.match(
                r"^(?:play(?:back)?|queue|add\s+to\s+(?:the\s+)?queue)\s+(?P<title>.+?)\s+by\s+(?P<artist>.+)$",
                prompt.strip(),
                re.IGNORECASE,
            )
            if by_m:
                query = query or by_m.group("title").strip().strip("\"'")
                artist_filter = artist_filter or by_m.group("artist").strip().strip("\"'")
            else:
                query = query or resolve._extract_search_query(prompt)

    # ── Now playing ───────────────────────────────────────────────────────────
    if action == "now_playing":
        try:
            data = await asyncio.to_thread(mpd._sync_now_playing)
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
            data = await asyncio.to_thread(mpd._sync_queue_view)
            return ToolResult(ok=True, data=data)
        except (musicpd.ConnectionError, ConnectionRefusedError, OSError) as exc:
            log.warning("music.queue_view | mpd_error=%s", exc)
            return ToolResult.failure("Could not reach MPD — is it running?", retryable=True)
        except Exception as exc:
            log.error("music.queue_view | unexpected=%s", exc)
            return ToolResult.failure(str(exc), retryable=False)

    # ── Outputs view ──────────────────────────────────────────────────────────
    if action == "outputs":
        try:
            data = await asyncio.to_thread(mpd._sync_get_outputs)
            return ToolResult(ok=True, data={"outputs": data})
        except (musicpd.ConnectionError, ConnectionRefusedError, OSError) as exc:
            log.warning("music.outputs | mpd_error=%s", exc)
            return ToolResult.failure("Could not reach MPD — is it running?", retryable=True)
        except Exception as exc:
            log.error("music.outputs | unexpected=%s", exc)
            return ToolResult.failure(str(exc), retryable=False)

    # ── Select output ─────────────────────────────────────────────────────────
    if action == "select_output":
        target_id = params.get("output_id")
        if target_id is None:
            return ToolResult.failure("output_id is required.", retryable=False)
        mode = str(params.get("mode", "exclusive"))
        try:
            data = await asyncio.to_thread(mpd._sync_set_output, str(target_id), mode)
            return ToolResult(ok=True, data={"action": "select_output", "ok": True, "outputs": data})
        except (musicpd.ConnectionError, ConnectionRefusedError, OSError) as exc:
            log.warning("music.select_output | mpd_error=%s", exc)
            return ToolResult.failure("Could not reach MPD — is it running?", retryable=True)
        except Exception as exc:
            log.error("music.select_output | unexpected=%s", exc)
            return ToolResult.failure(str(exc), retryable=False)

    # ── Playback control ──────────────────────────────────────────────────────
    if action == "control":
        cmd = control or resolve._extract_control_action(prompt)
        if cmd == "set_volume":
            raw_volume = params.get("volume")
            if raw_volume is None:
                return ToolResult.failure("Volume value is required.", retryable=False)
            try:
                applied = await asyncio.to_thread(mpd._sync_set_volume, int(raw_volume))
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
                await asyncio.to_thread(mpd._sync_play_pos, pos)
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
            await asyncio.to_thread(mpd._sync_control, cmd)
            return ToolResult(ok=True, data={"action": cmd, "ok": True})
        except (musicpd.ConnectionError, ConnectionRefusedError, OSError) as exc:
            log.warning("music.control | action=%s mpd_error=%s", cmd, exc)
            return ToolResult.failure("Could not reach MPD — is it running?", retryable=True)
        except Exception as exc:
            log.error("music.control | action=%s unexpected=%s", cmd, exc)
            err_msg = str(exc)
            if "not playing" in err_msg.lower():
                return ToolResult.failure("Nothing is currently playing.", retryable=False)
            return ToolResult.failure(err_msg, retryable=False)

    # ── Phase 8b: direct song_id resolution (rec engine path) ─────────────────
    if song_id is not None:
        try:
            track = await asyncio.to_thread(library._sync_get_by_id, song_id)
        except sqlite3.OperationalError:
            return ToolResult.failure(
                "Music library database is temporarily unavailable. Please retry.",
                retryable=True,
            )
        if not track:
            return ToolResult.failure(f"Song ID {song_id} not found in library.", retryable=False)
        try:
            if action == "queue":
                await asyncio.to_thread(mpd._sync_queue, track["url"])
                return ToolResult(ok=True, data={"action": "queue", "track": track, "tracks": None, "confidence": 1.0, "picked_from": 1})
            else:
                mpd_status = await asyncio.to_thread(mpd._sync_play, track["url"])
                return ToolResult(ok=True, data={"action": "play", "track": track, "tracks": None, "confidence": 1.0, "picked_from": 1, **mpd_status})
        except (musicpd.ConnectionError, ConnectionRefusedError, OSError) as exc:
            log.warning("music.song_id_play | mpd_error=%s", exc)
            return ToolResult.failure("Could not reach MPD — is it running?", retryable=True)

    # ── Explicit album / stored playlist ("mixtape") requests ─────────────────
    collection: dict[str, Any] | None = None
    if playlist_param:
        collection = {"kind": "playlist", "name": playlist_param, "weak": False}
    elif album_param:
        full = bool(re.search(rf"\b{resolve._WHOLE_WORD}\b", prompt, re.IGNORECASE))
        collection = {"kind": "album", "name": album_param, "artist": None, "full": full}
    elif action in ("play", "queue") and year_range is None:
        collection = resolve._parse_collection_request(query or resolve._extract_search_query(prompt))
        # "<title> by <artist>" requests are never stored playlists.
        if collection and collection["kind"] == "playlist" and artist_filter:
            collection = None

    if collection and collection["kind"] == "album":
        album_artist = collection.get("artist") or artist_filter or artist_param
        try:
            album_result = await _play_album(
                collection["name"], album_artist, action, exact=False, full=collection.get("full", False)
            )
        except sqlite3.OperationalError:
            return ToolResult.failure(
                "Music library database is temporarily unavailable. Please retry.",
                retryable=True,
            )
        if album_result is not None:
            return album_result
        by = f" by '{album_artist}'" if album_artist else ""
        return ToolResult.failure(
            f"No album matching '{collection['name']}'{by} in the library.", retryable=False
        )

    if collection and collection["kind"] == "playlist":
        try:
            match, available = await _find_playlist(collection["name"], fuzzy=True)
        except Exception as exc:
            if not collection["weak"]:
                log.warning("music.playlist_list | mpd_error=%s", exc)
                return ToolResult.failure("Could not reach MPD — is it running?", retryable=True)
            match, available = None, []
        if match:
            return await _play_playlist(match, action)
        if not collection["weak"]:
            names = ", ".join(sorted(available)) if available else "none saved yet"
            return ToolResult.failure(
                f"No playlist called '{collection['name']}'. Available playlists: {names}.",
                retryable=False,
            )
        # Weak "<x> mix" with no such playlist: "a jazz mix" → genre radio for "jazz".
        if resolve._resolve_genre_query(collection["name"]):
            query = collection["name"]

    # ── Phase 8b: artist radio (called directly from 8b fallback) ─────────────
    if artist_param:
        try:
            tracks = await asyncio.to_thread(radio.artist_radio, artist_param)
        except sqlite3.OperationalError:
            return ToolResult.failure(
                "Music library database is temporarily unavailable. Please retry.",
                retryable=True,
            )
        if not tracks:
            return ToolResult.failure(
                f"No songs found for artist '{artist_param}'.", retryable=False
            )
        try:
            if action == "queue":
                q_result = await asyncio.to_thread(mpd._sync_queue_tracks, tracks)
                queued = q_result.get("queued_tracks", tracks)
                mpd_status = {}
            else:
                mpd_status = await asyncio.to_thread(mpd._sync_play_tracks, tracks)
                queued = mpd_status.pop("queued_tracks", tracks)
        except FileNotFoundError as exc:
            log.warning("music.artist_radio | library_miss=%s", exc)
            return ToolResult.failure("No matching tracks are available in the Beets library.", retryable=False)
        except (musicpd.ConnectionError, ConnectionRefusedError, OSError) as exc:
            log.warning("music.artist_radio | mpd_error=%s", exc)
            return ToolResult.failure("Could not reach MPD — is it running?", retryable=True)
        return ToolResult(ok=True, data={
            "action": "queue" if action == "queue" else "play",
            "track": queued[0],
            "tracks": queued,
            "confidence": 1.0,
            "picked_from": len(queued),
            **mpd_status,
        })

    # ── Compound title + artist search ("title by artist") ────────────────────
    if artist_filter and query:
        # "Kind of Blue by Miles Davis": an exact album name wins unless a
        # track by that artist has exactly that title.
        try:
            if not await asyncio.to_thread(library._sync_is_title_or_artist, query, artist_filter):
                album_result = await _play_album(query, artist_filter, action, exact=True)
                if album_result is not None:
                    return album_result
        except sqlite3.Error as exc:
            log.info("music.album_probe_skipped | query=%r error=%s", query, exc)
        try:
            results = await asyncio.to_thread(library._sync_search_by_title_artist, query, artist_filter)
        except sqlite3.OperationalError:
            return ToolResult.failure(
                "Music library database is temporarily unavailable. Please retry.",
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
                    await asyncio.to_thread(mpd._sync_queue, top["url"])
                    return ToolResult(ok=True, data={"action": "queue", "track": top, "tracks": None, "confidence": confidence, "picked_from": len(results)})
                else:
                    mpd_status = await asyncio.to_thread(mpd._sync_play, top["url"])
                    return ToolResult(ok=True, data={"action": "play", "track": top, "tracks": None, "confidence": confidence, "picked_from": len(results), **mpd_status})
            except FileNotFoundError as exc:
                log.warning("music.title_artist_play | library_miss=%s", exc)
                return ToolResult.failure("No matching tracks are available in the Beets library.", retryable=False)
            except (musicpd.ConnectionError, ConnectionRefusedError, OSError) as exc:
                log.warning("music.title_artist_play | mpd_error=%s", exc)
                return ToolResult.failure("Could not reach MPD — is it running?", retryable=True)
        # No title+artist match — fall back to artist radio.
        log.info("music.title_artist_miss | title=%r artist_filter=%r trying artist radio", query, artist_filter)
        try:
            tracks = await asyncio.to_thread(radio.artist_radio, artist_filter)
        except sqlite3.OperationalError:
            return ToolResult.failure(
                "Music library database is temporarily unavailable. Please retry.",
                retryable=True,
            )
        if tracks:
            try:
                if action == "queue":
                    q_result = await asyncio.to_thread(mpd._sync_queue_tracks, tracks)
                    queued = q_result.get("queued_tracks", tracks)
                    mpd_status = {}
                else:
                    mpd_status = await asyncio.to_thread(mpd._sync_play_tracks, tracks)
                    queued = mpd_status.pop("queued_tracks", tracks)
            except FileNotFoundError as exc:
                log.warning("music.title_artist_radio_fallback | library_miss=%s", exc)
                return ToolResult.failure("No matching tracks are available in the Beets library.", retryable=False)
            except (musicpd.ConnectionError, ConnectionRefusedError, OSError) as exc:
                log.warning("music.title_artist_radio_fallback | mpd_error=%s", exc)
                return ToolResult.failure("Could not reach MPD — is it running?", retryable=True)
            return ToolResult(ok=True, data={
                "action": "queue" if action == "queue" else "play",
                "track": queued[0],
                "tracks": queued,
                "confidence": 0.7,
                "picked_from": len(queued),
                **mpd_status,
            })
        return ToolResult.failure(
            f"No tracks found for '{query}' by '{artist_filter}'.", retryable=False
        )

    # ── Year / decade range search ─────────────────────────────────────────────
    if year_range is not None:
        yr_start, yr_end = int(year_range[0]), int(year_range[1])
        try:
            all_year_tracks = await asyncio.to_thread(library._sync_search_by_year_range, yr_start, yr_end)
        except sqlite3.OperationalError:
            return ToolResult.failure(
                "Music library database is temporarily unavailable. Please retry.",
                retryable=True,
            )
        if not all_year_tracks:
            period = f"{yr_start}s" if yr_start != yr_end else str(yr_start)
            return ToolResult.failure(f"No tracks found from {period}.", retryable=False)
        rng = random.Random(int(time.time()))
        n_pick = radio._playlist_pick_count(len(all_year_tracks), requested_n=None)
        tracks = rng.sample(all_year_tracks, n_pick)
        log.info(
            "music.year_range | yr_start=%d yr_end=%d pool=%d picking=%d",
            yr_start, yr_end, len(all_year_tracks), n_pick,
        )
        try:
            if action == "queue":
                q_result = await asyncio.to_thread(mpd._sync_queue_tracks, tracks)
                queued = q_result.get("queued_tracks", tracks)
                mpd_status = {}
            else:
                mpd_status = await asyncio.to_thread(mpd._sync_play_tracks, tracks)
                queued = mpd_status.pop("queued_tracks", tracks)
        except FileNotFoundError as exc:
            log.warning("music.year_range_play | library_miss=%s", exc)
            return ToolResult.failure("No matching tracks are available in the Beets library.", retryable=False)
        except (musicpd.ConnectionError, ConnectionRefusedError, OSError) as exc:
            log.warning("music.year_range_play | mpd_error=%s", exc)
            return ToolResult.failure("Could not reach MPD — is it running?", retryable=True)
        return ToolResult(ok=True, data={
            "action": "queue" if action == "queue" else "play",
            "track": queued[0],
            "tracks": queued,
            "confidence": 0.9,
            "picked_from": len(queued),
            **mpd_status,
        })

    # ── Search (explicit action="search") ─────────────────────────────────────
    if action == "search":
        q = query or resolve._extract_search_query(prompt)
        if not q:
            return ToolResult.failure("No search query provided.", retryable=False)
        try:
            results = await asyncio.to_thread(library._sync_search, q)
        except sqlite3.OperationalError:
            return ToolResult.failure(
                "Music library database is temporarily unavailable. Please retry.",
                retryable=True,
            )
        return ToolResult(ok=True, data={"query": q, "results": results, "total": len(results)})

    # ── Play / Queue (default paths) ──────────────────────────────────────────
    q = query or resolve._extract_search_query(prompt)
    if not q:
        return ToolResult.failure("No track or artist specified.", retryable=False)

    # A bare stored-playlist name ("play chill") beats genre/DB matching.
    try:
        playlist_match, _ = await _find_playlist(resolve._strip_owner(q), fuzzy=False)
    except Exception as exc:
        log.info("music.playlist_probe_skipped | query=%r error=%s", q, exc)
        playlist_match = None
    if playlist_match:
        return await _play_playlist(playlist_match, action)

    # Genre-first resolver path for ambiguous playback requests.
    genre_match = resolve._resolve_genre_query(q)

    # Exact album name ("play Abbey Road") — unless it is also an artist or a
    # track title, so self-titled albums keep meaning the artist.
    if not genre_match:
        try:
            if not await asyncio.to_thread(library._sync_is_title_or_artist, q):
                album_result = await _play_album(q, None, action, exact=True)
                if album_result is not None:
                    return album_result
        except sqlite3.Error as exc:
            log.info("music.album_probe_skipped | query=%r error=%s", q, exc)

    if not genre_match and action != "queue":
        # Safety-net: if taxonomy matching misses, probe DB genre column directly.
        try:
            probe_genre_tracks = await asyncio.to_thread(library._sync_genre_songs, q)
        except sqlite3.OperationalError:
            return ToolResult.failure(
                "Music library database is temporarily unavailable. Please retry.",
                retryable=True,
            )
        if probe_genre_tracks:
            genre_match = q
            log.info(
                "music.play | genre_db_probe_hit query=%r matches=%d",
                q,
                len(probe_genre_tracks),
            )
    if genre_match and action != "queue":
        log.info("music.play | genre_first query=%r genre=%r", q, genre_match)
        try:
            tracks = await asyncio.to_thread(radio.genre_radio, genre_match)
        except sqlite3.OperationalError:
            return ToolResult.failure(
                "Music library database is temporarily unavailable. Please retry.",
                retryable=True,
            )
        if tracks:
            try:
                mpd_status = await asyncio.to_thread(mpd._sync_play_tracks, tracks)
                queued = mpd_status.pop("queued_tracks", tracks)
            except FileNotFoundError as exc:
                log.warning("music.genre_radio | library_miss=%s", exc)
                return ToolResult.failure("No matching tracks are available in the Beets library.", retryable=False)
            except (musicpd.ConnectionError, ConnectionRefusedError, OSError) as exc:
                log.warning("music.genre_radio | mpd_error=%s", exc)
                return ToolResult.failure("Could not reach MPD — is it running?", retryable=True)
            return ToolResult(ok=True, data={
                "action": "play",
                "track": queued[0],
                "tracks": queued,
                "genre": genre_match,
                "confidence": 0.9,
                "picked_from": len(queued),
                **mpd_status,
            })
        # Genre returned nothing — for decade terms, fall back to year column.
        _decade_m = re.match(r"^(\d{2})s$", genre_match)
        if _decade_m:
            _d = int(_decade_m.group(1))
            _yr = (2000 + _d) if _d < 30 else (1900 + _d)
            _yr_start, _yr_end = _yr, _yr + 9
            log.info(
                "music.play | genre_miss decade=%r falling_back_to_year range=%d-%d",
                genre_match, _yr_start, _yr_end,
            )
            try:
                _year_tracks = await asyncio.to_thread(library._sync_search_by_year_range, _yr_start, _yr_end)
            except sqlite3.OperationalError:
                return ToolResult.failure(
                    "Music library database is temporarily unavailable. Please retry.",
                )
            if _year_tracks:
                _rng = random.Random(int(time.time()))
                _n_pick = radio._playlist_pick_count(len(_year_tracks), requested_n=None)
                tracks = _rng.sample(_year_tracks, _n_pick)
                try:
                    mpd_status = await asyncio.to_thread(mpd._sync_play_tracks, tracks)
                    queued = mpd_status.pop("queued_tracks", tracks)
                except FileNotFoundError as exc:
                    log.warning("music.decade_year_fallback | library_miss=%s", exc)
                    return ToolResult.failure("No matching tracks are available in the Beets library.", retryable=False)
                except (musicpd.ConnectionError, ConnectionRefusedError, OSError) as exc:
                    log.warning("music.decade_year_fallback | mpd_error=%s", exc)
                    return ToolResult.failure("Could not reach MPD — is it running?", retryable=True)
                return ToolResult(ok=True, data={
                    "action": "play",
                    "track": queued[0],
                    "tracks": queued,
                    "confidence": 0.9,
                    "picked_from": len(queued),
                    **mpd_status,
                })

    try:
        results = await asyncio.to_thread(library._sync_search, q)
    except sqlite3.OperationalError:
        return ToolResult.failure(
            "Music library database is temporarily unavailable. Please retry.",
            retryable=True,
        )

    if not results:
        # No exact/LIKE match — try artist radio as automatic fallback.
        log.info("music.play | no_results query=%r trying artist radio", q)
        try:
            tracks = await asyncio.to_thread(radio.artist_radio, q)
        except sqlite3.OperationalError:
            return ToolResult.failure(
                "Music library database is temporarily unavailable. Please retry.",
                retryable=True,
            )
        if not tracks:
            return ToolResult.failure(
                f"No tracks or artists found matching '{q}'.", retryable=False
            )
        try:
            if action == "queue":
                q_result = await asyncio.to_thread(mpd._sync_queue_tracks, tracks)
                queued = q_result.get("queued_tracks", tracks)
                mpd_status = {}
            else:
                mpd_status = await asyncio.to_thread(mpd._sync_play_tracks, tracks)
                queued = mpd_status.pop("queued_tracks", tracks)
        except FileNotFoundError as exc:
            log.warning("music.artist_radio_fallback | library_miss=%s", exc)
            return ToolResult.failure("No matching tracks are available in the Beets library.", retryable=False)
        except (musicpd.ConnectionError, ConnectionRefusedError, OSError) as exc:
            log.warning("music.artist_radio_fallback | mpd_error=%s", exc)
            return ToolResult.failure("Could not reach MPD — is it running?", retryable=True)
        return ToolResult(ok=True, data={
            "action": "queue" if action == "queue" else "play",
            "track": queued[0],
            "tracks": queued,
            "confidence": 0.7,
            "picked_from": len(queued),
            **mpd_status,
        })

    # Artist-radio heuristic: if the query looks like an artist/genre name rather
    # than a song title, use artist_radio() to queue multiple tracks instead of
    # single-picking the top LIKE result.
    #
    # Criteria (both must be true):
    #   1. The normalised query doesn't appear in the top result's title
    #      (i.e. no title match → the LIKE hit was on artist, not title).
    #   2. The normalised query does appear in the top result's artist field
    #      (confirms it's an artist search, not e.g. an album search).
    #
    # Strip common genre suffixes ("music", "songs", "tracks") before comparing
    # so "classical music" matches artist "Classical".
    _q_norm = resolve._normalize_music_query(q)
    _title_miss = _q_norm not in results[0]["title"].lower()
    _artist_hit = _q_norm in results[0]["artist"].lower()
    log.info(
        "music.play | heuristic_check query_norm=%r title_miss=%s artist_hit=%s action=%s | top_title=%r top_artist=%r",
        _q_norm, _title_miss, _artist_hit, action, results[0]["title"].lower(), results[0]["artist"].lower(),
    )
    if _title_miss and _artist_hit and action != "queue":
        log.info(
            "music.play | artist_radio_heuristic query=%r artist=%r candidates=%d",
            q, results[0]["artist"], len(results),
        )
        try:
            tracks = await asyncio.to_thread(radio.artist_radio, q)
        except sqlite3.OperationalError:
            return ToolResult.failure(
                "Music library database is temporarily unavailable. Please retry.",
                retryable=True,
            )
        if tracks:
            try:
                mpd_status = await asyncio.to_thread(mpd._sync_play_tracks, tracks)
                queued = mpd_status.pop("queued_tracks", tracks)
            except FileNotFoundError as exc:
                log.warning("music.artist_radio_heuristic | library_miss=%s", exc)
                return ToolResult.failure("No matching tracks are available in the Beets library.", retryable=False)
            except (musicpd.ConnectionError, ConnectionRefusedError, OSError) as exc:
                log.warning("music.artist_radio_heuristic | mpd_error=%s", exc)
                return ToolResult.failure("Could not reach MPD — is it running?", retryable=True)
            return ToolResult(ok=True, data={
                "action": "play",
                "track": queued[0],
                "tracks": queued,
                "confidence": 0.85,
                "picked_from": len(queued),
                **mpd_status,
            })
        # artist_radio returned nothing (shouldn't happen if LIKE found results,
        # but fall through to single-pick as safety net).

    # Multi-word artist intent rescue: title matches like
    # "Michael Jackson - remix title" can suppress the heuristic above even when
    # there are clear artist-field matches in the result set.
    _artist_matches = [r for r in results if _q_norm in r["artist"].lower()]
    _multi_word_query = " " in _q_norm.strip()
    if action != "queue" and _multi_word_query and _artist_matches:
        preferred_artist = _artist_matches[0]["artist"]
        log.info(
            "music.play | artist_radio_multiword query=%r preferred_artist=%r artist_matches=%d candidates=%d",
            q,
            preferred_artist,
            len(_artist_matches),
            len(results),
        )
        try:
            tracks = await asyncio.to_thread(radio.artist_radio, preferred_artist)
        except sqlite3.OperationalError:
            return ToolResult.failure(
                "Music library database is temporarily unavailable. Please retry.",
                retryable=True,
            )
        if tracks:
            try:
                mpd_status = await asyncio.to_thread(mpd._sync_play_tracks, tracks)
                queued = mpd_status.pop("queued_tracks", tracks)
            except FileNotFoundError as exc:
                log.warning("music.artist_radio_multiword | library_miss=%s", exc)
                return ToolResult.failure("No matching tracks are available in the Beets library.", retryable=False)
            except (musicpd.ConnectionError, ConnectionRefusedError, OSError) as exc:
                log.warning("music.artist_radio_multiword | mpd_error=%s", exc)
                return ToolResult.failure("Could not reach MPD — is it running?", retryable=True)
            return ToolResult(ok=True, data={
                "action": "play",
                "track": queued[0],
                "tracks": queued,
                "confidence": 0.86,
                "picked_from": len(queued),
                **mpd_status,
            })

    # Auto-pick with availability fallback:
    # try ranked candidates in order, skipping stale Strawberry rows that MPD
    # no longer serves.
    first_missing: Exception | None = None
    for i, candidate in enumerate(results, start=1):
        confidence = candidate.get("score", 0.9)
        try:
            if action == "queue":
                await asyncio.to_thread(mpd._sync_queue, candidate["url"])
                log.info(
                    "music.auto_pick | query=%r picked=%r artist=%r confidence=%.3f candidates=%d index=%d",
                    q,
                    candidate["title"],
                    candidate["artist"],
                    confidence,
                    len(results),
                    i,
                )
                return ToolResult(ok=True, data={
                    "action": "queue", "track": candidate, "tracks": None,
                    "confidence": confidence, "picked_from": len(results),
                })
            mpd_status = await asyncio.to_thread(mpd._sync_play, candidate["url"])
            log.info(
                "music.auto_pick | query=%r picked=%r artist=%r confidence=%.3f candidates=%d index=%d",
                q,
                candidate["title"],
                candidate["artist"],
                confidence,
                len(results),
                i,
            )
            return ToolResult(ok=True, data={
                "action": "play", "track": candidate, "tracks": None,
                "confidence": confidence, "picked_from": len(results), **mpd_status,
            })
        except FileNotFoundError as exc:
            if first_missing is None:
                first_missing = exc
            log.warning(
                "music.auto_pick_skip_missing | query=%r index=%d title=%r error=%s",
                q,
                i,
                candidate.get("title", ""),
                exc,
            )
            continue
        except (musicpd.ConnectionError, ConnectionRefusedError, OSError) as exc:
            log.warning("music.play | mpd_error=%s", exc)
            return ToolResult.failure("Could not reach MPD — is it running?", retryable=True)
        except Exception as exc:
            log.error("music.play | unexpected=%s", exc)
            return ToolResult.failure(str(exc), retryable=False)

    if first_missing is not None:
        log.warning("music.play | library_miss=%s", first_missing)
        return ToolResult.failure("No matching tracks are available in the Beets library.", retryable=False)
    return ToolResult.failure(f"No tracks or artists found matching '{q}'.", retryable=False)
