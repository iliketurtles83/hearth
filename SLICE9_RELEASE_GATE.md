# Slice 9 Release Gate

Date: 2026-04-29

## Regression Snapshot

Command run:

- `cd backend && .venv/bin/python -m pytest tests/test_chat_sessions.py tests/test_router.py tests/test_weather.py tests/test_memory_isolation.py tests/test_graph.py tests/test_music.py -q`

Result:

- 131 passed
- 18 warnings
- Warning class: Chroma deprecation warnings about EmbeddingFunction `name()`
- No test failures or regressions in focused suite

## Manual UX Matrix

Status key:

- PASS: verified in this cycle
- PENDING: requires interactive browser/device validation

Checks:

- PENDING: Sidebar desktop behavior
  - Chats / Projects sections visible at top
  - Account + sign-out controls anchored at bottom-left
  - Session rows render as one-line titles
- PENDING: Sidebar mobile behavior
  - Sidebar slide-in and overlay close behavior
  - Account block remains reachable at bottom
- PENDING: Chat readability
  - Serif transcript and larger text remain readable on desktop and mobile
  - No overflow/cropping in user and assistant bubbles
- PENDING: Session flow
  - New creates one session per click
  - Selecting a session does not reorder list until a new message is sent
- PENDING: Music controls and hierarchy
  - Now playing area visually distinct from queue
  - Active queue track indicator updates correctly
  - Queue click jumps playback to selected track
  - Volume slider updates MPD volume and reflects state after refresh

## Env Migration Notes and Defaults

New/updated environment variables introduced in recent slices:

- `MUSIC_PLAYLIST_MIN_N` (default: `12`)
  - Lower bound for adaptive multi-track queue size.
- `MUSIC_PLAYLIST_MAX_N` (default: `24`)
  - Upper bound for adaptive multi-track queue size.

Music API behavior changes:

- `POST /music/control` now supports:
  - `play_pos`
  - `set_volume`
- `GET /music/now_playing` now includes:
  - `pos` (current queue position)
  - `volume` (current volume 0-100)

Compatibility notes:

- Existing clients using `pause|resume|next|stop` continue to work unchanged.
- Clients can ignore new `now_playing` fields safely.
- Volume requests are clamped server-side to 0-100.

## Release Recommendation

- Backend release gate: PASS (focused regression green).
- Frontend release gate: CONDITIONAL (manual UX matrix still pending signoff).
