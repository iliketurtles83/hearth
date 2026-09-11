# Hearth — Work Plan 1

Personal working backlog. Statuses verified against the code on 2026-09-05.
Legend: `[x]` done · `[ ]` to do · `[~]` in progress/partial · `[P]` parked/deferred

## Frontend / UI

### [x] Top bar removal
- Removed `#shell-topbar` wrapper + duplicate `#shell-brand` "Hearth" label; `#sidebar-toggle-btn` is now pinned top-left of `#app-shell` (z-index 40), independent of the sidebar so it survives collapse.
- Mic is already top-right, independent of the bar — done.

### [x] Sidebar visual improvements
- `[x]` borders removed (`style.css:251-257`) · `[x]` items share background (`style.css:255`)
- `[x]` hover highlights whole item (`style.css:259-261`) · `[x]` artist + title on one line (`message.js:688`) · `[x]` tighter song row spacing (`style.css:355-358`)
- `[x]` remove the "Queue" header text (`index.html:96`, `style.css:396-402`)

### [x] Sidebar menu improvements
- `[x]` sidebar uses total height · `[x]` collapse/expand on music + memory + sessions (`message.js:_bindCollapsiblePanels`)
- `[x]` remove the overall scroll bar; size each section independently (`#sidebar-main { overflow: hidden }`; expanded panels `flex: 1 1 0`, collapsed `flex: 0 0 auto`, each `.list` scrolls internally)
- `[x]` add collapse to the Sessions/chats panel (`#sessions-collapse-btn`, "New" stays visible when collapsed)

### [x] Per-chat settings
- `[x]` vertical three-dot (kebab) menu to the right of each chat title (`message.js` renderSessions)
- `[x]` small settings popover opened from the kebab (`.session-menu`, fixed-position, closes on outside-click / Escape)
- `[x]` move delete into the menu (was an inline `×`)
- `[x]` add rename — `PATCH /chat/sessions/{id}` (`session_routes.py`) backed by a `session_titles` table in `memory.py`; `list_sessions` surfaces `title`, UI falls back to first-message preview

### [x] Music window
- artist – song on one line (`message.js:651-653`, `style.css:343-348`) — done.

### [x] Settings menu (from username)
- `[x]` clicking `#auth-username` opens a settings menu — username is now a button; fixed-position `.settings-menu` popover (same pattern as the session kebab menu: closes on outside-click / Escape), `message.js` `_ensureSettingsMenu`/`openSettingsMenu`/`closeSettingsMenu`
- `[x]` theme toggle dark/light — `:root[data-theme="light"]` palette added (`style.css`); `frontend/theme.js` applies the stored theme in `<head>` before first paint (inline script not allowed by CSP `script-src 'self'`); persisted in `localStorage['ui.theme']`
- `[x]` display reasoning on/off — menu item shares state with sidebar `#reasoning-toggle-btn` via `_setReasoningVisible` (both labels stay in sync)
- `[x]` manual beets update button — new `POST /music/beets/update` (`memory_tool_routes.py`) → `run_beets_update()` in `main.py`: runs `beet -l <db> update <root>` then `beet -l <db> import -A <root>` (same no-autotag flags as the bootstrap import). Note: `beet update` has **no** `-A` flag and never autotags; `-A` is an `import`-only flag. Errors: 409 config/PATH, 503 failure/timeout
- `[x]` logout — menu item calls `window.hearthLogout` (now exported by `auth.js`); sidebar `#logout-btn` kept
- `[x]` manual consolidation trigger — menu item → existing `POST /memory/consolidate`; refreshes the memory list and reports success/failure in chat

### [x] Frontend cleanup (found during audit)
- `[x]` removed orphaned dead refs `sidebar-section-chats` / `#sidebar-sections` (`message.js` `_setSidebarSection` + listener + bootstrap call; `style.css` `#sidebar-top`, `#sidebar-sections`, `.sidebar-section-btn` rules)

## Backend / Config

### [~] Security — extract system-specific values to .env
- `[x]` config is env-driven via `os.getenv` across `routing_config`/`memory`/`main`/`embedding_router`/`graph`/`auth`/`tools` (79 distinct vars in project source, excl. `.venv`)
- `[ ]` refresh `.env.example` — it documented ~25 of those 79 (the old "60+" note was stale; an even higher "180" was `.venv` pollution). Added only the deployment-specific delta: `MPD_HOST`/`MPD_PORT`, `CORS_ORIGINS`/`SESSION_COOKIE_SECURE`, `TTS_ENGINE`, plus a commented bare-host block for `MEMORY_DB_PATH`/`CHROMA_PATH`/`AUTH_DB_PATH`/`GRAPH_CHECKPOINT_DB_PATH`. Deliberately NOT added (stay as code defaults — safe defaults, still overridable at deploy time): `MODEL_LOCAL` (redundant alias of `OPENAI_CHAT_MODEL`), `ROUTER_EMBEDDING_ENABLED` (already documented), `OLLAMA_VISION_MODEL` (stale name → the real var is `OPENAI_VISION_MODEL`, already documented), and all router/memory/chat/auth/music/voice tuning knobs.
- `[ ]` env-ify the remaining literals: `/dev/nvidia0` (`main.py:696`), `backend/models` dir (`main.py:396,662,684`), `../frontend` (`main.py:1131`), prompt filenames (`main.py:182,187`)

## Tests

### [~] Model values from .env / shared constants
- `[ ]` import `intents.CHAT_MODEL` / `intents.CLOUD_MODEL` instead of hardcoding; fix the divergent `gemma3:4b` fallbacks (`test_graph.py:46-47`, `test_responder_modality.py:43-44`)
- `[ ]` use `ROUTING_CONFIG.router_embed_model` for the embed model instead of the literal `nomic-embed-text` (`test_embedding_router.py:102,128`)
- `[ ]` review the hardcoded `setdefault`s in `test_router.py:17-18` and `test_swap_latency.py:25-26`

## Music ops

### [ ] Document beets update
- Document the command to run when the music library changes: `docker compose exec backend sh -c 'cd /beets && beet update /music'`. Flags confirmed: `beet update` has no autotag flag (it never autotags); to also pick up *new* files run `beet import -A /music` (the `-A` no-autotag flag is import-only). The settings-menu "Update music library" button now does both (`POST /music/beets/update`). Add to README.

## Tool calling / LLM-native — PARKED

### [P] LLM-native tool-calling migration
- Parked. We want to limit LLM calls to conserve GPU resources. Revisit later.
- Plan kept at `docs/PLAN-LLM-NATIVE-TOOL-CALLING.md` (marked PARKED).
- The "write JSON for a weather tool call" note is part of this — no JSON tool schema exists yet (string dispatch only). `WEATHER_FASTPATH.json` was deleted (conflicted with this).

## Docs hygiene (#10) — resolved
- `[x]` `docs/WORK_PLAN_1.md` → this organized plan
- `[x]` `docs/DEVELOPER_NOTES.md` → raw scratch notes (source for this plan); deleted
- `[x]` `docs/MEMORY.md` → kept as canonical memory design doc (Phase 1 done / Phase 2 backlog); 2 inaccuracies fixed
- `[x]` `docs/WEATHER_FASTPATH.json` → deleted
- `[x]` `docs/PLAN-LLM-NATIVE-TOOL-CALLING.md` → kept, marked PARKED
