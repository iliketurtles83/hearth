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

### [ ] Settings menu (from username)
- `[ ]` clicking `#auth-username` (`index.html:120`) opens a settings menu
- `[ ]` theme toggle dark/light (only a dark `:root` exists today, `style.css:3-18`)
- `[ ]` display reasoning on/off — exists as sidebar `#reasoning-toggle-btn`; surface it here too
- `[ ]` manual beets update button — does not exist
- `[ ]` logout — exists as sidebar `#logout-btn`; surface it here too
- `[ ]` manual consolidation trigger — does not exist (only a read-only status label)

### [ ] Frontend cleanup (found during audit)
- `[ ]` remove orphaned dead refs `sidebar-section-chats` / `#sidebar-sections` (`message.js:9`, `style.css:71-98`)

## Backend / Config

### [~] Security — extract system-specific values to .env
- `[x]` config is env-driven via `os.getenv` across `routing_config`/`memory`/`main`/`embedding_router`/`graph`/`auth`/`tools`
- `[ ]` refresh `.env.example` — documents 17 vars but code reads 60+ (add `MEMORY_DB_PATH`, `CHROMA_PATH`, `AUTH_DB_PATH`, `GRAPH_CHECKPOINT_DB_PATH`, `MODEL_LOCAL`, `OLLAMA_VISION_MODEL`, `CHAT_TOKEN_BUDGET`, `CHAT_MAX_TURNS`, `ROUTE_CONFIDENCE_THRESHOLD`, `ROUTER_EMBEDDING_ENABLED`, `MEMORY_TOP_N`, `MPD_HOST`/`MPD_PORT`, `WEATHER_UNITS`, `TTS_*`, `WAKEWORD_*`, `WHISPER_*`, `CORS_ORIGINS`, `SESSION_COOKIE_SECURE`)
- `[ ]` env-ify the remaining literals: `/dev/nvidia0` (`main.py:621`), `backend/models` dir (`main.py:334,587`), `../frontend` (`main.py:1035`), prompt filenames (`main.py:171`)

## Tests

### [~] Model values from .env / shared constants
- `[ ]` import `intents.CHAT_MODEL` / `intents.CLOUD_MODEL` instead of hardcoding; fix the divergent `gemma3:4b` fallbacks (`test_graph.py:46-47`, `test_responder_modality.py:43-44`)
- `[ ]` use `ROUTING_CONFIG.router_embed_model` for the embed model instead of the literal `nomic-embed-text` (`test_embedding_router.py:102,128`)
- `[ ]` review the hardcoded `setdefault`s in `test_router.py:17-18` and `test_swap_latency.py:25-26`

## Music ops

### [ ] Document beets update
- Document the command to run when the music library changes: `docker compose exec backend sh -c 'cd /beets && beet update /music'` (confirm flags, e.g. `-A` for no auto-tagging). Add to README.

## Tool calling / LLM-native — PARKED

### [P] LLM-native tool-calling migration
- Parked. We want to limit LLM calls to conserve GPU resources. Revisit later.
- Plan kept at `docs/PLAN-LLM-NATIVE-TOOL-CALLING.md` (marked PARKED).
- The "write JSON for a weather tool call" note is part of this — no JSON tool schema exists yet (string dispatch only). `WEATHER_FASTPATH.json` was deleted (conflicted with this).

## Docs hygiene (#10) — resolved
- `[x]` `docs/WORK_PLAN_1.md` → this organized plan
- `[ ]` `docs/DEVELOPER_NOTES.md` → raw scratch notes (source for this plan); kept for now, remove once this supersedes it
- `[x]` `docs/MEMORY.md` → kept as canonical memory design doc (Phase 1 done / Phase 2 backlog); 2 inaccuracies fixed
- `[x]` `docs/WEATHER_FASTPATH.json` → deleted
- `[x]` `docs/PLAN-LLM-NATIVE-TOOL-CALLING.md` → kept, marked PARKED
