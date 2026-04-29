# Product Battle Plan (Developer Notes -> Slices)

Date: 2026-04-29

This plan converts the developer notes into execution slices with dependencies, acceptance criteria, and verification.

## Execution Status

- [x] Slice 1 - Backend Config Hardening
- [x] Slice 2 - Test Config Alignment
- [x] Slice 3 - Music Resolver V1 (Genre-First Ambiguity Strategy)
- [x] Slice 4 - Playlist Length and Diversity Controls
- [x] Slice 5 - Session List UX Behavior
- [x] Slice 6 - Sidebar Information Architecture
- [x] Slice 7 - Typography and Readability Upgrade
- [x] Slice 8 - Music Panel UX Refresh
- [ ] Slice 9 - Final Polish and Release Gate (manual UX signoff pending)

## Product Direction (Locked)

- UX target: claude.ai-like experience.
- Sidebar future structure:
  - Top: `Chats` and `Projects` sections.
  - Middle: chat session list, one-line rows only.
  - Bottom-left: user identity + sign out controls.
- Typography:
  - Left sidebar: modern sans-serif.
  - Chat transcript (user + assistant messages): mild serif similar to Claude's reading style.
  - Chat message font size: larger than current baseline for readability.

## Operating Constraints

- Backend-first execution priority.
- Keep deterministic music fast-path behavior intact while improving ambiguity handling.
- Each slice must have test or verification criteria before marked done.
- Preserve existing auth/session boundaries and graph/checkpoint behavior.

## Slice Map

## Phase 0 - Baseline and Guardrails

Goal: Freeze behavior before changes.

Scope:
- Capture baseline focused regression run and known caveats.
- Confirm no pending breakages in chat/router/weather/music/memory/graph tests.

Acceptance:
- Focused suite passes from clean state.
- Any known warnings/failures are documented with owner and follow-up slice.

Verification:
- `cd backend && .venv/bin/python -m pytest tests/test_chat_sessions.py tests/test_router.py tests/test_weather.py tests/test_memory_isolation.py tests/test_graph.py tests/test_music.py -q`

## Slice 1 - Backend Config Hardening

Goal: Move system-specific values from code defaults into env-first config surfaces.

Scope:
- Centralize model/env/path-related runtime configuration.
- Reduce hardcoded machine-specific assumptions in runtime modules.
- Keep backward-compatible defaults for local dev.

Primary files:
- `backend/main.py`
- `backend/router.py`
- `backend/tools/music.py`

Acceptance:
- Runtime behavior is unchanged under existing `.env`.
- Environment override points are explicit and documented.
- No model/path literals required in runtime logic where env is intended.

Verification:
- Existing focused test suite still passes.

## Slice 2 - Test Config Alignment

Goal: Remove brittle hardcoded model values in tests.

Scope:
- Replace explicit model literals in tests with config-derived values/fixtures.
- Ensure tests remain deterministic by isolating env setup in each test module.

Primary files:
- `backend/tests/test_router.py`
- `backend/tests/test_graph.py`
- Any other backend test relying on model literals

Acceptance:
- Tests do not fail due to model-name changes in env.
- Test setup clearly states required env assumptions.

Verification:
- Router + graph + chat session tests pass with overridden model env values.

## Slice 3 - Music Resolver V1 (Genre-First Ambiguity Strategy)

Goal: Improve ambiguous playback requests like `play michael jackson` with a deterministic resolver.

Strategy (ordered):
1. Genre-tree match first.
2. Artist match second.
3. Existing search fallback.

Scope:
- Add candidate scoring for genre and artist paths.
- Add confidence logging for resolver decisions.
- Keep existing explicit command handling and tool contracts.

Primary files:
- `backend/tools/music.py`
- `genre-tree.txt`
- `backend/tests/test_music.py`

Acceptance:
- Ambiguous prompts route consistently via genre-first policy.
- Resolver logs selected path + confidence.
- No regression in explicit title/artist commands.

Verification:
- New and existing music tests pass, including fallback behavior.

## Slice 4 - Playlist Length and Diversity Controls

Goal: Prevent short queues for artist/genre playback.

Scope:
- Add configurable target queue sizing (floor and ceiling).
- Enforce dedupe and broaden candidate sampling when pool allows.
- Preserve deterministic behavior for seeded test paths.

Primary files:
- `backend/tools/music.py`
- `backend/tests/test_music.py`

Acceptance:
- `play <artist>` and genre-oriented prompts produce sufficiently long queues when library size permits.
- Queue generation degrades gracefully on small pools.

Verification:
- Tests cover minimum queue length policy and dedupe guarantees.

## Slice 5 - Session List UX Behavior

Goal: Make chat sessions feel claude.ai-like in flow and density.

Scope:
- Ensure `New` creates one new session per click and avoids accidental duplicate creation race.
- Session rows become one-line title only (no date text in row).
- Sort by `updated_at` descending.
- `updated_at` changes only when a chat message is added, not on simple selection.

Primary files:
- `frontend/message.js`
- `frontend/style.css`
- `backend/main.py` (if timestamp semantics need backend adjustment)

Acceptance:
- Session list is stable, compact, and sorted correctly.
- Selection does not mutate activity ordering unexpectedly.

Verification:
- Manual interaction matrix for create/select/send/delete.
- Add/adjust tests where feasible for timestamp semantics.

## Slice 6 - Sidebar Information Architecture

Goal: Establish the future sidebar skeleton now.

Scope:
- Introduce top sections for `Chats` and `Projects` (projects may be placeholder/non-functional initially).
- Move user/sign-out controls to bottom-left sidebar account block.
- Keep sidebar hide/show behavior and mobile interactions stable.

Primary files:
- `frontend/index.html`
- `frontend/message.js`
- `frontend/auth.js`
- `frontend/style.css`

Acceptance:
- Sidebar reflects new section structure.
- Account actions are bottom-left and accessible.
- Header no longer carries primary account controls.

Verification:
- Manual checks on desktop and mobile widths.

## Slice 7 - Typography and Readability Upgrade

Goal: Match claude.ai-like readability and visual hierarchy.

Scope:
- Sidebar font family switched to modern sans-serif stack.
- Chat transcript font switched to mild serif stack.
- Increase chat message font size and line-height for user and assistant bubbles.

Primary files:
- `frontend/style.css`
- `frontend/index.html` (font loading hooks if needed)

Acceptance:
- Sidebar and transcript typography are visually distinct and intentional.
- Chat text is noticeably larger and easier to read.
- No layout overflow regressions on narrow screens.

Verification:
- Visual QA on desktop and mobile breakpoints.

## Slice 8 - Music Panel UX Refresh

Goal: Make the music panel feel deliberate and less clumsy.

Scope:
- Improve now-playing vs queue visual separation.
- Add active-track affordance in queue.
- Add volume dial control (frontend + backend endpoint if MPD volume is wired).

Primary files:
- `frontend/index.html`
- `frontend/message.js`
- `frontend/style.css`
- `backend/main.py` and `backend/tools/music.py` (if volume endpoint/control added)

Acceptance:
- Music panel has clear hierarchy and pleasant interaction.
- Volume control works and state is reflected in UI.

Verification:
- Manual control checks: play/pause/next/stop/volume/queue click.

## Slice 9 - Final Polish and Release Gate

Goal: Close with confidence.

Scope:
- Run focused regression suite.
- Run manual UX matrix for sidebar/session/chat/music.
- Document env migration notes and any changed defaults.

Acceptance:
- Regression suite passes.
- UX behavior matches direction above.
- Deployment notes are written and actionable.

Verification:
- Backend focused pytest run.
- Manual QA checklist signoff.

## Suggested Execution Order

1. Phase 0
2. Slice 1
3. Slice 2
4. Slice 3
5. Slice 4
6. Slice 5
7. Slice 6
8. Slice 7
9. Slice 8
10. Slice 9

## Notes on Risk

- Genre-first matching can misclassify artist names that also look like genre terms; require confidence logging and fallback rules.
- Typography changes can cause subtle overflow issues in message bubbles and session rows; test mobile early.
- Session ordering semantics may require backend adjustment to prevent select-click from changing `updated_at`.
