# Hearth — Review Notes (Sep 4)

Scope: findings from a static code review **plus** a live `docker compose up -d --build` run and a real `/chat` smoke test (registered a throwaway user, streamed two conversations through Caddy).

Live-run facts that anchor the findings below:
- All 4 containers (`ollama`, `mpd`, `backend`, `caddy`) reached `healthy`.
- Ollama v0.21.0; `gemma4:e4b` (9.6GB) + `nomic-embed-text` already present in the `ollama` volume.
- GPU (RTX 3060, 12GB) **is** attached to the ollama container (`nvidia-smi -L` lists it inside).
- First `/chat` → `"Hello. I am Hearth."`; warm `/chat` (`27*43`) → `1161` in 3.8s.
- Backend log on startup: `embedding_router.failed | error=` (empty), and every request logged `embedding_route.fallback | reason=router_unavailable`.

## Severity legend

- **HIGH** — a core/intended behavior is wrong or a user-facing feature is broken.
- **MEDIUM** — degrades quality/latency/maintainability; workaround exists.
- **LOW** — hygiene, dead config, or documented-intentional footguns.

## Summary

| # | Severity | Issue |
|---|----------|-------|
| 1 | HIGH | Embedding intent router fails to build at startup → silent permanent heuristic fallback |
| 2 | MEDIUM | `/transcribe` non-functional — no Whisper model present |
| 3 | MEDIUM | ~83s cold-start first token; recurs after 5min idle (no chat-model warmup / keep_alive) |
| 4 | MEDIUM | `main.py` is a 1,234-line god module (routes + middleware + bootstrap + graph deps) |
| 5 | LOW-MED | Module-level `MemoryStore` on import — `import main` writes to state files |
| 6 | LOW | `OLLAMA_URL` unset → defaults to Docker hostname; bare-host local runs break |
| 7 | LOW | Dead env var `CLOUD_THRESHOLD` (not consumed; real knob is `ROUTE_CONFIDENCE_THRESHOLD`) |
| 8 | LOW | `graph.py` oversized nested node closures |
| 9 | LOW | `load_dotenv()` import-order footgun (documented-intentional) |
| 10 | LOW | Untracked WIP/scratch files + uncommitted `AGENTS.md` |
| 11 | LOW | No CI — script-only gates (accepted constraint) |

---

## 1. [HIGH] Embedding intent router fails to build at startup

**Description**
`/health` reports `embedding_router: false` and every request logs `embedding_route.fallback | reason=router_unavailable`. The router is built **once** in the app lifespan by `build_embedding_router()`, which embeds the entire exemplar bank (tool + dialogue) sequentially with an **8s** per-call timeout. On a cold Ollama the first embed call has to load `nomic-embed-text` into VRAM, which blows the 8s budget; the resulting exception is caught and the process permanently falls back to the deterministic heuristic (it is never retried, and the lazy fallback path in `_resolve_graph_runner` also constructs deps with `embedding_router=None`).

Two aggravating bugs make this hard to diagnose and to fix via config:
- The failure log uses `str(exc)`, which is empty for timeout/connection errors → `error=` prints nothing.
- The configured `ROUTER_EMBED_TIMEOUT_MS=10000` is **not** wired in — `build_embedding_router()` uses its own hardcoded `timeout_seconds=8.0`, and `_graph_lifespan` calls it with no timeout argument.

**Related files**
- `backend/main.py`
- `backend/embedding_router.py`
- `backend/graph.py`
- `.env`

**Related functions**
- `build_embedding_router()` — `embedding_router.py:363` (hardcoded `timeout_seconds=8.0`, no warmup, no retry)
- `_graph_lifespan()` — `main.py:433` (build at `:444`, failure log at `:451`, wired into deps at `:454`)
- `health()` — `main.py:1074` (reads `app.state.embedding_router`)
- `_make_graph_deps()` — `main.py:715`
- `_resolve_graph_runner()` — `main.py:748` (lazy rebuild with no router)
- `intent_classifier` fallback — `graph.py:467-469`

**Recommended fix**
1. In the failure handler, log `repr(exc)` and `type(exc).__name__` (never just `str(exc)`) so the cause is diagnosable.
2. Warm up the embedder first (one probe `/api/embeddings` call) before embedding the exemplar bank, so the first real call isn't the model-load.
3. Honor `ROUTING_CONFIG.router_embed_timeout_ms` by passing it into `build_embedding_router()` instead of the hardcoded 8s.
4. Retry the build with backoff, and/or rebuild lazily on first request rather than failing once for the lifetime of the process.

**Acceptance**
- On a cold Ollama, startup logs `embedding_router.ready | model=... dim=<n>`.
- `/health` returns `embedding_router: true`.
- A `/chat` request takes the embedding path (no `reason=router_unavailable` line).
- If the build does fail, the log includes the exception type/message (not an empty `error=`).
- Existing `test_router.py` / `test_embedding_router.py` still pass.

---

## 2. [MEDIUM] `/transcribe` non-functional — no Whisper model present

**Description**
`/transcribe` lazily loads a faster-whisper model (`WHISPER_MODEL` defaults to `base.en`, `main.py:161`) on first use via `get_whisper_model()`. No Whisper model exists under `backend/models/` (only the openWakeWord `.onnx` files and Kokoro TTS assets), and `scripts/download-models.sh` only covers openWakeWord — it does not fetch a Whisper model. So the first `/transcribe` call will fail (model not found) rather than transcribing.

**Related files**
- `backend/main.py`
- `backend/models/` (missing asset)
- `.env`
- `scripts/download-models.sh` (does not cover Whisper)

**Related functions**
- `get_whisper_model()` — `main.py:497`
- `transcribe` handler — `main.py:1142`
- `WHISPER_MODEL` / `WHISPER_DEVICE` / `WHISPER_COMPUTE_TYPE` — `main.py:161-163`

**Recommended fix**
- Make the Whisper model available to the backend (extend a download script or document a one-time `faster-whisper` pull / HF cache mount in the README).
- Return a clear, non-500 error (e.g. 503 with `code: MODEL_NOT_LOADED`) when the model isn't present, instead of a raw exception.
- Add model presence to `_validate_startup()` so it's caught at boot.

**Acceptance**
- `POST /transcribe` with a short audio clip returns `200` with a JSON transcript.
- If the model is genuinely absent, the endpoint returns a documented 503 (not 500) and startup validation warns.
- A script or README step guarantees the model is present before first use.

---

## 3. [MEDIUM] ~83s cold-start first token; recurs after 5min idle

**Description**
The first `/chat` after container start logged `first_token_ms=83618`. This is the one-time cost of loading the 9.6GB `gemma4:e4b` into the RTX 3060 (the GPU is correctly attached). Warm requests are ~4s. Ollama's default `keep_alive` is 5 minutes, so after 5 minutes of idle the model is unloaded and the next request pays the ~80s load again. There is no startup warmup that pre-loads the chat model.

**Related files**
- `docker-compose.yml` (ollama service)
- `backend/main.py`

**Related functions**
- `_graph_lifespan()` — `main.py:433` (natural place to add a post-graph warmup)
- ollama service definition — `docker-compose.yml:2`

**Recommended fix**
- After the graph is built in `_graph_lifespan`, issue a tiny throwaway `/api/chat` (or `/api/generate`) to `gemma4:e4b` to pre-load it into VRAM.
- Set a longer `OLLAMA_KEEP_ALIVE` (e.g. `30m`) on the ollama service environment so the model stays resident.

**Acceptance**
- First user `/chat` after `docker compose up` returns first-token well under ~10s.
- `ollama ps` shows `gemma4:e4b` loaded shortly after startup.
- No ~80s stall after periods of idle.

---

## 4. [MEDIUM] `main.py` is a 1,234-line god module

**Description**
`backend/main.py` mixes cross-cutting concerns (Auth + COOP/COEP middleware, `_validate_startup()`, Beets bootstrap, graph lifecycle/deps) with ~15 route handlers (`/chat`, session CRUD, `/tts`, `/ws/wake`, `/transcribe`, `/code`, health). Meanwhile `routes/` only contains `auth_routes.py` and `memory_tool_routes.py`, which proves the intended pattern (router factories included in `main`). The chat/voice/TTS/code endpoints should follow the same pattern.

**Related files**
- `backend/main.py`
- `backend/routes/`

**Related functions**
- `AuthMiddleware` — `main.py:333`; `COOPCOEPMiddleware` — `main.py:414`
- `_validate_startup()` — `main.py:275`; `_bootstrap_beets_library_if_empty()` — `main.py:224`
- Route handlers: `chat` (`:769`), session routes (`:972-1064`), `tts_synthesize` (`:1090`), `wake_websocket` (`:1107`), `transcribe` (`:1142`), `code` (`:1163`), `health` (`:1074`)

**Recommended fix**
- Extract the chat/session/voice/TTS/code endpoints into `routes/` modules exposing `create_*_router(...)` factories, mirroring `auth_routes.py`/`memory_tool_routes.py`, and `app.include_router(...)` them from `main.py`.
- Leave app construction, middleware, lifespan, and dependency factories in `main.py`.

**Acceptance**
- `main.py` no longer defines chat/voice/TTS/code route handlers (only wiring).
- New router modules are imported and included; the app starts cleanly.
- Full backend test suite passes and a `/chat` round-trip still works end-to-end.

---

## 5. [LOW-MED] Module-level `MemoryStore` on import

**Description**
`memory_store = MemoryStore(...)` runs at module import (`main.py:94`). `MemoryStore.__init__` creates directories, opens SQLite in WAL mode, runs `_init_db()`, and opens a Chroma `PersistentClient` — so a bare `import main` writes to `memory.db` and `chroma/`. This couples import to side effects, slows/complicates tests, and is a footgun when splitting routes into modules that import `main`.

**Related files**
- `backend/main.py`
- `backend/memory.py`

**Related functions**
- module-level `memory_store` — `main.py:94`
- `MemoryStore.__init__()` — `memory.py:137` (`_init_db` at `:255`, Chroma setup at `:172`)
- `_make_graph_deps()` — `main.py:715` (primary consumer)

**Recommended fix**
- Make `MemoryStore` lazy: construct it in `_graph_lifespan` / first use (or a cached getter) so importing `main` is side-effect free.

**Acceptance**
- `python -c "import main"` does not create or modify `memory.db`/`chroma` (default paths).
- Tests can import the app module without touching on-disk state; suite still passes.

---

## 6. [LOW] `OLLAMA_URL` unset → Docker hostname default breaks bare-host runs

**Description**
`.env` sets the model names but not `OLLAMA_URL`, so it defaults to `http://ollama:11434` (the Compose service hostname, `routing_config.py:36`). Correct inside Docker, unresolvable when running `uvicorn` directly on the host.

**Related files**
- `.env`
- `.env.example`
- `backend/routing_config.py`

**Related functions**
- `load_routing_config()` — `routing_config.py:36`

**Recommended fix**
- Document a host-local `OLLAMA_URL=http://localhost:11434` in `.env.example` for non-Docker dev (commented, since Docker uses the default).

**Acceptance**
- A local (non-Docker) `uvicorn` run reaches Ollama.
- `.env.example` contains explicit `OLLAMA_URL` guidance for both Docker and host-local cases.

---

## 7. [LOW] Dead env var `CLOUD_THRESHOLD`

**Description**
`.env` defines `CLOUD_THRESHOLD=300`, but no backend code reads `CLOUD_THRESHOLD`. The actual routing-confidence knob is `ROUTE_CONFIDENCE_THRESHOLD` (default `0.55`, `routing_config.py:37`; used at `intents.py:34` and `intents.py:323`). The `300` value looks like a leftover from an older config.

**Related files**
- `.env`
- `.env.example`
- `backend/routing_config.py`

**Related functions**
- `load_routing_config()` — `routing_config.py:37`

**Recommended fix**
- Remove `CLOUD_THRESHOLD`, or — if it was meant to configure the threshold — rename it to `ROUTE_CONFIDENCE_THRESHOLD` with a valid `0..1` value.

**Acceptance**
- No unused env vars in `.env`/`.env.example`; if a threshold value is intended, `ROUTE_CONFIDENCE_THRESHOLD` reflects it and a test asserts it.

---

## 8. [LOW] `graph.py` oversized nested node closures

**Description**
`graph.py` (1,020 lines) nests large logic inside the graph node closures (`intent_classifier` at `:385`, `responder` at `:723`), making it hard to unit-test the pure decision/response logic in isolation.

**Related files**
- `backend/graph.py`

**Related functions**
- `intent_classifier` — `graph.py:385`
- `responder` — `graph.py:723`

**Recommended fix**
- Extract the pure helpers (decision building, response assembly, token budgeting) into module-level functions and add focused unit tests.

**Acceptance**
- Node bodies are thin dispatchers calling tested module-level helpers.
- New helpers have unit tests; `test_graph.py` / `test_router.py` still pass.

---

## 9. [LOW] `load_dotenv()` import-order footgun

**Description**
`load_dotenv()` runs at `main.py:31`, before local imports, because env vars are read at import time (documented-intentional in AGENTS.md). It is a latent footgun: any module that reads env at import must be imported after this line.

**Related files**
- `backend/main.py`

**Related functions**
- top-level `load_dotenv()` — `main.py:31`

**Recommended fix**
- Keep it, but add a clear comment enforcing the ordering invariant, and/or centralize config loading so import-time env reads have a single well-defined entry point.

**Acceptance**
- The invariant is documented at the call site; no import-order regressions after route splitting (Issue 4).

---

## 10. [LOW] Untracked WIP/scratch files + uncommitted `AGENTS.md`

**Description**
Untracked at repo root: `create-skeleton.sh`, `repo_skeleton.md`, `docs/PLAN-LLM-NATIVE-TOOL-CALLING.md`, `docs/WEATHER_FASTPATH.json`, `docs/MEMORY.md`. `AGENTS.md` also had uncommitted edits. Mixed signal on what is intentional vs. scratch.

**Related files**
- `create-skeleton.sh`, `repo_skeleton.md`
- `docs/PLAN-LLM-NATIVE-TOOL-CALLING.md`, `docs/WEATHER_FASTPATH.json`, `docs/MEMORY.md`
- `AGENTS.md`

**Related functions**
- n/a

**Recommended fix**
- Commit the intentional docs, delete or move scratch files, or add them to `.gitignore` if they are generated.

**Acceptance**
- `git status` is clean, or every untracked file is intentionally tracked/ignored.

---

## 11. [LOW] No CI — script-only gates

**Description**
GitHub Actions was removed; the only quality gates are `scripts/review_baseline.sh` and `scripts/review_changed_tests.sh`, which must be run manually. Nothing enforces them on PR.

**Related files**
- `scripts/review_baseline.sh`
- `scripts/review_changed_tests.sh`
- `docs/review/` (checklists / known failures)

**Related functions**
- n/a

**Recommended fix**
- Either add a minimal CI workflow that runs `review_baseline.sh` (and `review_changed_tests.sh --allow-known-failures`) on PR, or explicitly document that script-only gating is an accepted constraint.

**Acceptance**
- If added: PRs run the gate and block on failure. If not: the accepted-constraint note lives in AGENTS.md/README.
