# Backend & Full-Stack Code Review (review2)

Scope: `backend/`, `frontend/`, and deployment (`docker-compose.yml`, `caddy/`,
`backend/Dockerfile`). This review supersedes nothing in `review.md` — it (a) verifies the
status of every issue raised in `review.md`, and (b) documents **new** findings not covered
there. Line references are approximate and may drift as the code changes.

---

## Executive Summary

The backend is in noticeably better shape than at the time of `review.md`. **11 of the 16**
previously-reported issues have been genuinely fixed (not papered over): `load_dotenv()` now
runs first, the memory-consolidation lock is released around blocking LLM calls, the SPA
catch-all serves the shell for browser navigations, fire-and-forget tasks are tracked with
done-callbacks, expired tokens are purged on startup, and the entire ad-hoc code-file write
surface (`code_file_routes.py`, `pending_code_writes`) has been removed — which dissolves a
whole cluster of the older findings.

What remains is a different, mostly *frontend and operational* risk profile that the
backend-only `review.md` never looked at:

- **Cross-site scripting is the headline risk.** Assistant/LLM output is rendered with
  `marked.parse()` straight into `innerHTML` with no sanitizer, and the memory panel
  interpolates `key`/`value` into `innerHTML` without the escaping the rest of the file uses.
  In a local-first single-user box this is "self-XSS", but the app authenticates multiple
  named users and stores cross-session memory, so it is a real stored-XSS surface.
- **Operational exposure.** Ollama is published on `0.0.0.0:11434`, the production image runs
  `uvicorn --reload` as root, and there is no rate-limiting on auth endpoints. None of these
  match the "local-first, nothing leaves the device" security posture in the docs.
- **A data-integrity bug in memory.** The `facts` table has no uniqueness constraint and
  consolidation marks summaries `consolidated` only *after* a non-atomic, lock-free LLM pass,
  so concurrent consolidation passes duplicate facts.

Overall health: **moderate-to-good for a local-first project, with a small number of
high-severity items that should be fixed before any multi-user or LAN-exposed deployment.**
Priority order: XSS (N1/N2) → memory duplication/race (N7/N8) → Ollama exposure (N4) →
auth rate-limiting (N5) → the rest.

---

## Part A — Status of `review.md` findings

> Per instructions: issues that remain valid are **not** re-documented below — see `review.md`
> for their full text. Issues that no longer apply are marked **NON-RELEVANT** with evidence.

| # | review.md issue | Status |
|---|---|---|
| 1 | `load_dotenv()` runs too late | **NON-RELEVANT** — now the first executable statement, before all local imports and `MemoryStore(...)` ([backend/main.py](backend/main.py#L25-L30)). |
| 2 | Consolidation holds DB lock during LLM calls | **NON-RELEVANT** — `consolidate_pending` is now 3-phase: read under lock → LLM extraction **without** lock → writes under lock ([backend/memory.py](backend/memory.py#L906-L1005)). |
| 3 | SPA catch-all returns 401 for unauth deep links | **NON-RELEVANT** — `_is_browser_navigation()` lets non-API `GET` navigations fall through to the SPA shell ([backend/main.py](backend/main.py#L300-L320)). |
| 4 | `/transcribe` unauthenticated | **STILL VALID (partially mitigated)** — an upload size cap was added ([backend/main.py](backend/main.py#L1075-L1085)), but it is still unauthenticated with no rate limiting. See `review.md` #4. |
| 5 | `_extract_candidates_llm_sync` broken running-loop branch | **NON-RELEVANT** — rewritten to always create a fresh loop on the worker thread; the broken `run_until_complete` branch is gone ([backend/memory.py](backend/memory.py#L547-L575)). |
| 6 | `stream_local` does not check HTTP status | **NON-RELEVANT** — now calls `resp.raise_for_status()` ([backend/main.py](backend/main.py#L585-L595)). |
| 7 | Fire-and-forget consolidation task GC'd / swallows errors | **NON-RELEVANT** — `_track_background_task()` keeps a strong ref and attaches a done-callback that logs exceptions ([backend/graph.py](backend/graph.py#L16-L31)). |
| 8 | Music fast-path skips conversation logging | **NON-RELEVANT** — the fast-path now logs both the user and assistant turns via `memory_store.log_turn` ([backend/main.py](backend/main.py#L740-L760)). |
| 9 | Duplicated path-safety/diff/confirm logic | **NON-RELEVANT** — `code_file_routes.py` and the duplicated workspace/diff/confirm logic no longer exist; the code-file write surface was removed. |
| 10 | `pending_code_writes` grows unbounded | **NON-RELEVANT** — `pending_code_writes` no longer exists. |
| 11 | In-memory `_pending_code_writes` breaks under multiple workers | **NON-RELEVANT (as cited)** — the cited structure is gone. (General multi-worker caveat for model singletons / in-memory session state still applies but is documented in `AGENTS.md`.) |
| 12 | Expired auth tokens never purged | **NON-RELEVANT** — `purge_expired_tokens()` is called in the graph lifespan startup ([backend/main.py](backend/main.py#L409-L414)). |
| 13 | Module-level graph built then discarded | **NON-RELEVANT** — replaced by lazy `_resolve_graph_runner()` that builds a fallback only on demand and logs a clear warning ([backend/main.py](backend/main.py#L640-L665)). |
| 14 | Wake-word threshold hardcoded | **NON-RELEVANT** — now `WAKEWORD_THRESHOLD = float(os.getenv("WAKEWORD_THRESHOLD", "0.5"))` ([backend/main.py](backend/main.py#L150)). |
| 15 | Embedding code-route vs heuristic mismatch | **NON-RELEVANT (superseded)** — `_decision_from_embedding` no longer consults the heuristic for code; write-vs-question is now decided separately by `is_write_like_code_request()` ([backend/graph.py](backend/graph.py#L298-L308), [backend/graph.py](backend/graph.py#L537-L549)). |
| 16 | Bandit B608 in `memory.py` | **STILL VALID** — `_forget_by_query` still builds `%{q}%` (passed as a bound parameter, so not injectable) and Bandit still flags it. See `review.md` #16; annotate with `# nosec B608`. |

---

## Part B — New findings

### N1 — Stored XSS in the memory panel (unescaped `innerHTML`)
- **Severity:** High
- **Location:** [frontend/message.js](frontend/message.js#L458-L466), `renderMemory()`
- **Issue:** Every other render path in this file escapes interpolated values with `_esc(...)`
  (sessions L427, queue L557), but `renderMemory` injects `item.key` and `item.value` raw:
  ```js
  div.innerHTML = `
    <div class="list-item-title">${item.key}</div>
    <div class="list-item-meta">${(item.value || '').slice(0, 90)}</div>
    ...`;
  ```
  `key`/`value` originate from user messages and LLM extraction (`facts`/`preferences`). A
  value like `<img src=x onerror=alert(document.cookie)>` stored as memory executes when the
  panel renders. Because memory is keyed by `user_id` and the app supports multiple users,
  this is stored XSS, not just self-XSS.
- **Fix:** Escape exactly like the sibling renderers:
  ```js
  div.innerHTML = `
    <div class="list-item-title">${_esc(item.key)}</div>
    <div class="list-item-meta">${_esc((item.value || '').slice(0, 90))}</div>
    <div class="list-item-meta">${_esc(tierLabel)}${consolidatedLabel ? ` · ${_esc(consolidatedLabel)}` : ''}</div>
    <div class="memory-actions">
      <button class="memory-delete-btn" data-id="${_esc(String(item.id))}">Delete</button>
    </div>`;
  ```

### N2 — Unsanitized markdown rendering of assistant output
- **Severity:** High
- **Location:** [frontend/message.js](frontend/message.js#L324) (`appendHistoryMessage`), [frontend/message.js](frontend/message.js#L791) (stream render)
- **Issue:** `bubble.innerHTML = marked.parse(text)` renders model output (and restored history)
  as HTML. `marked` does **not** sanitize by default (the old `sanitize` option was removed), so
  any raw HTML in the assistant stream — or echoed back from a malicious user turn the model
  quotes — is live in the DOM. LLM output is attacker-influenceable (prompt injection, quoted
  user content), making this a practical XSS sink.
- **Fix:** Sanitize before insertion. Add DOMPurify and wrap both call sites:
  ```js
  // index.html: <script src="/static/vendor/purify.min.js"></script>  (self-hosted)
  bubble.innerHTML = DOMPurify.sanitize(marked.parse(text || ''));
  ```
  Centralize in a `renderMarkdown(text)` helper so both sites stay consistent.

### N3 — Third-party script from CDN, unpinned and without SRI; no CSP
- **Severity:** Medium
- **Location:** [frontend/index.html](frontend/index.html#L11) (and the Google Fonts links L7-L9)
- **Issue:** `https://cdn.jsdelivr.net/npm/marked/marked.min.js` is loaded with no version pin
  and no `integrity`/SRI hash. This (a) contradicts the "local-first, nothing leaves the device"
  posture — the app fetches third-party JS on every load and breaks offline, and (b) is a
  supply-chain risk: a compromised/served-latest `marked` executes in the authenticated origin.
  There is also no `Content-Security-Policy` header anywhere (neither FastAPI middleware nor
  Caddy), so nothing constrains script sources as defense-in-depth for N1/N2.
- **Fix:** Vendor `marked` (and DOMPurify) under `frontend/` and serve from `/static`, pinned to
  a known version. Add a CSP at the Caddy edge, e.g.:
  ```
  header {
    Content-Security-Policy "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'"
    X-Content-Type-Options "nosniff"
    Referrer-Policy "no-referrer"
  }
  ```

### N4 — Ollama published on `0.0.0.0:11434` (unauthenticated inference engine on the LAN)
- **Severity:** High
- **Location:** [docker-compose.yml](docker-compose.yml#L6-L8) (`ollama.ports`)
- **Issue:** The whole architecture funnels browser traffic through Caddy and keeps the backend
  on `expose` only — but Ollama itself is published to the host on all interfaces. Anyone on the
  LAN can call `http://<host>:11434` directly: run arbitrary prompts (compute abuse), pull/delete
  models, and read whatever the local models can produce. This bypasses the auth boundary the
  rest of the stack carefully maintains.
- **Fix:** Don't publish the port, or bind it to loopback only. The backend reaches Ollama over
  the compose network, so the host mapping is usually unnecessary:
  ```yaml
  ollama:
    # remove "ports:" entirely, OR bind to localhost if host tools need it:
    ports:
      - "127.0.0.1:11434:11434"
  ```

### N5 — No rate limiting / lockout on auth; username enumeration via timing
- **Severity:** Medium
- **Location:** [backend/auth.py](backend/auth.py#L175-L215) (`login`), [backend/routes/auth_routes.py](backend/routes/auth_routes.py)
- **Issue:** `login`/`register` have no throttling, attempt counter, or lockout, so credentials
  are brute-forceable (scrypt slows but does not stop this). Additionally, when the username is
  unknown, `login` returns immediately *without* computing scrypt, while a known username pays the
  scrypt cost — a measurable timing oracle for username enumeration.
- **Fix:** Add per-IP/per-username rate limiting (e.g. `slowapi` or a small in-memory token bucket
  keyed on `X-Real-IP`, which Caddy already forwards). For the timing oracle, always run a dummy
  scrypt against a fixed salt when the user is missing so both branches take constant time.

### N6 — Production image runs `uvicorn --reload` as root
- **Severity:** Medium
- **Location:** [backend/Dockerfile](backend/Dockerfile#L6)
- **Issue:** `CMD ["uvicorn", "main:app", ... "--reload"]` ships the dev auto-reloader in the
  production image — it spawns a file-watcher/reloader subprocess, increases memory/CPU, and
  reloads on any bind-mounted file change. The container also runs as root (no `USER`), so a
  backend RCE is a root-in-container compromise with the music/beets bind mounts attached.
- **Fix:** Drop `--reload` for production (gate it behind an env-driven compose `command`
  override for dev) and add a non-root user:
  ```dockerfile
  RUN adduser --disabled-password --gecos "" app && chown -R app /app
  USER app
  CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
  ```

### N7 — `facts` table has no uniqueness constraint → duplicate facts accumulate
- **Severity:** High
- **Location:** [backend/memory.py](backend/memory.py#L134-L143) (schema), [backend/memory.py](backend/memory.py#L978-L985) (`consolidate_pending` insert), `ingest_user_message` insert
- **Issue:** `preferences` has `CREATE UNIQUE INDEX idx_preferences_user_key (user_id, key)` and
  upserts with `ON CONFLICT`, but `facts` has only a non-unique `idx_facts_user_id` and inserts
  with a plain `INSERT`. Every consolidation pass and every `ingest_user_message` re-inserts the
  same `(user_id, key, value)` rows, so `facts` grows without bound and retrieval surfaces
  duplicates. This degrades recall quality and inflates the DB over time.
- **Fix:** Add `CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_user_key ON facts(user_id, key)` and
  switch fact inserts to `INSERT ... ON CONFLICT(user_id, key) DO UPDATE SET value=excluded.value,
  created_at=excluded.created_at`. (Note: this also changes `memory_id = facts:{lastrowid}` — read
  the row id back like the preferences branch already does.) Backfill/dedupe existing rows in a
  one-time migration.

### N8 — Consolidation race: summaries marked `consolidated` only after a lock-free LLM pass
- **Severity:** High
- **Location:** [backend/memory.py](backend/memory.py#L906-L1005) (`consolidate_pending`), [backend/graph.py](backend/graph.py#L943-L957) (trigger)
- **Issue:** `memory_writer` spawns `consolidate_pending` as a background task whenever
  `unconsolidated >= threshold`, on *every* qualifying turn. `consolidate_pending` reads the
  unconsolidated summaries in Phase 1, runs the (slow) LLM extraction in Phase 2 **without the
  lock**, and only sets `consolidated = 1` in Phase 3. Two consolidation tasks overlapping in time
  both read the same Phase-1 rows and both promote them → duplicated facts (compounded by N7,
  which has no unique constraint to absorb the duplicates).
- **Fix:** Claim the rows atomically before releasing the lock — mark them `consolidated = 1` (or a
  `claimed` state) inside the Phase-1 transaction, then extract/write. On extraction failure, roll
  the claim back. Alternatively serialize consolidation with a single-flight guard
  (`asyncio.Lock` / a "consolidation in progress" flag) so only one pass runs at a time.

### N9 — Raw exception text streamed to the client (information disclosure)
- **Severity:** Medium
- **Location:** [backend/main.py](backend/main.py#L838-L840) (`/chat`), [backend/main.py](backend/main.py#L1133-L1135) (`/code`), [backend/main.py](backend/main.py#L730-L735) (music fast-path)
- **Issue:** Stream error handlers emit the raw exception to the user:
  ```python
  except Exception as exc:
      yield f"data: {json.dumps({'text': f'⚠ Error: {exc}'})}\n\n"
  ```
  `str(exc)` can contain internal paths, model names, upstream URLs, or SQL fragments, leaking
  implementation detail to any authenticated client and complicating a clean error contract.
- **Fix:** Log the full exception server-side (already done) but send a generic, stable message to
  the client, e.g. `{'text': '⚠ Something went wrong. Please try again.', 'code': 'INTERNAL'}`.

### N10 — Dead schema + stale docs for the removed code-file write surface
- **Severity:** Low
- **Location:** [backend/app_schemas.py](backend/app_schemas.py#L66-L69) (`WriteRequest`), [README.md](README.md#L63-L65)
- **Issue:** `WriteRequest` is no longer imported anywhere (the `/code/files` routes were removed),
  yet `README.md` still advertises `GET/PUT /code/files/{file_path}` as live endpoints. Dead code
  plus doc drift misleads onboarding and implies a security-sensitive write surface that does not
  exist. (`AGENTS.md` similarly references `router.py`/`code_file_routes.py` that are gone.)
- **Fix:** Delete `WriteRequest` and remove the `/code/files` section from `README.md` (and the
  stale path references in `AGENTS.md`).

### N11 — Single global lock + single shared SQLite connection serializes all memory I/O
- **Severity:** Medium
- **Location:** [backend/memory.py](backend/memory.py#L100-L107) (`__init__`), used by every method
- **Issue:** `MemoryStore` shares one `sqlite3` connection guarded by one `threading.Lock`. Every
  history load, turn log, retrieval, and consolidation write contends on that single lock. Because
  these run via `asyncio.to_thread`, concurrent chat requests fully serialize on memory access —
  a throughput ceiling that worsens as sessions/turns grow, and any slow query (or the in-lock
  Phase-3 writes) stalls all other requests.
- **Fix:** Enable WAL (`PRAGMA journal_mode=WAL`) and move to a connection-per-thread (or a small
  pool) so reads run concurrently; reserve the lock only for write serialization. SQLite with WAL
  supports concurrent readers + a single writer, which fits this workload well.

### N12 — `_esc()` does not escape quotes; relies on values never landing in attribute context
- **Severity:** Low
- **Location:** [frontend/message.js](frontend/message.js#L501-L503)
- **Issue:** `_esc` replaces only `&`, `<`, `>` — not `"` or `'`. It is currently used in text
  contexts (and in N1's fix), but any future use inside an HTML attribute (`title="${_esc(x)}"`)
  would be breakable with a quote. This is a latent footgun rather than a live bug.
- **Fix:** Also escape quotes: `.replace(/"/g, '&quot;').replace(/'/g, '&#39;')`, and prefer
  `element.textContent` / `setAttribute` over template-string `innerHTML` for dynamic values.

### N13 — New `anthropic.Anthropic` client constructed per cloud call
- **Severity:** Low
- **Location:** [backend/main.py](backend/main.py#L615-L625) (`stream_cloud`)
- **Issue:** `stream_cloud` instantiates a fresh `anthropic.Anthropic(api_key=...)` on every
  request and reads `ANTHROPIC_API_KEY` each time. Beyond the minor per-call setup cost, if the key
  is unset it constructs an unusable client and fails deep inside the stream rather than failing
  fast with a clear "cloud unavailable" message.
- **Fix:** Build the client once at startup (module/app-state singleton) and short-circuit with a
  clean error if `ANTHROPIC_API_KEY` is absent before attempting the stream.

---

## Notes / Confirmed-good (not findings)

- Auth uses scrypt at OWASP-minimum params with `secrets.compare_digest` and opaque 32-byte
  tokens looked up server-side — solid token design ([backend/auth.py](backend/auth.py#L36-L41)).
- All Beets/SQLite queries in `tools/music.py` and `memory.py` use bound parameters; the only
  string interpolation builds `%{q}%` wildcards passed *as* parameters — no SQL injection.
- Image upload validation enforces MIME allow-list + 25 MB cap before decode
  ([backend/main.py](backend/main.py#L600-L625)).
- Session ownership is checked before returning/deleting session data
  ([backend/main.py](backend/main.py#L880-L960)).
- CORS is added outermost (after auth) and disables credentials when origin is `*` — correct.
</content>
</invoke>
