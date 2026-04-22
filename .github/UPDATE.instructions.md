1. Epic: Stabilize LAN Access and Single-Origin Serving  
- Goal: Ensure the app works reliably from desktop, phone, and tablet on your local network.  
- Tasks:
  - Serve frontend from backend origin consistently.
  - Replace hardcoded frontend backend URL usage with relative paths.
  - Add startup/health checks and clearer failure logs.
  - Verify container bind/expose settings (`0.0.0.0`, mapped ports).  
- Acceptance:
  - `http://<LAN-IP>:8000` loads UI and chat works from another device.
  - Voice WebSocket connects without looped reconnect.  
- Estimate: 1-2 days  
- Depends on: none  
- Labels: `epic`, `network`, `infra`, `priority-high`

2. Bug: Make Frontend API Calls LAN-Safe  
- Goal: Remove localhost coupling in browser requests.  
- Tasks:
  - Use `/chat`, `/transcribe`, `/ws/wake` everywhere.
  - Ensure no `http://localhost:*` remains in runtime fetch/websocket code.  
- Acceptance:
  - Same build works on host and remote LAN clients.  
- Estimate: 2-4 hours  
- Depends on: Issue 1  
- Labels: `bug`, frontend, `network`

3. Feature: Harden Wake-Word Input Pipeline (“Computer, …”)  
- Goal: Stable wake-to-transcribe loop with explicit state transitions.  
- Tasks:
  - Add startup validation for required ONNX model files.
  - Keep wake model filename + prediction key aligned.
  - Add guard window to prevent retrigger right after wake.
  - Improve logging of WS close codes/reasons and wake scores.  
- Acceptance:
  - Mic enters sleeping state and stays stable.
  - Wake phrase triggers one capture/transcribe cycle and returns to sleeping.  
- Estimate: 1-2 days  
- Depends on: Issues 1-2  
- Labels: `feature`, `voice`, backend, frontend, `priority-high`

4. Feature: Smarter Routing for Complex Queries  
- Goal: Use local model by default, escalate to stronger model only when needed.  
- Tasks:
  - Add intent categories (`quick-local`, `reasoning-heavy`, `external-data-needed`, `memory-needed`).
  - Add confidence score + fallback policy.
  - Emit route/latency/fallback telemetry.
  - Keep model badge visible in UI.  
- Acceptance:
  - Simple prompts stay local.
  - Complex prompts reliably route/escalate.  
- Estimate: 1-2 days  
- Depends on: Issue 1  
- Labels: `feature`, `routing`, `llm`, backend

5. Feature: Add SQLite Memory Layer  
- Goal: Persist useful user facts/preferences across restarts with controls.  
- Tasks:
  - Add schema (`facts`, `preferences`, `summaries`, retrieval metadata).
  - Implement write policy for high-value memory only.
  - Add retrieval for relevant memory snippets per query.
  - Add CRUD endpoints for view/delete/clear.  
- Acceptance:
  - Preferences survive restart.
  - User can inspect and remove memory items.  
- Estimate: 2-3 days  
- Depends on: Issue 4  
- Labels: `feature`, `memory`, backend, `database`, `priority-high`

6. Feature: Weather Tool Integration  
- Goal: First practical external tool with memory-backed default location.  
- Tasks:
  - Build weather provider adapter and normalized response schema.
  - Add endpoint and route weather intents to tool path.
  - Support “weather in <city>” override.
  - Graceful offline/API failure responses.  
- Acceptance:
  - “What’s the weather?” uses stored default location.
  - Override city works reliably.  
- Estimate: 1 day  
- Depends on: Issue 5  
- Labels: `feature`, `tools`, `weather`, backend

7. Feature: Local Music Library Playback  
- Goal: Search and control music from local collection via text/voice.  
- Tasks:
  - Build media indexer + metadata DB table.
  - Add search/control/queue endpoints.
  - Implement playback mode decision (browser playback vs backend daemon).
  - Add ambiguity handling and confirmations.  
- Acceptance:
  - “Play <song/artist>” works end to end.
  - Pause/resume/next/status works.  
- Estimate: 3-5 days  
- Depends on: Issues 3, 5  
- Labels: `feature`, `music`, `tools`, backend, frontend

8. Feature: TNG-Style Female Voice Output (TTS)  
- Goal: Spoken responses with configurable style and interruption behavior.  
- Tasks:
  - Add pluggable TTS backend service.
  - Add voice preset tuning (rate/pitch/prosody).
  - Add brief/full spoken response modes.
  - Add barge-in: wake phrase interrupts TTS and resumes listening.  
- Acceptance:
  - Assistant speaks responses with stable quality.
  - Voice playback does not block normal chat flow.  
- Estimate: 2-4 days  
- Depends on: Issues 3-4 (and ideally 6 for tool response parity)  
- Labels: `feature`, `tts`, `voice`, frontend, backend

9. Cross-Cutting: Observability and Reliability  
- Goal: Make debugging and uptime practical before scaling features.  
- Tasks:
  - Structured logs (routing, wake, transcription, tools).
  - Retry/backoff for transient failures.
  - Standardized user-facing error payloads.  
- Acceptance:
  - Failures are diagnosable from logs without guesswork.  
- Estimate: 1 day  
- Depends on: parallel with Issues 3-8  
- Labels: `reliability`, `observability`, backend, `infra`

10. Cross-Cutting: Test Matrix and Smoke Automation  
- Goal: Prevent regressions while features are added.  
- Tasks:
  - Backend tests: `/chat`, `/transcribe`, `/ws/wake`, weather endpoint.
  - Frontend smoke tests: streaming markdown, mic toggle, reconnect handling.
  - Manual LAN matrix checklist (desktop/phone/tablet).  
- Acceptance:
  - Critical flows covered with repeatable test steps.  
- Estimate: 1-2 days initial, ongoing updates  
- Depends on: start after Issues 1-3, expand continuously  
- Labels: `testing`, `qa`, backend, frontend

Suggested first sprint (ordered):
1. Issue 1  
2. Issue 2  
3. Issue 3  
4. Issue 4  
5. Issue 5  
6. Issue 6

If you want, I can next generate ready-to-paste GitHub issue bodies in exact template format (`Title`, `Problem`, `Scope`, `Tasks`, `Acceptance`, `Out of Scope`, `Estimate`, `Dependencies`, `Labels`) for the first 6 items.