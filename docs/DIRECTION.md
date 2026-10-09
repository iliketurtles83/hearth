# Direction

> **Status (2026-10-07):** thinking, not decided. Written after a review of the local-AI stack (`~/stack`). Revisit
> once the first stack tools (notes, search) exist.

## The goal

A private personal assistant and second brain, **fully hands-free on the phone, on the go**, plus a GUI on the
phone. Privacy first, local preferred. Coding is out of scope (solved by opencode).

Judge work by daily use: does it answer or do something I actually ask for? Music polish was fun but drifted from
this.

## Requests that define "done"

- Plan my week
- Play a genre / band / album — **works**
- Continue with project X (pick up a project's notes and context)
- "I've got carrots, pork and cabbage" → recipe, shown as a card
- Record a voice note
- Search the latest news on a topic
- What's on my calendar
- Whatever I can't think of yet → log every request Hearth fails at; that log is the backlog

## Hearth's role

Hearth is the **voice front-end and orchestrator**, not the home of data or tools.

- **In Hearth**: wake word, STT, TTS, barge-in, deterministic fast paths (music), UI cards, intent routing.
- **Outside Hearth (in the stack)**: data in plain formats (`me.md`, a markdown notes folder, CalDAV calendar,
  SearXNG) and tools as MCP servers. Hearth reaches them as an MCP client (`langchain-mcp-adapters` on LangGraph).
- Why: the second brain survives if Hearth is replaced, each tool is testable with curl, and Hermes / opencode /
  Open WebUI share the same tools and memory instead of each keeping its own.
- Open: Hermes as the brain with Hearth as voice-only front-end. Leaning Hearth (own prompt, own latency). The
  tools-outside design keeps this open.

## Memory

- Lesson from Hindsight: automatic recall injected ~80 memories per turn and made answers worse; it also changes
  the prompt prefix every turn, which defeats prompt caching (prefill is the iGPU's weak spot).
- Prefer a small curated `me.md` (< ~2k tokens) loaded at the top of the system prompt, shared by all agents.
  Agents propose additions to an inbox file; I merge by hand.
- Bulk knowledge (notes, documents) goes behind a search tool the model calls on demand, not into every prompt.
- Consequence for [MEMORY.md](MEMORY.md): its hybrid SQLite + Chroma recall is the same design that failed in
  Hindsight. Demote vector recall to a tool, or turn it off, once `me.md` exists.

## Model (done 2026-10-07)

Chat runs on `qwen36-35b-a3b-mtp` with thinking off (`CHAT_ENABLE_THINKING=false`); vision stays on `gemma-4-12b`
(the MTP 35B has no mmproj). Time to first answer token: 35B thinking on 5.7 s, off 1.05 s, Gemma 4 12B 1.24 s.

## Hands-free on the go

- Today wake word runs server-side: the browser streams the mic to `/ws/wake`. On a phone that dies when the screen
  locks (no background mic in mobile browsers) and streams audio over cellular continuously.
- Path: a small native Android app — first as the default assistant / Bluetooth headset-button trigger (no
  always-on mic), then on-device wake word (openWakeWord or Porcupine), streaming to Hearth over Tailscale only
  after the wake.
- iOS can't run a third-party background wake word; there the ceiling is the headset button or Shortcuts.

## Open questions

- **Cloud fallback** is configured (`MODEL_CLOUD` + key): "reasoning-heavy" requests go to Anthropic. "Plan my
  week" with calendar data would leave the machine. Disable, or require explicit opt-in per request?
- STT is `base.en` (English only). Multilingual model if Estonian dictation matters.
- Calendar source: Radicale (tiny CalDAV/CardDAV, DAVx5 on Android) vs Nextcloud (only if also hosting files).
