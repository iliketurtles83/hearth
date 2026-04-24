from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from pydantic import BaseModel
import httpx
import anthropic
import os
import json
import logging
import re
import tempfile
import time
from threading import Lock
from uuid import uuid4
import numpy as np
from dotenv import load_dotenv
from router import route as router_route, classify_intent, LOCAL_MODEL, CLOUD_MODEL, CHAT_MODEL, CODER_MODEL
from memory import memory_store
import tools
from tools.base import ToolResult

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("assistant")

SESSION_COOKIE_NAME = os.getenv("CHAT_SESSION_COOKIE", "assistant_session")
SESSION_IDLE_TTL_SECONDS = int(os.getenv("CHAT_SESSION_IDLE_TTL_SECONDS", "1800"))
SESSION_MAX_ITEMS = int(os.getenv("CHAT_SESSION_MAX_ITEMS", "200"))
CHAT_TOKEN_BUDGET = int(os.getenv("CHAT_TOKEN_BUDGET", "1500"))
CHAT_MAX_TURNS = int(os.getenv("CHAT_MAX_TURNS", "24"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434").rstrip("/")
CHAT_SUMMARY_TRIGGER_MESSAGES = int(os.getenv("CHAT_SUMMARY_TRIGGER_MESSAGES", "18"))
CHAT_SUMMARY_KEEP_RECENT_MESSAGES = int(os.getenv("CHAT_SUMMARY_KEEP_RECENT_MESSAGES", "8"))
CHAT_SUMMARY_MAX_CHARS = int(os.getenv("CHAT_SUMMARY_MAX_CHARS", "1400"))

# ── HTTPS / CORS / cookie policy (Phase 0b) ───────────────────────────────────
# CORS_ORIGINS: comma-separated list of allowed origins, e.g.
#   CORS_ORIGINS=https://192.168.1.42,https://assistant.lan
# Default '*' preserves the Phase 1 permissive behaviour for plain-HTTP dev.
# Set to the exact Caddy origin(s) once HTTPS is in use.
_cors_origins_raw = os.getenv("CORS_ORIGINS", "*")
_CORS_ORIGINS: list[str] = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
# allow_credentials requires a non-wildcard origin per the CORS spec.
_CORS_CREDENTIALS: bool = _CORS_ORIGINS != ["*"]

# SESSION_COOKIE_SECURE: set to 'true' when the browser-facing edge is HTTPS
# (i.e. when Caddy is in use). Tells the browser to send the cookie only over
# HTTPS connections. The backend itself may still run plain HTTP internally.
SESSION_COOKIE_SECURE: bool = os.getenv("SESSION_COOKIE_SECURE", "false").lower() == "true"

_session_store_lock = Lock()
_session_store: dict[str, dict] = {}

# ── Startup validation ─────────────────────────────────────────────────────────
def _validate_startup() -> None:
    _models_dir = os.path.join(os.path.dirname(__file__), "models")
    required_models = ["computer_v2.onnx", "melspectrogram.onnx", "embedding_model.onnx"]
    missing_models = [m for m in required_models if not os.path.isfile(os.path.join(_models_dir, m))]
    if missing_models:
        log.warning("Missing ONNX model files (wake-word will fail): %s", missing_models)
        log.warning("Run: bash scripts/download-models.sh")

    if not os.getenv("ANTHROPIC_API_KEY"):
        log.warning("ANTHROPIC_API_KEY not set — cloud model fallback will be unavailable")

    log.info(
        "Startup OK | chat_model=%s | coder_model=%s | ollama=%s | cors_origins=%s | cookie_secure=%s",
        CHAT_MODEL, CODER_MODEL, OLLAMA_URL, _CORS_ORIGINS, SESSION_COOKIE_SECURE,
    )

_validate_startup()

# ── Cross-Origin isolation headers (required for SharedArrayBuffer / vad-web) ──
class COOPCOEPMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        # 'credentialless' (vs 'require-corp') still enables SharedArrayBuffer
        # (needed by vad-web's threaded WASM) while allowing cross-origin CDN
        # resources that don't set Cross-Origin-Resource-Policy headers.
        response.headers["Cross-Origin-Embedder-Policy"] = "credentialless"
        return response

app = FastAPI()

app.add_middleware(COOPCOEPMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=_CORS_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── openWakeWord model (lazy-loaded on first WebSocket connection) ──
_oww_model = None

def get_oww_model():
    global _oww_model
    if _oww_model is None:
        from openwakeword.model import Model
        _models_dir = os.path.join(os.path.dirname(__file__), "models")
        # v0.6.0 removed bundled backbone models — pass explicit paths so AudioFeatures
        # doesn't look in the (empty) library resources directory.
        _oww_model = Model(
            wakeword_models=[os.path.join(_models_dir, "computer_v2.onnx")],
            inference_framework="onnx",
            melspec_model_path=os.path.join(_models_dir, "melspectrogram.onnx"),
            embedding_model_path=os.path.join(_models_dir, "embedding_model.onnx"),
        )
    return _oww_model

# ── faster-whisper model (lazy-loaded on first /transcribe call) ──
_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        device = "cuda" if os.path.exists("/dev/nvidia0") else "cpu"
        compute = "float16" if device == "cuda" else "int8"
        _whisper_model = WhisperModel("base.en", device=device, compute_type=compute)
    return _whisper_model

class ChatRequest(BaseModel):
    message: str
    system: str = "You are a helpful personal assistant. Be concise and accurate."


class SessionSelectRequest(BaseModel):
    session_id: str


def _error_response(message: str, code: str, retryable: bool, status_code: int = 400) -> JSONResponse:
    return JSONResponse({"error": message, "code": code, "retryable": retryable}, status_code=status_code)


def _set_session_cookie(response: Response, session_id: str) -> None:
    """Apply consistent session-cookie attributes across all endpoints."""
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        httponly=True,
        samesite="lax",
        secure=SESSION_COOKIE_SECURE,
        max_age=SESSION_IDLE_TTL_SECONDS,
    )


def _estimate_tokens(text: str) -> int:
    # Lightweight token estimate for bounded context decisions.
    return max(1, len(text) // 4)


def _cleanup_expired_sessions(now: float) -> None:
    expired = [
        session_id
        for session_id, session in _session_store.items()
        if now - session["updated_at"] > SESSION_IDLE_TTL_SECONDS
    ]
    for session_id in expired:
        del _session_store[session_id]
    if expired:
        log.info("chat.session.evicted_expired | count=%d", len(expired))


def _evict_oldest_sessions_if_needed() -> None:
    if len(_session_store) <= SESSION_MAX_ITEMS:
        return
    overflow = len(_session_store) - SESSION_MAX_ITEMS
    oldest_first = sorted(_session_store.items(), key=lambda item: item[1]["updated_at"])
    for session_id, _ in oldest_first[:overflow]:
        del _session_store[session_id]
    log.info("chat.session.evicted_capacity | count=%d", overflow)


def _get_or_create_session(session_id: str | None) -> tuple[str, bool]:
    now = time.time()
    with _session_store_lock:
        _cleanup_expired_sessions(now)
        effective_id = session_id if session_id and session_id in _session_store else str(uuid4())
        created = effective_id not in _session_store
        if created:
            _session_store[effective_id] = {
                "messages": [],
                "summary": "",
                "summary_message_count": 0,
                "created_at": now,
                "updated_at": now,
            }
            log.info("chat.session.created | session_id=%s", effective_id)
        else:
            _session_store[effective_id]["updated_at"] = now
        _evict_oldest_sessions_if_needed()
        return effective_id, created


def _select_history_for_budget(
    messages: list[dict],
    system: str,
    current_user_message: str,
    summary_text: str,
) -> tuple[list[dict], int, bool, int]:
    summary_tokens = _estimate_tokens(summary_text) if summary_text else 0
    history_budget = max(
        0,
        CHAT_TOKEN_BUDGET
        - _estimate_tokens(system)
        - _estimate_tokens(current_user_message)
        - summary_tokens
        - 32,
    )
    selected_reversed: list[dict] = []
    used_tokens = 0
    truncated = False
    max_messages = max(1, CHAT_MAX_TURNS * 2)
    candidates = messages[-max_messages:]

    for message in reversed(candidates):
        cost = _estimate_tokens(message["content"]) + 4
        if used_tokens + cost > history_budget:
            truncated = True
            continue
        selected_reversed.append(message)
        used_tokens += cost

    selected = list(reversed(selected_reversed))
    if len(messages) > len(selected):
        truncated = True
    return selected, used_tokens, truncated, summary_tokens


def _normalize_summary_line(text: str, max_len: int = 200) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"


def _summarize_messages_chunk(messages: list[dict]) -> str:
    lines: list[str] = []
    for message in messages:
        role = message.get("role")
        content = _normalize_summary_line(str(message.get("content", "")))
        if not content:
            continue
        if role == "user":
            lines.append(f"- User: {content}")
        elif role == "assistant":
            lines.append(f"- Assistant: {content}")
        if len(lines) >= 14:
            break
    return "\n".join(lines)


def _truncate_summary(summary: str) -> str:
    if len(summary) <= CHAT_SUMMARY_MAX_CHARS:
        return summary
    tail = summary[-CHAT_SUMMARY_MAX_CHARS :]
    first_newline = tail.find("\n")
    if first_newline > 0:
        return tail[first_newline + 1 :]
    return tail


def _update_session_summary_if_needed(session_id: str) -> tuple[bool, int, int]:
    now = time.time()
    with _session_store_lock:
        session = _session_store.get(session_id)
        if not session:
            return False, 0, 0

        messages = list(session.get("messages", []))
        keep_recent = max(2, CHAT_SUMMARY_KEEP_RECENT_MESSAGES)
        trigger = max(keep_recent + 2, CHAT_SUMMARY_TRIGGER_MESSAGES)
        if len(messages) <= trigger:
            return False, int(session.get("summary_message_count", 0) or 0), len(session.get("summary", ""))

        older = messages[:-keep_recent]
        if not older:
            return False, int(session.get("summary_message_count", 0) or 0), len(session.get("summary", ""))

        already_summarized = int(session.get("summary_message_count", 0) or 0)
        if already_summarized >= len(older):
            return False, already_summarized, len(session.get("summary", ""))

        new_slice = older[already_summarized:]
        chunk_summary = _summarize_messages_chunk(new_slice)
        if not chunk_summary:
            return False, already_summarized, len(session.get("summary", ""))

        existing_summary = str(session.get("summary", "") or "")
        combined = (
            f"{existing_summary}\n{chunk_summary}".strip()
            if existing_summary
            else chunk_summary
        )
        combined = _truncate_summary(combined)

        session["summary"] = combined
        session["summary_message_count"] = len(older)
        session["updated_at"] = now
        _session_store[session_id] = session
        return True, len(older), len(combined)


def _build_local_prompt(history: list[dict], current_user_message: str) -> str:
    if not history:
        return current_user_message

    role_map = {"user": "User", "assistant": "Assistant"}
    lines = ["Conversation so far:"]
    for message in history:
        role = role_map.get(message.get("role", ""), "User")
        lines.append(f"{role}: {message['content']}")
    lines.append("")
    lines.append(f"User: {current_user_message}")
    lines.append("Assistant:")
    return "\n".join(lines)


def _augment_system_with_session_summary(system: str, summary_text: str) -> str:
    if not summary_text:
        return system
    return "\n".join(
        [
            system,
            "",
            "Session summary of older messages (use as context for continuity):",
            summary_text,
        ]
    )


def _augment_system_with_memories(system: str, memory_hits: list[dict]) -> str:
    if not memory_hits:
        return system

    lines = [
        system,
        "",
        "Relevant user memory (apply only if directly helpful to this request):",
        "If a memory item is not clearly relevant, ignore it.",
    ]
    for hit in memory_hits[:5]:
        lines.append(f"- {hit['text']}")
    return "\n".join(lines)


def _should_inject_memory(decision_intent: str, memory_hits: list[dict], user_message: str) -> bool:
    if not memory_hits:
        return False
    if decision_intent == "memory-needed":
        return True

    # For non-memory intents, only inject when there is strong lexical overlap.
    terms = [t for t in re.findall(r"[a-z0-9]+", user_message.lower()) if len(t) > 2][:10]
    if not terms:
        return False

    top_text = " ".join(str(h.get("text", "")).lower() for h in memory_hits[:3])
    overlap = sum(1 for t in terms if t in top_text)
    return overlap >= 2


def _session_preview_text(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")[:80]
    return ""


def _list_sessions() -> list[dict]:
    with _session_store_lock:
        ordered = sorted(_session_store.items(), key=lambda item: item[1]["updated_at"], reverse=True)
        return [
            {
                "session_id": sid,
                "created_at": data["created_at"],
                "updated_at": data["updated_at"],
                "message_count": len(data.get("messages", [])),
                "preview": _session_preview_text(data.get("messages", [])),
            }
            for sid, data in ordered
        ]


def _append_session_message(session_id: str, role: str, content: str) -> None:
    now = time.time()
    with _session_store_lock:
        session = _session_store.get(session_id)
        if not session:
            return
        session["messages"].append({"role": role, "content": content, "ts": now})
        session["updated_at"] = now

async def stream_local(request: ChatRequest, model_name: str = CHAT_MODEL):
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", f"{OLLAMA_URL}/api/generate", json={
            "model": model_name,
            "prompt": request.message,
            "system": request.system,
            "stream": True,
        }) as resp:
            async for line in resp.aiter_lines():
                if line:
                    data = json.loads(line)
                    yield data.get("response", "")
                    if data.get("done"):
                        break

async def stream_cloud(system: str, messages: list[dict]):  # type: ignore[override]
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    with client.messages.stream(
        model=CLOUD_MODEL,
        max_tokens=2048,
        system=system,
        messages=messages,  # type: ignore[arg-type]
    ) as stream:
        for text in stream.text_stream:
            yield text

@app.post("/chat")
async def chat(request: ChatRequest, http_request: Request):
    cookie_session_id = http_request.cookies.get(SESSION_COOKIE_NAME)
    session_id, session_created = _get_or_create_session(cookie_session_id)

    summary_updated, summary_message_count, summary_char_count = _update_session_summary_if_needed(session_id)

    with _session_store_lock:
        session = _session_store[session_id]
        session_messages = list(session["messages"])
        session_summary = str(session.get("summary", "") or "")

    selected_history, history_tokens, truncated, summary_tokens = _select_history_for_budget(
        messages=session_messages,
        system=request.system,
        current_user_message=request.message,
        summary_text=session_summary,
    )

    memory_hits_all = memory_store.retrieve(request.message)

    decision = await router_route(request.message)
    inject_memory = _should_inject_memory(decision.intent, memory_hits_all, request.message)
    memory_hits = memory_hits_all if inject_memory else []
    system_with_summary = _augment_system_with_session_summary(request.system, session_summary)
    augmented_system = _augment_system_with_memories(system_with_summary, memory_hits)

    log.info(
        "chat.route | session_id=%s intent=%s confidence=%.3f route=%s model=%s "
        "planner_status=%s needs_memory=%s tool=%s "
        "total_messages=%d included_messages=%d estimated_history_tokens=%d summary_tokens=%d "
        "summary_updated=%s summary_message_count=%d summary_chars=%d truncated=%s",
        session_id,
        decision.intent,
        decision.confidence,
        "cloud" if decision.use_cloud else "local",
        decision.model,
        decision.planner_status,
        decision.needs_memory,
        decision.tool,
        len(session_messages),
        len(selected_history),
        history_tokens,
        summary_tokens,
        summary_updated,
        summary_message_count,
        summary_char_count,
        truncated,
    )
    if decision.reasoning_summary:
        log.debug("chat.planner_reasoning | session_id=%s reasoning=%s", session_id, decision.reasoning_summary)

    # Build history for each backend.
    # Local: flatten into a single prompt string (Ollama /api/generate).
    # Cloud: pass as a proper messages array (Anthropic messages API).
    local_prompt = _build_local_prompt(selected_history, request.message)
    local_request = ChatRequest(message=local_prompt, system=augmented_system)

    cloud_messages = [
        {"role": m["role"], "content": m["content"]} for m in selected_history
    ]
    cloud_messages.append({"role": "user", "content": request.message})

    async def generate():
        nonlocal decision
        assistant_accumulated = ""
        start_time = time.monotonic()
        first_token_time: float | None = None
        active_model = decision.model
        fallback_used = False

        yield f"data: {json.dumps({'model': active_model, 'intent': decision.intent, 'confidence': decision.confidence})}\n\n"

        # ── Tool dispatch ─────────────────────────────────────────────────────
        # When the router sets decision.tool, call the registered tool module
        # and summarize its normalized output through the local chat model.
        # Raw API data is never sent directly to the client.
        if decision.tool:
            tool_result: ToolResult = await tools.dispatch(
                decision.tool,
                {"prompt": request.message, "memory": memory_store},
            )
            log.info(
                "chat.tool | session_id=%s tool=%s ok=%s retryable=%s",
                session_id, decision.tool, tool_result.ok, tool_result.retryable,
            )
            if tool_result.ok:
                # Build a summarization prompt: ask the local model to turn the
                # normalized data dict into a natural-language response.
                tool_data_str = json.dumps(tool_result.data, ensure_ascii=False)
                summary_prompt = (
                    f"You are a helpful assistant. Based on the following structured data, "
                    f"write a concise, natural-language response to the user's request.\n"
                    f"User request: {request.message}\n"
                    f"Data: {tool_data_str}\n"
                    f"Keep your response brief and include all relevant values with units."
                )
                summary_request = ChatRequest(message=summary_prompt, system=augmented_system)
                try:
                    async for chunk in stream_local(summary_request, model_name=CHAT_MODEL):
                        if first_token_time is None:
                            first_token_time = time.monotonic()
                        assistant_accumulated += chunk
                        yield f"data: {json.dumps({'text': chunk})}\n\n"
                except Exception as exc:
                    log.error("chat.tool_summary_error | session_id=%s error=%s", session_id, exc)
                    yield f"data: {json.dumps({'text': f'⚠ Could not summarize tool response: {exc}'})}\n\n"
            else:
                error_msg = tool_result.error or "The tool returned no data."
                assistant_accumulated = error_msg
                yield f"data: {json.dumps({'text': error_msg})}\n\n"

            # Skip stream_local / stream_cloud for tool responses — we are done.
            completion_time = time.monotonic()
            first_token_ms = (first_token_time - start_time) * 1000 if first_token_time else -1
            completion_ms = (completion_time - start_time) * 1000
            log.info(
                "chat.telemetry | session_id=%s intent=%s confidence=%.3f route=tool "
                "model=%s fallback=False first_token_ms=%.0f completion_ms=%.0f response_tokens_approx=%d",
                session_id, decision.intent, decision.confidence, active_model,
                first_token_ms, completion_ms, _estimate_tokens(assistant_accumulated),
            )
            _append_session_message(session_id, "user", request.message)
            _append_session_message(session_id, "assistant", assistant_accumulated.strip())
            memory_store.ingest_user_message(request.message, source="chat")
            return
        # ── End tool dispatch ─────────────────────────────────────────────────

        primary = (
            stream_cloud(augmented_system, cloud_messages)
            if decision.use_cloud
            else stream_local(local_request, model_name=decision.model)
        )

        try:
            async for chunk in primary:
                if first_token_time is None:
                    first_token_time = time.monotonic()
                assistant_accumulated += chunk
                yield f"data: {json.dumps({'text': chunk})}\n\n"

        except Exception as exc:
            if decision.use_cloud:
                # Graceful degradation: fall back to local with a visible notice.
                log.warning("chat.cloud_fallback | session_id=%s error=%s", session_id, exc)
                fallback_used = True
                active_model = CHAT_MODEL
                yield f"data: {json.dumps({'notice': 'Cloud unavailable \u2014 responding with local model'})}\n\n"
                yield f"data: {json.dumps({'model': active_model, 'intent': decision.intent, 'confidence': decision.confidence, 'fallback': True})}\n\n"
                async for chunk in stream_local(local_request, model_name=CHAT_MODEL):
                    if first_token_time is None:
                        first_token_time = time.monotonic()
                    assistant_accumulated += chunk
                    yield f"data: {json.dumps({'text': chunk})}\n\n"
            else:
                log.error("chat.local_error | session_id=%s error=%s", session_id, exc)
                yield f"data: {json.dumps({'text': f'⚠ Error: {exc}'})}\n\n"

        completion_time = time.monotonic()
        first_token_ms = (first_token_time - start_time) * 1000 if first_token_time else -1
        completion_ms = (completion_time - start_time) * 1000

        log.info(
            "chat.telemetry | session_id=%s intent=%s confidence=%.3f route=%s "
            "model=%s fallback=%s first_token_ms=%.0f completion_ms=%.0f response_tokens_approx=%d",
            session_id,
            decision.intent,
            decision.confidence,
            "cloud" if decision.use_cloud else "local",
            active_model,
            fallback_used,
            first_token_ms,
            completion_ms,
            _estimate_tokens(assistant_accumulated),
        )

        _append_session_message(session_id, "user", request.message)
        _append_session_message(session_id, "assistant", assistant_accumulated.strip())

        previous_user_message = next(
            (m.get("content", "") for m in reversed(session_messages) if m.get("role") == "user"),
            None,
        )

        memory_result = memory_store.ingest_user_message(
            request.message,
            source="chat",
            previous_user_message=previous_user_message,
        )
        log.info(
            "chat.memory | session_id=%s retrieved=%d injected=%d status=%s saved=%d blocked=%d needs_confirmation=%d candidates=%d explicit=%s",
            session_id,
            len(memory_hits_all),
            len(memory_hits),
            memory_result.get("status", "none"),
            len(memory_result.get("saved", [])),
            len(memory_result.get("blocked", [])),
            len(memory_result.get("needs_confirmation", [])),
            int(memory_result.get("candidates", 0)),
            bool(memory_result.get("explicit", False)),
        )

        yield (
            "data: "
            + json.dumps(
                {
                    "memory": {
                        "status": memory_result.get("status", "none"),
                        "saved": len(memory_result.get("saved", [])),
                        "blocked": len(memory_result.get("blocked", [])),
                        "needs_confirmation": len(memory_result.get("needs_confirmation", [])),
                        "deleted": int(memory_result.get("deleted", 0) or 0),
                        "explicit": bool(memory_result.get("explicit", False)),
                        "hint": (
                            "Memory needs confirmation. Say 'remember this' to store it."
                            if memory_result.get("status") == "needs-confirmation"
                            else ""
                        ),
                    }
                }
            )
            + "\n\n"
        )

        yield "data: [DONE]\n\n"

    response = StreamingResponse(generate(), media_type="text/event-stream")
    _set_session_cookie(response, session_id)
    if session_created:
        log.info("chat.session.cookie_set | session_id=%s", session_id)
    return response


@app.delete("/chat/session")
async def reset_chat_session(http_request: Request):
    cookie_session_id = http_request.cookies.get(SESSION_COOKIE_NAME)
    session_id, created = _get_or_create_session(cookie_session_id)
    with _session_store_lock:
        session = _session_store.get(session_id, {"messages": [], "summary": "", "summary_message_count": 0})
        cleared_messages = len(session["messages"])
        session["messages"] = []
        session["summary"] = ""
        session["summary_message_count"] = 0
        session["updated_at"] = time.time()
        _session_store[session_id] = session

    log.info(
        "chat.session.reset | session_id=%s cleared_messages=%d was_new=%s",
        session_id,
        cleared_messages,
        created,
    )
    response = JSONResponse({"ok": True, "session_id": session_id, "cleared_messages": cleared_messages})
    _set_session_cookie(response, session_id)
    return response


@app.get("/chat/sessions")
async def list_chat_sessions(http_request: Request):
    current_session = http_request.cookies.get(SESSION_COOKIE_NAME)
    return JSONResponse({"sessions": _list_sessions(), "current_session_id": current_session})


@app.delete("/chat/sessions/{session_id}")
async def delete_chat_session(session_id: str, http_request: Request):
    cookie_session_id = http_request.cookies.get(SESSION_COOKIE_NAME)
    active_session_id: str | None = None

    with _session_store_lock:
        if session_id not in _session_store:
            return _error_response("Session not found", "SESSION_NOT_FOUND", False, status_code=404)
        del _session_store[session_id]

        # If the deleted session was active, prefer an existing remaining session.
        if cookie_session_id == session_id and _session_store:
            active_session_id = max(
                _session_store.items(),
                key=lambda item: item[1]["updated_at"],
            )[0]
    log.info("chat.session.deleted | session_id=%s", session_id)

    # If the deleted session was active and nothing remains, create a fresh session.
    if cookie_session_id == session_id and active_session_id is None:
        active_session_id, _ = _get_or_create_session(None)

    payload = {"ok": True, "session_id": session_id}
    if active_session_id:
        payload["active_session_id"] = active_session_id
    response = JSONResponse(payload)

    if active_session_id:
        _set_session_cookie(response, active_session_id)
    return response


@app.post("/chat/session/new")
async def create_chat_session():
    session_id, _ = _get_or_create_session(None)
    response = JSONResponse({"ok": True, "session_id": session_id})
    _set_session_cookie(response, session_id)
    return response


@app.post("/chat/session/select")
async def select_chat_session(payload: SessionSelectRequest):
    with _session_store_lock:
        if payload.session_id not in _session_store:
            return _error_response("Session not found", "SESSION_NOT_FOUND", False, status_code=404)
    response = JSONResponse({"ok": True, "session_id": payload.session_id})
    _set_session_cookie(response, payload.session_id)
    return response


@app.get("/chat/session/messages")
async def get_chat_session_messages(http_request: Request):
    cookie_session_id = http_request.cookies.get(SESSION_COOKIE_NAME)
    session_id, _ = _get_or_create_session(cookie_session_id)
    with _session_store_lock:
        messages = list(_session_store.get(session_id, {}).get("messages", []))
    return JSONResponse({"session_id": session_id, "messages": messages})


@app.get("/memory")
async def list_memory(limit: int = Query(default=200, ge=1, le=500), offset: int = Query(default=0, ge=0)):
    return JSONResponse(memory_store.list_items(limit=limit, offset=offset))


@app.delete("/memory/{memory_id}")
async def delete_memory(memory_id: str):
    if not memory_store.delete_item(memory_id):
        return _error_response("Memory item not found", "MEMORY_NOT_FOUND", False, status_code=404)
    return JSONResponse({"ok": True, "id": memory_id})


@app.delete("/memory")
async def clear_memory():
    counts = memory_store.clear_all()
    return JSONResponse({"ok": True, "cleared": counts})


class WeatherRequest(BaseModel):
    location: str | None = None


@app.post("/weather")
async def weather(request: WeatherRequest):
    """Direct weather endpoint.

    Returns the normalized weather data dict for the given location (or stored default).
    Suitable for frontend calls and future LangGraph tool nodes.
    """
    from tools import weather as weather_tool  # type: ignore[attr-defined]
    result = await weather_tool.run({
        "prompt": f"weather in {request.location}" if request.location else "",
        "memory": memory_store,
        "location": request.location,
    })
    if not result.ok:
        status = 503 if result.retryable else 422
        return _error_response(result.error, "WEATHER_ERROR", result.retryable, status_code=status)
    return JSONResponse(result.data)


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Wake-word WebSocket ────────────────────────────────────────────────────────
# Browser sends raw binary frames: 1280 int16 samples (80ms @ 16kHz).
# Server replies with {"event": "wake"} when the wake word is detected.
@app.websocket("/ws/wake")
async def wake_websocket(ws: WebSocket):
    await ws.accept()
    log.info("Wake WebSocket connected from %s", ws.client)
    model = get_oww_model()
    model.reset()  # clear any stale state from a previous session
    try:
        while True:
            data = await ws.receive_bytes()
            # Keep as int16 — the library's melspectrogram model requires int16 PCM input.
            # Converting to float32 here would silently zero-out all samples when the
            # library casts back to int16, causing the model to see only silence.
            samples = np.frombuffer(data, dtype=np.int16)
            raw_prediction = model.predict(samples)
            # openWakeWord can return either a dict or a tuple where index 0 is the dict.
            if isinstance(raw_prediction, tuple):
                prediction = raw_prediction[0] if raw_prediction else {}
            else:
                prediction = raw_prediction
            if not isinstance(prediction, dict):
                prediction = {}
            score = float(prediction.get("computer_v2", 0.0) or 0.0)
            log.debug("Wake score: %.3f (threshold: 0.5)", score)
            if score > 0.5:
                log.info("Wake word detected — score: %.3f", score)
                await ws.send_json({"event": "wake", "score": round(float(score), 3)})
                model.reset()
    except WebSocketDisconnect as exc:
        log.info("Wake WebSocket disconnected — code: %d, reason: %s", exc.code, exc.reason or "(none)")


# ── Transcription endpoint ─────────────────────────────────────────────────────
@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    whisper = get_whisper_model()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        segments, _ = whisper.transcribe(tmp_path, language="en", vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments).strip()
    finally:
        os.unlink(tmp_path)
    return JSONResponse({"text": text})


# ── Static frontend — MUST be last ────────────────────────────────────────────
_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="static")