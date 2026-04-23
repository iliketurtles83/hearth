from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
import httpx
import anthropic
import os
import json
import logging
import tempfile
import time
from threading import Lock
from uuid import uuid4
import numpy as np
from dotenv import load_dotenv
from router import classify_intent, LOCAL_MODEL, CLOUD_MODEL

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

    ollama_url = os.getenv("OLLAMA_URL", "http://ollama:11434")
    log.info("Startup OK | local_model=%s | ollama=%s", LOCAL_MODEL, ollama_url)

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
    allow_origins=["*"],
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
            _session_store[effective_id] = {"messages": [], "created_at": now, "updated_at": now}
            log.info("chat.session.created | session_id=%s", effective_id)
        else:
            _session_store[effective_id]["updated_at"] = now
        _evict_oldest_sessions_if_needed()
        return effective_id, created


def _select_history_for_budget(messages: list[dict], system: str, current_user_message: str) -> tuple[list[dict], int, bool]:
    history_budget = max(0, CHAT_TOKEN_BUDGET - _estimate_tokens(system) - _estimate_tokens(current_user_message) - 32)
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
    return selected, used_tokens, truncated


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


def _append_session_message(session_id: str, role: str, content: str) -> None:
    now = time.time()
    with _session_store_lock:
        session = _session_store.get(session_id)
        if not session:
            return
        session["messages"].append({"role": role, "content": content, "ts": now})
        session["updated_at"] = now

async def stream_local(request: ChatRequest):
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", "http://ollama:11434/api/generate", json={
            "model": LOCAL_MODEL,
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
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text

@app.post("/chat")
async def chat(request: ChatRequest, http_request: Request):
    cookie_session_id = http_request.cookies.get(SESSION_COOKIE_NAME)
    session_id, session_created = _get_or_create_session(cookie_session_id)

    with _session_store_lock:
        session_messages = list(_session_store[session_id]["messages"])

    selected_history, history_tokens, truncated = _select_history_for_budget(
        messages=session_messages,
        system=request.system,
        current_user_message=request.message,
    )

    decision = classify_intent(request.message)

    log.info(
        "chat.route | session_id=%s intent=%s confidence=%.3f route=%s model=%s "
        "total_messages=%d included_messages=%d estimated_history_tokens=%d truncated=%s",
        session_id,
        decision.intent,
        decision.confidence,
        "cloud" if decision.use_cloud else "local",
        decision.model,
        len(session_messages),
        len(selected_history),
        history_tokens,
        truncated,
    )

    # Build history for each backend.
    # Local: flatten into a single prompt string (Ollama /api/generate).
    # Cloud: pass as a proper messages array (Anthropic messages API).
    local_prompt = _build_local_prompt(selected_history, request.message)
    local_request = ChatRequest(message=local_prompt, system=request.system)

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

        primary = (
            stream_cloud(request.system, cloud_messages)
            if decision.use_cloud
            else stream_local(local_request)
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
                active_model = LOCAL_MODEL
                yield f"data: {json.dumps({'notice': 'Cloud unavailable — responding with local model'})}\n\n"
                yield f"data: {json.dumps({'model': active_model, 'intent': decision.intent, 'confidence': decision.confidence, 'fallback': True})}\n\n"
                async for chunk in stream_local(local_request):
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

        yield "data: [DONE]\n\n"

    response = StreamingResponse(generate(), media_type="text/event-stream")
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=SESSION_IDLE_TTL_SECONDS,
    )
    if session_created:
        log.info("chat.session.cookie_set | session_id=%s", session_id)
    return response


@app.delete("/chat/session")
async def reset_chat_session(http_request: Request):
    cookie_session_id = http_request.cookies.get(SESSION_COOKIE_NAME)
    session_id, created = _get_or_create_session(cookie_session_id)
    with _session_store_lock:
        session = _session_store.get(session_id, {"messages": []})
        cleared_messages = len(session["messages"])
        session["messages"] = []
        session["updated_at"] = time.time()
        _session_store[session_id] = session

    log.info(
        "chat.session.reset | session_id=%s cleared_messages=%d was_new=%s",
        session_id,
        cleared_messages,
        created,
    )
    response = JSONResponse({"ok": True, "session_id": session_id, "cleared_messages": cleared_messages})
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=SESSION_IDLE_TTL_SECONDS,
    )
    return response

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
            prediction = model.predict(samples)
            score = prediction.get("computer_v2", 0.0)
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