from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from pydantic import BaseModel
import httpx
import anthropic
import os
import json
import logging
import tempfile
import numpy as np
from dotenv import load_dotenv
from router import should_use_cloud, LOCAL_MODEL, CLOUD_MODEL

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("assistant")

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

async def stream_cloud(request: ChatRequest):
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    with client.messages.stream(
        model=CLOUD_MODEL,
        max_tokens=2048,
        system=request.system,
        messages=[{"role": "user", "content": request.message}],
    ) as stream:
        for text in stream.text_stream:
            yield text

@app.post("/chat")
async def chat(request: ChatRequest):
    use_cloud = should_use_cloud(request.message)
    streamer = stream_cloud(request) if use_cloud else stream_local(request)
    model_used = CLOUD_MODEL if use_cloud else LOCAL_MODEL

    async def generate():
        yield f"data: {json.dumps({'model': model_used})}\n\n"
        async for chunk in streamer:
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

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