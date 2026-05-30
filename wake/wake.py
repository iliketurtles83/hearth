"""
wake.py — Standalone wakeword detection backend (extracted from backend/main.py).

Implements:
  - GET  /ws/wake   — WebSocket endpoint: browser streams 1280-sample int16 PCM
                      frames (80 ms @ 16 kHz); server replies {"event":"wake","score":…}
                      when the wake word is detected.
  - POST /transcribe — Whisper transcription endpoint (receives a WAV file,
                       returns {"text": "…"}).

Environment variables:
  WAKEWORD_MODEL_FILE      — filename of the .onnx wake-word model (default: computer_v2.onnx)
  OWW_MELSPEC_MODEL_FILE   — filename of the melspectrogram backbone (default: melspectrogram.onnx)
  OWW_EMBEDDING_MODEL_FILE — filename of the embedding backbone (default: embedding_model.onnx)
  MODELS_DIR               — directory that contains the .onnx model files
                             (default: ./models relative to this file)
  WHISPER_MODEL            — faster-whisper model size/name (default: base.en)
  WHISPER_DEVICE           — "cuda" or "cpu" (auto-detected when unset)
  WHISPER_COMPUTE_TYPE     — "float16", "int8", etc. (auto-selected when unset)

Dependencies (install separately):
  fastapi uvicorn[standard] openwakeword faster-whisper numpy python-dotenv
"""

import logging
import os
import tempfile

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
WAKEWORD_MODEL_FILE = os.getenv("WAKEWORD_MODEL_FILE", "computer_v2.onnx")
OWW_MELSPEC_MODEL_FILE = os.getenv("OWW_MELSPEC_MODEL_FILE", "melspectrogram.onnx")
OWW_EMBEDDING_MODEL_FILE = os.getenv("OWW_EMBEDDING_MODEL_FILE", "embedding_model.onnx")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "").strip().lower()
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "").strip().lower()

_models_dir = os.getenv("MODELS_DIR", os.path.join(os.path.dirname(__file__), "models"))

# ── openWakeWord model (lazy-loaded on first WebSocket connection) ─────────────
_oww_model = None


def get_oww_model():
    global _oww_model
    if _oww_model is None:
        from openwakeword.model import Model

        # v0.6.0 removed bundled backbone models — pass explicit paths so
        # AudioFeatures doesn't look in the (empty) library resources directory.
        _oww_model = Model(
            wakeword_models=[os.path.join(_models_dir, WAKEWORD_MODEL_FILE)],
            inference_framework="onnx",
            melspec_model_path=os.path.join(_models_dir, OWW_MELSPEC_MODEL_FILE),
            embedding_model_path=os.path.join(_models_dir, OWW_EMBEDDING_MODEL_FILE),
        )
    return _oww_model


# ── faster-whisper model (lazy-loaded on first /transcribe call) ──────────────
_whisper_model = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        device = WHISPER_DEVICE or ("cuda" if os.path.exists("/dev/nvidia0") else "cpu")
        compute = WHISPER_COMPUTE_TYPE or ("float16" if device == "cuda" else "int8")
        _whisper_model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute)
    return _whisper_model


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Wake Word Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/wake")
async def wake_websocket(ws: WebSocket):
    """
    Browser sends raw binary frames: 1280 int16 samples (80 ms @ 16 kHz).
    Server replies with {"event": "wake", "score": <float>} when the wake
    word is detected.
    """
    await ws.accept()
    log.info("Wake WebSocket connected from %s", ws.client)
    model = get_oww_model()
    model.reset()  # clear any stale state from a previous session
    try:
        while True:
            data = await ws.receive_bytes()
            # Keep as int16 — the library's melspectrogram model requires int16
            # PCM input.  Converting to float32 here would silently zero-out all
            # samples when the library casts back to int16.
            samples = np.frombuffer(data, dtype=np.int16)
            raw_prediction = model.predict(samples)
            # openWakeWord can return either a dict or a tuple where index 0 is
            # the dict.
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
        log.info(
            "Wake WebSocket disconnected — code: %d, reason: %s",
            exc.code,
            exc.reason or "(none)",
        )


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Transcribe a WAV audio file and return the detected text."""
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


# ── Run ────────────────────────────────────────────────────────────────────────
# uvicorn wake:app --host 0.0.0.0 --port 8020
