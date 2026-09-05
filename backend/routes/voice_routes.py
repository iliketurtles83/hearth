import os
import tempfile

import numpy as np
from fastapi import APIRouter, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse


def create_voice_router(services) -> APIRouter:
    router = APIRouter()

    log = services.log
    get_oww_model = services.get_oww_model
    wakeword_threshold = services.wakeword_threshold
    max_transcribe_bytes = services.max_transcribe_bytes
    error_response = services.error_response
    # NOTE: get_whisper_model is accessed via `services.` at call time (not bound to
    # a local) so tests can monkeypatch main.services.get_whisper_model.

    @router.websocket("/ws/wake")
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
                log.debug("Wake score: %.3f (threshold: %.3f)", score, wakeword_threshold)
                if score > wakeword_threshold:
                    log.info("Wake word detected — score: %.3f", score)
                    await ws.send_json({"event": "wake", "score": round(float(score), 3)})
                    model.reset()
        except WebSocketDisconnect as exc:
            log.info("Wake WebSocket disconnected — code: %d, reason: %s", exc.code, exc.reason or "(none)")

    @router.post("/transcribe")
    async def transcribe(audio: UploadFile = File(...)):
        # /transcribe is auth-protected; still cap upload size to avoid unbounded reads.
        raw = await audio.read(max_transcribe_bytes + 1)
        if len(raw) > max_transcribe_bytes:
            return JSONResponse(
                {"error": f"Audio too large. Maximum is {max_transcribe_bytes // (1024 * 1024)} MB."},
                status_code=413,
            )
        try:
            whisper = services.get_whisper_model()
        except Exception as exc:
            # Missing model (no local copy + no reachable HF cache), an offline
            # download, or a device/compute failure all surface here. Return a
            # documented 503 instead of a raw 500.
            log.error("whisper.load_failed | type=%s error=%r", type(exc).__name__, exc)
            return error_response(
                "Speech-to-text model is not available. Run 'bash scripts/download-whisper-model.sh' to install it, then retry.",
                "MODEL_NOT_LOADED",
                retryable=False,
                status_code=503,
            )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        try:
            segments, _ = whisper.transcribe(tmp_path, language="en", vad_filter=True)
            text = " ".join(seg.text.strip() for seg in segments).strip()
        finally:
            os.unlink(tmp_path)
        return JSONResponse({"text": text})

    return router
