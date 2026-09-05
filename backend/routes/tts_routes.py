from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

from app_schemas import TTSRequest


def create_tts_router(services) -> APIRouter:
    router = APIRouter()

    tts = services.tts
    log = services.log
    tts_error_status = services.tts_error_status
    error_response = services.error_response

    @router.post("/tts")
    async def tts_synthesize(request: TTSRequest):
        try:
            audio = await tts.synthesize(request.text)
            return Response(content=audio, media_type="audio/wav")
        except Exception as exc:
            payload = tts.error_to_payload(exc)
            code = str(payload.get("code", "TTS_UNKNOWN_ERROR"))
            retryable = bool(payload.get("retryable", False))
            message = str(payload.get("error", "Unknown TTS error"))
            status = tts_error_status(code, retryable)
            log.warning("tts.error | code=%s retryable=%s message=%s", code, retryable, message)
            return error_response(message, code, retryable, status_code=status)

    return router
