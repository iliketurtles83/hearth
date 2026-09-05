import json
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse


def create_code_router(services) -> APIRouter:
    router = APIRouter()

    CodeRequest = services.code_request
    log = services.log
    chat_model = services.chat_model
    code_default_system_prompt = services.code_default_system_prompt
    session_cookie_name = services.session_cookie_name
    normalize_chat_source = services.normalize_chat_source
    set_session_cookie = services.set_session_cookie
    resolve_graph_runner = services.resolve_graph_runner
    checkpoint_config = services.checkpoint_config
    voice_tts_metadata = services.voice_tts_metadata
    stream_error_payload = services.stream_error_payload

    @router.post("/code", summary="Stream code-question responses via graph with local code intent bias")
    async def code(request: CodeRequest, http_request: Request):
        user_id: str = http_request.state.user_id
        cookie_session_id = http_request.cookies.get(session_cookie_name)
        session_id = cookie_session_id or str(uuid4())
        session_created = cookie_session_id is None
        code_source = normalize_chat_source(request.source)
        effective_system = request.system or code_default_system_prompt

        graph_state = {
            "user_id": user_id,
            "session_id": session_id,
            "message": request.message,
            "system": effective_system,
            "source": code_source,
            "force_code": True,
        }

        graph_runner = resolve_graph_runner()

        async def generate():
            assistant_accumulated = ""
            active_model = chat_model

            try:
                async for event in graph_runner.astream(
                    graph_state,
                    config=checkpoint_config(session_id),
                    stream_mode="custom",
                ):
                    if "meta" in event:
                        meta = event["meta"]
                        active_model = meta.get("model", chat_model)
                        yield f"data: {json.dumps({'model': active_model, 'intent': meta.get('intent', 'code'), 'confidence': meta.get('confidence', 1.0)})}\n\n"
                    elif "text" in event:
                        chunk = event["text"]
                        assistant_accumulated += chunk
                        yield f"data: {json.dumps({'text': chunk})}\n\n"
                    elif "notice" in event:
                        yield f"data: {json.dumps({'notice': event['notice']})}\n\n"
            except Exception as exc:
                log.error("code.graph_error | session_id=%s error=%s", session_id, exc)
                yield f"data: {json.dumps(stream_error_payload())}\n\n"

            voice_meta = voice_tts_metadata(code_source)
            if voice_meta is not None:
                yield "data: " + json.dumps(voice_meta) + "\n\n"

            yield "data: [DONE]\n\n"

        response = StreamingResponse(generate(), media_type="text/event-stream")
        set_session_cookie(response, session_id)
        if session_created:
            log.info("code.session.cookie_set | session_id=%s", session_id)
        return response

    return router
