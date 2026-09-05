import asyncio
import json
import time
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse


def create_chat_router(services) -> APIRouter:
    router = APIRouter()

    # Bind shared state/callables to locals so handler bodies read identically to
    # the pre-refactor code. `services` is the AppServices object owned by main.py.
    ChatRequest = services.chat_request
    memory_store = services.memory_store
    tools = services.tools
    ToolResult = services.ToolResult
    log = services.log
    chat_model = services.chat_model
    chat_default_system_prompt = services.chat_default_system_prompt
    session_cookie_name = services.session_cookie_name
    normalize_chat_source = services.normalize_chat_source
    set_session_cookie = services.set_session_cookie
    validate_image = services.validate_image
    resolve_graph_runner = services.resolve_graph_runner
    get_state_graph_runner = services.get_state_graph_runner
    checkpoint_config = services.checkpoint_config
    estimate_tokens = services.estimate_tokens
    voice_tts_metadata = services.voice_tts_metadata
    stream_error_payload = services.stream_error_payload
    parse_music_command = services.parse_music_command
    format_music_response = services.format_music_response
    error_response = services.error_response

    @router.post("/chat")
    async def chat(request: ChatRequest, http_request: Request):
        user_id: str = http_request.state.user_id
        cookie_session_id = http_request.cookies.get(session_cookie_name)
        session_id = cookie_session_id or str(uuid4())
        session_created = cookie_session_id is None
        chat_source = normalize_chat_source(request.source)
        effective_system = request.system or chat_default_system_prompt

        # ── Deterministic music fast-path ─────────────────────────────────────
        # Check before graph routing so clear music commands never touch the LLM.
        music_cmd = parse_music_command(request.message)
        if music_cmd is not None:
            music_cmd["prompt"] = request.message
            music_cmd["user_id"] = user_id

            async def generate_music():
                yield f"data: {json.dumps({'model': 'music', 'intent': 'music', 'confidence': 1.0})}\n\n"
                try:
                    tool_result: ToolResult = await tools.dispatch("music", music_cmd)
                except Exception as exc:
                    log.error("chat.music_fast | session_id=%s error=%s", session_id, exc)
                    yield f"data: {json.dumps(stream_error_payload('MUSIC_COMMAND_FAILED'))}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                log.info(
                    "chat.music_fast | session_id=%s action=%s ok=%s retryable=%s",
                    session_id, music_cmd.get("action"), tool_result.ok, tool_result.retryable,
                )
                response_text = format_music_response(tool_result, music_cmd)
                # Persist the turn so music interactions appear in session history and
                # provide follow-up context (mirrors graph.memory_writer logging).
                if user_id:
                    try:
                        await asyncio.to_thread(
                            memory_store.log_turn,
                            session_id,
                            user_id,
                            "user",
                            request.message,
                        )
                        await asyncio.to_thread(
                            memory_store.log_turn,
                            session_id,
                            user_id,
                            "assistant",
                            response_text,
                        )
                    except Exception as exc:
                        log.warning("chat.music_fast.log_turn | session_id=%s error=%s", session_id, exc)
                yield f"data: {json.dumps({'text': response_text})}\n\n"
                yield "data: [DONE]\n\n"

            fast_response = StreamingResponse(generate_music(), media_type="text/event-stream")
            set_session_cookie(fast_response, session_id)
            return fast_response
        # ── End music fast-path ───────────────────────────────────────────────

        # Validate image payload if present
        image_error = validate_image(request.image_base64, request.image_mime)
        if image_error:
            log.warning("chat.image_invalid | session_id=%s reason=%s", session_id, image_error)
            return JSONResponse({"error": image_error, "code": "INVALID_IMAGE"}, status_code=422)

        graph_state = {
            "user_id": user_id,
            "session_id": session_id,
            "message": request.message,
            "system": effective_system,
            "source": chat_source,
            "modality": "voice" if chat_source == "voice" else "chat",
            # Pass image through state (ephemeral, not persisted to memory)
            "image_base64": request.image_base64,
            "image_mime": request.image_mime,
        }
        graph_runner = resolve_graph_runner()

        async def generate():
            assistant_accumulated = ""
            start_time = time.monotonic()
            first_token_time: float | None = None
            active_model = chat_model
            intent_for_log = "quick-local"
            confidence_for_log = 1.0
            route_for_log = "local"
            fallback_used = False

            try:
                async for event in graph_runner.astream(
                    graph_state,
                    config=checkpoint_config(session_id),
                    stream_mode="custom",
                ):
                    if "meta" in event:
                        meta = event["meta"]
                        active_model = meta.get("model", chat_model)
                        intent_for_log = meta.get("intent", "")
                        confidence_for_log = float(meta.get("confidence", 0.0))
                        route_for_log = meta.get("route_type", "local")
                        log.info(
                            "chat.route | session_id=%s source=%s intent=%s confidence=%.3f route=%s model=%s "
                            "planner_status=%s needs_memory=%s tool=%s",
                            session_id, chat_source, intent_for_log, confidence_for_log, route_for_log,
                            active_model, meta.get("planner_status", ""), meta.get("needs_memory", False),
                            meta.get("tool"),
                        )
                        if meta.get("reasoning_summary"):
                            log.debug("chat.planner_reasoning | session_id=%s reasoning=%s", session_id, meta["reasoning_summary"])
                        yield f"data: {json.dumps({'model': active_model, 'intent': intent_for_log, 'confidence': confidence_for_log, 'route_type': route_for_log, 'planner_status': meta.get('planner_status', ''), 'reasoning_summary': meta.get('reasoning_summary', '')})}\n\n"
                    elif "text" in event:
                        chunk = event["text"]
                        if first_token_time is None:
                            first_token_time = time.monotonic()
                        assistant_accumulated += chunk
                        yield f"data: {json.dumps({'text': chunk})}\n\n"
                    elif "thinking" in event:
                        yield f"data: {json.dumps({'thinking': event['thinking']})}\n\n"
                    elif "notice" in event:
                        fallback_used = True
                        yield f"data: {json.dumps({'notice': event['notice']})}\n\n"
                    elif "memory" in event:
                        yield f"data: {json.dumps({'memory': event['memory']})}\n\n"
                    elif event.get("fallback"):
                        active_model = event.get("model", chat_model)
                        yield f"data: {json.dumps(event)}\n\n"
            except Exception as exc:
                log.error("chat.graph_error | session_id=%s error=%s", session_id, exc)
                yield f"data: {json.dumps(stream_error_payload())}\n\n"

            completion_time = time.monotonic()
            first_token_ms = (first_token_time - start_time) * 1000 if first_token_time else -1
            completion_ms = (completion_time - start_time) * 1000
            log.info(
                "chat.telemetry | session_id=%s intent=%s confidence=%.3f route=%s "
                "model=%s fallback=%s first_token_ms=%.0f completion_ms=%.0f response_tokens_approx=%d",
                session_id, intent_for_log, confidence_for_log, route_for_log,
                active_model, fallback_used, first_token_ms, completion_ms,
                estimate_tokens(assistant_accumulated),
            )

            voice_meta = voice_tts_metadata(chat_source)
            # Images are visual; suppress auto-TTS for vision responses.
            if intent_for_log == "vision":
                voice_meta = None
            if voice_meta is not None:
                yield "data: " + json.dumps(voice_meta) + "\n\n"

            yield "data: [DONE]\n\n"

        response = StreamingResponse(generate(), media_type="text/event-stream")
        set_session_cookie(response, session_id)
        if session_created:
            log.info("chat.session.cookie_set | session_id=%s", session_id)
        return response

    @router.get("/graph/state/{session_id}")
    async def get_graph_state(session_id: str, http_request: Request):
        user_id: str = http_request.state.user_id

        # Verify session ownership before checking graph availability.
        if not memory_store.session_exists_for_user(session_id, user_id):
            return error_response("Session not found", "SESSION_NOT_FOUND", False, status_code=404)

        graph_runner = get_state_graph_runner()
        if graph_runner is None:
            return error_response("Graph not initialized", "GRAPH_UNAVAILABLE", True, status_code=503)

        try:
            if hasattr(graph_runner, "aget_state"):
                snapshot = await graph_runner.aget_state(checkpoint_config(session_id))
            else:
                snapshot = graph_runner.get_state(checkpoint_config(session_id))
        except Exception as exc:
            log.error("graph.state.error | session_id=%s error=%s", session_id, exc)
            return error_response("Graph state unavailable", "GRAPH_STATE_UNAVAILABLE", True, status_code=503)

        return JSONResponse(
            {
                "session_id": session_id,
                "state": getattr(snapshot, "values", {}) or {},
                "next": list(getattr(snapshot, "next", ()) or ()),
                "metadata": getattr(snapshot, "metadata", {}) or {},
            }
        )

    return router
