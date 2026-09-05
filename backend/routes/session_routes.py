from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app_schemas import SessionSelectRequest


def create_session_router(services) -> APIRouter:
    router = APIRouter()

    memory_store = services.memory_store
    session_cookie_name = services.session_cookie_name
    set_session_cookie = services.set_session_cookie
    error_response = services.error_response
    # NOTE: clear_checkpoint_thread is accessed via `services.` at call time (not
    # bound to a local) so tests can monkeypatch main.services.clear_checkpoint_thread.

    @router.get("/chat/sessions")
    async def list_chat_sessions(http_request: Request):
        user_id: str = http_request.state.user_id
        current_session_id = http_request.cookies.get(session_cookie_name)
        sessions = memory_store.list_sessions(user_id)
        # Validate that current_session_id belongs to the user; strip foreign cookies.
        if current_session_id and not any(s["session_id"] == current_session_id for s in sessions):
            current_session_id = None
        return JSONResponse(
            {
                "sessions": sessions,
                "current_session_id": current_session_id,
            }
        )

    @router.get("/chat/session/messages")
    async def get_chat_session_messages(http_request: Request):
        user_id: str = http_request.state.user_id
        cookie_session_id = http_request.cookies.get(session_cookie_name)
        if cookie_session_id:
            # Check ownership: if session exists for another user, this is a stale/foreign cookie.
            if (
                memory_store.session_exists(cookie_session_id)
                and not memory_store.session_exists_for_user(cookie_session_id, user_id)
            ):
                # Foreign session — generate a new one for this user.
                session_id = str(uuid4())
            else:
                session_id = cookie_session_id
        else:
            session_id = str(uuid4())
        turns = memory_store.get_session_turns(session_id, user_id, limit=500)
        response = JSONResponse({"session_id": session_id, "messages": turns})
        set_session_cookie(response, session_id)
        return response

    @router.post("/chat/session/new")
    async def create_chat_session(http_request: Request):
        _ = http_request
        session_id = str(uuid4())
        response = JSONResponse({"ok": True, "session_id": session_id})
        set_session_cookie(response, session_id)
        return response

    @router.post("/chat/session/select")
    async def select_chat_session(
        payload: SessionSelectRequest,
        http_request: Request,
    ):
        user_id: str = http_request.state.user_id
        sessions = memory_store.list_sessions(user_id)
        if not any(s.get("session_id") == payload.session_id for s in sessions):
            return error_response("Session not found", "SESSION_NOT_FOUND", False, status_code=404)

        response = JSONResponse({"ok": True, "session_id": payload.session_id})
        set_session_cookie(response, payload.session_id)
        return response

    @router.delete("/chat/sessions/{session_id}")
    async def delete_chat_session(
        session_id: str,
        http_request: Request,
    ):
        user_id: str = http_request.state.user_id
        current_session_id = http_request.cookies.get(session_cookie_name)

        if not memory_store.session_exists_for_user(session_id, user_id):
            return error_response("Session not found", "SESSION_NOT_FOUND", False, status_code=404)

        memory_store.delete_session(session_id, user_id)
        await services.clear_checkpoint_thread(session_id)

        next_session_id: str | None = None
        if current_session_id == session_id:
            sessions = memory_store.list_sessions(user_id)
            next_session_id = sessions[0]["session_id"] if sessions else str(uuid4())

        payload: dict[str, object] = {"ok": True, "session_id": session_id}
        if next_session_id:
            payload["active_session_id"] = next_session_id

        response = JSONResponse(payload)
        if next_session_id:
            set_session_cookie(response, next_session_id)
        return response

    @router.delete("/chat/session")
    async def reset_chat_session(http_request: Request):
        user_id: str = http_request.state.user_id
        session_id = http_request.cookies.get(session_cookie_name) or str(uuid4())
        memory_store.reset_session(session_id, user_id)
        await services.clear_checkpoint_thread(session_id)
        response = JSONResponse({"ok": True, "session_id": session_id})
        set_session_cookie(response, session_id)
        return response

    return router
