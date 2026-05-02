from __future__ import annotations

import time
from typing import Callable

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app_schemas import LoginRequest, RegisterRequest
from auth import AuthError


def create_auth_router(
    *,
    auth_service,
    auth_cookie_name: str,
    session_cookie_secure: bool,
    extract_bearer_token: Callable[[Request], str | None],
    error_response: Callable[[str, str, bool, int], JSONResponse],
) -> APIRouter:
    router = APIRouter()

    def _auth_cookie(response, token: str, expires_at: float) -> None:
        max_age = max(0, int(expires_at - time.time()))
        response.set_cookie(
            key=auth_cookie_name,
            value=token,
            httponly=True,
            samesite="lax",
            secure=session_cookie_secure,
            max_age=max_age,
        )

    @router.post("/auth/register")
    async def auth_register(payload: RegisterRequest):
        try:
            result = auth_service.register(
                payload.username,
                payload.password,
                device_name=payload.device_name,
                persistent=payload.persistent,
            )
        except AuthError as exc:
            return error_response(str(exc), exc.code, False, status_code=exc.status)
        response = JSONResponse(result, status_code=201)
        _auth_cookie(response, result["token"], result["expires_at"])
        return response

    @router.post("/auth/login")
    async def auth_login(payload: LoginRequest):
        try:
            result = auth_service.login(
                payload.username,
                payload.password,
                device_name=payload.device_name,
                persistent=payload.persistent,
            )
        except AuthError as exc:
            return error_response(str(exc), exc.code, False, status_code=exc.status)
        response = JSONResponse(result)
        _auth_cookie(response, result["token"], result["expires_at"])
        return response

    @router.post("/auth/logout")
    async def auth_logout(http_request: Request):
        token = extract_bearer_token(http_request)
        if token:
            auth_service.revoke_token(token)
        response = JSONResponse({"ok": True})
        response.delete_cookie(auth_cookie_name)
        return response

    @router.post("/auth/logout/all")
    async def auth_logout_all(http_request: Request):
        user_id: str = http_request.state.user_id
        count = auth_service.revoke_all_tokens(user_id)
        response = JSONResponse({"ok": True, "revoked": count})
        response.delete_cookie(auth_cookie_name)
        return response

    @router.get("/auth/me")
    async def auth_me(http_request: Request):
        user_id: str = http_request.state.user_id
        info = auth_service.get_user(user_id)
        if not info:
            return error_response("User not found.", "USER_NOT_FOUND", False, status_code=404)
        return JSONResponse(info)

    return router
