from __future__ import annotations

import os
import time
from collections import deque
from threading import Lock
from typing import Callable

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app_schemas import LoginRequest, RegisterRequest
from auth import AuthError


AUTH_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("AUTH_RATE_LIMIT_WINDOW_SECONDS", "60"))
AUTH_LOGIN_RATE_LIMIT_MAX_ATTEMPTS = int(os.getenv("AUTH_LOGIN_RATE_LIMIT_MAX_ATTEMPTS", "10"))
AUTH_REGISTER_RATE_LIMIT_MAX_ATTEMPTS = int(os.getenv("AUTH_REGISTER_RATE_LIMIT_MAX_ATTEMPTS", "5"))


class _SlidingWindowRateLimiter:
    def __init__(self, window_seconds: int):
        self.window_seconds = max(1, int(window_seconds))
        self._lock = Lock()
        self._buckets: dict[str, deque[float]] = {}

    def _prune(self, bucket: deque[float], now: float) -> None:
        cutoff = now - self.window_seconds
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

    def is_limited(self, key: str, max_attempts: int, now: float | None = None) -> tuple[bool, int]:
        now = now if now is not None else time.time()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                return False, 0
            self._prune(bucket, now)
            if len(bucket) < max_attempts:
                return False, 0
            retry_after = max(1, int(self.window_seconds - (now - bucket[0])))
            return True, retry_after

    def add_attempt(self, key: str, now: float | None = None) -> None:
        now = now if now is not None else time.time()
        with self._lock:
            bucket = self._buckets.setdefault(key, deque())
            self._prune(bucket, now)
            bucket.append(now)

    def clear(self, key: str) -> None:
        with self._lock:
            self._buckets.pop(key, None)


_auth_rate_limiter = _SlidingWindowRateLimiter(AUTH_RATE_LIMIT_WINDOW_SECONDS)


def _client_ip(request: Request) -> str:
    # Trust Caddy-forwarded headers when present; otherwise use direct client host.
    x_real_ip = request.headers.get("x-real-ip", "").strip()
    if x_real_ip:
        return x_real_ip
    xff = request.headers.get("x-forwarded-for", "").strip()
    if xff:
        return xff.split(",", 1)[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def create_auth_router(
    *,
    auth_service,
    auth_cookie_name: str,
    session_cookie_secure: bool,
    extract_bearer_token: Callable[[Request], str | None],
    error_response: Callable[[str, str, bool, int], JSONResponse],
) -> APIRouter:
    router = APIRouter()

    def _rate_limited_response(retry_after: int) -> JSONResponse:
        resp = error_response(
            "Too many authentication attempts. Please try again shortly.",
            "RATE_LIMITED",
            True,
            status_code=429,
        )
        resp.headers["Retry-After"] = str(max(1, retry_after))
        return resp

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
    async def auth_register(payload: RegisterRequest, http_request: Request):
        client_ip = _client_ip(http_request)
        username = (payload.username or "").strip().lower()

        ip_limited, ip_retry_after = _auth_rate_limiter.is_limited(
            f"register_ip:{client_ip}",
            AUTH_REGISTER_RATE_LIMIT_MAX_ATTEMPTS,
        )
        user_limited, user_retry_after = _auth_rate_limiter.is_limited(
            f"register_user:{username}",
            AUTH_REGISTER_RATE_LIMIT_MAX_ATTEMPTS,
        )
        if ip_limited or user_limited:
            return _rate_limited_response(max(ip_retry_after, user_retry_after))

        try:
            result = auth_service.register(
                payload.username,
                payload.password,
                device_name=payload.device_name,
                persistent=payload.persistent,
            )
        except AuthError as exc:
            _auth_rate_limiter.add_attempt(f"register_ip:{client_ip}")
            _auth_rate_limiter.add_attempt(f"register_user:{username}")
            return error_response(str(exc), exc.code, False, status_code=exc.status)

        _auth_rate_limiter.clear(f"register_ip:{client_ip}")
        _auth_rate_limiter.clear(f"register_user:{username}")
        response = JSONResponse(result, status_code=201)
        _auth_cookie(response, result["token"], result["expires_at"])
        return response

    @router.post("/auth/login")
    async def auth_login(payload: LoginRequest, http_request: Request):
        client_ip = _client_ip(http_request)
        username = (payload.username or "").strip().lower()

        ip_limited, ip_retry_after = _auth_rate_limiter.is_limited(
            f"login_ip:{client_ip}",
            AUTH_LOGIN_RATE_LIMIT_MAX_ATTEMPTS,
        )
        user_limited, user_retry_after = _auth_rate_limiter.is_limited(
            f"login_user:{username}",
            AUTH_LOGIN_RATE_LIMIT_MAX_ATTEMPTS,
        )
        if ip_limited or user_limited:
            return _rate_limited_response(max(ip_retry_after, user_retry_after))

        try:
            result = auth_service.login(
                payload.username,
                payload.password,
                device_name=payload.device_name,
                persistent=payload.persistent,
            )
        except AuthError as exc:
            _auth_rate_limiter.add_attempt(f"login_ip:{client_ip}")
            _auth_rate_limiter.add_attempt(f"login_user:{username}")
            return error_response(str(exc), exc.code, False, status_code=exc.status)

        _auth_rate_limiter.clear(f"login_ip:{client_ip}")
        _auth_rate_limiter.clear(f"login_user:{username}")
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
