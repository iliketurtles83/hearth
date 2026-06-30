from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth import AuthError
from routes.auth_routes import create_auth_router
from routes import auth_routes


class _FakeAuthService:
    def __init__(self) -> None:
        self.login_calls = 0
        self.register_calls = 0

    def login(self, username: str, password: str, device_name: str | None = None, persistent: bool = False) -> dict:
        self.login_calls += 1
        if username == "alice" and password == "correct-password":
            return {
                "user_id": "alice",
                "username": "alice",
                "token": "a" * 64,
                "expires_at": 9999999999.0,
            }
        raise AuthError("Invalid username or password.", code="INVALID_CREDENTIALS", status=401)

    def register(self, username: str, password: str, device_name: str | None = None, persistent: bool = False) -> dict:
        self.register_calls += 1
        if username == "taken":
            raise AuthError("Username is already taken.", code="USERNAME_TAKEN", status=409)
        return {
            "user_id": username.lower(),
            "username": username.lower(),
            "token": "b" * 64,
            "expires_at": 9999999999.0,
        }

    def revoke_token(self, _token: str) -> bool:
        return True

    def revoke_all_tokens(self, _user_id: str) -> int:
        return 1

    def get_user(self, user_id: str) -> dict | None:
        return {"user_id": user_id, "username": user_id}



def _error_response(message: str, code: str, retryable: bool, status_code: int = 400):
    from fastapi.responses import JSONResponse

    return JSONResponse(
        {"error": message, "code": code, "retryable": retryable},
        status_code=status_code,
    )



def _build_client() -> tuple[TestClient, _FakeAuthService]:
    # Reset limiter state to isolate tests.
    auth_routes._auth_rate_limiter._buckets.clear()

    svc = _FakeAuthService()
    app = FastAPI()
    app.include_router(
        create_auth_router(
            auth_service=svc,
            auth_cookie_name="auth_token",
            session_cookie_secure=False,
            extract_bearer_token=lambda _req: None,
            error_response=_error_response,
        )
    )
    return TestClient(app), svc



def test_login_rate_limited_after_repeated_failures():
    client, svc = _build_client()

    for _ in range(auth_routes.AUTH_LOGIN_RATE_LIMIT_MAX_ATTEMPTS):
        resp = client.post(
            "/auth/login",
            json={"username": "alice", "password": "wrong", "persistent": False},
            headers={"x-real-ip": "10.1.2.3"},
        )
        assert resp.status_code == 401

    limited = client.post(
        "/auth/login",
        json={"username": "alice", "password": "wrong", "persistent": False},
        headers={"x-real-ip": "10.1.2.3"},
    )
    assert limited.status_code == 429
    assert limited.json()["code"] == "RATE_LIMITED"
    assert int(limited.headers.get("Retry-After", "0")) >= 1
    assert svc.login_calls == auth_routes.AUTH_LOGIN_RATE_LIMIT_MAX_ATTEMPTS



def test_register_rate_limited_after_repeated_failures():
    client, svc = _build_client()

    for _ in range(auth_routes.AUTH_REGISTER_RATE_LIMIT_MAX_ATTEMPTS):
        resp = client.post(
            "/auth/register",
            json={"username": "taken", "password": "longpassword", "persistent": False},
            headers={"x-real-ip": "10.9.8.7"},
        )
        assert resp.status_code == 409

    limited = client.post(
        "/auth/register",
        json={"username": "taken", "password": "longpassword", "persistent": False},
        headers={"x-real-ip": "10.9.8.7"},
    )
    assert limited.status_code == 429
    assert limited.json()["code"] == "RATE_LIMITED"
    assert int(limited.headers.get("Retry-After", "0")) >= 1
    assert svc.register_calls == auth_routes.AUTH_REGISTER_RATE_LIMIT_MAX_ATTEMPTS
