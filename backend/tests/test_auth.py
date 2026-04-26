"""Tests for AuthService — registration, login, token lifecycle, error paths."""
import time
import pytest

from auth import AuthService, AuthError


@pytest.fixture
def svc(tmp_path):
    return AuthService(str(tmp_path / "auth.db"))


# ── Registration ──────────────────────────────────────────────────────────────

def test_register_returns_token(svc):
    result = svc.register("alice", "hunter2secret")
    assert "token" in result
    assert result["username"] == "alice"
    assert result["user_id"] == "alice"


def test_register_normalises_username_to_lowercase(svc):
    result = svc.register("BOB", "password123")
    assert result["user_id"] == "bob"
    assert result["username"] == "bob"


def test_register_duplicate_username_raises(svc):
    svc.register("alice", "password123")
    with pytest.raises(AuthError) as exc_info:
        svc.register("alice", "different")
    assert exc_info.value.code == "USERNAME_TAKEN"


def test_register_short_password_raises(svc):
    with pytest.raises(AuthError) as exc_info:
        svc.register("dave", "short")
    assert exc_info.value.code == "PASSWORD_TOO_SHORT"


def test_register_empty_username_raises(svc):
    with pytest.raises(AuthError):
        svc.register("", "validpassword")


# ── Login ─────────────────────────────────────────────────────────────────────

def test_login_correct_password(svc):
    svc.register("alice", "hunter2secret")
    result = svc.login("alice", "hunter2secret")
    assert "token" in result
    assert result["username"] == "alice"


def test_login_wrong_password_raises(svc):
    svc.register("alice", "hunter2secret")
    with pytest.raises(AuthError) as exc_info:
        svc.login("alice", "wrongpassword")
    assert exc_info.value.code == "INVALID_CREDENTIALS"
    assert exc_info.value.status == 401


def test_login_unknown_user_raises(svc):
    with pytest.raises(AuthError) as exc_info:
        svc.login("nobody", "password123")
    assert exc_info.value.code == "INVALID_CREDENTIALS"


def test_login_case_insensitive_username(svc):
    svc.register("alice", "hunter2secret")
    result = svc.login("ALICE", "hunter2secret")
    assert result["user_id"] == "alice"


# ── Token verification ────────────────────────────────────────────────────────

def test_verify_token_valid(svc):
    result = svc.register("alice", "hunter2secret")
    user_id = svc.verify_token(result["token"])
    assert user_id == "alice"


def test_verify_token_invalid(svc):
    assert svc.verify_token("notarealtoken") is None


def test_verify_token_empty(svc):
    assert svc.verify_token("") is None


def test_verify_token_after_revoke(svc):
    result = svc.register("alice", "hunter2secret")
    token = result["token"]
    svc.revoke_token(token)
    assert svc.verify_token(token) is None


# ── Token TTL ─────────────────────────────────────────────────────────────────

def test_expired_token_returns_none(svc):
    """Create a token that expires immediately and verify it's rejected."""
    import sqlite3, time, secrets

    result = svc.register("alice", "hunter2secret")
    # Manually insert an already-expired token
    expired_token = secrets.token_hex(32)
    now = time.time()
    with sqlite3.connect(svc._db_path) as conn:
        conn.execute(
            "INSERT INTO auth_tokens(token, user_id, device_name, created_at, last_used_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (expired_token, "alice", "test", now, now, now - 1),
        )
    assert svc.verify_token(expired_token) is None


# ── Revocation ────────────────────────────────────────────────────────────────

def test_revoke_all_tokens(svc):
    reg = svc.register("alice", "hunter2secret")
    login = svc.login("alice", "hunter2secret")
    count = svc.revoke_all_tokens("alice")
    assert count == 2
    assert svc.verify_token(reg["token"]) is None
    assert svc.verify_token(login["token"]) is None


# ── get_user ──────────────────────────────────────────────────────────────────

def test_get_user_exists(svc):
    svc.register("alice", "hunter2secret")
    user = svc.get_user("alice")
    assert user is not None
    assert user["username"] == "alice"
    assert "pw_hash" not in user


def test_get_user_unknown(svc):
    assert svc.get_user("ghost") is None


# ── Persistent vs session tokens ──────────────────────────────────────────────

def test_persistent_token_has_longer_ttl(svc):
    """Persistent tokens should expire later than session tokens."""
    import sqlite3

    res_session = svc.register("alice", "hunter2secret", persistent=False)
    res_persist = svc.login("alice", "hunter2secret", persistent=True)

    def get_expires(token):
        with sqlite3.connect(svc._db_path) as conn:
            row = conn.execute(
                "SELECT expires_at FROM auth_tokens WHERE token=?", (token,)
            ).fetchone()
        return row[0] if row else None

    session_exp = get_expires(res_session["token"])
    persist_exp = get_expires(res_persist["token"])
    assert persist_exp > session_exp
