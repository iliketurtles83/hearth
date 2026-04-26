"""
Authentication service — named users with persistent bearer tokens.

Design:
  - Users register once (username + password).  Passwords hashed with scrypt.
  - Login returns an opaque token stored in the ``auth_tokens`` SQLite table.
  - Tokens have a configurable TTL (default 30 days for device tokens,
    8 hours for browser-session tokens).
  - ``verify_token(token)`` returns the user_id, or None for invalid/expired.

Token format: random 32-byte hex string (no embedded user info — looked up in DB).

Environment variables:
  AUTH_TOKEN_TTL_SECONDS   long-lived device token lifetime  (default: 2592000 = 30 days)
  AUTH_SESSION_TTL_SECONDS short-lived browser token lifetime (default: 28800  = 8 hours)
  AUTH_MIN_PASSWORD_LEN    minimum password length           (default: 8)
"""

from __future__ import annotations

import logging
import os
import re
import secrets
import sqlite3
import time
from hashlib import scrypt
from threading import Lock

log = logging.getLogger("assistant.auth")

AUTH_TOKEN_TTL_SECONDS: int = int(os.getenv("AUTH_TOKEN_TTL_SECONDS", str(60 * 60 * 24 * 30)))
AUTH_SESSION_TTL_SECONDS: int = int(os.getenv("AUTH_SESSION_TTL_SECONDS", str(60 * 60 * 8)))
AUTH_MIN_PASSWORD_LEN: int = int(os.getenv("AUTH_MIN_PASSWORD_LEN", "8"))

_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_]{3,30}$")

# scrypt parameters — OWASP-recommended minimum.
_SCRYPT_N = 16384
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32


class AuthError(Exception):
    """Raised for expected authentication failures (wrong password, user exists, etc.)."""

    def __init__(self, message: str, code: str, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


class AuthService:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id       TEXT PRIMARY KEY,
                    username      TEXT NOT NULL UNIQUE,
                    pw_hash       TEXT NOT NULL,
                    pw_salt       TEXT NOT NULL,
                    created_at    REAL NOT NULL,
                    last_login_at REAL
                );

                CREATE TABLE IF NOT EXISTS auth_tokens (
                    token        TEXT PRIMARY KEY,
                    user_id      TEXT NOT NULL REFERENCES users(user_id),
                    device_name  TEXT,
                    created_at   REAL NOT NULL,
                    last_used_at REAL NOT NULL,
                    expires_at   REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_auth_tokens_user_id
                    ON auth_tokens(user_id);
                """
            )
            self._conn.commit()

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _hash_password(password: str, salt: bytes) -> str:
        dk = scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=_SCRYPT_N,
            r=_SCRYPT_R,
            p=_SCRYPT_P,
            dklen=_SCRYPT_DKLEN,
        )
        return dk.hex()

    @staticmethod
    def _new_salt() -> bytes:
        return secrets.token_bytes(16)

    @staticmethod
    def _new_token() -> str:
        return secrets.token_hex(32)

    @staticmethod
    def _user_id_from_username(username: str) -> str:
        """Derive a stable, URL-safe user_id from a username (lowercased)."""
        return username.lower()

    # ── public API ────────────────────────────────────────────────────────────

    def register(
        self,
        username: str,
        password: str,
        device_name: str | None = None,
        persistent: bool = False,
    ) -> dict:
        """Create a new user and return a token.

        Raises AuthError on:
          - invalid username format (code: INVALID_USERNAME)
          - password too short     (code: PASSWORD_TOO_SHORT)
          - username taken         (code: USERNAME_TAKEN, status 409)
        """
        if not _USERNAME_RE.match(username):
            raise AuthError(
                "Username must be 3–30 characters: letters, digits, underscores only.",
                code="INVALID_USERNAME",
                status=400,
            )
        username = username.lower()  # normalise to lowercase; user_id == username
        if len(password) < AUTH_MIN_PASSWORD_LEN:
            raise AuthError(
                f"Password must be at least {AUTH_MIN_PASSWORD_LEN} characters.",
                code="PASSWORD_TOO_SHORT",
                status=400,
            )

        user_id = self._user_id_from_username(username)
        salt = self._new_salt()
        pw_hash = self._hash_password(password, salt)
        now = time.time()
        ttl = AUTH_TOKEN_TTL_SECONDS if persistent else AUTH_SESSION_TTL_SECONDS
        token = self._new_token()

        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute(
                    "INSERT INTO users (user_id, username, pw_hash, pw_salt, created_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (user_id, username, pw_hash, salt.hex(), now),
                )
            except sqlite3.IntegrityError:
                raise AuthError("Username is already taken.", code="USERNAME_TAKEN", status=409)

            cur.execute(
                "INSERT INTO auth_tokens (token, user_id, device_name, created_at, last_used_at, expires_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (token, user_id, device_name, now, now, now + ttl),
            )
            self._conn.commit()

        log.info("auth.register | user_id=%s persistent=%s", user_id, persistent)
        return {"user_id": user_id, "username": username, "token": token, "expires_at": now + ttl}

    def login(
        self,
        username: str,
        password: str,
        device_name: str | None = None,
        persistent: bool = False,
    ) -> dict:
        """Verify credentials and issue a new token.

        Raises AuthError on:
          - user not found / wrong password (code: INVALID_CREDENTIALS, status 401)
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT user_id, username, pw_hash, pw_salt FROM users WHERE username = ?",
                (username.lower(),),
            ).fetchone()

        if row is None:
            raise AuthError("Invalid username or password.", code="INVALID_CREDENTIALS", status=401)

        expected = self._hash_password(password, bytes.fromhex(row["pw_salt"]))
        if not secrets.compare_digest(expected, row["pw_hash"]):
            raise AuthError("Invalid username or password.", code="INVALID_CREDENTIALS", status=401)

        now = time.time()
        ttl = AUTH_TOKEN_TTL_SECONDS if persistent else AUTH_SESSION_TTL_SECONDS
        token = self._new_token()

        with self._lock:
            self._conn.execute(
                "INSERT INTO auth_tokens (token, user_id, device_name, created_at, last_used_at, expires_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (token, row["user_id"], device_name, now, now, now + ttl),
            )
            self._conn.execute(
                "UPDATE users SET last_login_at = ? WHERE user_id = ?",
                (now, row["user_id"]),
            )
            self._conn.commit()

        log.info("auth.login | user_id=%s persistent=%s", row["user_id"], persistent)
        return {
            "user_id": row["user_id"],
            "username": row["username"],
            "token": token,
            "expires_at": now + ttl,
        }

    def verify_token(self, token: str) -> str | None:
        """Return the user_id for a valid, non-expired token; None otherwise.

        Also touches ``last_used_at`` so monitoring can detect stale tokens.
        """
        if not token or len(token) != 64:  # 32 bytes → 64 hex chars
            return None

        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT user_id, expires_at FROM auth_tokens WHERE token = ?",
                (token,),
            ).fetchone()
            if row is None or now > row["expires_at"]:
                return None
            self._conn.execute(
                "UPDATE auth_tokens SET last_used_at = ? WHERE token = ?",
                (now, token),
            )
            self._conn.commit()
        return str(row["user_id"])

    def revoke_token(self, token: str) -> bool:
        """Delete a single token (logout from one device)."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM auth_tokens WHERE token = ?", (token,))
            self._conn.commit()
        return cur.rowcount > 0

    def revoke_all_tokens(self, user_id: str) -> int:
        """Delete all tokens for a user (global logout)."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM auth_tokens WHERE user_id = ?", (user_id,)
            )
            self._conn.commit()
        log.info("auth.revoke_all | user_id=%s count=%d", user_id, cur.rowcount)
        return cur.rowcount

    def get_user(self, user_id: str) -> dict | None:
        """Return public user info or None if not found."""
        with self._lock:
            row = self._conn.execute(
                "SELECT user_id, username, created_at, last_login_at FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "user_id": row["user_id"],
            "username": row["username"],
            "created_at": row["created_at"],
            "last_login_at": row["last_login_at"],
        }

    def purge_expired_tokens(self) -> int:
        """Remove all expired tokens. Call periodically to keep the table lean."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM auth_tokens WHERE expires_at < ?", (time.time(),)
            )
            self._conn.commit()
        return cur.rowcount