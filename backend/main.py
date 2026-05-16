from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from contextlib import asynccontextmanager
import asyncio
import base64
import httpx
import anthropic
import importlib
import importlib.util
import os
import json
import logging
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from threading import Lock
from uuid import uuid4
import numpy as np
from dotenv import load_dotenv
from router import route as router_route, CLOUD_MODEL, CHAT_MODEL, CODER_MODEL
from memory import MemoryStore
from graph import (
    build_assistant_graph,
    AssistantGraphDependencies,
    create_assistant_graph,
    checkpoint_config,
    default_checkpoint_path,
)
from auth import AuthService
from music_fastpath import parse_music_command, format_music_response
from routes.auth_routes import create_auth_router
from routes.code_file_routes import create_code_file_router
from routes.memory_tool_routes import create_memory_tool_router
from app_schemas import (
    ChatRequest as BaseChatRequest,
    TTSRequest,
    CodeRequest as BaseCodeRequest,
    SessionSelectRequest,
)
import tts

_BACKEND_DIR = os.path.dirname(__file__)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


def _import_local_tools_module():
    mod = importlib.import_module("tools")
    if hasattr(mod, "dispatch"):
        return mod

    tools_dir = Path(_BACKEND_DIR) / "tools"
    init_file = tools_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "assistant_backend_tools",
        str(init_file),
        submodule_search_locations=[str(tools_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load backend tools package")

    fallback = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fallback
    spec.loader.exec_module(fallback)
    return fallback


tools = _import_local_tools_module()
ToolResult = importlib.import_module(f"{tools.__name__}.base").ToolResult


async def _run_weather_tool(params: dict):
    weather_tool = importlib.import_module(f"{tools.__name__}.weather")
    return await weather_tool.run(params)



_memory_db_default = os.path.join(os.path.dirname(__file__), "memory.db")
_chroma_default = os.path.join(os.path.dirname(__file__), "chroma")
memory_store = MemoryStore(
    db_path=os.getenv("MEMORY_DB_PATH", _memory_db_default),
    chroma_path=os.getenv("CHROMA_PATH", _chroma_default),
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("assistant")

SESSION_COOKIE_NAME = os.getenv("CHAT_SESSION_COOKIE", "assistant_session")
SESSION_IDLE_TTL_SECONDS = int(os.getenv("CHAT_SESSION_IDLE_TTL_SECONDS", "1800"))
SESSION_MAX_ITEMS = int(os.getenv("CHAT_SESSION_MAX_ITEMS", "200"))
CHAT_TOKEN_BUDGET = int(os.getenv("CHAT_TOKEN_BUDGET", "1500"))
CHAT_MAX_TURNS = int(os.getenv("CHAT_MAX_TURNS", "24"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434").rstrip("/")
# Phase 14 — vision model: defaults to CHAT_MODEL (gemma:e4b is multimodal)
OLLAMA_VISION_MODEL: str = (
    os.getenv("OLLAMA_VISION_MODEL")
    or CHAT_MODEL
)
CHAT_SUMMARY_TRIGGER_MESSAGES = int(os.getenv("CHAT_SUMMARY_TRIGGER_MESSAGES", "18"))
CHAT_SUMMARY_KEEP_RECENT_MESSAGES = int(os.getenv("CHAT_SUMMARY_KEEP_RECENT_MESSAGES", "8"))
CHAT_SUMMARY_MAX_CHARS = int(os.getenv("CHAT_SUMMARY_MAX_CHARS", "1400"))
MEMORY_CONSOLIDATION_MODE = os.getenv("MEMORY_CONSOLIDATION_MODE", "on-session-end").strip().lower()
MEMORY_CONSOLIDATION_INTERVAL_SECONDS = int(os.getenv("MEMORY_CONSOLIDATION_INTERVAL_SECONDS", "86400"))
MEMORY_CONSOLIDATION_BATCH_SIZE = int(os.getenv("MEMORY_CONSOLIDATION_BATCH_SIZE", "50"))


def _load_hearth_prompt(filename: str, env_var: str, fallback: str) -> str:
    """Load Hearth's character prompt from file, with env-var and hardcoded fallbacks."""
    prompt_path = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(prompt_path):
        with open(prompt_path, encoding="utf-8") as _f:
            text = _f.read().strip()
        if text:
            return text
    return os.getenv(env_var, fallback)


CHAT_DEFAULT_SYSTEM_PROMPT = _load_hearth_prompt(
    "hearth_prompt.txt",
    "CHAT_DEFAULT_SYSTEM_PROMPT",
    "You are a helpful personal assistant. Be concise and accurate.",
)
CODE_DEFAULT_SYSTEM_PROMPT = _load_hearth_prompt(
    "hearth_coder_prompt.txt",
    "CODE_DEFAULT_SYSTEM_PROMPT",
    "You are a helpful coding assistant. Be concise and accurate.",
)

class ChatRequest(BaseChatRequest):
    system: str = CHAT_DEFAULT_SYSTEM_PROMPT


class CodeRequest(BaseCodeRequest):
    system: str = CODE_DEFAULT_SYSTEM_PROMPT

WAKEWORD_MODEL_FILE = os.getenv("WAKEWORD_MODEL_FILE", "computer_v2.onnx")
OWW_MELSPEC_MODEL_FILE = os.getenv("OWW_MELSPEC_MODEL_FILE", "melspectrogram.onnx")
OWW_EMBEDDING_MODEL_FILE = os.getenv("OWW_EMBEDDING_MODEL_FILE", "embedding_model.onnx")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "").strip().lower()
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "").strip().lower()

# ── Model swap latency baseline (measured 2026-04-28, RTX 3060 12 GB NVMe) ────
# Run backend/tests/test_swap_latency.py to re-measure after hardware changes.
# Measured cold-swap latency (gemma3:4b ↔ qwen2.5-coder:7b, n=10 each):
#   gemma3:4b→qwen2.5-coder:7b: median=0.2s  min=0.2s  max=1.9s
#   qwen2.5-coder:7b→gemma3:4b: median=0.3s  min=0.3s  max=2.4s
#   Overall median: 0.3s — imperceptible; loading-state UX not required.
# Interpretation: Ollama caches model weights in system RAM after GPU eviction
# (keep_alive=0). First-ever load hits disk (~2s); subsequent swaps are RAM→GPU
# re-pin only (~0.2-0.3s). Skip visible loading-state badge in Phase 10b.

# ── HTTPS / CORS / cookie policy (Phase 0b) ───────────────────────────────────
# CORS_ORIGINS: comma-separated list of allowed origins, e.g.
#   CORS_ORIGINS=https://192.168.1.42,https://assistant.lan
# Default '*' preserves the Phase 1 permissive behaviour for plain-HTTP dev.
# Set to the exact Caddy origin(s) once HTTPS is in use.
_cors_origins_raw = os.getenv("CORS_ORIGINS", "*")
_CORS_ORIGINS: list[str] = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
# allow_credentials requires a non-wildcard origin per the CORS spec.
_CORS_CREDENTIALS: bool = _CORS_ORIGINS != ["*"]

# SESSION_COOKIE_SECURE: set to 'true' when the browser-facing edge is HTTPS
# (i.e. when Caddy is in use). Tells the browser to send the cookie only over
# HTTPS connections. The backend itself may still run plain HTTP internally.
SESSION_COOKIE_SECURE: bool = os.getenv("SESSION_COOKIE_SECURE", "false").lower() == "true"
AUTH_COOKIE_NAME: str = os.getenv("AUTH_COOKIE_NAME", "auth_token")

_session_store_lock = Lock()
_session_store: dict[str, dict] = {}
_code_write_lock = Lock()
_pending_code_writes: dict[str, dict] = {}
_consolidation_loop_task: asyncio.Task | None = None

# ── Auth service (shared singleton) ───────────────────────────────────────────
_auth_db_default = os.path.join(os.path.dirname(__file__), "auth.db")
auth_service = AuthService(os.getenv("AUTH_DB_PATH", _auth_db_default))

# ── Startup validation ─────────────────────────────────────────────────────────
def _required_wake_models() -> list[str]:
    models = [WAKEWORD_MODEL_FILE, OWW_MELSPEC_MODEL_FILE, OWW_EMBEDDING_MODEL_FILE]
    return [m for m in models if m]


def _beets_db_has_items(db_path: str) -> bool:
    """Return True when Beets DB exists and has at least one item."""
    if not db_path or not os.path.isfile(db_path):
        return False

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='items'"
        )
        if cur.fetchone() is None:
            return False
        return conn.execute("SELECT 1 FROM items LIMIT 1").fetchone() is not None
    except sqlite3.Error as exc:
        log.warning("beets.bootstrap_check_failed | db=%s error=%s", db_path, exc)
        return False
    finally:
        if conn is not None:
            conn.close()


def _bootstrap_beets_library_if_empty() -> None:
    """Run `beet import -A` once when the Beets library database is empty."""
    beets_db = os.getenv(
        "BEETS_DB_PATH",
        os.path.join(os.path.expanduser("~"), ".config", "beets", "library.db"),
    )
    if _beets_db_has_items(beets_db):
        log.info("beets.bootstrap_skip | db=%s reason=already_populated", beets_db)
        return

    music_root = os.getenv("MUSIC_ROOT", "").strip()
    if not music_root:
        music_path = os.getenv("MUSIC_PATH", "").strip()
        hint = (
            "Set MUSIC_ROOT=/music (Docker) or to your local music directory (non-Docker)."
            if music_path
            else "Set MUSIC_ROOT to the directory used by Beets import (e.g. /music in Docker)."
        )
        log.warning(
            "beets.bootstrap_skip | db=%s reason=missing_music_root env=MUSIC_ROOT hint=%s",
            beets_db,
            hint,
        )
        return
    if not os.path.isdir(music_root):
        log.warning(
            "beets.bootstrap_skip | db=%s reason=invalid_music_root path=%s",
            beets_db,
            music_root,
        )
        return

    beet_bin = shutil.which("beet")
    if not beet_bin:
        log.warning(
            "beets.bootstrap_skip | db=%s reason=beet_not_found hint='Install beets or include it in container image'",
            beets_db,
        )
        return

    cmd = [beet_bin, "-l", beets_db, "import", "-A", music_root]
    log.info("beets.bootstrap_start | db=%s music_root=%s", beets_db, music_root)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        log.info("beets.bootstrap_done | db=%s", beets_db)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip().splitlines()
        tail = stderr[-1] if stderr else str(exc)
        log.warning("beets.bootstrap_failed | db=%s error=%s", beets_db, tail)


def _validate_startup() -> None:
    _models_dir = os.path.join(os.path.dirname(__file__), "models")
    required_models = _required_wake_models()
    missing_models = [m for m in required_models if not os.path.isfile(os.path.join(_models_dir, m))]
    if missing_models:
        log.warning("Missing ONNX model files (wake-word will fail): %s", missing_models)
        log.warning("Run: bash scripts/download-models.sh")

    if not os.getenv("ANTHROPIC_API_KEY"):
        log.warning("ANTHROPIC_API_KEY not set — cloud model fallback will be unavailable")

    # Phase 10b: code workspace
    _code_root = os.getenv("CODE_WORKSPACE_ROOT", "")
    if not _code_root:
        log.warning(
            "CODE_WORKSPACE_ROOT not set — code_tool node will refuse all file operations. "
            "Set it in .env and restart."
        )
    elif not os.path.isdir(_code_root):
        log.warning(
            "CODE_WORKSPACE_ROOT=%s does not exist or is not a directory — "
            "create it and mount it into the container before using the code tool.",
            _code_root,
        )
    else:
        # Start background workspace indexer
        _index_paths_raw = os.getenv("CODE_INDEX_PATHS", "")
        _index_paths = [p.strip() for p in _index_paths_raw.split() if p.strip()] or None
        from tools.code_indexer import start_background_index
        start_background_index(_code_root, os.getenv("CHROMA_PATH", _chroma_default), _index_paths)

    _bootstrap_beets_library_if_empty()

    log.info(
        "Startup OK | chat_model=%s | coder_model=%s | ollama=%s | cors_origins=%s | cookie_secure=%s",
        CHAT_MODEL, CODER_MODEL, OLLAMA_URL, _CORS_ORIGINS, SESSION_COOKIE_SECURE,
    )

_validate_startup()

# ── Auth middleware ────────────────────────────────────────────────────────────
# Resolves the bearer token (from Authorization header or auth_token cookie)
# and attaches user_id to request.state.  Returns 401 for protected routes
# that have no valid token.
_UNPROTECTED_PATHS = frozenset(["/health", "/", "/transcribe", "/ws/wake"])


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        # Static files, auth endpoints, and health are always open.
        # Static files are mounted at the application root ("/") so requests
        # like /message.js or /auth.js must be exempt from auth checks. Use
        # an extension whitelist to detect common static asset requests.
        _, ext = os.path.splitext(path)
        static_exts = {
            ".js",
            ".mjs",
            ".css",
            ".map",
            ".png",
            ".jpg",
            ".jpeg",
            ".svg",
            ".ico",
            ".wasm",
            ".woff2",
            ".woff",
            ".ttf",
            ".mp3",
            ".wav",
        }

        # Allow unauthenticated access to a small set of endpoints and static files.
        # NOTE: do NOT globally exempt the `/auth/` prefix — endpoints like
        # `/auth/me` must remain protected so they can validate bearer tokens.
        if (
            path in _UNPROTECTED_PATHS
            or path.startswith("/static")
            or path in ("/auth/login", "/auth/register")
            or (ext and ext.lower() in static_exts)
        ):
            request.state.user_id = None
            return await call_next(request)

        token = _extract_bearer_token(request)
        user_id = auth_service.verify_token(token) if token else None
        request.state.user_id = user_id

        if user_id is None:
            return JSONResponse(
                {"error": "Authentication required.", "code": "UNAUTHORIZED", "retryable": False},
                status_code=401,
            )
        return await call_next(request)


def _extract_bearer_token(request: Request) -> str | None:
    """Return the raw token from Authorization header or auth_token cookie."""
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    return request.cookies.get(AUTH_COOKIE_NAME)


# ── Cross-Origin isolation headers (required for SharedArrayBuffer / vad-web) ──
class COOPCOEPMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        # 'credentialless' (vs 'require-corp') still enables SharedArrayBuffer
        # (needed by vad-web's threaded WASM) while allowing cross-origin CDN
        # resources that don't set Cross-Origin-Resource-Policy headers.
        response.headers["Cross-Origin-Embedder-Policy"] = "credentialless"
        return response

# Module-level graph instance — initialized after stream_local/stream_cloud are defined.
# The lifespan (Slice 5) will replace this with a checkpointed version.
_assistant_graph = None  # type: ignore[assignment]


@asynccontextmanager
async def _graph_lifespan(_app: FastAPI):
    global _assistant_graph, _consolidation_loop_task
    async with create_assistant_graph(
        _make_graph_deps(),
        checkpoint_path=default_checkpoint_path(),
    ) as checkpointed_graph:
        _assistant_graph = checkpointed_graph
        _app.state.assistant_graph = checkpointed_graph
        if MEMORY_CONSOLIDATION_MODE == "interval":
            _consolidation_loop_task = asyncio.create_task(_consolidation_loop())
            log.info(
                "memory.consolidation.loop_started | mode=interval interval_seconds=%d",
                max(60, MEMORY_CONSOLIDATION_INTERVAL_SECONDS),
            )
        log.info("graph.ready | checkpointer=sqlite path=%s", default_checkpoint_path())
        try:
            yield
        finally:
            if _consolidation_loop_task:
                _consolidation_loop_task.cancel()
                _consolidation_loop_task = None

app = FastAPI(lifespan=_graph_lifespan)

app.add_middleware(COOPCOEPMiddleware)
# Auth middleware must be added after COOP/COEP so it runs on the resolved request.
app.add_middleware(AuthMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=_CORS_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── openWakeWord model (lazy-loaded on first WebSocket connection) ──
_oww_model = None

def get_oww_model():
    global _oww_model
    if _oww_model is None:
        from openwakeword.model import Model
        _models_dir = os.path.join(os.path.dirname(__file__), "models")
        # v0.6.0 removed bundled backbone models — pass explicit paths so AudioFeatures
        # doesn't look in the (empty) library resources directory.
        _oww_model = Model(
            wakeword_models=[os.path.join(_models_dir, WAKEWORD_MODEL_FILE)],
            inference_framework="onnx",
            melspec_model_path=os.path.join(_models_dir, OWW_MELSPEC_MODEL_FILE),
            embedding_model_path=os.path.join(_models_dir, OWW_EMBEDDING_MODEL_FILE),
        )
    return _oww_model

# ── faster-whisper model (lazy-loaded on first /transcribe call) ──
_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        if WHISPER_DEVICE:
            device = WHISPER_DEVICE
        else:
            device = "cuda" if os.path.exists("/dev/nvidia0") else "cpu"
        compute = WHISPER_COMPUTE_TYPE or ("float16" if device == "cuda" else "int8")
        _whisper_model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute)
    return _whisper_model


def _error_response(message: str, code: str, retryable: bool, status_code: int = 400) -> JSONResponse:
    return JSONResponse({"error": message, "code": code, "retryable": retryable}, status_code=status_code)


def _tts_error_status(code: str, retryable: bool) -> int:
    client_errors = {
        "TTS_INVALID_TEXT",
        "TTS_TEXT_TOO_LONG",
        "TTS_ENGINE_INVALID",
        "TTS_PIPER_CONFIG_INVALID",
        "TTS_KOKORO_CONFIG_INVALID",
    }
    unavailable_errors = {
        "TTS_ENGINE_UNAVAILABLE",
        "TTS_ENGINE_INIT_FAILED",
        "TTS_PIPER_MODEL_MISSING",
        "TTS_PIPER_MODEL_NOT_FOUND",
        "TTS_PIPER_BIN_NOT_FOUND",
        "TTS_PIPER_PITCH_UNSUPPORTED",
        "TTS_KOKORO_UNAVAILABLE",
        "TTS_KOKORO_INIT_FAILED",
        "TTS_KOKORO_BAD_RUNTIME",
    }

    if code in client_errors:
        return 400
    if code in unavailable_errors:
        return 503
    if retryable:
        return 502
    return 500


app.include_router(
    create_auth_router(
        auth_service=auth_service,
        auth_cookie_name=AUTH_COOKIE_NAME,
        session_cookie_secure=SESSION_COOKIE_SECURE,
        extract_bearer_token=_extract_bearer_token,
        error_response=_error_response,
    )
)

app.include_router(
    create_memory_tool_router(
        memory_store=memory_store,
        memory_consolidation_batch_size=MEMORY_CONSOLIDATION_BATCH_SIZE,
        error_response=_error_response,
        dispatch_tool=tools.dispatch,
        run_weather=_run_weather_tool,
    )
)

app.include_router(
    create_code_file_router(
        code_write_lock=_code_write_lock,
        pending_code_writes=_pending_code_writes,
        log=log,
    )
)


def _normalize_chat_source(source: str | None) -> str:
    s = (source or "text").strip().lower()
    return s if s in {"text", "voice"} else "text"


def _voice_tts_metadata(chat_source: str) -> dict | None:
    if chat_source != "voice":
        return None
    return {
        "voice": {
            "source": "voice",
            "tts_endpoint": "/tts",
            "tts_ready": True,
        }
    }


def _set_session_cookie(response: Response, session_id: str) -> None:
    """Apply consistent session-cookie attributes across all endpoints."""
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        httponly=True,
        samesite="lax",
        secure=SESSION_COOKIE_SECURE,
        max_age=SESSION_IDLE_TTL_SECONDS,
    )


def _estimate_tokens(text: str) -> int:
    # Lightweight token estimate for bounded context decisions.
    return max(1, len(text) // 4)


def _cleanup_expired_sessions(now: float) -> None:
    expired = [
        session_id
        for session_id, session in _session_store.items()
        if now - session["updated_at"] > SESSION_IDLE_TTL_SECONDS
    ]
    for session_id in expired:
        del _session_store[session_id]
    if expired:
        log.info("chat.session.evicted_expired | count=%d", len(expired))


def _evict_oldest_sessions_if_needed() -> None:
    if len(_session_store) <= SESSION_MAX_ITEMS:
        return
    overflow = len(_session_store) - SESSION_MAX_ITEMS
    oldest_first = sorted(_session_store.items(), key=lambda item: item[1]["updated_at"])
    for session_id, _ in oldest_first[:overflow]:
        del _session_store[session_id]
    log.info("chat.session.evicted_capacity | count=%d", overflow)


def _session_owned_by(session: dict | None, user_id: str) -> bool:
    return bool(session) and str(session.get("user_id") or "") == user_id


def _get_or_create_session(user_id: str, session_id: str | None) -> tuple[str, bool]:
    now = time.time()
    with _session_store_lock:
        _cleanup_expired_sessions(now)
        existing_session = _session_store.get(session_id) if session_id else None
        effective_id = session_id if session_id and _session_owned_by(existing_session, user_id) else str(uuid4())
        created = effective_id not in _session_store
        if created:
            _session_store[effective_id] = {
                "user_id": user_id,
                "messages": [],
                "summary": "",
                "summary_message_count": 0,
                "created_at": now,
                "updated_at": now,
            }
            log.info("chat.session.created | session_id=%s user_id=%s", effective_id, user_id)
        _evict_oldest_sessions_if_needed()
        return effective_id, created


def _select_history_for_budget(
    messages: list[dict],
    system: str,
    current_user_message: str,
    summary_text: str,
) -> tuple[list[dict], int, bool, int]:
    summary_tokens = _estimate_tokens(summary_text) if summary_text else 0
    history_budget = max(
        0,
        CHAT_TOKEN_BUDGET
        - _estimate_tokens(system)
        - _estimate_tokens(current_user_message)
        - summary_tokens
        - 32,
    )
    selected_reversed: list[dict] = []
    used_tokens = 0
    truncated = False
    max_messages = max(1, CHAT_MAX_TURNS * 2)
    candidates = messages[-max_messages:]

    for message in reversed(candidates):
        cost = _estimate_tokens(message["content"]) + 4
        if used_tokens + cost > history_budget:
            truncated = True
            continue
        selected_reversed.append(message)
        used_tokens += cost

    selected = list(reversed(selected_reversed))
    if len(messages) > len(selected):
        truncated = True
    return selected, used_tokens, truncated, summary_tokens


def _normalize_summary_line(text: str, max_len: int = 200) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"


def _summarize_messages_chunk(messages: list[dict]) -> str:
    lines: list[str] = []
    for message in messages:
        role = message.get("role")
        content = _normalize_summary_line(str(message.get("content", "")))
        if not content:
            continue
        if role == "user":
            lines.append(f"- User: {content}")
        elif role == "assistant":
            lines.append(f"- Assistant: {content}")
        if len(lines) >= 14:
            break
    return "\n".join(lines)


def _truncate_summary(summary: str) -> str:
    if len(summary) <= CHAT_SUMMARY_MAX_CHARS:
        return summary
    tail = summary[-CHAT_SUMMARY_MAX_CHARS :]
    first_newline = tail.find("\n")
    if first_newline > 0:
        return tail[first_newline + 1 :]
    return tail


def _build_episodic_record_text(session: dict) -> str:
    session_summary = str(session.get("summary", "") or "").strip()
    messages = list(session.get("messages", []))
    recent = _summarize_messages_chunk(messages[-14:]) if messages else ""

    if session_summary and recent:
        merged = f"{session_summary}\n{recent}".strip()
    else:
        merged = session_summary or recent
    return _truncate_summary(merged.strip())


async def _persist_session_episodic_snapshot(session_id: str, session_snapshot: dict, reason: str) -> None:
    user_id = str(session_snapshot.get("user_id") or "")
    if not user_id:
        return

    summary_text = _build_episodic_record_text(session_snapshot)
    if not summary_text:
        return

    try:
        row_id = await asyncio.to_thread(memory_store.save_summary, user_id, session_id, summary_text)
        log.info(
            "memory.episodic.saved | session_id=%s user_id=%s row_id=%s reason=%s",
            session_id,
            user_id,
            row_id,
            reason,
        )
    except Exception as exc:
        log.error(
            "memory.episodic.save_failed | session_id=%s user_id=%s reason=%s error=%s",
            session_id,
            user_id,
            reason,
            exc,
        )
        return

    if MEMORY_CONSOLIDATION_MODE != "on-session-end":
        return

    try:
        stats = await asyncio.to_thread(
            memory_store.consolidate_pending,
            user_id,
            MEMORY_CONSOLIDATION_BATCH_SIZE,
        )
        log.info(
            "memory.consolidation.on_session_end | user_id=%s processed=%d promoted=%d blocked=%d",
            user_id,
            int(stats.get("processed", 0)),
            int(stats.get("promoted", 0)),
            int(stats.get("blocked", 0)),
        )
    except Exception as exc:
        log.error("memory.consolidation.failed | user_id=%s error=%s", user_id, exc)


def _spawn_episodic_persist_task(session_id: str, session_snapshot: dict, reason: str) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(_persist_session_episodic_snapshot(session_id, session_snapshot, reason))


async def _consolidation_loop() -> None:
    interval = max(60, MEMORY_CONSOLIDATION_INTERVAL_SECONDS)
    while True:
        await asyncio.sleep(interval)
        try:
            stats = await asyncio.to_thread(memory_store.consolidate_pending, None, MEMORY_CONSOLIDATION_BATCH_SIZE)
            log.info(
                "memory.consolidation.interval | processed=%d promoted=%d blocked=%d",
                int(stats.get("processed", 0)),
                int(stats.get("promoted", 0)),
                int(stats.get("blocked", 0)),
            )
        except Exception as exc:
            log.error("memory.consolidation.interval_failed | error=%s", exc)


def _update_session_summary_if_needed(session_id: str) -> tuple[bool, int, int]:
    now = time.time()
    with _session_store_lock:
        session = _session_store.get(session_id)
        if not session:
            return False, 0, 0

        messages = list(session.get("messages", []))
        keep_recent = max(2, CHAT_SUMMARY_KEEP_RECENT_MESSAGES)
        trigger = max(keep_recent + 2, CHAT_SUMMARY_TRIGGER_MESSAGES)
        if len(messages) <= trigger:
            return False, int(session.get("summary_message_count", 0) or 0), len(session.get("summary", ""))

        older = messages[:-keep_recent]
        if not older:
            return False, int(session.get("summary_message_count", 0) or 0), len(session.get("summary", ""))

        already_summarized = int(session.get("summary_message_count", 0) or 0)
        if already_summarized >= len(older):
            return False, already_summarized, len(session.get("summary", ""))

        new_slice = older[already_summarized:]
        chunk_summary = _summarize_messages_chunk(new_slice)
        if not chunk_summary:
            return False, already_summarized, len(session.get("summary", ""))

        existing_summary = str(session.get("summary", "") or "")
        combined = (
            f"{existing_summary}\n{chunk_summary}".strip()
            if existing_summary
            else chunk_summary
        )
        combined = _truncate_summary(combined)

        session["summary"] = combined
        session["summary_message_count"] = len(older)
        session["updated_at"] = now
        _session_store[session_id] = session
        return True, len(older), len(combined)


def _build_local_prompt(history: list[dict], current_user_message: str) -> str:
    if not history:
        return current_user_message

    role_map = {"user": "User", "assistant": "Assistant"}
    lines = ["Conversation so far:"]
    for message in history:
        role = role_map.get(message.get("role", ""), "User")
        lines.append(f"{role}: {message['content']}")
    lines.append("")
    lines.append(f"User: {current_user_message}")
    lines.append("Assistant:")
    return "\n".join(lines)


def _augment_system_with_session_summary(system: str, summary_text: str) -> str:
    if not summary_text:
        return system
    return "\n".join(
        [
            system,
            "",
            "Session summary of older messages (use as context for continuity):",
            summary_text,
        ]
    )


def _augment_system_with_memories(system: str, memory_hits: list[dict]) -> str:
    if not memory_hits:
        return system

    lines = [
        system,
        "",
        "Relevant user memory (apply only if directly helpful to this request):",
        "If a memory item is not clearly relevant, ignore it.",
    ]
    for hit in memory_hits[:5]:
        lines.append(f"- {hit['text']}")
    return "\n".join(lines)


def _should_inject_memory(decision_intent: str, memory_hits: list[dict], user_message: str) -> bool:
    if not memory_hits:
        return False
    if decision_intent == "memory-needed":
        return True

    # For non-memory intents, only inject when there is strong lexical overlap.
    terms = [t for t in re.findall(r"[a-z0-9]+", user_message.lower()) if len(t) > 2][:10]
    if not terms:
        return False

    top_text = " ".join(str(h.get("text", "")).lower() for h in memory_hits[:3])
    overlap = sum(1 for t in terms if t in top_text)
    return overlap >= 2


def _session_preview_text(messages: list[dict]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "")[:80]
    return ""


def _list_sessions(user_id: str) -> list[dict]:
    with _session_store_lock:
        ordered = sorted(
            (
                (sid, data)
                for sid, data in _session_store.items()
                if _session_owned_by(data, user_id)
            ),
            key=lambda item: item[1]["updated_at"],
            reverse=True,
        )
        return [
            {
                "session_id": sid,
                "created_at": data["created_at"],
                "updated_at": data["updated_at"],
                "message_count": len(data.get("messages", [])),
                "preview": _session_preview_text(data.get("messages", [])),
            }
            for sid, data in ordered
        ]


def _append_session_message(session_id: str, role: str, content: str) -> None:
    now = time.time()
    with _session_store_lock:
        session = _session_store.get(session_id)
        if not session:
            return
        session["messages"].append({"role": role, "content": content, "ts": now})
        session["updated_at"] = now

async def stream_local(request: ChatRequest, model_name: str = CHAT_MODEL):
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", f"{OLLAMA_URL}/api/generate", json={
            "model": model_name,
            "prompt": request.message,
            "system": request.system,
            "stream": True,
        }) as resp:
            async for line in resp.aiter_lines():
                if line:
                    data = json.loads(line)
                    yield data.get("response", "")
                    if data.get("done"):
                        break


# ── Phase 14: vision helpers ──────────────────────────────────────────────────

_ALLOWED_VISION_MIME = frozenset({"image/png", "image/jpeg", "image/webp"})
_MAX_IMAGE_BYTES = 25 * 1024 * 1024  # 25 MB


def _validate_image(image_base64: str | None, image_mime: str | None) -> str | None:
    """Return an error string if the image payload is invalid, else None."""
    if image_base64 is None:
        return None
    if image_mime not in _ALLOWED_VISION_MIME:
        return f"Unsupported image type '{image_mime}'. Allowed: image/png, image/jpeg, image/webp."
    try:
        raw = base64.b64decode(image_base64, validate=True)
    except Exception:
        return "Image data is not valid base64."
    if len(raw) > _MAX_IMAGE_BYTES:
        mb = len(raw) / (1024 * 1024)
        return f"Image too large ({mb:.1f} MB). Maximum is 25 MB."
    return None


async def stream_local_vision(
    request: ChatRequest,
    image_base64: str,
    image_mime: str,
    model_name: str = OLLAMA_VISION_MODEL,
):
    """Ollama /api/chat endpoint with image tokens (multimodal forward pass)."""
    user_msg: dict = {"role": "user", "content": request.message, "images": [image_base64]}
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": request.system or CHAT_DEFAULT_SYSTEM_PROMPT},
            user_msg,
        ],
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    data = json.loads(line)
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        yield chunk
                    if data.get("done"):
                        break

async def stream_cloud(system: str, messages: list[dict]):  # type: ignore[override]
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    with client.messages.stream(
        model=CLOUD_MODEL,
        max_tokens=2048,
        system=system,
        messages=messages,  # type: ignore[arg-type]
    ) as stream:
        for text in stream.text_stream:
            yield text


# ── LangGraph dependency wiring ───────────────────────────────────────────────
# Late-binding proxies: each closure looks up the current module-level name at
# *call* time, so monkeypatch.setattr(main, "router_route", fake) in tests
# propagates transparently through the graph.

def _make_graph_deps() -> AssistantGraphDependencies:
    async def _late_router_route(msg: str):
        return await router_route(msg)

    async def _late_stream_local(req, model_name=None):
        async for chunk in stream_local(req, model_name):  # type: ignore[arg-type]
            yield chunk

    async def _late_stream_cloud(system: str, messages: list):
        async for chunk in stream_cloud(system, messages):
            yield chunk

    async def _late_stream_local_vision(req, image_b64: str, image_mime: str):
        async for chunk in stream_local_vision(req, image_b64, image_mime):  # type: ignore[arg-type]
            yield chunk

    async def _late_tool_dispatch(tool_name: str, params: dict):
        return await tools.dispatch(tool_name, params)

    return AssistantGraphDependencies(
        memory_store=memory_store,
        router_route=_late_router_route,
        stream_local=_late_stream_local,
        stream_cloud=_late_stream_cloud,
        stream_local_vision=_late_stream_local_vision,
        tool_dispatch=_late_tool_dispatch,
        chat_model=CHAT_MODEL,
        cloud_model=CLOUD_MODEL,
        coder_model=CODER_MODEL,
        vision_model=OLLAMA_VISION_MODEL,
        chroma_path=os.getenv("CHROMA_PATH", _chroma_default),
    )


_assistant_graph = build_assistant_graph(_make_graph_deps())
log.info("graph.ready | checkpointer=none (no-checkpoint mode; Slice 5 adds AsyncSqliteSaver)")


@app.post("/chat")
async def chat(request: ChatRequest, http_request: Request):
    user_id: str = http_request.state.user_id
    cookie_session_id = http_request.cookies.get(SESSION_COOKIE_NAME)
    session_id, session_created = _get_or_create_session(user_id, cookie_session_id)
    chat_source = _normalize_chat_source(request.source)
    effective_system = request.system or CHAT_DEFAULT_SYSTEM_PROMPT

    # ── Deterministic music fast-path ─────────────────────────────────────────
    # Check before router_route() so clear music commands never touch the LLM.
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
                msg = f"⚠ Music command failed: {exc}"
                yield f"data: {json.dumps({'text': msg})}\n\n"
                yield "data: [DONE]\n\n"
                return
            log.info(
                "chat.music_fast | session_id=%s action=%s ok=%s retryable=%s",
                session_id, music_cmd.get("action"), tool_result.ok, tool_result.retryable,
            )
            response_text = format_music_response(tool_result, music_cmd)
            yield f"data: {json.dumps({'text': response_text})}\n\n"
            _append_session_message(session_id, "user", request.message)
            _append_session_message(session_id, "assistant", response_text)
            yield "data: [DONE]\n\n"

        fast_response = StreamingResponse(generate_music(), media_type="text/event-stream")
        _set_session_cookie(fast_response, session_id)
        return fast_response
    # ── End music fast-path ───────────────────────────────────────────────────

    summary_updated, summary_message_count, summary_char_count = _update_session_summary_if_needed(session_id)

    # ── Phase 14: validate image if present ──────────────────────────────────
    image_error = _validate_image(request.image_base64, request.image_mime)
    if image_error:
        log.warning("chat.image_invalid | session_id=%s reason=%s", session_id, image_error)
        return JSONResponse({"error": image_error, "code": "INVALID_IMAGE"}, status_code=422)

    with _session_store_lock:
        session = _session_store[session_id]
        session_messages = list(session["messages"])
        session_summary = str(session.get("summary", "") or "")

    previous_user_message = next(
        (m.get("content", "") for m in reversed(session_messages) if m.get("role") == "user"),
        None,
    )

    graph_state = {
        "user_id": user_id,
        "session_id": session_id,
        "message": request.message,
        "system": effective_system,
        "source": chat_source,
        "modality": "voice" if chat_source == "voice" else "chat",
        "history": session_messages,
        "session_summary": session_summary,
        # Phase 14: pass image through state (ephemeral, not persisted to memory)
        "image_base64": request.image_base64,
        "image_mime": request.image_mime,
    }
    graph_runner = getattr(app.state, "assistant_graph", _assistant_graph)

    async def generate():
        assistant_accumulated = ""
        start_time = time.monotonic()
        first_token_time: float | None = None
        active_model = CHAT_MODEL
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
                    active_model = meta.get("model", CHAT_MODEL)
                    intent_for_log = meta.get("intent", "")
                    confidence_for_log = float(meta.get("confidence", 0.0))
                    route_for_log = meta.get("route_type", "local")
                    log.info(
                        "chat.route | session_id=%s source=%s intent=%s confidence=%.3f route=%s model=%s "
                        "planner_status=%s needs_memory=%s tool=%s total_messages=%d "
                        "summary_updated=%s summary_message_count=%d summary_chars=%d",
                        session_id, chat_source, intent_for_log, confidence_for_log, route_for_log,
                        active_model, meta.get("planner_status", ""), meta.get("needs_memory", False),
                        meta.get("tool"), len(session_messages), summary_updated,
                        summary_message_count, summary_char_count,
                    )
                    if meta.get("reasoning_summary"):
                        log.debug("chat.planner_reasoning | session_id=%s reasoning=%s", session_id, meta["reasoning_summary"])
                    yield f"data: {json.dumps({'model': active_model, 'intent': intent_for_log, 'confidence': confidence_for_log})}\n\n"
                elif "text" in event:
                    chunk = event["text"]
                    if first_token_time is None:
                        first_token_time = time.monotonic()
                    assistant_accumulated += chunk
                    yield f"data: {json.dumps({'text': chunk})}\n\n"
                elif "notice" in event:
                    fallback_used = True
                    yield f"data: {json.dumps({'notice': event['notice']})}\n\n"
                elif event.get("fallback"):
                    active_model = event.get("model", CHAT_MODEL)
                    yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            log.error("chat.graph_error | session_id=%s error=%s", session_id, exc)
            yield f"data: {json.dumps({'text': f'⚠ Error: {exc}'})}\n\n"

        completion_time = time.monotonic()
        first_token_ms = (first_token_time - start_time) * 1000 if first_token_time else -1
        completion_ms = (completion_time - start_time) * 1000
        log.info(
            "chat.telemetry | session_id=%s intent=%s confidence=%.3f route=%s "
            "model=%s fallback=%s first_token_ms=%.0f completion_ms=%.0f response_tokens_approx=%d",
            session_id, intent_for_log, confidence_for_log, route_for_log,
            active_model, fallback_used, first_token_ms, completion_ms,
            _estimate_tokens(assistant_accumulated),
        )

        _append_session_message(session_id, "user", request.message)
        _append_session_message(session_id, "assistant", assistant_accumulated.strip())

        memory_result = memory_store.ingest_user_message(
            user_id,
            request.message,
            source=chat_source,
            previous_user_message=previous_user_message,
        )
        log.info(
            "chat.memory | session_id=%s status=%s saved=%d blocked=%d needs_confirmation=%d candidates=%d explicit=%s",
            session_id,
            memory_result.get("status", "none"),
            len(memory_result.get("saved", [])),
            len(memory_result.get("blocked", [])),
            len(memory_result.get("needs_confirmation", [])),
            int(memory_result.get("candidates", 0)),
            bool(memory_result.get("explicit", False)),
        )

        yield (
            "data: "
            + json.dumps({
                "memory": {
                    "status": memory_result.get("status", "none"),
                    "saved": len(memory_result.get("saved", [])),
                    "blocked": len(memory_result.get("blocked", [])),
                    "needs_confirmation": len(memory_result.get("needs_confirmation", [])),
                    "deleted": int(memory_result.get("deleted", 0) or 0),
                    "explicit": bool(memory_result.get("explicit", False)),
                    "hint": (
                        "Memory needs confirmation. Say 'remember this' to store it."
                        if memory_result.get("status") == "needs-confirmation"
                        else ""
                    ),
                }
            })
            + "\n\n"
        )

        voice_meta = _voice_tts_metadata(chat_source)
        # Phase 14: images are visual — suppress auto-TTS for vision responses.
        if intent_for_log == "vision":
            voice_meta = None
        if voice_meta is not None:
            yield "data: " + json.dumps(voice_meta) + "\n\n"

        yield "data: [DONE]\n\n"

    response = StreamingResponse(generate(), media_type="text/event-stream")
    _set_session_cookie(response, session_id)
    if session_created:
        log.info("chat.session.cookie_set | session_id=%s", session_id)
    return response


@app.get("/graph/state/{session_id}")
async def get_graph_state(session_id: str, http_request: Request):
    user_id: str = http_request.state.user_id
    with _session_store_lock:
        if not _session_owned_by(_session_store.get(session_id), user_id):
            return _error_response("Session not found", "SESSION_NOT_FOUND", False, status_code=404)

    graph_runner = getattr(app.state, "assistant_graph", _assistant_graph)
    if graph_runner is None:
        return _error_response("Graph not initialized", "GRAPH_UNAVAILABLE", True, status_code=503)

    try:
        if hasattr(graph_runner, "aget_state"):
            snapshot = await graph_runner.aget_state(checkpoint_config(session_id))
        else:
            snapshot = graph_runner.get_state(checkpoint_config(session_id))
    except Exception as exc:
        log.error("graph.state.error | session_id=%s error=%s", session_id, exc)
        return _error_response("Graph state unavailable", "GRAPH_STATE_UNAVAILABLE", True, status_code=503)

    return JSONResponse(
        {
            "session_id": session_id,
            "state": getattr(snapshot, "values", {}) or {},
            "next": list(getattr(snapshot, "next", ()) or ()),
            "metadata": getattr(snapshot, "metadata", {}) or {},
        }
    )


@app.delete("/chat/session")
async def reset_chat_session(http_request: Request):
    user_id: str = http_request.state.user_id
    cookie_session_id = http_request.cookies.get(SESSION_COOKIE_NAME)
    session_id, created = _get_or_create_session(user_id, cookie_session_id)
    snapshot: dict | None = None
    with _session_store_lock:
        session = _session_store.get(session_id, {"messages": [], "summary": "", "summary_message_count": 0})
        cleared_messages = len(session["messages"])
        if cleared_messages > 0 or str(session.get("summary", "") or "").strip():
            snapshot = dict(session)
            snapshot["messages"] = list(session.get("messages", []))
        session["messages"] = []
        session["summary"] = ""
        session["summary_message_count"] = 0
        session["updated_at"] = time.time()
        _session_store[session_id] = session

    if snapshot is not None:
        _spawn_episodic_persist_task(session_id, snapshot, "session_reset")

    log.info(
        "chat.session.reset | session_id=%s cleared_messages=%d was_new=%s",
        session_id,
        cleared_messages,
        created,
    )
    response = JSONResponse({"ok": True, "session_id": session_id, "cleared_messages": cleared_messages})
    _set_session_cookie(response, session_id)
    return response


@app.get("/chat/sessions")
async def list_chat_sessions(http_request: Request):
    user_id: str = http_request.state.user_id
    current_session = http_request.cookies.get(SESSION_COOKIE_NAME)
    with _session_store_lock:
        current_owned = _session_owned_by(_session_store.get(current_session), user_id) if current_session else False
    return JSONResponse(
        {
            "sessions": _list_sessions(user_id),
            "current_session_id": current_session if current_owned else None,
        }
    )


@app.delete("/chat/sessions/{session_id}")
async def delete_chat_session(session_id: str, http_request: Request):
    user_id: str = http_request.state.user_id
    cookie_session_id = http_request.cookies.get(SESSION_COOKIE_NAME)
    active_session_id: str | None = None
    snapshot: dict | None = None

    with _session_store_lock:
        session = _session_store.get(session_id)
        if not _session_owned_by(session, user_id):
            return _error_response("Session not found", "SESSION_NOT_FOUND", False, status_code=404)
        if session and (len(session.get("messages", [])) > 0 or str(session.get("summary", "") or "").strip()):
            snapshot = dict(session)
            snapshot["messages"] = list(session.get("messages", []))
        del _session_store[session_id]

        # If the deleted session was active, prefer an existing remaining session.
        if cookie_session_id == session_id:
            owned_sessions = [
                (sid, data)
                for sid, data in _session_store.items()
                if _session_owned_by(data, user_id)
            ]
            if owned_sessions:
                active_session_id = max(
                    owned_sessions,
                    key=lambda item: item[1]["updated_at"],
                )[0]

    if snapshot is not None:
        _spawn_episodic_persist_task(session_id, snapshot, "session_delete")

    log.info("chat.session.deleted | session_id=%s user_id=%s", session_id, user_id)

    # If the deleted session was active and nothing remains, create a fresh session.
    if cookie_session_id == session_id and active_session_id is None:
        active_session_id, _ = _get_or_create_session(user_id, None)

    payload = {"ok": True, "session_id": session_id}
    if active_session_id:
        payload["active_session_id"] = active_session_id
    response = JSONResponse(payload)

    if active_session_id:
        _set_session_cookie(response, active_session_id)
    return response


@app.post("/chat/session/new")
async def create_chat_session(http_request: Request):
    user_id: str = http_request.state.user_id
    previous_session_id = http_request.cookies.get(SESSION_COOKIE_NAME)
    previous_snapshot: dict | None = None

    if previous_session_id:
        with _session_store_lock:
            prev = _session_store.get(previous_session_id)
            if _session_owned_by(prev, user_id) and prev and (
                len(prev.get("messages", [])) > 0 or str(prev.get("summary", "") or "").strip()
            ):
                previous_snapshot = dict(prev)
                previous_snapshot["messages"] = list(prev.get("messages", []))

    session_id, _ = _get_or_create_session(user_id, None)

    if previous_snapshot is not None and previous_session_id and previous_session_id != session_id:
        _spawn_episodic_persist_task(previous_session_id, previous_snapshot, "session_new")

    response = JSONResponse({"ok": True, "session_id": session_id})
    _set_session_cookie(response, session_id)
    return response


@app.post("/chat/session/select")
async def select_chat_session(payload: SessionSelectRequest, http_request: Request):
    user_id: str = http_request.state.user_id
    current_session_id = http_request.cookies.get(SESSION_COOKIE_NAME)
    previous_snapshot: dict | None = None

    with _session_store_lock:
        if not _session_owned_by(_session_store.get(payload.session_id), user_id):
            return _error_response("Session not found", "SESSION_NOT_FOUND", False, status_code=404)

        prev = _session_store.get(current_session_id) if current_session_id else None
        if (
            current_session_id
            and current_session_id != payload.session_id
            and _session_owned_by(prev, user_id)
            and prev
            and (len(prev.get("messages", [])) > 0 or str(prev.get("summary", "") or "").strip())
        ):
            previous_snapshot = dict(prev)
            previous_snapshot["messages"] = list(prev.get("messages", []))

    if previous_snapshot is not None and current_session_id:
        _spawn_episodic_persist_task(current_session_id, previous_snapshot, "session_switch")

    response = JSONResponse({"ok": True, "session_id": payload.session_id})
    _set_session_cookie(response, payload.session_id)
    return response


@app.get("/chat/session/messages")
async def get_chat_session_messages(http_request: Request):
    user_id: str = http_request.state.user_id
    cookie_session_id = http_request.cookies.get(SESSION_COOKIE_NAME)
    session_id, _ = _get_or_create_session(user_id, cookie_session_id)
    with _session_store_lock:
        messages = list(_session_store.get(session_id, {}).get("messages", []))
    # Set the session cookie so the browser anchors to this session after every
    # page load. Without this, POST /chat could pick up a stale/missing cookie
    # and silently create a new session, losing context on refresh.
    response = JSONResponse({"session_id": session_id, "messages": messages})
    _set_session_cookie(response, session_id)
    return response


# Compatibility wrappers retained for existing unit tests that call these
# handlers directly from the main module.
async def list_episodic_memory(
    http_request: Request,
    limit: int = 200,
    offset: int = 0,
    consolidated: bool | None = None,
):
    user_id: str = http_request.state.user_id
    return JSONResponse(memory_store.list_episodic(user_id, limit=limit, offset=offset, consolidated=consolidated))


async def consolidate_memory(http_request: Request):
    user_id: str = http_request.state.user_id
    stats = await asyncio.to_thread(memory_store.consolidate_pending, user_id, MEMORY_CONSOLIDATION_BATCH_SIZE)
    return JSONResponse({"ok": True, "stats": stats})


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.head("/health")
async def health_head():
    # Respond to HEAD probes (no body).
    return Response(status_code=200)


@app.post("/tts")
async def tts_synthesize(request: TTSRequest):
    try:
        audio = await tts.synthesize(request.text)
        return Response(content=audio, media_type="audio/wav")
    except Exception as exc:
        payload = tts.error_to_payload(exc)
        code = str(payload.get("code", "TTS_UNKNOWN_ERROR"))
        retryable = bool(payload.get("retryable", False))
        message = str(payload.get("error", "Unknown TTS error"))
        status = _tts_error_status(code, retryable)
        log.warning("tts.error | code=%s retryable=%s message=%s", code, retryable, message)
        return _error_response(message, code, retryable, status_code=status)


# Legacy asset routes (compatibility with clients requesting root-mounted files)
@app.get("/style.css", include_in_schema=False)
async def legacy_style():
    return FileResponse(os.path.join(_frontend_dir, "style.css"))


@app.get("/auth.js", include_in_schema=False)
async def legacy_auth_js():
    return FileResponse(os.path.join(_frontend_dir, "auth.js"))


@app.get("/message.js", include_in_schema=False)
async def legacy_message_js():
    return FileResponse(os.path.join(_frontend_dir, "message.js"))


@app.get("/voice.js", include_in_schema=False)
async def legacy_voice_js():
    return FileResponse(os.path.join(_frontend_dir, "voice.js"))


@app.get("/favicon.ico", include_in_schema=False)
async def legacy_favicon():
    path = os.path.join(_frontend_dir, "favicon.ico")
    if os.path.exists(path):
        return FileResponse(path)
    return Response(status_code=404)


# Browser sends raw binary frames: 1280 int16 samples (80ms @ 16kHz).
# Server replies with {"event": "wake"} when the wake word is detected.
@app.websocket("/ws/wake")
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
            log.debug("Wake score: %.3f (threshold: 0.5)", score)
            if score > 0.5:
                log.info("Wake word detected — score: %.3f", score)
                await ws.send_json({"event": "wake", "score": round(float(score), 3)})
                model.reset()
    except WebSocketDisconnect as exc:
        log.info("Wake WebSocket disconnected — code: %d, reason: %s", exc.code, exc.reason or "(none)")


# ── Transcription endpoint ─────────────────────────────────────────────────────
@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    whisper = get_whisper_model()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        segments, _ = whisper.transcribe(tmp_path, language="en", vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments).strip()
    finally:
        os.unlink(tmp_path)
    return JSONResponse({"text": text})


@app.post("/code", summary="Stream code generation/editing via graph code_tool")
async def code(request: CodeRequest, http_request: Request):
    user_id: str = http_request.state.user_id
    cookie_session_id = http_request.cookies.get(SESSION_COOKIE_NAME)
    session_id, session_created = _get_or_create_session(user_id, cookie_session_id)
    code_source = _normalize_chat_source(request.source)
    effective_system = request.system or CODE_DEFAULT_SYSTEM_PROMPT

    with _session_store_lock:
        session = _session_store[session_id]
        session_messages = list(session["messages"])
        session_summary = str(session.get("summary", "") or "")

    graph_state = {
        "user_id": user_id,
        "session_id": session_id,
        "message": request.message,
        "system": effective_system,
        "source": code_source,
        "history": session_messages,
        "session_summary": session_summary,
        "force_code": True,
    }

    graph_runner = getattr(app.state, "assistant_graph", _assistant_graph)

    async def generate():
        assistant_accumulated = ""
        active_model = CODER_MODEL

        try:
            async for event in graph_runner.astream(
                graph_state,
                config=checkpoint_config(session_id),
                stream_mode="custom",
            ):
                if "meta" in event:
                    meta = event["meta"]
                    active_model = meta.get("model", CODER_MODEL)
                    yield f"data: {json.dumps({'model': active_model, 'intent': meta.get('intent', 'code'), 'confidence': meta.get('confidence', 1.0)})}\n\n"
                elif "text" in event:
                    chunk = event["text"]
                    assistant_accumulated += chunk
                    yield f"data: {json.dumps({'text': chunk})}\n\n"
                elif "notice" in event:
                    yield f"data: {json.dumps({'notice': event['notice']})}\n\n"
        except Exception as exc:
            log.error("code.graph_error | session_id=%s error=%s", session_id, exc)
            yield f"data: {json.dumps({'text': f'⚠ Error: {exc}'})}\n\n"

        _append_session_message(session_id, "user", request.message)
        _append_session_message(session_id, "assistant", assistant_accumulated.strip())

        voice_meta = _voice_tts_metadata(code_source)
        if voice_meta is not None:
            yield "data: " + json.dumps(voice_meta) + "\n\n"

        yield "data: [DONE]\n\n"

    response = StreamingResponse(generate(), media_type="text/event-stream")
    _set_session_cookie(response, session_id)
    if session_created:
        log.info("code.session.cookie_set | session_id=%s", session_id)
    return response


# ── Static frontend — MUST be last ────────────────────────────────────────────
_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_frontend_dir):
    # Serve static assets under /static so API routes (e.g. /health) are
    # not intercepted by the static files app which would return 404.
    app.mount("/static", StaticFiles(directory=_frontend_dir), name="static")

    # Serve the SPA entrypoint for root and unknown paths (client-side routing).
    @app.get("/", include_in_schema=False)
    async def _index():
        return FileResponse(os.path.join(_frontend_dir, "index.html"))

    @app.get("/{full_path:path}", include_in_schema=False)
    async def _spa_catchall(full_path: str):
        return FileResponse(os.path.join(_frontend_dir, "index.html"))