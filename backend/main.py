from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Request, Query, HTTPException
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

# Load .env BEFORE importing local modules — intents, routing_config, memory and
# other modules read os.getenv() at import time, so the environment must be
# populated first or those module-level constants capture stale defaults.
load_dotenv()

from intents import CLOUD_MODEL, CHAT_MODEL, CODER_MODEL
from embedding_router import build_embedding_router
from memory import MemoryStore
from routing_config import ROUTING_CONFIG
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
from routes.project_routes import create_project_router
from app_schemas import (
    ChatRequest as BaseChatRequest,
    TTSRequest,
    CodeRequest as BaseCodeRequest,
    SessionSelectRequest,
)
import tts
from projects import ProjectStore, ProjectError
from tools.workspace import WorkspacePathError, resolve_workspace_path

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("assistant")

SESSION_COOKIE_NAME = os.getenv("CHAT_SESSION_COOKIE", "assistant_session")
SESSION_IDLE_TTL_SECONDS = int(os.getenv("CHAT_SESSION_IDLE_TTL_SECONDS", "1800"))
SESSION_MAX_ITEMS = int(os.getenv("CHAT_SESSION_MAX_ITEMS", "200"))
CHAT_TOKEN_BUDGET = ROUTING_CONFIG.chat_token_budget
CHAT_MAX_TURNS = ROUTING_CONFIG.chat_max_turns
OLLAMA_URL = ROUTING_CONFIG.ollama_url
# Vision model defaults to CHAT_MODEL (gemma:e4b is multimodal)
OLLAMA_VISION_MODEL: str = (
    os.getenv("OLLAMA_VISION_MODEL")
    or CHAT_MODEL
)
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
WAKEWORD_THRESHOLD = float(os.getenv("WAKEWORD_THRESHOLD", "0.5"))

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
# re-pin only (~0.2-0.3s). Skip visible loading-state badge.

# ── HTTPS / CORS / cookie policy ───────────────────────────────────────────────
# CORS_ORIGINS: comma-separated list of allowed origins, e.g.
#   CORS_ORIGINS=https://192.168.1.42,https://assistant.lan
# Default '*' preserves permissive behaviour for plain-HTTP dev.
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


_code_write_lock = Lock()
# NOTE: single-worker assumption. _pending_code_writes (and the lazily-loaded
# model singletons / fallback graph below) are per-process in-memory state. Do
# NOT run the backend with `uvicorn --workers N` (N > 1): a write confirmation
# could land on a different worker than the one that created the pending entry,
# producing spurious "Pending write not found" 404s. Move this state to the
# SQLite store before scaling to multiple workers.
_pending_code_writes: dict[str, dict] = {}

# ── Auth service (shared singleton) ───────────────────────────────────────────
_auth_db_default = os.path.join(os.path.dirname(__file__), "auth.db")
auth_service = AuthService(os.getenv("AUTH_DB_PATH", _auth_db_default))
project_store = ProjectStore(
    db_path=os.getenv("AUTH_DB_PATH", _auth_db_default),
    code_workspace_root=os.getenv("CODE_WORKSPACE_ROOT", ""),
)

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

    # Code workspace
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

# Path prefixes that correspond to JSON API routes and must always be
# auth-checked, even for browser navigations.  Frontend API calls use fetch()
# (Sec-Fetch-Mode: cors/same-origin), so they never look like navigations.
_API_PATH_PREFIXES = (
    "/chat",
    "/code",
    "/graph",
    "/memory",
    "/music",
    "/weather",
    "/tts",
    "/projects",
    "/auth/me",
    "/auth/logout",
)


def _is_browser_navigation(request: Request) -> bool:
    """True for a top-level browser navigation (page load / deep-link refresh).

    Browsers set ``Sec-Fetch-Mode: navigate`` for address-bar navigations,
    link clicks, and refreshes.  fetch()/XHR calls use ``cors``/``same-origin``
    instead, so this reliably distinguishes a page request from an API call.
    Falls back to the Accept header for older clients that omit Sec-Fetch-*.
    """
    if request.method != "GET":
        return False
    fetch_mode = request.headers.get("sec-fetch-mode")
    if fetch_mode:
        return fetch_mode == "navigate"
    return "text/html" in request.headers.get("accept", "")


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

        # Browser navigations (page loads / deep-link refreshes) to non-API
        # paths must reach the SPA catch-all so it can serve index.html, not a
        # 401 JSON body.  The SPA bootstraps and authenticates client-side.
        # API routes stay protected because fetch() requests are not navigations.
        if _is_browser_navigation(request) and not path.startswith(_API_PATH_PREFIXES):
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
    global _assistant_graph
    try:
        purged = auth_service.purge_expired_tokens()
        if purged:
            log.info("auth.tokens.purged | count=%d", purged)
    except Exception as exc:
        log.warning("auth.tokens.purge_failed | error=%s", exc)
    embed_router = None
    try:
        embed_router, embed_snapshot = await build_embedding_router()
        log.info(
            "embedding_router.ready | model=%s dim=%d",
            embed_snapshot.model,
            embed_snapshot.dim,
        )
    except Exception as exc:
        log.warning("embedding_router.failed | error=%s | using heuristic fallback", exc)

    async with create_assistant_graph(
        _make_graph_deps(embedding_router=embed_router),
        checkpoint_path=default_checkpoint_path(),
    ) as checkpointed_graph:
        _assistant_graph = checkpointed_graph
        _app.state.assistant_graph = checkpointed_graph
        _app.state.embedding_router = embed_router
        log.info("graph.ready | checkpointer=sqlite path=%s", default_checkpoint_path())
        yield

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

app.include_router(
    create_project_router(
        project_store=project_store,
        chroma_path=os.getenv("CHROMA_PATH", _chroma_default),
        coder_model=CODER_MODEL,
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


def _resolve_project_context(user_id: str, raw_project_id: str | None) -> tuple[str, str]:
    project_id = (raw_project_id or "").strip()
    if not project_id:
        return "", ""
    try:
        project = project_store.get_project(project_id, user_id)
    except ProjectError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    folder_name = str(project.get("folder_name", "")).strip()
    try:
        project_folder = resolve_workspace_path(project_store.code_workspace_root, folder_name)
    except WorkspacePathError as exc:
        raise HTTPException(status_code=400, detail="Path traversal is not allowed") from exc
    return project_id, project_folder


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


async def stream_local(request: ChatRequest, model_name: str = CHAT_MODEL):
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", f"{OLLAMA_URL}/api/generate", json={
            "model": model_name,
            "prompt": request.message,
            "system": request.system,
            "stream": True,
        }) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    data = json.loads(line)
                    yield data.get("response", "")
                    if data.get("done"):
                        break


# ── Vision helpers ─────────────────────────────────────────────────────────────

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
# Late-binding stream/tool proxies keep graph dependencies patchable in tests.

def _make_graph_deps(*, embedding_router=None) -> AssistantGraphDependencies:
    async def _unused_router_route(_msg: str):
        return None

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
        embedding_router=embedding_router,
        router_route=_unused_router_route,
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


def _resolve_graph_runner():
    """Return the active checkpointed graph, building a no-checkpoint fallback lazily.

    The checkpointed graph is installed on ``app.state.assistant_graph`` by the
    lifespan handler. If it is missing (lifespan failed or hasn't run yet), build
    a no-checkpoint graph on first use and log a clear warning — running on it
    means conversation persistence is silently disabled.
    """
    runner = getattr(app.state, "assistant_graph", None)
    if runner is not None:
        return runner
    global _assistant_graph
    if _assistant_graph is None:
        log.warning(
            "graph.fallback | checkpointed graph unavailable — building no-checkpoint "
            "fallback graph; conversation persistence is DISABLED"
        )
        _assistant_graph = build_assistant_graph(_make_graph_deps())
    return _assistant_graph


@app.post("/chat")
async def chat(request: ChatRequest, http_request: Request):
    user_id: str = http_request.state.user_id
    cookie_session_id = http_request.cookies.get(SESSION_COOKIE_NAME)
    session_id = cookie_session_id or str(uuid4())
    session_created = cookie_session_id is None
    project_id, project_folder = _resolve_project_context(user_id, request.project_id)
    chat_source = _normalize_chat_source(request.source)
    effective_system = request.system or CHAT_DEFAULT_SYSTEM_PROMPT

    # ── Deterministic music fast-path ─────────────────────────────────────────
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
                msg = f"⚠ Music command failed: {exc}"
                yield f"data: {json.dumps({'text': msg})}\n\n"
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
                        project_id or None,
                    )
                    await asyncio.to_thread(
                        memory_store.log_turn,
                        session_id,
                        user_id,
                        "assistant",
                        response_text,
                        project_id or None,
                    )
                except Exception as exc:
                    log.warning("chat.music_fast.log_turn | session_id=%s error=%s", session_id, exc)
            yield f"data: {json.dumps({'text': response_text})}\n\n"
            yield "data: [DONE]\n\n"

        fast_response = StreamingResponse(generate_music(), media_type="text/event-stream")
        _set_session_cookie(fast_response, session_id)
        return fast_response
    # ── End music fast-path ───────────────────────────────────────────────────

    # Validate image payload if present
    image_error = _validate_image(request.image_base64, request.image_mime)
    if image_error:
        log.warning("chat.image_invalid | session_id=%s reason=%s", session_id, image_error)
        return JSONResponse({"error": image_error, "code": "INVALID_IMAGE"}, status_code=422)

    graph_state = {
        "user_id": user_id,
        "session_id": session_id,
        "message": request.message,
        "system": effective_system,
        "source": chat_source,
        "project_id": project_id,
        "project_folder": project_folder,
        "modality": "voice" if chat_source == "voice" else "chat",
        # Pass image through state (ephemeral, not persisted to memory)
        "image_base64": request.image_base64,
        "image_mime": request.image_mime,
    }
    graph_runner = _resolve_graph_runner()

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
                        "planner_status=%s needs_memory=%s tool=%s",
                        session_id, chat_source, intent_for_log, confidence_for_log, route_for_log,
                        active_model, meta.get("planner_status", ""), meta.get("needs_memory", False),
                        meta.get("tool"),
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
                elif "memory" in event:
                    yield f"data: {json.dumps({'memory': event['memory']})}\n\n"
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

        voice_meta = _voice_tts_metadata(chat_source)
        # Images are visual; suppress auto-TTS for vision responses.
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


async def _clear_checkpoint_thread(session_id: str) -> None:
    graph_runner = getattr(app.state, "assistant_graph", None)
    if graph_runner is None:
        return
    checkpointer = getattr(graph_runner, "checkpointer", None)
    if checkpointer is None:
        return
    try:
        if hasattr(checkpointer, "adelete_thread"):
            await checkpointer.adelete_thread(session_id)
        elif hasattr(checkpointer, "delete_thread"):
            await asyncio.to_thread(checkpointer.delete_thread, session_id)
    except Exception as exc:
        log.warning("checkpoint_cleanup_failed | session_id=%s | %s", session_id, exc)


@app.get("/chat/sessions")
async def list_chat_sessions(http_request: Request, project_id: str | None = Query(default=None)):
    user_id: str = http_request.state.user_id
    current_session_id = http_request.cookies.get(SESSION_COOKIE_NAME)
    sessions = memory_store.list_sessions(user_id, project_id)
    return JSONResponse(
        {
            "sessions": sessions,
            "current_session_id": current_session_id,
        }
    )


@app.get("/chat/session/messages")
async def get_chat_session_messages(http_request: Request, project_id: str | None = Query(default=None)):
    user_id: str = http_request.state.user_id
    session_id = http_request.cookies.get(SESSION_COOKIE_NAME) or str(uuid4())
    turns = memory_store.get_session_turns(session_id, user_id, limit=500, project_id=project_id)
    response = JSONResponse({"session_id": session_id, "messages": turns})
    _set_session_cookie(response, session_id)
    return response


@app.post("/chat/session/new")
async def create_chat_session(http_request: Request):
    _ = http_request
    session_id = str(uuid4())
    response = JSONResponse({"ok": True, "session_id": session_id})
    _set_session_cookie(response, session_id)
    return response


@app.post("/chat/session/select")
async def select_chat_session(
    payload: SessionSelectRequest,
    http_request: Request,
    project_id: str | None = Query(default=None),
):
    user_id: str = http_request.state.user_id
    sessions = memory_store.list_sessions(user_id, project_id)
    if not any(s.get("session_id") == payload.session_id for s in sessions):
        return _error_response("Session not found", "SESSION_NOT_FOUND", False, status_code=404)

    response = JSONResponse({"ok": True, "session_id": payload.session_id})
    _set_session_cookie(response, payload.session_id)
    return response


@app.delete("/chat/sessions/{session_id}")
async def delete_chat_session(
    session_id: str,
    http_request: Request,
    project_id: str | None = Query(default=None),
):
    user_id: str = http_request.state.user_id
    current_session_id = http_request.cookies.get(SESSION_COOKIE_NAME)

    memory_store.delete_session(session_id, user_id, project_id=project_id)
    await _clear_checkpoint_thread(session_id)

    next_session_id: str | None = None
    if current_session_id == session_id:
        sessions = memory_store.list_sessions(user_id, project_id)
        next_session_id = sessions[0]["session_id"] if sessions else str(uuid4())

    payload: dict[str, object] = {"ok": True, "session_id": session_id}
    if next_session_id:
        payload["active_session_id"] = next_session_id

    response = JSONResponse(payload)
    if next_session_id:
        _set_session_cookie(response, next_session_id)
    return response


@app.delete("/chat/session")
async def reset_chat_session(http_request: Request, project_id: str | None = Query(default=None)):
    user_id: str = http_request.state.user_id
    session_id = http_request.cookies.get(SESSION_COOKIE_NAME) or str(uuid4())
    memory_store.reset_session(session_id, user_id, project_id=project_id)
    await _clear_checkpoint_thread(session_id)
    response = JSONResponse({"ok": True, "session_id": session_id})
    _set_session_cookie(response, session_id)
    return response


@app.get("/health")
async def health():
    graph_ready = getattr(app.state, "assistant_graph", None) is not None
    embed_router = getattr(app.state, "embedding_router", None)
    return {
        "status": "ok" if graph_ready else "starting",
        "embedding_router": embed_router is not None,
    }


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
            log.debug("Wake score: %.3f (threshold: %.3f)", score, WAKEWORD_THRESHOLD)
            if score > WAKEWORD_THRESHOLD:
                log.info("Wake word detected — score: %.3f", score)
                await ws.send_json({"event": "wake", "score": round(float(score), 3)})
                model.reset()
    except WebSocketDisconnect as exc:
        log.info("Wake WebSocket disconnected — code: %d, reason: %s", exc.code, exc.reason or "(none)")


# ── Transcription endpoint ─────────────────────────────────────────────────────
_MAX_TRANSCRIBE_BYTES = int(os.getenv("MAX_TRANSCRIBE_BYTES", str(25 * 1024 * 1024)))  # 25 MB


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    # /transcribe is unauthenticated (voice wake flow), so cap the upload size to
    # avoid an unbounded-read / compute-DoS surface before touching Whisper.
    raw = await audio.read(_MAX_TRANSCRIBE_BYTES + 1)
    if len(raw) > _MAX_TRANSCRIBE_BYTES:
        return JSONResponse(
            {"error": f"Audio too large. Maximum is {_MAX_TRANSCRIBE_BYTES // (1024 * 1024)} MB."},
            status_code=413,
        )
    whisper = get_whisper_model()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(raw)
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
    session_id = cookie_session_id or str(uuid4())
    session_created = cookie_session_id is None
    project_id, project_folder = _resolve_project_context(user_id, request.project_id)
    code_source = _normalize_chat_source(request.source)
    effective_system = request.system or CODE_DEFAULT_SYSTEM_PROMPT

    graph_state = {
        "user_id": user_id,
        "session_id": session_id,
        "message": request.message,
        "system": effective_system,
        "source": code_source,
        "project_id": project_id,
        "project_folder": project_folder,
        "force_code": True,
    }

    graph_runner = _resolve_graph_runner()

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
