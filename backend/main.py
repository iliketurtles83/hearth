from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
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
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load .env BEFORE importing local modules — intents, routing_config, memory and
# other modules read os.getenv() at import time, so the environment must be
# populated first or those module-level constants capture stale defaults.
load_dotenv()

from intents import CLOUD_MODEL, CHAT_MODEL
from embedding_router import (
    get_embedding_router,
    get_embedding_router_error,
    get_embedding_router_snapshot,
    warmup_embedding_router,
)
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
from routes.memory_tool_routes import create_memory_tool_router
from routes.chat_routes import create_chat_router
from routes.session_routes import create_session_router
from routes.tts_routes import create_tts_router
from routes.voice_routes import create_voice_router
from routes.code_routes import create_code_router
from routes.health_routes import create_health_router
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
_memory_store: MemoryStore | None = None


def get_memory_store() -> MemoryStore:
    """Return the shared MemoryStore, constructing it on first use.

    Lazy on purpose: MemoryStore.__init__ creates directories, opens SQLite in
    WAL mode, and opens a Chroma PersistentClient, so constructing it at import
    time would make a bare ``import main`` write to state files.
    """
    global _memory_store
    if _memory_store is None:
        _memory_store = MemoryStore(
            db_path=os.getenv("MEMORY_DB_PATH", _memory_db_default),
            chroma_path=os.getenv("CHROMA_PATH", _chroma_default),
        )
    return _memory_store


def __getattr__(name: str):
    # PEP 562: keep the conventional ``main.memory_store`` attribute working
    # (tests, ``from main import memory_store``) without building the store at
    # import time. First access constructs and caches the singleton.
    if name == "memory_store":
        return get_memory_store()
    if name == "auth_service":
        return get_auth_service()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
# Startup chat-model warmup: pre-loads the (large) chat model into VRAM so the
# first /chat doesn't pay the one-time model-load. Kept in sync with the
# OLLAMA_KEEP_ALIVE the ollama service is started with.
CHAT_MODEL_WARMUP = os.getenv("CHAT_MODEL_WARMUP", "true").strip().lower() == "true"
CHAT_MODEL_WARMUP_TIMEOUT_S = float(os.getenv("CHAT_MODEL_WARMUP_TIMEOUT_S", "180"))
CHAT_MODEL_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
MEMORY_CONSOLIDATION_BATCH_SIZE = int(os.getenv("MEMORY_CONSOLIDATION_BATCH_SIZE", "50"))
_GENERIC_STREAM_ERROR_TEXT = "⚠ Something went wrong. Please try again."


def _stream_error_payload(code: str = "INTERNAL") -> dict[str, str]:
    return {
        "text": _GENERIC_STREAM_ERROR_TEXT,
        "code": code,
    }


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
# Optional explicit local model dir (downloaded by scripts/download-whisper-model.sh).
# When empty, we auto-detect backend/models/whisper/<WHISPER_MODEL>.
WHISPER_MODEL_DIR = os.getenv("WHISPER_MODEL_DIR", "").strip()

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


# ── Auth service (shared singleton) ───────────────────────────────────────────
_auth_db_default = os.path.join(os.path.dirname(__file__), "auth.db")
_auth_service: AuthService | None = None


def get_auth_service() -> AuthService:
    """Return the shared AuthService, constructing it on first use.

    Lazy for the same reason as get_memory_store: AuthService.__init__ opens
    SQLite (creating auth.db), so a bare ``import main`` should not write
    state files.
    """
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService(os.getenv("AUTH_DB_PATH", _auth_db_default))
    return _auth_service

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

    _whisper_dir = WHISPER_MODEL_DIR or os.path.join(_models_dir, "whisper", WHISPER_MODEL)
    if not os.path.isfile(os.path.join(_whisper_dir, "model.bin")):
        log.warning("Whisper STT model not found locally — /transcribe will return 503")
        log.warning("Run: bash scripts/download-whisper-model.sh")

    if not os.getenv("ANTHROPIC_API_KEY"):
        log.warning("ANTHROPIC_API_KEY not set — cloud model fallback will be unavailable")

    _bootstrap_beets_library_if_empty()

    log.info(
        "Startup OK | chat_model=%s | ollama=%s | cors_origins=%s | cookie_secure=%s",
        CHAT_MODEL, OLLAMA_URL, _CORS_ORIGINS, SESSION_COOKIE_SECURE,
    )

_validate_startup()

# ── Auth middleware ────────────────────────────────────────────────────────────
# Resolves the bearer token (from Authorization header or auth_token cookie)
# and attaches user_id to request.state.  Returns 401 for protected routes
# that have no valid token.
_UNPROTECTED_PATHS = frozenset(["/health", "/", "/ws/wake"])

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
        user_id = get_auth_service().verify_token(token) if token else None
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


_CSP_POLICY = (
    "default-src 'self'; "
    "base-uri 'self'; "
    "object-src 'none'; "
    "frame-ancestors 'none'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    "font-src 'self' data:; "
    "img-src 'self' data: blob:; "
    "connect-src 'self' ws: wss:; "
    "media-src 'self' data: blob:"
)


# ── Cross-Origin isolation headers (required for SharedArrayBuffer / vad-web) ──
class COOPCOEPMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        # 'credentialless' (vs 'require-corp') still enables SharedArrayBuffer
        # (needed by vad-web's threaded WASM) while allowing cross-origin CDN
        # resources that don't set Cross-Origin-Resource-Policy headers.
        response.headers["Cross-Origin-Embedder-Policy"] = "credentialless"
        response.headers["Content-Security-Policy"] = _CSP_POLICY
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        return response

# Module-level graph instance — initialized after stream_local/stream_cloud are defined.
# The lifespan (Slice 5) will replace this with a checkpointed version.
_assistant_graph = None  # type: ignore[assignment]


@asynccontextmanager
async def _graph_lifespan(_app: FastAPI):
    global _assistant_graph
    try:
        purged = get_auth_service().purge_expired_tokens()
        if purged:
            log.info("auth.tokens.purged | count=%d", purged)
    except Exception as exc:
        log.warning("auth.tokens.purge_failed | error=%s", exc)
    # Kick off the chat-model warmup up front so the ~80s cold load overlaps
    # the (fast) embedding-router warmup and graph build below. It is awaited
    # before we start serving (see end of this lifespan).
    chat_warmup_task = asyncio.create_task(warmup_chat_model())
    embed_router = None
    # Retry the router build with backoff: the backend can start before
    # Ollama is fully ready (depends_on only waits for container start), and
    # the first embed call also has to load the model into VRAM.
    max_attempts = max(1, int(os.getenv("ROUTER_EMBEDDING_RETRIES", "3")))
    retry_backoff = max(0.0, float(os.getenv("ROUTER_EMBEDDING_RETRY_BACKOFF_SECONDS", "2.0")))
    warm_ok = False
    for attempt in range(1, max_attempts + 1):
        warm_ok = await warmup_embedding_router(force_refresh=True)
        if warm_ok:
            break
        if attempt < max_attempts:
            log.info(
                "embedding_router.retry | attempt=%d/%d backoff=%.1fs error=%s",
                attempt,
                max_attempts,
                retry_backoff * attempt,
                get_embedding_router_error(),
            )
            await asyncio.sleep(retry_backoff * attempt)
    if warm_ok:
        embed_router = get_embedding_router()
        embed_snapshot = get_embedding_router_snapshot()
        log.info(
            "embedding_router.ready | model=%s dim=%d",
            embed_snapshot.model,
            embed_snapshot.dim,
        )
    else:
        log.warning(
            "embedding_router.failed | error=%s | using heuristic fallback",
            get_embedding_router_error(),
        )

    try:
        async with create_assistant_graph(
            _make_graph_deps(embedding_router=embed_router),
            checkpoint_path=default_checkpoint_path(),
        ) as checkpointed_graph:
            _assistant_graph = checkpointed_graph
            _app.state.assistant_graph = checkpointed_graph
            _app.state.embedding_router = embed_router
            log.info("graph.ready | checkpointer=sqlite path=%s", default_checkpoint_path())
            # Await the chat-model warmup (started above, running in parallel) so
            # the model is resident before we start serving — guarantees the
            # first /chat takes the fast path. warmup_chat_model never raises.
            if chat_warmup_task is not None:
                try:
                    await chat_warmup_task
                except Exception as exc:  # defensive; warmup already self-guards
                    log.warning("chat_warmup.await_failed | error=%s", exc if str(exc) else repr(exc))
            yield
    finally:
        # If the graph build failed before we reached the await, don't leave the
        # warmup task orphaned (it would hold an open httpx client past shutdown).
        if chat_warmup_task is not None and not chat_warmup_task.done():
            chat_warmup_task.cancel()

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

def _resolve_whisper_model_source():
    # Prefer a local, persistent model dir (downloaded by
    # scripts/download-whisper-model.sh) over the HuggingFace size string, so a
    # Docker container does not re-fetch the model into an ephemeral cache on the
    # first /transcribe call. backend/models is bind-mounted (./backend:/app), so
    # the downloaded dir survives rebuilds.
    if WHISPER_MODEL_DIR:
        return WHISPER_MODEL_DIR
    default_dir = os.path.join(os.path.dirname(__file__), "models", "whisper", WHISPER_MODEL)
    if os.path.isfile(os.path.join(default_dir, "model.bin")):
        return default_dir
    return WHISPER_MODEL

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        if WHISPER_DEVICE:
            device = WHISPER_DEVICE
        else:
            device = "cuda" if os.path.exists("/dev/nvidia0") else "cpu"
        compute = WHISPER_COMPUTE_TYPE or ("float16" if device == "cuda" else "int8")
        model_source = _resolve_whisper_model_source()
        log.info("whisper.model_source | source=%s device=%s compute=%s", model_source, device, compute)
        _whisper_model = WhisperModel(model_source, device=device, compute_type=compute)
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
        get_auth_service=get_auth_service,
        auth_cookie_name=AUTH_COOKIE_NAME,
        session_cookie_secure=SESSION_COOKIE_SECURE,
        extract_bearer_token=_extract_bearer_token,
        error_response=_error_response,
    )
)

app.include_router(
    create_memory_tool_router(
        get_memory_store=get_memory_store,
        memory_consolidation_batch_size=MEMORY_CONSOLIDATION_BATCH_SIZE,
        error_response=_error_response,
        dispatch_tool=tools.dispatch,
        run_weather=_run_weather_tool,
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


def _ollama_think_setting() -> bool | str:
    raw = (os.getenv("OLLAMA_THINK", "true") or "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"low", "medium", "high", "max"}:
        return raw
    return True


async def warmup_chat_model() -> bool:
    """Pre-load the chat model into VRAM so the first /chat is fast.

    Mirrors warmup_embedding_router(): a single minimal /api/generate probe
    with a generous timeout absorbs the cold model load at startup. Returns
    True on success. On failure it logs a diagnosable error (never an empty
    ``error=``) and returns False; a warmup failure must never block startup.
    """
    if not CHAT_MODEL_WARMUP:
        log.info("chat_warmup.skipped | reason=disabled")
        return False
    payload = {
        "model": CHAT_MODEL,
        "prompt": "warmup",
        "stream": False,
        "num_predict": 1,
        "keep_alive": CHAT_MODEL_KEEP_ALIVE,
    }
    try:
        async with httpx.AsyncClient(timeout=CHAT_MODEL_WARMUP_TIMEOUT_S) as client:
            resp = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
            resp.raise_for_status()
        log.info("chat_warmup.ready | model=%s", CHAT_MODEL)
        return True
    except Exception as exc:
        message = str(exc)
        log.warning(
            "chat_warmup.failed | model=%s error=%s",
            CHAT_MODEL,
            message if message else repr(exc),
        )
        return False


async def stream_local(request: ChatRequest, model_name: str = CHAT_MODEL):
    think_mode = _ollama_think_setting()
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", f"{OLLAMA_URL}/api/generate", json={
            "model": model_name,
            "prompt": request.message,
            "system": request.system,
            "stream": True,
            "think": think_mode,
        }) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    data = json.loads(line)
                    thinking = data.get("thinking") or data.get("message", {}).get("thinking", "")
                    if thinking:
                        yield {"thinking": thinking}

                    text = data.get("response") or data.get("message", {}).get("content", "")
                    if text:
                        yield {"text": text}

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
    """Ollama /api/chat endpoint with image tokens (multimodal forward pass).

    The system prompt is embedded into the user message content because
    Ollama's /api/chat endpoint ignores the "system" role for most models.
    """
    system_content = request.system or CHAT_DEFAULT_SYSTEM_PROMPT
    user_msg: dict = {
        "role": "user",
        "content": f"{system_content}\n\n{request.message}",
        "images": [image_base64],
    }
    payload = {
        "model": model_name,
        "messages": [user_msg],
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
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Cloud model is unavailable: ANTHROPIC_API_KEY is not configured")

    # Reuse one client instance per process to avoid per-request setup churn.
    global _anthropic_client
    try:
        client = _anthropic_client
    except NameError:
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
        client = _anthropic_client

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
        memory_store=get_memory_store(),
        embedding_router=embedding_router,
        router_route=_unused_router_route,
        stream_local=_late_stream_local,
        stream_cloud=_late_stream_cloud,
        stream_local_vision=_late_stream_local_vision,
        tool_dispatch=_late_tool_dispatch,
        chat_model=CHAT_MODEL,
        cloud_model=CLOUD_MODEL,
        vision_model=OLLAMA_VISION_MODEL,
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


def _get_state_graph_runner():
    """Return the graph used for /graph/state introspection (checkpointed or fallback)."""
    return getattr(app.state, "assistant_graph", _assistant_graph)


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


@dataclass
class AppServices:
    """Shared state + dependency callables handed to the route factories.

    Owned by main.py. Route modules read these at call time so tests can
    monkeypatch main.services.<attr> and the shared singletons (tools/tts are
    the same objects as main.tools / main.tts). ``memory_store`` is the lazy
    getter (main.get_memory_store); route modules call it inside handlers so
    the store is only constructed on first real use, never at import time.
    """

    app: object
    log: object
    memory_store: object
    tools: object
    tts: object
    resolve_graph_runner: object
    get_state_graph_runner: object
    clear_checkpoint_thread: object
    set_session_cookie: object
    get_whisper_model: object
    get_oww_model: object
    normalize_chat_source: object
    voice_tts_metadata: object
    validate_image: object
    estimate_tokens: object
    tts_error_status: object
    stream_error_payload: object
    error_response: object
    checkpoint_config: object
    ToolResult: object
    chat_request: object
    code_request: object
    tts_request: object
    session_select_request: object
    parse_music_command: object
    format_music_response: object
    chat_model: str
    chat_default_system_prompt: str
    code_default_system_prompt: str
    session_cookie_name: str
    wakeword_threshold: float
    max_transcribe_bytes: int


_MAX_TRANSCRIBE_BYTES = int(os.getenv("MAX_TRANSCRIBE_BYTES", str(25 * 1024 * 1024)))  # 25 MB


services = AppServices(
    app=app,
    log=log,
    memory_store=get_memory_store,
    tools=tools,
    tts=tts,
    resolve_graph_runner=_resolve_graph_runner,
    get_state_graph_runner=_get_state_graph_runner,
    clear_checkpoint_thread=_clear_checkpoint_thread,
    set_session_cookie=_set_session_cookie,
    get_whisper_model=get_whisper_model,
    get_oww_model=get_oww_model,
    normalize_chat_source=_normalize_chat_source,
    voice_tts_metadata=_voice_tts_metadata,
    validate_image=_validate_image,
    estimate_tokens=_estimate_tokens,
    tts_error_status=_tts_error_status,
    stream_error_payload=_stream_error_payload,
    error_response=_error_response,
    checkpoint_config=checkpoint_config,
    ToolResult=ToolResult,
    chat_request=ChatRequest,
    code_request=CodeRequest,
    tts_request=TTSRequest,
    session_select_request=SessionSelectRequest,
    parse_music_command=parse_music_command,
    format_music_response=format_music_response,
    chat_model=CHAT_MODEL,
    chat_default_system_prompt=CHAT_DEFAULT_SYSTEM_PROMPT,
    code_default_system_prompt=CODE_DEFAULT_SYSTEM_PROMPT,
    session_cookie_name=SESSION_COOKIE_NAME,
    wakeword_threshold=WAKEWORD_THRESHOLD,
    max_transcribe_bytes=_MAX_TRANSCRIBE_BYTES,
)

app.include_router(create_health_router(services))
app.include_router(create_chat_router(services))
app.include_router(create_session_router(services))
app.include_router(create_tts_router(services))
app.include_router(create_voice_router(services))
app.include_router(create_code_router(services))


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
