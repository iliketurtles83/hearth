from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    system: str | None = None
    source: str = "text"
    project_id: str | None = None
    # Phase 14 — vision input
    image_base64: str | None = None  # raw base64, no data-URI prefix
    image_mime: str | None = None    # "image/png" | "image/jpeg" | "image/webp"


class TTSRequest(BaseModel):
    text: str


class CodeRequest(BaseModel):
    message: str
    system: str | None = None
    source: str = "text"
    project_id: str | None = None


class SessionSelectRequest(BaseModel):
    session_id: str


class WeatherRequest(BaseModel):
    location: str | None = None


class MusicSearchRequest(BaseModel):
    query: str


class MusicPlayRequest(BaseModel):
    query: str | None = None
    song_id: int | None = None
    artist: str | None = None


class MusicQueueRequest(BaseModel):
    query: str | None = None
    song_id: int | None = None


class MusicControlRequest(BaseModel):
    action: str
    pos: int | None = None
    volume: int | None = None


class RegisterRequest(BaseModel):
    username: str
    password: str
    device_name: str | None = None
    persistent: bool = False


class LoginRequest(BaseModel):
    username: str
    password: str
    device_name: str | None = None
    persistent: bool = False


class WriteRequest(BaseModel):
    content: str
    confirm: bool = False
    request_id: str | None = None
