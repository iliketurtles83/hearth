from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    system: str | None = None
    source: str = "text"
    # Phase 14 — vision input
    image_base64: str | None = None  # raw base64, no data-URI prefix
    image_mime: str | None = None    # "image/png" | "image/jpeg" | "image/webp"


class TTSRequest(BaseModel):
    text: str


class CodeRequest(BaseModel):
    message: str
    system: str | None = None
    source: str = "text"


class SessionSelectRequest(BaseModel):
    session_id: str


class SessionRenameRequest(BaseModel):
    title: str


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


class MusicOutputSelectRequest(BaseModel):
    output_id: str
    mode: str = "exclusive"


class MusicTimingReport(BaseModel):
    """Client-side play-latency probe: ms marks since the music request was sent."""
    mode: str = Field(max_length=16)
    output: str = Field(max_length=16)
    marks: dict[str, int] = Field(max_length=16)
    stream_lag_s: float = 0.0



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
