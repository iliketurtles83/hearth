from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from app_schemas import (
    MusicControlRequest,
    MusicPlayRequest,
    MusicQueueRequest,
    MusicSearchRequest,
    WeatherRequest,
)


def create_memory_tool_router(
    *,
    memory_store,
    memory_consolidation_batch_size: int,
    error_response: Callable[[str, str, bool, int], JSONResponse],
    dispatch_tool: Callable[[str, dict], Awaitable],
    run_weather: Callable[[dict], Awaitable],
) -> APIRouter:
    router = APIRouter()

    async def _music_run(params: dict) -> JSONResponse:
        result = await dispatch_tool("music", params)
        if not result.ok:
            status = 503 if result.retryable else 422
            return error_response(result.error, "MUSIC_ERROR", result.retryable, status_code=status)
        return JSONResponse(result.data)

    @router.get("/memory")
    async def list_memory(
        http_request: Request,
        limit: int = Query(default=200, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ):
        user_id: str = http_request.state.user_id
        return JSONResponse(memory_store.list_items(user_id, limit=limit, offset=offset))

    @router.get("/memory/episodic")
    async def list_episodic_memory(
        http_request: Request,
        limit: int = Query(default=200, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        consolidated: bool | None = Query(default=None),
    ):
        user_id: str = http_request.state.user_id
        return JSONResponse(memory_store.list_episodic(user_id, limit=limit, offset=offset, consolidated=consolidated))

    @router.post("/memory/consolidate")
    async def consolidate_memory(http_request: Request):
        user_id: str = http_request.state.user_id
        stats = await asyncio.to_thread(
            memory_store.consolidate_pending,
            user_id,
            memory_consolidation_batch_size,
        )
        return JSONResponse({"ok": True, "stats": stats})

    @router.delete("/memory/{memory_id}")
    async def delete_memory(memory_id: str, http_request: Request):
        user_id: str = http_request.state.user_id
        if not memory_store.delete_item(user_id, memory_id):
            return error_response("Memory item not found", "MEMORY_NOT_FOUND", False, status_code=404)
        return JSONResponse({"ok": True, "id": memory_id})

    @router.delete("/memory")
    async def clear_memory(http_request: Request):
        user_id: str = http_request.state.user_id
        counts = memory_store.clear_all(user_id)
        return JSONResponse({"ok": True, "cleared": counts})

    @router.post("/weather")
    async def weather(request: WeatherRequest, http_request: Request):
        user_id: str = http_request.state.user_id
        result = await run_weather(
            {
                "prompt": f"weather in {request.location}" if request.location else "",
                "user_id": user_id,
                "memory": memory_store,
                "location": request.location,
            }
        )
        if not result.ok:
            status = 503 if result.retryable else 422
            return error_response(result.error, "WEATHER_ERROR", result.retryable, status_code=status)
        return JSONResponse(result.data)

    @router.post("/music/search")
    async def music_search(request: MusicSearchRequest):
        return await _music_run({"action": "search", "query": request.query, "prompt": request.query})

    @router.post("/music/play")
    async def music_play(request: MusicPlayRequest):
        return await _music_run(
            {
                "action": "play",
                "query": request.query,
                "song_id": request.song_id,
                "artist": request.artist,
                "prompt": request.query or request.artist or "",
            }
        )

    @router.post("/music/queue")
    async def music_queue_add(request: MusicQueueRequest):
        return await _music_run(
            {
                "action": "queue",
                "query": request.query,
                "song_id": request.song_id,
                "prompt": request.query or "",
            }
        )

    @router.post("/music/control")
    async def music_control(request: MusicControlRequest):
        if request.action not in ("pause", "resume", "next", "stop", "play_pos", "set_volume"):
            return error_response(
                f"Unknown action '{request.action}'. Use: pause, resume, next, stop, play_pos, set_volume.",
                "MUSIC_INVALID_ACTION",
                False,
                status_code=400,
            )
        return await _music_run(
            {
                "action": "control",
                "control": request.action,
                "pos": request.pos,
                "volume": request.volume,
                "prompt": "",
            }
        )

    @router.get("/music/now_playing")
    async def music_now_playing():
        return await _music_run({"action": "now_playing", "prompt": ""})

    @router.get("/music/queue")
    async def music_queue_view():
        return await _music_run({"action": "queue_view", "prompt": ""})

    return router
