"""OpenAI-compatible function/tool schemas for Hearth assistant tools."""

from __future__ import annotations

from typing import Any

HEARTH_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Get current weather or forecast for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City or location name, e.g. Helsinki, Tallinn, Paris, Tokyo",
                    },
                    "timeframe": {
                        "type": "string",
                        "description": "Timeframe, e.g. today, tomorrow, this weekend",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "music",
            "description": "Search, play, queue, or control music playback",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "play",
                            "queue",
                            "pause",
                            "resume",
                            "next",
                            "stop",
                            "shuffle",
                            "now_playing",
                        ],
                        "description": "Playback action to perform",
                    },
                    "query": {
                        "type": "string",
                        "description": "Track title, album, artist, or genre to search and play/queue",
                    },
                    "artist": {
                        "type": "string",
                        "description": "Specific artist name if requesting songs by an artist",
                    },
                },
                "required": ["action"],
            },
        },
    },
]
