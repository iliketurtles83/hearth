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
                        "description": "Track title, artist, or genre to search and play/queue",
                    },
                    "artist": {
                        "type": "string",
                        "description": "Artist name: alone plays a mix of their songs; with query or album it narrows the match",
                    },
                    "album": {
                        "type": "string",
                        "description": "Album or compilation name when the user asks for a whole album/record; plays it in track order",
                    },
                    "playlist": {
                        "type": "string",
                        "description": "Name of a saved playlist or mixtape (e.g. 'chill' for 'my chill mixtape')",
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "timer",
            "description": "Set, list, or cancel timers and reminders",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["set", "list", "cancel"],
                        "description": "Timer action to perform",
                    },
                    "duration_minutes": {
                        "type": "number",
                        "description": "Duration in minutes for the timer",
                    },
                    "label": {
                        "type": "string",
                        "description": "Label or reminder text for the timer",
                    },
                    "timer_id": {
                        "type": "string",
                        "description": "ID of the timer to cancel",
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression or convert between units",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate, e.g. '15% of 87.50', 'sqrt(144)', '2 ** 16', '5 miles to km'",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "datetime",
            "description": "Get current time in a timezone, day of week for a date, or countdown to an event",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The date/time query, e.g. 'time in Tokyo', 'what day is March 15 2026', 'days until Christmas'",
                    },
                    "timezone": {
                        "type": "string",
                        "description": "Optional timezone name or city, e.g. 'Tokyo', 'America/New_York'",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

