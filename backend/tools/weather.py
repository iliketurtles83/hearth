"""
Weather tool — Open-Meteo adapter (no API key required).

Normalised ToolResult.data schema:
{
    "location":     str,   # human-readable "City, Country"
    "temperature":  float, # current temperature
    "feels_like":   float, # apparent temperature
    "humidity":     int,   # relative humidity %
    "wind_speed":   float, # wind speed
    "condition":    str,   # human-readable weather condition
    "units": {
        "temperature": "°C" | "°F",
        "wind_speed":  "km/h" | "mph",
    }
}

Location precedence:
  1. Inline override in prompt ("weather in Tallinn")
  2. Stored preference "default_location" from memory
  3. → ToolResult.failure asking user to set a default location

Environment variables:
  WEATHER_UNITS         celsius | fahrenheit  (default: celsius)
  WEATHER_TIMEOUT_MS    HTTP timeout ms       (default: 5000)
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

import httpx

from tools.base import ToolResult
import tools as _registry

log = logging.getLogger("assistant.tools.weather")

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_UNITS: str = os.getenv("WEATHER_UNITS", "celsius").lower()
WEATHER_TIMEOUT_MS: int = int(os.getenv("WEATHER_TIMEOUT_MS", "5000"))

# WMO weather interpretation codes → human-readable strings.
# https://open-meteo.com/en/docs#weathervariables
_WMO_CODES: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Heavy freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

# Regex for inline location override.
# Allows up to 3 optional words between "weather" and the preposition, so
# "what is the weather like in Tallinn" works ("like" is the intervening word).
_LOCATION_RE = re.compile(
    r"\bweather\b(?:\s+\w+){0,3}?\s+(?:in|for|at|near|around)\s+"
    r"([A-Za-z][A-Za-z\s,.\-]{1,60}?)(?:\?|$|\.)",
    re.IGNORECASE,
)

# Trailing time/context words that are not part of a city name.
_LOCATION_TRAILING_RE = re.compile(
    r"\s+(?:right\s+now|at\s+the\s+moment|this\s+\w+|today|tonight|tomorrow|now|currently)\s*$",
    re.IGNORECASE,
)


def wmo_condition(code: int) -> str:
    """Map a WMO weather code to a human-readable condition string."""
    return _WMO_CODES.get(code, f"Unknown condition (code {code})")


def extract_location(prompt: str) -> str | None:
    """Extract an inline location from a user prompt, or return None."""
    match = _LOCATION_RE.search(prompt.strip())
    if match:
        location = match.group(1).strip().rstrip(",. ")
        location = _LOCATION_TRAILING_RE.sub("", location).strip().rstrip(",. ")
        return location if location else None
    return None


async def _geocode(city: str, timeout_s: float) -> tuple[float, float, str]:
    """Geocode *city* using Open-Meteo geocoding API.

    Returns (latitude, longitude, display_name).
    Raises ValueError when the city is not found.
    Raises httpx.HTTPError on network failures.
    """
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.get(
            _GEOCODE_URL,
            params={"name": city, "count": 1, "language": "en", "format": "json"},
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()

    results = data.get("results") or []
    if not results:
        raise ValueError(f"Location not found: '{city}'")

    r = results[0]
    lat: float = float(r["latitude"])
    lon: float = float(r["longitude"])
    name: str = r.get("name", city)
    country: str = r.get("country", "")
    display = f"{name}, {country}".strip(", ")
    log.debug("weather.geocode | query=%s → %s (%.4f, %.4f)", city, display, lat, lon)
    return lat, lon, display


async def _fetch_current(
    lat: float, lon: float, units: str, timeout_s: float
) -> dict[str, Any]:
    """Fetch current weather from Open-Meteo forecast API.

    Returns the normalized ToolResult.data dict.
    Raises httpx.HTTPError on network failures.
    """
    temp_unit = "fahrenheit" if units == "fahrenheit" else "celsius"
    wind_unit = "mph" if units == "fahrenheit" else "kmh"

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.get(
            _FORECAST_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": (
                    "temperature_2m,apparent_temperature,"
                    "relative_humidity_2m,wind_speed_10m,weather_code"
                ),
                "temperature_unit": temp_unit,
                "wind_speed_unit": wind_unit,
            },
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()

    current: dict[str, Any] = data.get("current", {})
    units_out: dict[str, str] = {
        "temperature": "°F" if temp_unit == "fahrenheit" else "°C",
        "wind_speed": "mph" if wind_unit == "mph" else "km/h",
    }
    return {
        "temperature": current.get("temperature_2m"),
        "feels_like": current.get("apparent_temperature"),
        "humidity": current.get("relative_humidity_2m"),
        "wind_speed": current.get("wind_speed_10m"),
        "condition": wmo_condition(int(current.get("weather_code", 0))),
        "units": units_out,
    }


async def run(params: dict[str, Any]) -> ToolResult:
    """Entry point called by tools.dispatch().

    params:
        prompt  (str)       — original user message
        memory  (MemoryStore-like) — must expose .get_preference(key) -> str | None
        location (str | None) — optional override (used by /weather direct endpoint)
    """
    prompt: str = params.get("prompt", "")
    memory = params.get("memory")
    units: str = WEATHER_UNITS
    timeout_s: float = WEATHER_TIMEOUT_MS / 1000.0

    # Location resolution.
    location: str | None = params.get("location") or extract_location(prompt)
    if not location and memory is not None:
        location = memory.get_preference("default_location")

    if not location:
        return ToolResult.failure(
            "I don't know which location to check. "
            "You can say 'weather in <city>' or set a default: 'my default location is <city>'.",
            retryable=False,
        )

    log.info("weather.run | location=%s units=%s", location, units)

    try:
        lat, lon, display_name = await _geocode(location, timeout_s)
    except ValueError as exc:
        return ToolResult.failure(str(exc), retryable=False)
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        log.warning("weather.geocode_network_error | %s", exc)
        return ToolResult.failure(
            "Could not reach the geocoding service — please check your internet connection.",
            retryable=True,
        )
    except httpx.HTTPStatusError as exc:
        log.warning("weather.geocode_http_error | status=%s", exc.response.status_code)
        return ToolResult.failure(
            f"Geocoding service returned an error ({exc.response.status_code}).",
            retryable=exc.response.status_code >= 500,
        )

    try:
        weather_data = await _fetch_current(lat, lon, units, timeout_s)
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        log.warning("weather.forecast_network_error | %s", exc)
        return ToolResult.failure(
            "Could not reach the weather service — please check your internet connection.",
            retryable=True,
        )
    except httpx.HTTPStatusError as exc:
        log.warning("weather.forecast_http_error | status=%s", exc.response.status_code)
        return ToolResult.failure(
            f"Weather service returned an error ({exc.response.status_code}).",
            retryable=exc.response.status_code >= 500,
        )

    data = {"location": display_name, **weather_data}
    log.info(
        "weather.run | result location=%s temp=%s%s condition=%s",
        display_name,
        data.get("temperature"),
        data.get("units", {}).get("temperature", ""),
        data.get("condition"),
    )
    return ToolResult(ok=True, data=data)


# Self-register when the module is imported.
import sys as _sys
_registry.register("weather", _sys.modules[__name__])
