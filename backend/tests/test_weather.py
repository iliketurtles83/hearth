"""
Weather tool tests — Phase 6B.

Covers:
- extract_location: inline override regex (in/for/at/near)
- wmo_condition: known codes + unknown code fallback
- run(): location from prompt / from memory / missing location
- run(): geocode not found → failure
- run(): geocode network error → retryable failure
- run(): geocode HTTP 500 → retryable failure
- run(): geocode HTTP 404 → non-retryable failure
- run(): forecast network error → retryable failure
- run(): success path → ToolResult.ok + normalized schema
- run(): units=fahrenheit propagated
"""
from __future__ import annotations

import json
import sys
import types

import httpx
import pytest
import respx

# ── Stub the memory module so we don't import chromadb in tests ───────────────
# We only need the get_preference/set_preference interface.
_memory_stub = types.ModuleType("memory")
sys.modules.setdefault("memory", _memory_stub)

# ── Import the module under test ──────────────────────────────────────────────
# tools/__init__.py calls _auto_register() which imports tools.weather;
# weather.py calls _registry.register() on itself. Import order matters.
import tools  # noqa: E402  (must be after stub)
import tools.weather as w  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

class _FakeMemory:
    """Minimal duck-typed MemoryStore for tests."""

    def __init__(self, prefs: dict[str, str] | None = None) -> None:
        self._prefs: dict[str, str] = prefs or {}

    def get_preference(self, key: str) -> str | None:
        return self._prefs.get(key)

    def set_preference(self, key: str, value: str) -> None:
        self._prefs[key] = value


def _geocode_response(name: str = "London", country: str = "United Kingdom",
                       lat: float = 51.5, lon: float = -0.12) -> dict:
    return {"results": [{"name": name, "country": country, "latitude": lat, "longitude": lon}]}


def _forecast_response(temp: float = 18.0, feels: float = 16.5,
                        humidity: int = 72, wind: float = 15.0, code: int = 2) -> dict:
    return {
        "current": {
            "temperature_2m": temp,
            "apparent_temperature": feels,
            "relative_humidity_2m": humidity,
            "wind_speed_10m": wind,
            "weather_code": code,
        }
    }


# ── extract_location tests ────────────────────────────────────────────────────

class TestExtractLocation:
    def test_in_city(self):
        assert w.extract_location("What is the weather in London?") == "London"

    def test_for_city(self):
        assert w.extract_location("weather for Tokyo") == "Tokyo"

    def test_at_city(self):
        assert w.extract_location("weather at Berlin") == "Berlin"

    def test_near_city(self):
        assert w.extract_location("weather near Paris") == "Paris"

    def test_no_override_returns_none(self):
        assert w.extract_location("What is the weather?") is None

    def test_just_weather_returns_none(self):
        assert w.extract_location("weather") is None

    def test_multi_word_city(self):
        result = w.extract_location("weather in New York")
        assert result == "New York"

    def test_like_in_preposition(self):
        result = w.extract_location("what is the weather like in Tallinn, Estonia today?")
        assert result == "Tallinn, Estonia"

    def test_trailing_right_now_stripped(self):
        result = w.extract_location("weather in Zurich, Switzerland right now")
        assert result == "Zurich, Switzerland"

    def test_trailing_today_stripped(self):
        result = w.extract_location("weather in Berlin, Germany today")
        assert result == "Berlin, Germany"

    def test_trailing_tonight_stripped(self):
        result = w.extract_location("weather in Tokyo tonight?")
        assert result == "Tokyo"

    def test_trailing_currently_stripped(self):
        result = w.extract_location("weather in London currently")
        assert result == "London"


# ── wmo_condition tests ───────────────────────────────────────────────────────

class TestWmoCondition:
    def test_known_code_0(self):
        assert w.wmo_condition(0) == "Clear sky"

    def test_known_code_95(self):
        assert w.wmo_condition(95) == "Thunderstorm"

    def test_unknown_code_returns_fallback(self):
        result = w.wmo_condition(9999)
        assert "9999" in result


# ── run() tests ───────────────────────────────────────────────────────────────

class TestWeatherRun:
    def _params(self, prompt: str = "", location: str | None = None,
                memory: _FakeMemory | None = None) -> dict:
        return {
            "prompt": prompt,
            "location": location,
            "memory": memory or _FakeMemory(),
        }

    @pytest.mark.asyncio
    async def test_no_location_returns_failure(self):
        result = await w.run(self._params(prompt="What is the weather?"))
        assert not result.ok
        assert not result.retryable
        assert "location" in result.error.lower()

    @pytest.mark.asyncio
    async def test_location_from_memory(self):
        mem = _FakeMemory({"default_location": "Edinburgh"})
        with respx.mock() as mock:
            mock.get(w._GEOCODE_URL).mock(
                return_value=httpx.Response(200, json=_geocode_response("Edinburgh", "United Kingdom", 55.9, -3.2))
            )
            mock.get(w._FORECAST_URL).mock(
                return_value=httpx.Response(200, json=_forecast_response())
            )
            result = await w.run(self._params(memory=mem))
        assert result.ok
        assert "Edinburgh" in result.data["location"]

    @pytest.mark.asyncio
    async def test_inline_location_overrides_memory(self):
        mem = _FakeMemory({"default_location": "Edinburgh"})
        with respx.mock() as mock:
            mock.get(w._GEOCODE_URL).mock(
                return_value=httpx.Response(200, json=_geocode_response("Tokyo", "Japan", 35.68, 139.69))
            )
            mock.get(w._FORECAST_URL).mock(
                return_value=httpx.Response(200, json=_forecast_response())
            )
            result = await w.run(self._params(
                prompt="weather in Tokyo", memory=mem
            ))
        assert result.ok
        assert "Tokyo" in result.data["location"]

    @pytest.mark.asyncio
    async def test_geocode_not_found_returns_failure(self):
        with respx.mock() as mock:
            mock.get(w._GEOCODE_URL).mock(
                return_value=httpx.Response(200, json={"results": []})
            )
            result = await w.run(self._params(location="XYZNotACity"))
        assert not result.ok
        assert not result.retryable

    @pytest.mark.asyncio
    async def test_geocode_network_error_is_retryable(self):
        with respx.mock() as mock:
            mock.get(w._GEOCODE_URL).mock(side_effect=httpx.ConnectError("refused"))
            result = await w.run(self._params(location="London"))
        assert not result.ok
        assert result.retryable

    @pytest.mark.asyncio
    async def test_geocode_timeout_is_retryable(self):
        with respx.mock() as mock:
            mock.get(w._GEOCODE_URL).mock(side_effect=httpx.TimeoutException("timeout"))
            result = await w.run(self._params(location="London"))
        assert not result.ok
        assert result.retryable

    @pytest.mark.asyncio
    async def test_geocode_http_500_is_retryable(self):
        with respx.mock() as mock:
            mock.get(w._GEOCODE_URL).mock(return_value=httpx.Response(500))
            result = await w.run(self._params(location="London"))
        assert not result.ok
        assert result.retryable

    @pytest.mark.asyncio
    async def test_geocode_http_404_not_retryable(self):
        with respx.mock() as mock:
            mock.get(w._GEOCODE_URL).mock(return_value=httpx.Response(404))
            result = await w.run(self._params(location="London"))
        assert not result.ok
        assert not result.retryable

    @pytest.mark.asyncio
    async def test_forecast_network_error_is_retryable(self):
        with respx.mock() as mock:
            mock.get(w._GEOCODE_URL).mock(
                return_value=httpx.Response(200, json=_geocode_response())
            )
            mock.get(w._FORECAST_URL).mock(side_effect=httpx.ConnectError("refused"))
            result = await w.run(self._params(location="London"))
        assert not result.ok
        assert result.retryable

    @pytest.mark.asyncio
    async def test_success_normalized_schema(self):
        with respx.mock() as mock:
            mock.get(w._GEOCODE_URL).mock(
                return_value=httpx.Response(200, json=_geocode_response())
            )
            mock.get(w._FORECAST_URL).mock(
                return_value=httpx.Response(200, json=_forecast_response(
                    temp=18.0, feels=16.5, humidity=72, wind=15.0, code=2
                ))
            )
            result = await w.run(self._params(location="London"))

        assert result.ok
        d = result.data
        assert d["location"] == "London, United Kingdom"
        assert d["temperature"] == 18.0
        assert d["feels_like"] == 16.5
        assert d["humidity"] == 72
        assert d["wind_speed"] == 15.0
        assert d["condition"] == "Partly cloudy"
        assert "°C" in d["units"]["temperature"]
        assert "km/h" in d["units"]["wind_speed"]

    @pytest.mark.asyncio
    async def test_fahrenheit_units(self):
        original_units = w.WEATHER_UNITS
        w.WEATHER_UNITS = "fahrenheit"
        try:
            with respx.mock() as mock:
                mock.get(w._GEOCODE_URL).mock(
                    return_value=httpx.Response(200, json=_geocode_response())
                )
                mock.get(w._FORECAST_URL).mock(
                    return_value=httpx.Response(200, json=_forecast_response(temp=64.4))
                )
                result = await w.run(self._params(location="London"))
        finally:
            w.WEATHER_UNITS = original_units

        assert result.ok
        assert "°F" in result.data["units"]["temperature"]
        assert "mph" in result.data["units"]["wind_speed"]

    @pytest.mark.asyncio
    async def test_result_contains_no_provider_field_names(self):
        """ToolResult.data must not contain raw Open-Meteo field names."""
        with respx.mock() as mock:
            mock.get(w._GEOCODE_URL).mock(
                return_value=httpx.Response(200, json=_geocode_response())
            )
            mock.get(w._FORECAST_URL).mock(
                return_value=httpx.Response(200, json=_forecast_response())
            )
            result = await w.run(self._params(location="London"))

        assert result.ok
        provider_fields = {
            "temperature_2m", "apparent_temperature",
            "relative_humidity_2m", "wind_speed_10m", "weather_code",
        }
        assert not provider_fields.intersection(result.data.keys())
