import pytest
from datetime import datetime
from tools.datetime_tool import run, format_datetime_response
from tools.base import ToolResult

class _FakeMemory:
    pass

@pytest.fixture
def fake_memory():
    return _FakeMemory()

@pytest.mark.asyncio
async def test_time_in_city():
    params = {"prompt": "what time is it in Tokyo"}
    result = await run(params)
    assert not result.error
    assert result.data["query_type"] == "time"
    assert result.data["timezone"] == "Asia/Tokyo"
    assert result.data["city"].lower() == "tokyo"
    assert "day_of_week" in result.data
    assert "local_time" in result.data

@pytest.mark.asyncio
async def test_time_in_city_case_insensitive():
    params = {"prompt": "time in LONDON"}
    result = await run(params)
    assert not result.error
    assert result.data["query_type"] == "time"
    assert result.data["timezone"] == "Europe/London"

@pytest.mark.asyncio
async def test_time_local():
    params = {"prompt": "what time is it"}
    result = await run(params)
    assert not result.error
    assert result.data["query_type"] == "time"
    assert result.data["timezone"] == "Local"
    assert result.data["city"] == "Local"

@pytest.mark.asyncio
async def test_time_iana_zone():
    params = {"prompt": "time in America/New_York"}
    result = await run(params)
    assert not result.error
    assert result.data["query_type"] == "time"
    assert result.data["timezone"] == "America/New_York"

@pytest.mark.asyncio
async def test_date_day_of_week():
    params = {"prompt": "what day is 2026-03-15"}
    result = await run(params)
    assert not result.error
    assert result.data["query_type"] == "date"
    assert result.data["date"] == "2026-03-15"
    assert result.data["day_of_week"] == "Sunday"

@pytest.mark.asyncio
async def test_date_natural():
    params = {"prompt": "what day of the week is March 15, 2026"}
    result = await run(params)
    assert not result.error
    assert result.data["query_type"] == "date"
    assert result.data["date"] == "2026-03-15"
    assert result.data["day_of_week"] == "Sunday"

@pytest.mark.asyncio
async def test_countdown_christmas():
    params = {"prompt": "how many days until Christmas"}
    result = await run(params)
    assert not result.error
    assert result.data["query_type"] == "countdown"
    assert "days_remaining" in result.data
    assert result.data["event"].lower() == "christmas"

@pytest.mark.asyncio
async def test_countdown_specific_date():
    params = {"prompt": "how many days until January 1, 2030"}
    result = await run(params)
    assert not result.error
    assert result.data["query_type"] == "countdown"
    assert result.data["target_date"] == "2030-01-01"

@pytest.mark.asyncio
async def test_countdown_new_year():
    params = {"prompt": "days until new year"}
    result = await run(params)
    assert not result.error
    assert result.data["query_type"] == "countdown"
    assert result.data["event"].lower() == "new year"

@pytest.mark.asyncio
async def test_unknown_city():
    params = {"prompt": "time in Atlantis"}
    result = await run(params)
    assert result.error is not None
    assert "Atlantis".lower() in result.error.lower()

def test_format_time_response():
    res = ToolResult(ok=True, data={
        "query_type": "time", "timezone": "Asia/Tokyo", "city": "Tokyo", 
        "local_time": "2026-09-26 00:49", "day_of_week": "Saturday", 
        "date": "2026-09-26", "utc_offset": "+09:00"
    })
    formatted = format_datetime_response(res)
    assert formatted == "It's **00:49** (Saturday) in Tokyo (UTC+09:00)."

def test_format_date_response():
    res = ToolResult(ok=True, data={
        "query_type": "date", "date": "2026-03-15", "day_of_week": "Sunday"
    })
    formatted = format_datetime_response(res)
    assert formatted == "March 15, 2026 is a **Sunday**."

def test_format_countdown_response():
    res = ToolResult(ok=True, data={
        "query_type": "countdown", "event": "Christmas", "target_date": "2026-12-25", "days_remaining": 91
    })
    formatted = format_datetime_response(res)
    assert formatted == "**91 days** until Christmas (December 25, 2026)."

@pytest.mark.asyncio
async def test_today_date():
    params = {"prompt": "what is today's date"}
    result = await run(params)
    assert not result.error
    assert result.data["query_type"] == "date"
    assert "date" in result.data
    assert "day_of_week" in result.data
