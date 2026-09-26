import asyncio
import pytest
from tools.timer import (
    run,
    set_fire_callback,
    _TIMERS,
    format_timer_response,
)

class _FakeMemory:
    pass

@pytest.fixture(autouse=True)
def clear_timers():
    _TIMERS.clear()
    set_fire_callback(None)
    yield

@pytest.mark.asyncio
async def test_set_timer_basic():
    params = {
        "action": "set",
        "duration_minutes": 5
    }
    result = await run(params)
    assert result.ok
    assert "id" in result.data
    assert result.data["duration_minutes"] == 5
    assert "fires_at" in result.data
    assert result.data["label"] is None

@pytest.mark.asyncio
async def test_set_timer_with_label():
    params = {
        "action": "set",
        "duration_minutes": 10,
        "label": "check oven"
    }
    result = await run(params)
    assert result.ok
    assert result.data["label"] == "check oven"

@pytest.mark.asyncio
async def test_set_timer_missing_duration():
    params = {
        "action": "set"
    }
    result = await run(params)
    assert not result.ok
    assert "duration_minutes parameter is required" in result.error

@pytest.mark.asyncio
async def test_list_timers_empty():
    params = {"action": "list"}
    result = await run(params)
    assert result.ok
    assert result.data["timers"] == []

@pytest.mark.asyncio
async def test_list_timers_with_active():
    await run({"action": "set", "duration_minutes": 5, "label": "t1"})
    await run({"action": "set", "duration_minutes": 10, "label": "t2"})
    
    result = await run({"action": "list"})
    assert result.ok
    assert len(result.data["timers"]) == 2
    labels = {t["label"] for t in result.data["timers"]}
    assert labels == {"t1", "t2"}

@pytest.mark.asyncio
async def test_cancel_timer():
    set_result = await run({"action": "set", "duration_minutes": 5, "label": "t1"})
    timer_id = set_result.data["id"]
    
    cancel_result = await run({"action": "cancel", "timer_id": timer_id})
    assert cancel_result.ok
    assert cancel_result.data["cancelled"] is True
    assert cancel_result.data["id"] == timer_id
    
    list_result = await run({"action": "list"})
    assert len(list_result.data["timers"]) == 0

@pytest.mark.asyncio
async def test_cancel_nonexistent():
    result = await run({"action": "cancel", "timer_id": "nope"})
    assert not result.ok
    assert "not found" in result.error

def test_format_timer_response_set():
    class DummyRes:
        ok = True
        data = {"id": "1", "label": "check oven", "duration_minutes": 20}
    
    txt = format_timer_response(DummyRes(), "set")
    assert txt == 'Timer set: "check oven" — fires in 20 minutes.'

def test_format_timer_response_list():
    class DummyRes:
        ok = True
        data = {
            "timers": [
                {"id": "1", "label": "check oven", "fired": False, "remaining_seconds": 900},
                {"id": "2", "label": "meeting", "fired": False, "remaining_seconds": 5400}
            ]
        }
    txt = format_timer_response(DummyRes(), "list")
    assert "You have 2 active timers:" in txt
    assert '1. "check oven" — 15 minutes remaining' in txt
    assert '2. "meeting" — fires in 1 hour 30 minutes' in txt

def test_format_timer_response_empty_list():
    class DummyRes:
        ok = True
        data = {"timers": []}
    txt = format_timer_response(DummyRes(), "list")
    assert txt == "No active timers."

def test_format_timer_response_cancel():
    class DummyRes:
        ok = True
        data = {"id": "1", "label": "check oven"}
    txt = format_timer_response(DummyRes(), "cancel")
    assert txt == 'Timer "check oven" cancelled.'

@pytest.mark.asyncio
async def test_fire_callback():
    fired_timer = None
    def cb(t):
        nonlocal fired_timer
        fired_timer = t

    set_fire_callback(cb)
    
    await run({"action": "set", "duration_minutes": 0.01, "label": "fast"})
    
    await asyncio.sleep(0.8) 
    
    assert fired_timer is not None
    assert fired_timer.label == "fast"
    assert fired_timer.fired is True

@pytest.mark.asyncio
async def test_cleanup_old_fired():
    await run({"action": "set", "duration_minutes": 0.01, "label": "fast"})
    await asyncio.sleep(0.8)
    
    import datetime
    from tools.timer import _TIMERS
    timer_id = list(_TIMERS.keys())[0]
    
    _TIMERS[timer_id].fires_at = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=2)
    
    result = await run({"action": "list"})
    assert result.ok
    assert len(result.data["timers"]) == 0
    assert len(_TIMERS) == 0
