import asyncio
import datetime
import logging
import re
import uuid
import sys as _sys
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass

from tools.base import ToolResult
from tools import base as _registry
_registry_mod = __import__("tools")

logger = logging.getLogger("assistant.tools.timer")


def parse_timer_prompt(prompt: str) -> dict:
    p = prompt.strip().lower()

    if any(w in p for w in ("cancel", "stop", "delete", "remove")) and any(w in p for w in ("timer", "reminder", "alarm")):
        m = re.search(r"(?:cancel|stop|delete|remove)\s+(?:the\s+)?(?:timer|reminder|alarm)?\s*(?:called|named)?\s*([a-zA-Z0-9_\-]+)?", p)
        target = m.group(1).strip() if m and m.group(1) else ""
        if target in ("the", "a", "my", "timer", "reminder", "alarm", ""):
            target = ""
        return {"action": "cancel", "timer_id": target, "label": target}

    if any(w in p for w in ("what timer", "what reminder", "list timer", "list reminder", "show timer", "show reminder", "my timer", "my reminder", "time left", "time remaining", "active timer")):
        return {"action": "list"}

    duration_minutes = None
    hr_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h\b)", p)
    min_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:minutes?|mins?|m\b)", p)
    sec_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s\b)", p)

    total_mins = 0.0
    matched = False
    if hr_m:
        total_mins += float(hr_m.group(1)) * 60.0
        matched = True
    if min_m:
        total_mins += float(min_m.group(1))
        matched = True
    if sec_m:
        total_mins += float(sec_m.group(1)) / 60.0
        matched = True

    if not matched:
        num_m = re.search(r"(?:timer|reminder|alarm|in|for)\s+(?:for\s+)?(\d+(?:\.\d+)?)\b", p)
        if num_m:
            total_mins = float(num_m.group(1))
            matched = True

    if matched:
        duration_minutes = total_mins

    label = None
    label_m = re.search(r"(?:remind me to|reminder to|remind me)\s+(.+?)(?:\s+(?:in|for|\bat)\s+\d+|\s*$)", p)
    if label_m:
        label = label_m.group(1).strip()
    else:
        label_m2 = re.search(r"(?:called|named|for)\s+([a-zA-Z0-9_\s]+?)(?:\s+(?:for|in)\s+\d+|\s*$)", p)
        if label_m2:
            cand = label_m2.group(1).strip()
            if not re.match(r"^\d+(?:\.\d+)?\s*(?:hours?|minutes?|seconds?)?", cand):
                label = cand

    return {
        "action": "set" if duration_minutes is not None else None,
        "duration_minutes": duration_minutes,
        "label": label,
    }


@dataclass
class TimerEntry:
    id: str
    label: Optional[str]
    duration_minutes: float
    created_at: datetime.datetime
    fires_at: datetime.datetime
    task: Optional[asyncio.Task]
    fired: bool


_TIMERS: Dict[str, TimerEntry] = {}
_on_fire_callback: Optional[Callable[[TimerEntry], Any]] = None


def set_fire_callback(cb: Callable[[TimerEntry], Any]) -> None:
    global _on_fire_callback
    _on_fire_callback = cb


async def _timer_worker(timer_id: str, duration_seconds: float):
    try:
        await asyncio.sleep(duration_seconds)
    except asyncio.CancelledError:
        return

    if timer_id in _TIMERS:
        timer = _TIMERS[timer_id]
        timer.fired = True
        logger.info(f"Timer {timer_id} ('{timer.label}') fired.")
        if _on_fire_callback:
            try:
                if asyncio.iscoroutinefunction(_on_fire_callback):
                    await _on_fire_callback(timer)
                else:
                    _on_fire_callback(timer)
            except Exception as e:
                logger.error(f"Error in timer callback: {e}")


def _cleanup_old_fired_timers():
    now = datetime.datetime.now(datetime.timezone.utc)
    to_delete = []
    for t_id, timer in _TIMERS.items():
        if timer.fired:
            if (now - timer.fires_at).total_seconds() > 3600:
                to_delete.append(t_id)
    for t_id in to_delete:
        del _TIMERS[t_id]


async def run(params: dict) -> ToolResult:
    action = params.get("action")
    prompt = params.get("prompt", "")

    if not action and prompt:
        parsed = parse_timer_prompt(prompt)
        action = parsed.get("action")
        if action == "set":
            if params.get("duration_minutes") is None and parsed.get("duration_minutes") is not None:
                params["duration_minutes"] = parsed.get("duration_minutes")
            if not params.get("label") and parsed.get("label"):
                params["label"] = parsed.get("label")
        elif action == "cancel":
            target = parsed.get("timer_id", "")
            if not params.get("timer_id") and target:
                if target in _TIMERS:
                    params["timer_id"] = target
                else:
                    for t_id, t_entry in _TIMERS.items():
                        if t_entry.label and target.lower() in t_entry.label.lower():
                            params["timer_id"] = t_id
                            break
            if not params.get("timer_id"):
                active = [t for t in _TIMERS.values() if not t.fired]
                if active:
                    params["timer_id"] = active[-1].id

    if not action:
        return ToolResult(ok=False, error="action parameter is required", data={})

    if action == "set":
        duration_minutes = params.get("duration_minutes")
        if duration_minutes is None:
            return ToolResult(ok=False, error="duration_minutes parameter is required for 'set'", data={})
        
        try:
            duration_minutes = float(duration_minutes)
        except ValueError:
            return ToolResult(ok=False, error="duration_minutes must be a number", data={})

        label = params.get("label")
        timer_id = uuid.uuid4().hex[:8]
        now = datetime.datetime.now(datetime.timezone.utc)
        fires_at = now + datetime.timedelta(minutes=duration_minutes)

        timer_entry = TimerEntry(
            id=timer_id,
            label=label,
            duration_minutes=duration_minutes,
            created_at=now,
            fires_at=fires_at,
            task=None,
            fired=False
        )

        duration_seconds = duration_minutes * 60.0
        timer_entry.task = asyncio.create_task(_timer_worker(timer_id, duration_seconds))
        _TIMERS[timer_id] = timer_entry

        return ToolResult(
            ok=True,
            data={
                "action": "set",
                "id": timer_id,
                "label": label,
                "duration_minutes": duration_minutes,
                "fires_at": fires_at.isoformat()
            }
        )

    elif action == "list":
        _cleanup_old_fired_timers()
        now = datetime.datetime.now(datetime.timezone.utc)
        
        timers_data = []
        for t_id, timer in _TIMERS.items():
            remaining_seconds = max(0.0, (timer.fires_at - now).total_seconds())
            if timer.fired:
                remaining_seconds = 0.0

            timers_data.append({
                "id": t_id,
                "label": timer.label,
                "duration_minutes": timer.duration_minutes,
                "fires_at": timer.fires_at.isoformat(),
                "fired": timer.fired,
                "remaining_seconds": remaining_seconds
            })

        return ToolResult(
            ok=True,
            data={
                "action": "list",
                "timers": timers_data
            }
        )

    elif action == "cancel":
        timer_id = params.get("timer_id")
        if not timer_id:
            return ToolResult(ok=False, error="timer_id parameter is required for 'cancel'", data={})

        if timer_id not in _TIMERS:
            return ToolResult(ok=False, error=f"Timer with id '{timer_id}' not found", data={})

        timer = _TIMERS[timer_id]
        if timer.task and not timer.task.done():
            timer.task.cancel()

        label = timer.label
        del _TIMERS[timer_id]

        return ToolResult(
            ok=True,
            data={
                "action": "cancel",
                "id": timer_id,
                "label": label,
                "cancelled": True
            }
        )

    else:
        return ToolResult(ok=False, error=f"Unknown action: {action}", data={})


def format_timer_response(result: ToolResult, action: str = "") -> str:
    if not result.ok:
        return f"Error: {result.error}"

    act = action or result.data.get("action", "")
    if not act:
        if "timers" in result.data:
            act = "list"
        elif "cancelled" in result.data:
            act = "cancel"
        elif "fires_at" in result.data or "duration_minutes" in result.data:
            act = "set"

    if act == "set":
        data = result.data
        label = data.get("label")
        duration = data.get("duration_minutes")
        if label:
            return f'Timer set: "{label}" — fires in {duration} minutes.'
        else:
            return f"Timer set — fires in {duration} minutes."

    elif act == "list":
        timers = result.data.get("timers", [])
        if not timers:
            return "No active timers."
        
        active_timers = [t for t in timers if not t["fired"]]
        if not active_timers:
            return "No active timers."

        lines = [f"You have {len(active_timers)} active timers:"]
        for i, t in enumerate(active_timers, start=1):
            label = f'"{t["label"]}"' if t.get("label") else "Timer"
            remaining = t.get("remaining_seconds", 0)
            
            if remaining < 60:
                time_str = f"{int(remaining)} seconds remaining"
            elif remaining < 3600:
                mins = int(remaining // 60)
                time_str = f"{mins} minutes remaining"
            else:
                hours = int(remaining // 3600)
                mins = int((remaining % 3600) // 60)
                if mins > 0:
                    time_str = f"fires in {hours} hour{'s' if hours != 1 else ''} {mins} minutes"
                else:
                    time_str = f"fires in {hours} hour{'s' if hours != 1 else ''}"

            lines.append(f"{i}. {label} — {time_str}")
        return "\n".join(lines)

    elif act == "cancel":
        data = result.data
        label = data.get("label")
        if label:
            return f'Timer "{label}" cancelled.'
        else:
            id_val = data.get("id", "")
            return f'Timer {id_val} cancelled.'

    return "Done."


_registry_mod.register("timer", _sys.modules[__name__])
