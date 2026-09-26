import sys as _sys
import re
from datetime import datetime, date, time, timezone, timedelta
try:
    import zoneinfo
except ImportError:
    from backports import zoneinfo

import logging
from tools.base import ToolResult
from tools import base as _registry

logger = logging.getLogger("assistant.tools.datetime")

_CITY_TO_TZ = {
    "tokyo": "Asia/Tokyo", "london": "Europe/London", "new york": "America/New_York",
    "paris": "Europe/Paris", "berlin": "Europe/Berlin", "sydney": "Australia/Sydney",
    "los angeles": "America/Los_Angeles", "chicago": "America/Chicago",
    "mumbai": "Asia/Kolkata", "dubai": "Asia/Dubai", "singapore": "Asia/Singapore",
    "hong kong": "Asia/Hong_Kong", "seoul": "Asia/Seoul", "beijing": "Asia/Shanghai",
    "moscow": "Europe/Moscow", "istanbul": "Europe/Istanbul", "cairo": "Africa/Cairo",
    "johannesburg": "Africa/Johannesburg", "toronto": "America/Toronto",
    "vancouver": "America/Vancouver", "helsinki": "Europe/Helsinki",
    "tallinn": "Europe/Tallinn", "stockholm": "Europe/Stockholm",
    "oslo": "Europe/Oslo", "copenhagen": "Europe/Copenhagen",
    "amsterdam": "Europe/Amsterdam", "rome": "Europe/Rome", "madrid": "Europe/Madrid",
    "lisbon": "Europe/Lisbon", "athens": "Europe/Athens", "warsaw": "Europe/Warsaw",
    "bangkok": "Asia/Bangkok", "jakarta": "Asia/Jakarta",
    "sao paulo": "America/Sao_Paulo", "mexico city": "America/Mexico_City",
    "denver": "America/Denver", "phoenix": "America/Phoenix",
}

_KNOWN_EVENTS = {
    "christmas": (12, 25), "new year": (1, 1), "new years": (1, 1),
    "valentine": (2, 14), "valentines": (2, 14),
    "halloween": (10, 31), "independence day": (7, 4),
}

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
MONTHS_SHORT = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

def _get_tz(location_str: str):
    location_str = location_str.lower().strip()
    if location_str in _CITY_TO_TZ:
        return zoneinfo.ZoneInfo(_CITY_TO_TZ[location_str])
    
    available = zoneinfo.available_timezones()
    for tz in available:
        if location_str == tz.lower() or location_str == tz.lower().replace("_", " "):
            return zoneinfo.ZoneInfo(tz)
    
    # Try fuzzy prefix
    for tz in available:
        if tz.lower().endswith("/" + location_str.replace(" ", "_")):
            return zoneinfo.ZoneInfo(tz)
            
    return None

def _parse_date(date_str: str) -> date:
    date_str = date_str.lower().strip()
    now = datetime.now()
    
    # ISO format 2026-03-15
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", date_str)
    if match:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        
    # March 15, 2026
    match = re.search(r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", date_str)
    if match:
        month_str = match.group(1)
        day = int(match.group(2))
        year = int(match.group(3))
        month = MONTHS.index(month_str) + 1 if month_str in MONTHS else MONTHS_SHORT.index(month_str) + 1
        return date(year, month, day)

    # March 15
    match = re.search(r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(\d{1,2})(?:st|nd|rd|th)?", date_str)
    if match:
        month_str = match.group(1)
        day = int(match.group(2))
        month = MONTHS.index(month_str) + 1 if month_str in MONTHS else MONTHS_SHORT.index(month_str) + 1
        
        target_date = date(now.year, month, day)
        if target_date < now.date():
            target_date = date(now.year + 1, month, day)
        return target_date

    return None

async def run(params: dict) -> ToolResult:
    prompt = params.get("prompt", "").lower()
    
    # Check for countdown
    if "until" in prompt or "days to" in prompt:
        event_match = re.search(r"(?:until|to)\s+(.*)", prompt)
        if event_match:
            event_raw = event_match.group(1).replace("?", "").strip()
            
            # Check known events
            target_date = None
            event_name = event_raw.title()
            for key, (month, day) in _KNOWN_EVENTS.items():
                if key in event_raw:
                    now = datetime.now()
                    target_date = date(now.year, month, day)
                    if target_date < now.date():
                        target_date = date(now.year + 1, month, day)
                    event_name = key.title()
                    break
            
            # Try parsing date
            if not target_date:
                target_date = _parse_date(event_raw)
            
            if target_date:
                now = datetime.now().date()
                delta = target_date - now
                days_remaining = delta.days
                formatted_date = target_date.strftime("%B %-d, %Y") if hasattr(target_date, "strftime") else target_date.isoformat()
                if "%-d" not in target_date.strftime("%B %-d, %Y"):
                   formatted_date = target_date.strftime("%B %d, %Y").replace(" 0", " ")
                
                return ToolResult(ok=True, 
                    data={
                        "query_type": "countdown",
                        "event": event_name,
                        "target_date": target_date.isoformat(),
                        "days_remaining": days_remaining,
                        "formatted": f"{days_remaining} days until {event_name} ({formatted_date})"
                    }
                )
    
    # Check for date query
    if "what day is" in prompt or "what day of the week is" in prompt:
        date_str_match = re.search(r"(?:is)\s+(.*)", prompt)
        if date_str_match:
            date_str = date_str_match.group(1).replace("?", "").strip()
            parsed_date = _parse_date(date_str)
            if parsed_date:
                day_of_week = parsed_date.strftime("%A")
                formatted_date = parsed_date.strftime("%A, %B %-d, %Y")
                if "%-d" not in formatted_date:
                    formatted_date = parsed_date.strftime("%A, %B %d, %Y").replace(" 0", " ")
                return ToolResult(ok=True, 
                    data={
                        "query_type": "date",
                        "date": parsed_date.isoformat(),
                        "day_of_week": day_of_week,
                        "formatted": formatted_date
                    }
                )
                
    if "today's date" in prompt or "date today" in prompt or "what is today" in prompt:
         now = datetime.now()
         day_of_week = now.strftime("%A")
         formatted_date = now.strftime("%A, %B %-d, %Y")
         if "%-d" not in formatted_date:
             formatted_date = now.strftime("%A, %B %d, %Y").replace(" 0", " ")
         return ToolResult(ok=True, 
             data={
                 "query_type": "date",
                 "date": now.date().isoformat(),
                 "day_of_week": day_of_week,
                 "formatted": formatted_date
             }
         )
         
    # Check for time query
    location_match = re.search(r"(?:in|for)\s+([a-zA-Z\s_/-]+?)(?:\?|$)", prompt)
    location_str = None
    if location_match:
        location_str = location_match.group(1).replace("?", "").strip()
        if location_str == "it": # what time is it
             location_str = None
    elif params.get("timezone"):
        location_str = params.get("timezone")
        
    tz = None
    city_name = "Local"
    if location_str:
        tz = _get_tz(location_str)
        if tz:
            city_name = location_str.title()
        else:
            return ToolResult.failure(error=f"Timezone or city '{location_str}' not found.")
            
    now = datetime.now(tz)
    day_of_week = now.strftime("%A")
    local_time = now.strftime("%Y-%m-%d %H:%M")
    date_str = now.strftime("%Y-%m-%d")
    
    utc_offset = now.strftime("%z")
    if utc_offset:
        sign = utc_offset[0]
        hrs = utc_offset[1:3]
        mins = utc_offset[3:5]
        utc_offset_str = f"{sign}{hrs}:{mins}"
    else:
        utc_offset_str = "+00:00"
        
    return ToolResult(ok=True, 
        data={
            "query_type": "time",
            "timezone": str(now.tzinfo) if tz else "Local",
            "city": city_name,
            "local_time": local_time,
            "day_of_week": day_of_week,
            "date": date_str,
            "utc_offset": utc_offset_str
        }
    )

def format_datetime_response(result: ToolResult) -> str:
    if result.error:
        return f"Error: {result.error}"
        
    data = result.data
    query_type = data.get("query_type")
    
    if query_type == "time":
        time_str = data["local_time"].split(" ")[1]
        day_of_week = data["day_of_week"]
        city = data["city"]
        utc_offset = data["utc_offset"]
        return f"It's **{time_str}** ({day_of_week}) in {city} (UTC{utc_offset})."
    
    elif query_type == "date":
        date_iso = data["date"]
        day_of_week = data["day_of_week"]
        date_obj = date.fromisoformat(date_iso)
        month = date_obj.strftime("%B")
        day = date_obj.day
        year = date_obj.year
        return f"{month} {day}, {year} is a **{day_of_week}**."
        
    elif query_type == "countdown":
        event = data["event"]
        target_date_iso = data["target_date"]
        days_remaining = data["days_remaining"]
        date_obj = date.fromisoformat(target_date_iso)
        month = date_obj.strftime("%B")
        day = date_obj.day
        year = date_obj.year
        return f"**{days_remaining} days** until {event} ({month} {day}, {year})."
        
    return "Unknown query type."

_registry_mod = __import__("tools")
_registry_mod.register("datetime", _sys.modules[__name__])
