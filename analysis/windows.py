"""Natural-language and compact time-window parsing.

"last 10 minutes", "previous ten minutes", "past hour", "15m", "2h", "today",
"all", or an explicit ISO range "2026-01-01T10:00/2026-01-01T11:00".

Windows are anchored either to *now* (live screen data) or, when the data
ends earlier than now (an uploaded file of historical measurements), to the
latest data timestamp — and the anchor is always reported so old data is
never presented as current.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

import pandas as pd

from core.models import TimeWindow, utcnow

WORD_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
    "forty-five": 45, "fifty": 50, "sixty": 60, "ninety": 90, "half an": 0.5, "half a": 0.5,
}
UNIT_SECONDS = {
    "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
    "d": 86400, "day": 86400, "days": 86400, "w": 604800, "week": 604800, "weeks": 604800,
}
_NUM = r"(\d+(?:\.\d+)?|" + "|".join(sorted(map(re.escape, WORD_NUMBERS), key=len, reverse=True)) + r")"
_UNIT = r"(" + "|".join(sorted(map(re.escape, UNIT_SECONDS), key=len, reverse=True)) + r")\b"
DURATION_RE = re.compile(rf"\b{_NUM}\s*-?\s*{_UNIT}", re.I)
BARE_UNIT_RE = re.compile(r"\b(?:last|past|previous|prior|this|the)\s+(second|minute|hour|day|week)\b", re.I)


class WindowError(ValueError):
    pass


def parse_duration(text: str) -> timedelta | None:
    m = DURATION_RE.search(text)
    if m:
        raw = m.group(1).lower()
        n = float(WORD_NUMBERS.get(raw, raw))
        return timedelta(seconds=n * UNIT_SECONDS[m.group(2).lower()])
    m2 = BARE_UNIT_RE.search(text)
    if m2:
        return timedelta(seconds=UNIT_SECONDS[m2.group(1).lower()])
    return None


def parse_window(text: str | None, data_end: datetime | None = None, data_start: datetime | None = None, now: datetime | None = None) -> TimeWindow:
    """Parse ``text`` into a window. ``data_end`` is the latest timestamp of the
    relevant data; if it is older than the window would allow, the window is
    anchored to it (anchor='latest-data')."""
    now = now or utcnow()
    t = (text or "all").strip().lower()
    if "/" in t and re.match(r"^\d{4}-", t):
        a, b = t.split("/", 1)
        try:
            start, end = pd.Timestamp(a), pd.Timestamp(b)
        except ValueError as exc:
            raise WindowError(f"Could not read the time range '{text}'.") from exc
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        if end <= start:
            raise WindowError("The end of the range must be after its start.")
        return TimeWindow(start.to_pydatetime(), end.to_pydatetime(), label=text or "", anchor="explicit")
    if t in ("all", "everything", "all time", "entire", "whole", ""):
        start = data_start or datetime(1970, 1, 1, tzinfo=now.tzinfo)
        return TimeWindow(start, max(now, data_end or now), label="all available data", anchor="all")
    anchor_end, anchor = now, "now"
    if data_end is not None and data_end < now - timedelta(minutes=2):
        anchor_end, anchor = data_end, "latest-data"
    if t in ("today",):
        start = anchor_end.replace(hour=0, minute=0, second=0, microsecond=0)
        return TimeWindow(start, anchor_end, label="today", anchor=anchor)
    dur = parse_duration(t)
    if dur is None:
        raise WindowError(f"I could not understand the time window '{text}'. Try e.g. 'last 10 minutes' or '2h'.")
    if dur <= timedelta(0):
        raise WindowError("A time window must be longer than zero.")
    previous = bool(re.search(r"\b(previous|prior|preceding|before that|earlier)\b", t))
    if previous:
        # "the previous ten minutes" = the block immediately before the latest ten minutes
        end = anchor_end - dur
        return TimeWindow(end - dur, end, label=f"previous {_fmt(dur)}", anchor=anchor)
    return TimeWindow(anchor_end - dur, anchor_end, label=f"last {_fmt(dur)}", anchor=anchor)


def window_after(event_time: datetime, duration: timedelta | None = None, now: datetime | None = None) -> TimeWindow:
    now = now or utcnow()
    end = min(now, event_time + duration) if duration else now
    return TimeWindow(event_time, end, label="after the event", anchor="event")


def window_before(event_time: datetime, duration: timedelta) -> TimeWindow:
    return TimeWindow(event_time - duration, event_time, label=f"{_fmt(duration)} before the event", anchor="event")


def _fmt(d: timedelta) -> str:
    s = int(d.total_seconds())
    for unit, n in (("week", 604800), ("day", 86400), ("hour", 3600), ("minute", 60)):
        if s >= n and s % n == 0:
            k = s // n
            return f"{k} {unit}{'s' if k != 1 else ''}"
    return f"{s} seconds"
