"""
sentinel-query: relative time parser.

Converts natural language time expressions into (start_ts, end_ts) UTC Unix
timestamp ranges.  This runs in pure Python *before* the LLM call so we can
inject precise UTC values into the system prompt rather than asking the model
to do arithmetic on Unix timestamps itself (which 8B models do poorly).

The LLM sees: "Current time: 1710000000  (2024-03-09 20:00:00 UTC)"
and only needs to map the user's phrase to a named bucket
("last_week", "yesterday", etc.).

However, we also keep this module usable standalone for validation and testing.

Strategy
────────
1. A small hand-written pattern matcher handles the common cases perfectly.
2. The LLM fills in the gaps for ambiguous expressions ("a few months ago",
   "around springtime last year") — the LLM receives `now_ts` and the grammar
   constrains it to emit integer timestamps, so arithmetic errors are bounded.
3. If the LLM emits timestamps that fail sanity checks (e.g. start > end,
   timestamps in the far future), we fall back to Tier-2 / Tier-3 retrieval.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _midnight(dt: datetime) -> datetime:
    """Return dt with time zeroed to 00:00:00 UTC."""
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _start_of_week(dt: datetime) -> datetime:
    """Monday 00:00:00 of the ISO week containing *dt*."""
    return _midnight(dt - timedelta(days=dt.weekday()))


def _start_of_month(dt: datetime) -> datetime:
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _start_of_year(dt: datetime) -> datetime:
    return dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)


def _ts(dt: datetime) -> int:
    return int(dt.timestamp())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TimeRange:
    """A (start_ts, end_ts) pair in UTC Unix seconds.  Either may be None."""

    __slots__ = ("start_ts", "end_ts")

    def __init__(self, start_ts: Optional[int], end_ts: Optional[int]) -> None:
        self.start_ts = start_ts
        self.end_ts = end_ts

    def as_dict(self) -> Optional[dict]:
        """Return None if no constraint; otherwise a dict for the JSON schema."""
        if self.start_ts is None and self.end_ts is None:
            return None
        d: dict = {}
        if self.start_ts is not None:
            d["start_ts"] = self.start_ts
        if self.end_ts is not None:
            d["end_ts"] = self.end_ts
        return d

    def __repr__(self) -> str:
        return f"TimeRange(start={self.start_ts}, end={self.end_ts})"


def parse_time_expression(text: str, now_ts: Optional[int] = None) -> TimeRange:
    """
    Extract a time range from *text* using pattern matching.

    Returns a TimeRange with None fields if no time expression is recognised.
    *now_ts*: current time as a UTC Unix timestamp.  Defaults to time.time().
    """
    now = (
        datetime.fromtimestamp(now_ts, tz=timezone.utc)
        if now_ts is not None
        else datetime.now(tz=timezone.utc)
    )
    t = text.lower()

    # ── Today ─────────────────────────────────────────────────────────────
    if re.search(r"\btoday\b", t):
        start = _midnight(now)
        return TimeRange(_ts(start), _ts(now))

    # ── Yesterday ─────────────────────────────────────────────────────────
    if re.search(r"\byesterday\b", t):
        yesterday = _midnight(now) - timedelta(days=1)
        return TimeRange(_ts(yesterday), _ts(_midnight(now)))

    # ── Last N days ───────────────────────────────────────────────────────
    m = re.search(r"\blast\s+(\d+)\s+days?\b", t)
    if m:
        n = int(m.group(1))
        return TimeRange(_ts(now - timedelta(days=n)), _ts(now))

    # ── This week ─────────────────────────────────────────────────────────
    if re.search(r"\bthis\s+week\b", t):
        return TimeRange(_ts(_start_of_week(now)), _ts(now))

    # ── Last week ─────────────────────────────────────────────────────────
    if re.search(r"\blast\s+week\b", t):
        start_this = _start_of_week(now)
        start_last = start_this - timedelta(weeks=1)
        return TimeRange(_ts(start_last), _ts(start_this))

    # ── Last N weeks ──────────────────────────────────────────────────────
    m = re.search(r"\blast\s+(\d+)\s+weeks?\b", t)
    if m:
        n = int(m.group(1))
        return TimeRange(_ts(now - timedelta(weeks=n)), _ts(now))

    # ── This month ────────────────────────────────────────────────────────
    if re.search(r"\bthis\s+month\b", t):
        return TimeRange(_ts(_start_of_month(now)), _ts(now))

    # ── Last month ────────────────────────────────────────────────────────
    if re.search(r"\blast\s+month\b", t):
        start_this = _start_of_month(now)
        # Go back one month (handle Jan → Dec year rollover)
        if now.month == 1:
            start_last = start_this.replace(year=now.year - 1, month=12)
        else:
            start_last = start_this.replace(month=now.month - 1)
        return TimeRange(_ts(start_last), _ts(start_this))

    # ── Last N months ─────────────────────────────────────────────────────
    m = re.search(r"\blast\s+(\d+)\s+months?\b", t)
    if m:
        n = int(m.group(1))
        return TimeRange(_ts(now - timedelta(days=n * 30)), _ts(now))

    # ── A few months ago ──────────────────────────────────────────────────
    if re.search(r"\ba\s+few\s+months?\s+ago\b", t):
        return TimeRange(_ts(now - timedelta(days=90)), _ts(now - timedelta(days=30)))

    # ── Recently / recently ───────────────────────────────────────────────
    if re.search(r"\brecently\b|\bjust\s+recently\b", t):
        return TimeRange(_ts(now - timedelta(days=7)), _ts(now))

    # ── This year ─────────────────────────────────────────────────────────
    if re.search(r"\bthis\s+year\b", t):
        return TimeRange(_ts(_start_of_year(now)), _ts(now))

    # ── Last year ─────────────────────────────────────────────────────────
    if re.search(r"\blast\s+year\b", t):
        start_this = _start_of_year(now)
        start_last = start_this.replace(year=now.year - 1)
        return TimeRange(_ts(start_last), _ts(start_this))

    # ── Named months: "in March", "last March", "March 2023" ─────────────
    _MONTHS = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4,
        "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    for name, num in _MONTHS.items():
        # "March 2023" or "in March 2023"
        m = re.search(rf"\b{name}\s+(20\d\d)\b", t)
        if m:
            year = int(m.group(1))
            start = datetime(year, num, 1, tzinfo=timezone.utc)
            # end = first day of next month
            if num == 12:
                end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                end = datetime(year, num + 1, 1, tzinfo=timezone.utc)
            return TimeRange(_ts(start), _ts(end))

        # "last March" or "in March" (no year) — most recent occurrence
        if re.search(rf"\b(?:last\s+|in\s+){name}\b", t):
            year = now.year if now.month > num else now.year - 1
            start = datetime(year, num, 1, tzinfo=timezone.utc)
            if num == 12:
                end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                end = datetime(year, num + 1, 1, tzinfo=timezone.utc)
            return TimeRange(_ts(start), _ts(end))

    # ── No time expression found ──────────────────────────────────────────
    return TimeRange(None, None)


def validate_range(start_ts: Optional[int], end_ts: Optional[int]) -> bool:
    """
    Sanity-check a timestamp range emitted by the LLM.

    Returns False if the range is clearly wrong so the caller can discard it
    and fall back to no time constraint.
    """
    now = int(time.time())
    if start_ts is not None and end_ts is not None:
        if start_ts > end_ts:
            return False
    if start_ts is not None and start_ts > now + 86400:
        # Start is in the future — nonsensical for a file search.
        return False
    if end_ts is not None and end_ts > now + 86400:
        return False
    # Reject timestamps before 2000-01-01 (likely LLM arithmetic overflow).
    epoch_2000 = 946684800
    if start_ts is not None and start_ts < epoch_2000:
        return False
    return True


def format_ts_for_prompt(now_ts: int) -> str:
    """Human-readable string injected into the LLM system prompt."""
    dt = datetime.fromtimestamp(now_ts, tz=timezone.utc)
    return f"{now_ts}  ({dt.strftime('%Y-%m-%d %H:%M:%S UTC')})"
