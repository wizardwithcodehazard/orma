"""
clock.py - Calendar time in the user's timezone, and dates the user said in words. Timestamps are UTC
epoch seconds everywhere; only days (what "yesterday" or a log line's date means) depend on the zone. A
zone of None means the system's local zone, so an Indian user's day starts at midnight IST, not UTC.
Small models should not do date arithmetic: they copy the phrase ("last Saturday") and this module
resolves it against when it was said.
"""

import re
from datetime import date, datetime, timezone, tzinfo
from functools import lru_cache


@lru_cache(maxsize=8)
def zone(name: str) -> tzinfo | None:
    """'' is the system's local zone (None), 'UTC' is UTC, anything else an IANA name such as Asia/Kolkata
    (on Windows that needs the tzdata package)."""
    if not name:
        return None
    if name.upper() == "UTC":
        return timezone.utc
    from zoneinfo import ZoneInfo

    return ZoneInfo(name)


def _system_offset() -> tzinfo:
    """The system zone's current fixed offset: the fallback where the OS cannot convert a time (Windows rejects
    times around 1970 and before)."""
    return datetime.now().astimezone().tzinfo


def _localize(naive: datetime) -> datetime:
    """A naive wall-clock time read in the system zone."""
    try:
        return naive.astimezone()
    except (OSError, OverflowError, ValueError):
        return naive.replace(tzinfo=_system_offset())


def local(stamp: float, tz: tzinfo | None) -> datetime:
    """The aware wall-clock time of stamp in tz (None: the system zone, daylight saving included)."""
    if tz is not None:
        return datetime.fromtimestamp(stamp, tz)
    utc = datetime.fromtimestamp(stamp, timezone.utc)
    try:
        return utc.astimezone()
    except (OSError, OverflowError, ValueError):
        return utc.astimezone(_system_offset())


def day(stamp: float, tz: tzinfo | None) -> date:
    return local(stamp, tz).date()


def iso_day(stamp: float, tz: tzinfo | None) -> str:
    return day(stamp, tz).isoformat()


def midnight(d: date, tz: tzinfo | None) -> float:
    """The timestamp at which day d starts in tz."""
    start = datetime(d.year, d.month, d.day)
    return (start.replace(tzinfo=tz) if tz is not None else _localize(start)).timestamp()


def parse_time(value: str, default: float | None, tz: tzinfo | None = None) -> float | None:
    """An ISO date or datetime ('2026', '2026-03', '2026-03-07', '2026-03-07T10:00Z'); a value without its own
    offset is read in tz. default when value is empty or not ISO."""
    if not value:
        return default
    value = value.strip()
    if re.fullmatch(r"\d{4}", value):
        value += "-01-01"
    elif re.fullmatch(r"\d{4}-\d{2}", value):
        value += "-01"
    if value[-1:] in ("Z", "z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=tz) if tz is not None else _localize(parsed)
        return parsed.timestamp()
    except (ValueError, OverflowError, OSError):
        return default


@lru_cache(maxsize=1024)
def resolve(phrase: str, said_at: float, tz: tzinfo | None, date_order: str = "") -> float | None:
    """The start of the day a date phrase refers to, read as said at said_at: ISO first, then natural language
    ('last Saturday', '3 days ago', 'in March 2024') via dateparser, preferring the past. Leading words are
    dropped one at a time until something parses, because dateparser misses some qualifiers ('last Saturday'
    parses as 'Saturday', 'since 2019' as '2019'). None when nothing parses or dateparser is not installed."""
    try:
        import dateparser
    except ImportError:
        dateparser = None
    settings = {"RELATIVE_BASE": local(said_at, tz).replace(tzinfo=None), "PREFER_DATES_FROM": "past",
                "PREFER_DAY_OF_MONTH": "first", "RETURN_AS_TIMEZONE_AWARE": False}
    if date_order:
        settings["DATE_ORDER"] = date_order
    words = phrase.split()
    for i in range(len(words)):  # longest suffix first, so "in March 2024" is not cut down to "2024"
        suffix = " ".join(words[i:])
        stamp = parse_time(suffix, None, tz)  # ISO (a bare year stays January 1st, as dateparser would not)
        if stamp is not None:
            return stamp
        parsed = dateparser.parse(suffix, languages=["en"], settings=settings) if dateparser else None
        if parsed is not None:
            return midnight(parsed.date(), tz)
    return None
