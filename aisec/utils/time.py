"""
AISec timestamp utilities.

All timestamps in AISec are UTC ISO-8601 strings.
This module provides consistent helpers used everywhere.

Why strings not datetime objects?
  Strings serialise to JSON without extra handling,
  are human-readable in log files, and are unambiguous
  about timezone (always UTC, always explicit).
"""

from __future__ import annotations

from datetime import datetime, timezone


# ── Timestamp creation ────────────────────────────────────────────────────────

def now_utc() -> str:
    """
    Return the current UTC time as an ISO-8601 string.

    Example:
        '2025-05-03T22:14:05.123456+00:00'
    """
    return datetime.now(timezone.utc).isoformat()


def from_timestamp(ts: float) -> str:
    """
    Convert a Unix timestamp (float) to a UTC ISO-8601 string.

    Args:
        ts: Seconds since the Unix epoch (UTC).

    Returns:
        ISO-8601 formatted UTC string.
    """
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


# ── Timestamp parsing ─────────────────────────────────────────────────────────

def parse_utc(ts_string: str) -> datetime:
    """
    Parse a UTC ISO-8601 string back into a datetime object.

    Args:
        ts_string: A string produced by now_utc() or from_timestamp().

    Returns:
        timezone-aware datetime object in UTC.

    Raises:
        ValueError: If the string is not a valid ISO-8601 datetime.
    """
    dt = datetime.fromisoformat(ts_string)
    if dt.tzinfo is None:
        raise ValueError(
            f"Timestamp '{ts_string}' has no timezone. "
            "AISec requires explicit UTC timestamps."
        )
    return dt.astimezone(timezone.utc)


# ── Comparison ────────────────────────────────────────────────────────────────

def seconds_between(earlier: str, later: str) -> float:
    """
    Return the number of seconds between two UTC ISO-8601 timestamps.

    Args:
        earlier: The earlier timestamp string.
        later:   The later timestamp string.

    Returns:
        Elapsed seconds as a float. Negative if order is reversed.
    """
    t1 = parse_utc(earlier)
    t2 = parse_utc(later)
    return (t2 - t1).total_seconds()


def is_within_seconds(ts: str, seconds: float) -> bool:
    """
    Return True if the given timestamp is within `seconds` of now.

    Used by the heartbeat monitor to check if AISec is still alive.

    Args:
        ts:      A UTC ISO-8601 timestamp string.
        seconds: The maximum allowed age in seconds.

    Returns:
        True if the timestamp is recent enough, False otherwise.
    """
    elapsed = seconds_between(ts, now_utc())
    return 0.0 <= elapsed <= seconds