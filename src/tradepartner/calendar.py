"""NSE trading calendar and IST helpers (spec 5, 12)."""

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from tradepartner.paths import config_dir

IST = ZoneInfo("Asia/Kolkata")
SESSION_OPEN = time(9, 15)
SESSION_CLOSE = time(15, 30)


def to_ist(ts: datetime) -> datetime:
    """Display helper: storage is UTC, display is IST."""
    if ts.tzinfo is None:
        raise ValueError("naive datetime; storage timestamps must be timezone-aware")
    return ts.astimezone(IST)


def utc_now() -> datetime:
    return datetime.now(UTC)


def session_close_utc(day: date) -> datetime:
    """When a daily bar's information becomes available (15:30 IST), in UTC."""
    return datetime.combine(day, SESSION_CLOSE, tzinfo=IST).astimezone(UTC)


def session_open_utc(day: date) -> datetime:
    return datetime.combine(day, SESSION_OPEN, tzinfo=IST).astimezone(UTC)


@dataclass(frozen=True)
class TradingCalendar:
    holidays: frozenset[date] = frozenset()
    extra_trading_days: frozenset[date] = frozenset()

    @classmethod
    def from_config(cls, path: Path | None = None) -> "TradingCalendar":
        p = path or config_dir() / "holidays.yaml"
        data = yaml.safe_load(p.read_text()) if p.exists() else {}
        data = data or {}
        return cls(
            frozenset(_as_date(d) for d in data.get("holidays", [])),
            frozenset(_as_date(d) for d in data.get("extra_trading_days", [])),
        )

    def is_trading_day(self, day: date) -> bool:
        if day in self.extra_trading_days:
            return True
        return day.weekday() < 5 and day not in self.holidays

    def next_trading_day(self, day: date) -> date:
        d = day + timedelta(days=1)
        while not self.is_trading_day(d):
            d += timedelta(days=1)
        return d

    def prev_trading_day(self, day: date) -> date:
        d = day - timedelta(days=1)
        while not self.is_trading_day(d):
            d -= timedelta(days=1)
        return d

    def trading_days(self, start: date, end: date) -> list[date]:
        """Inclusive range."""
        out: list[date] = []
        d = start
        while d <= end:
            if self.is_trading_day(d):
                out.append(d)
            d += timedelta(days=1)
        return out

    def add_trading_days(self, day: date, n: int) -> date:
        d = day
        step = 1 if n >= 0 else -1
        for _ in range(abs(n)):
            d = self.next_trading_day(d) if step > 0 else self.prev_trading_day(d)
        return d


def _as_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def indian_fy(day: date) -> str:
    """Financial year label, e.g. 2026-04-01 -> '2026-27'."""
    start = day.year if day.month >= 4 else day.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def fy_bounds(fy: str) -> tuple[date, date]:
    start = int(fy[:4])
    return date(start, 4, 1), date(start + 1, 3, 31)


def add_months(start: date, months: int) -> date:
    """``start`` plus calendar months, clamped to the target month's last day."""
    y, m = divmod(start.month - 1 + months, 12)
    year, month = start.year + y, m + 1
    last = (date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)).day
    return date(year, month, min(start.day, last))


def months_between_exceeds(start: date, end: date, months: int) -> bool:
    """True when ``end`` is strictly after ``start`` + ``months`` calendar months."""
    return end > add_months(start, months)


def fmt_ist(ts: datetime) -> str:
    return to_ist(ts).strftime("%Y-%m-%d %H:%M IST")
