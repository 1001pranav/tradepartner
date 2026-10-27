"""Synthetic market data helpers shared by tests (no real data)."""

from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from tradepartner.calendar import TradingCalendar

IST = ZoneInfo("Asia/Kolkata")


def daily_rows(
    start: date,
    end: date,
    cal: TradingCalendar,
    base: float = 100.0,
    step: float = 0.5,
    volume: int = 100_000,
) -> list[dict[str, Any]]:
    rows = []
    price = base
    for d in cal.trading_days(start, end):
        rows.append(
            {
                "date": datetime(d.year, d.month, d.day, tzinfo=IST),
                "open": round(price, 2),
                "high": round(price + 1, 2),
                "low": round(price - 1, 2),
                "close": round(price + step, 2),
                "volume": volume,
            }
        )
        price += step
    return rows


class FakeKite:
    """Mock Kite client: serves synthetic daily candles and an instrument master."""

    def __init__(self, cal: TradingCalendar, fail_tokens: set[int] | None = None) -> None:
        self.cal = cal
        self.calls: list[tuple[int, date, date, str]] = []
        self.fail_tokens = fail_tokens or set()
        self.instrument_rows: list[dict[str, Any]] = []

    def historical_data(
        self, instrument_token: int, from_date: date, to_date: date, interval: str
    ) -> list[dict[str, Any]]:
        self.calls.append((instrument_token, from_date, to_date, interval))
        if instrument_token in self.fail_tokens:
            raise RuntimeError("boom")
        return daily_rows(from_date, to_date, self.cal, base=100 + instrument_token % 7)

    def instruments(self, exchange: str | None = None) -> list[dict[str, Any]]:
        return self.instrument_rows


def make_cal() -> TradingCalendar:
    return TradingCalendar(holidays=frozenset({date(2026, 1, 26)}))


def dt(y: int, m: int, d: int, h: int = 0, mi: int = 0) -> datetime:
    return datetime(y, m, d, h, mi, tzinfo=IST)


__all__ = ["FakeKite", "daily_rows", "dt", "make_cal", "timedelta"]
