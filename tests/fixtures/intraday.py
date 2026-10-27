"""Synthetic 5-minute bars (no real data): 75 bars per session, some days with ORB breakouts."""

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd

from tests.fixtures.synth import bdays
from tradepartner.calendar import IST

BARS_PER_DAY = 75  # 09:15-15:30 in 5-minute steps


def session_bar_ends(day: date) -> list[datetime]:
    start = datetime(day.year, day.month, day.day, 9, 15, tzinfo=IST)
    return [(start + timedelta(minutes=5 * (i + 1))).astimezone(UTC) for i in range(BARS_PER_DAY)]


def synth_bars(
    symbol_seed: int, n_days: int = 90, start: date = date(2025, 1, 1), momentum: float = 0.3
) -> pd.DataFrame:
    rng = np.random.default_rng(symbol_seed)
    rows = []
    price = 50_000  # paise
    for day in bdays(start, n_days):
        trend = rng.normal(0, 0.0004)
        prev = price
        for i, end in enumerate(session_bar_ends(day)):
            ret = trend + rng.normal(0, 0.0012)
            if i > 0:
                ret += momentum * (prev - price) / max(price, 1) * 0 + momentum * trend
            o = price
            c = max(1000, int(o * (1 + ret)))
            hi, lo = max(o, c) + int(o * 0.0004), min(o, c) - int(o * 0.0004)
            vol = int(
                rng.integers(8_000, 14_000) * (2.5 if (i in (3, 4, 5) and trend > 0.0005) else 1.0)
            )
            rows.append(
                {
                    "ts": pd.Timestamp(end),
                    "date": day,
                    "open": o,
                    "high": hi,
                    "low": lo,
                    "close": c,
                    "volume": vol,
                }
            )
            price = c
    return pd.DataFrame(rows)


def breakout_day(symbol_price: int = 100_000, day: date = date(2025, 6, 2)) -> pd.DataFrame:
    """A hand-built session: flat opening range, then a high-volume breakout above it at bar 8."""
    rows = []
    price = symbol_price
    for i, end in enumerate(session_bar_ends(day)[:12]):
        if i < 3:
            o, c, hi, lo, vol = (
                price,
                price + (10 if i % 2 else -10),
                price + 60,
                price - 60,
                10_000,
            )
        elif i < 7:
            o, c, hi, lo, vol = price, price + 5, price + 30, price - 30, 10_000
        elif i == 7:
            o, c, hi, lo, vol = price, price + 300, price + 320, price - 5, 40_000
        else:
            o, c, hi, lo, vol = price, price + 20, price + 40, price - 10, 12_000
        rows.append(
            {
                "ts": pd.Timestamp(end),
                "date": day,
                "open": o,
                "high": hi,
                "low": lo,
                "close": c,
                "volume": vol,
            }
        )
        price = c
    return pd.DataFrame(rows)
