"""Data validation checks and report (spec 5: data/quality.py)."""

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from tradepartner.calendar import TradingCalendar
from tradepartner.data.candles import CandleStore, real_gaps
from tradepartner.data.corporate_actions import Action

JUMP_THRESHOLD = 0.20  # |daily return| above this must be explained by a corporate action


@dataclass(frozen=True)
class Issue:
    symbol: str
    kind: str
    detail: str
    severity: str = "warn"  # warn | error


def check_symbol(
    symbol: str,
    df: pd.DataFrame,
    calendar: TradingCalendar,
    actions: list[Action] | None = None,
) -> list[Issue]:
    issues: list[Issue] = []
    if df.empty:
        return [Issue(symbol, "empty", "no candles", "error")]
    if df["ts"].duplicated().any():
        issues.append(Issue(symbol, "duplicate_ts", f"{df['ts'].duplicated().sum()} rows", "error"))
    if not df["ts"].is_monotonic_increasing:
        issues.append(Issue(symbol, "not_sorted", "timestamps not increasing", "error"))
    px = df[["open", "high", "low", "close"]]
    bad_px = (px <= 0).any(axis=1).to_numpy()
    if bad_px.any():
        first_good = int(np.argmax(~bad_px)) if (~bad_px).any() else len(bad_px)
        # zero-price bars before the first real bar are pre-listing junk (the loader drops them);
        # a zero price in the middle of real data is a genuine error
        leading_only = bool(bad_px[first_good:].sum() == 0)
        issues.append(
            Issue(
                symbol,
                "non_positive_price",
                f"{int(bad_px.sum())} bars with price <= 0"
                + (" before the first real bar (ignored)" if leading_only else ""),
                "warn" if leading_only else "error",
            )
        )
    bad = (df["high"] < df[["open", "close", "low"]].max(axis=1)) | (
        df["low"] > df[["open", "close", "high"]].min(axis=1)
    )
    if bad.any():
        issues.append(Issue(symbol, "ohlc_inconsistent", f"{int(bad.sum())} bars", "error"))
    if (df["volume"] < 0).any():
        issues.append(Issue(symbol, "negative_volume", "volume < 0", "error"))
    zero_vol = int((df["volume"] == 0).sum())
    if zero_vol:
        issues.append(Issue(symbol, "zero_volume", f"{zero_vol} bars"))
    gaps = real_gaps(df, calendar)  # a stock has no history before it listed
    if gaps:
        issues.append(Issue(symbol, "gaps", f"{len(gaps)} missing days, first {gaps[0]}"))
    issues.extend(_unexplained_jumps(symbol, df, actions or []))
    return issues


def find_jumps(df: pd.DataFrame, actions: list[Action]) -> list[tuple[date, float]]:
    """(day, return) for every daily move above the threshold with no corporate action within
    three days."""
    ret = df["close"].astype("float64").pct_change()
    out: list[tuple[date, float]] = []
    for i in ret.index[ret.abs() > JUMP_THRESHOLD]:
        day: date = df.loc[i, "date"]
        if not any(abs((a.ex_date - day).days) <= 3 for a in actions):
            out.append((day, float(ret[i])))
    return out


def _unexplained_jumps(symbol: str, df: pd.DataFrame, actions: list[Action]) -> list[Issue]:
    return [
        Issue(symbol, "unexplained_jump", f"{r:+.1%} on {day}; missing split/bonus?")
        for day, r in find_jumps(df, actions)
    ]


def report(
    store: CandleStore,
    symbols: list[str],
    calendar: TradingCalendar,
    actions_by_symbol: dict[str, list[Action]] | None = None,
    interval: str = "day",
) -> list[Issue]:
    issues: list[Issue] = []
    for sym in symbols:
        issues.extend(
            check_symbol(
                sym, store.read(interval, sym), calendar, (actions_by_symbol or {}).get(sym)
            )
        )
    return issues


def stale_symbols(store: CandleStore, symbols: list[str], expected_last: date) -> list[str]:
    """Symbols whose latest bar is older than ``expected_last`` (e.g. not refreshed today)."""
    return [
        s for s in symbols if (last := store.last_date("day", s)) is None or last < expected_last
    ]
