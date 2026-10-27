"""Simple rule baselines (spec 10.5). A model that cannot beat these after costs is not deployed.

* Swing: buy-and-hold Nifty 50 (see ``benchmarks``) and a 20-day momentum top-N rule.
* Positional: equal-weight universe (see ``benchmarks``) and a 12-1 month momentum top-N rule.
Both rules run through the same risk engine as everything else.
"""

from datetime import date

import numpy as np
import pandas as pd

from tradepartner.backtest.view import MarketView
from tradepartner.strategies.base import Candidate, round_tick

ATR_N = 14


def atr(h: pd.DataFrame, n: int = ATR_N) -> float:
    if len(h) < n + 1:
        return float("nan")
    prev = h["close"].shift(1)
    tr = pd.concat(
        [h["high"] - h["low"], (h["high"] - prev).abs(), (h["low"] - prev).abs()], axis=1
    ).max(axis=1)
    return float(tr.iloc[-n:].mean())


def make_candidate(
    symbol: str, close: float, stop_distance: float, reward_risk: float, horizon: int,
    score: float, reason: str, entry_buffer: float = 0.005, max_stop_pct: float = 0.07,
) -> Candidate | None:  # fmt: skip
    if not np.isfinite(stop_distance) or stop_distance <= 0:
        return None
    entry = round_tick(close * 100 * (1 + entry_buffer))
    dist = min(stop_distance * 100, entry * max_stop_pct)
    stop = round_tick(entry - dist)
    target = round_tick(entry + reward_risk * (entry - stop))
    if stop <= 0 or stop >= entry:
        return None
    return Candidate(symbol, entry, stop, target, horizon, score, reason)


class Momentum20TopN:
    """Swing baseline: the N strongest 20-day performers above their 50-day average."""

    name = "baseline_momentum20"

    def __init__(self, n: int = 5, horizon_days: int = 5, reward_risk: float = 2.0) -> None:
        self.n, self.horizon, self.rr = n, horizon_days, reward_risk

    def candidates(self, as_of: date, view: MarketView) -> list[Candidate]:
        scored: list[tuple[float, str, float, float]] = []
        for sym in view.symbols_as_of(as_of):
            h = view.history(sym, as_of, 60)
            if len(h) < 51 or h["date"].iloc[-1] != as_of:
                continue
            close = float(h["close"].iloc[-1])
            ret20 = close / float(h["close"].iloc[-21]) - 1
            if ret20 <= 0 or close <= float(h["close"].iloc[-50:].mean()):
                continue
            scored.append((ret20, sym, close, atr(h)))
        scored.sort(key=lambda t: (-t[0], t[1]))
        out = []
        for ret20, sym, close, a in scored[: self.n]:
            c = make_candidate(sym, close, 2 * a, self.rr, self.horizon, ret20, f"20d {ret20:+.1%}")
            if c:
                out.append(c)
        return out


class Momentum12m1TopN:
    """Positional baseline: monthly, the top N by 12-1 month momentum (skip the last month)."""

    name = "baseline_momentum12_1"

    def __init__(self, n: int = 10, horizon_days: int = 60, reward_risk: float = 2.0) -> None:
        self.n, self.horizon, self.rr = n, horizon_days, reward_risk

    @staticmethod
    def _is_first_trading_day_of_month(as_of: date, view: MarketView) -> bool:
        h = view.index_history(as_of)
        days = list(h["date"]) if len(h) else view.trading_days()
        i = days.index(as_of) if as_of in days else -1
        return i > 0 and days[i - 1].month != as_of.month

    def candidates(self, as_of: date, view: MarketView) -> list[Candidate]:
        if not self._is_first_trading_day_of_month(as_of, view):
            return []
        scored: list[tuple[float, str, float]] = []
        for sym in view.symbols_as_of(as_of):
            h = view.history(sym, as_of, 260)
            if len(h) < 253 or h["date"].iloc[-1] != as_of:
                continue
            mom = float(h["close"].iloc[-22]) / float(h["close"].iloc[-253]) - 1
            if mom > 0:
                scored.append((mom, sym, float(h["close"].iloc[-1])))
        scored.sort(key=lambda t: (-t[0], t[1]))
        out = []
        for mom, sym, close in scored[: self.n]:
            c = make_candidate(sym, close, close * 0.07, self.rr, self.horizon, mom,
                                f"12-1m {mom:+.1%}")  # fmt: skip
            if c:
                out.append(c)
        return out
