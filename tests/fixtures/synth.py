"""Deterministic synthetic markets (no real data). Same seed -> identical bars."""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from tradepartner.backtest.view import MarketView
from tradepartner.settings import Settings, load_settings


def bdays(start: date, n: int) -> list[date]:
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def bars_from_closes(days: list[date], closes: list[float], vol: float = 3_000_000) -> pd.DataFrame:
    rows = []
    prev = closes[0]
    for d, c in zip(days, closes, strict=True):
        o = prev
        rows.append({"date": d, "open": o, "high": max(o, c) * 1.005, "low": min(o, c) * 0.995,
                     "close": c, "volume": vol})  # fmt: skip
        prev = c
    return pd.DataFrame(rows)


def synth_view(
    n_symbols: int = 12,
    n_days: int = 700,
    seed: int = 7,
    start: date = date(2023, 1, 2),
    momentum: float = 0.0,
) -> MarketView:
    rng = np.random.default_rng(seed)
    days = bdays(start, n_days)
    bars: dict[str, pd.DataFrame] = {}
    paths = []
    for i in range(n_symbols):
        drift = rng.normal(0.0004, 0.0003)
        vol = rng.uniform(0.01, 0.02)
        rets = rng.normal(drift, vol, n_days)
        if momentum:
            for t in range(1, n_days):
                rets[t] += momentum * rets[t - 1]  # makes past returns predictive
        closes = 500 * np.exp(np.cumsum(rets))
        paths.append(closes / closes[0])
        bars[f"SYM{i:02d}"] = bars_from_closes(days, [round(float(c), 2) for c in closes])
    idx = np.mean(paths, axis=0) * 10_000
    index = bars_from_closes(days, [round(float(c), 2) for c in idx])
    return MarketView(bars, index, {f"SYM{i:02d}": f"SECTOR{i % 6}" for i in range(n_symbols)})


def settings_with_capital(capital_inr: int = 1_000_000, **risk_updates: object) -> Settings:
    s = load_settings()
    risk = s.risk.model_copy(update={"capital_base_inr": capital_inr, **risk_updates})
    return s.model_copy(update={"risk": risk})
