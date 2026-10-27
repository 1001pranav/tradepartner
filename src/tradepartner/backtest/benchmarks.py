"""Benchmarks that are not run through the risk engine (spec 10.5): buy-and-hold index and an
equal-weight universe with monthly rebalancing."""

from datetime import date

import numpy as np
import pandas as pd

from tradepartner.backtest.view import MarketView


def buy_and_hold(view: MarketView, start: date, end: date, capital_paise: int) -> pd.Series:
    """Nifty 50 buy-and-hold equity (paise), starting at ``capital_paise`` on the first bar."""
    idx = view.index
    if idx is None or idx.empty:
        return pd.Series(dtype="int64")
    part = idx[(idx["date"] >= start) & (idx["date"] <= end)]
    if part.empty:
        return pd.Series(dtype="int64")
    close = part["close"].astype("float64")
    eq = capital_paise * close / close.iloc[0]
    return pd.Series(eq.round().astype("int64").to_numpy(), index=list(part["date"]))


def equal_weight_universe(
    view: MarketView, start: date, end: date, capital_paise: int, cost_bps: float = 10.0
) -> pd.Series:
    """Equal-weight over symbols with data, rebalanced on the first trading day of each month,
    paying ``cost_bps`` on the traded fraction. Membership is the (static) view universe."""
    days = [d for d in view.trading_days() if start <= d <= end]
    if not days:
        return pd.Series(dtype="int64")
    closes = view.close_matrix(days).to_numpy(dtype="float64")
    n_sym = closes.shape[1]
    rets = np.zeros_like(closes)
    rets[1:] = closes[1:] / closes[:-1] - 1
    rets = np.nan_to_num(rets, nan=0.0)
    equity = float(capital_paise)
    weights = np.zeros(n_sym)
    out: dict[date, int] = {}
    prev_month: int | None = None
    for i, d in enumerate(days):
        if prev_month is None or d.month != prev_month:
            live = ~np.isnan(closes[i])
            target = np.zeros(n_sym)
            if live.any():
                target[live] = 1.0 / live.sum()
            equity -= equity * float(np.abs(target - weights).sum()) * cost_bps / 10_000
            weights = target
        equity *= 1 + float((weights * rets[i]).sum())
        weights = weights * (1 + rets[i])
        total = float(weights.sum())
        weights = weights / total if total > 0 else weights
        out[d] = round(equity)
        prev_month = d.month
    return pd.Series(out, dtype="int64")
