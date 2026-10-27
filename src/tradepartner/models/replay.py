"""Replay: hide the future, predict it, then reveal what happened (spec 10.6, in miniature).

Train ONLY on data whose labels were fully known before ``as_of``, predict the picks for the
close of ``as_of``, then reveal how those picks actually did over the next ``days`` trading days.
Nothing here trades or registers a model; it is a transparent look at how the model behaves.
"""

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from tradepartner.backtest.view import MarketView
from tradepartner.features import build, labels
from tradepartner.models import train
from tradepartner.models.registry import ModelError
from tradepartner.settings import Settings


@dataclass(frozen=True)
class Pick:
    symbol: str
    score: float
    actual_return: float | None  # next open -> close after ``days``; None if not yet known
    vs_index: float | None


@dataclass(frozen=True)
class ReplayResult:
    as_of: date
    days: int
    train_rows: int
    train_last_date: date
    picks: list[Pick]
    index_return: float | None
    universe_mean: float | None  # average of every stock, the "do nothing clever" yardstick

    @property
    def hit_rate(self) -> float | None:
        known = [p for p in self.picks if p.vs_index is not None]
        return (
            sum(p.vs_index > 0 for p in known if p.vs_index is not None) / len(known)
            if known
            else None
        )


def _window_return(df: pd.DataFrame, as_of: date, days: int) -> float | None:
    pos = df.index[df["date"] == as_of]
    if len(pos) == 0:
        return None
    i = int(pos[0])
    if i + days >= len(df):
        return None  # that part of the future has not happened yet in our data
    return float(df["close"].iloc[i + days] / df["open"].iloc[i + 1] - 1)


def replay(
    view: MarketView,
    settings: Settings,
    as_of: date,
    days: int,
    *,
    horizon: int = 5,
    top_k: int = 5,
    seed: int = 42,
) -> ReplayResult:
    if view.index is None:
        raise ModelError("the Nifty 50 index series is required")
    all_days = view.trading_days()
    if as_of not in all_days:
        raise ModelError(f"{as_of} is not a trading day in the stored data")
    matrix = build.build_matrix(view)
    stock = {s: view.history(s, all_days[-1]) for s in view.symbols}
    cost = labels.round_trip_cost_fraction(settings.costs)
    lab = labels.swing_labels(stock, view.index, horizon, cost + labels.EXTRA_MARGIN)

    pos = {d: i for i, d in enumerate(all_days)}
    # labels of a training row at day d end at d+horizon: they must be known by as_of
    cutoff = pos[as_of] - horizon - 1
    cols = build.PRICE_ONLY_COLUMNS
    data = matrix[cols].join(lab[["y"]], how="inner")
    data = data[data["y"].notna()]
    dpos = np.array([pos[d] for d in data.index.get_level_values("date")])
    train_rows = data.loc[dpos <= cutoff]
    if len(train_rows) < 500:
        raise ModelError("not enough history before that date to train (need at least ~500 rows)")
    params = {
        **train.base_params("binary", seed),
        "num_leaves": 15,
        "learning_rate": 0.05,
        "min_child_samples": 50,
        "bagging_freq": 1,
        "feature_fraction": 0.8,
    }
    booster = train.fit(train_rows[cols], train_rows["y"].to_numpy(), params, 150)

    today = matrix.xs(as_of, level="date")[cols]
    members = set(view.symbols_as_of(as_of))
    has_data = today.notna().to_numpy().any(axis=1)
    today = today.loc[today.index.isin(members) & has_data]
    scores = pd.Series(booster.predict(today), index=today.index).sort_values(ascending=False)

    idx_ret = _window_return(view.index, as_of, days)
    picks = []
    for sym, sc in scores.head(top_k).items():
        r = _window_return(stock[str(sym)], as_of, days)
        picks.append(
            Pick(str(sym), float(sc), r, None if r is None or idx_ret is None else r - idx_ret)
        )
    everyone = [
        r for s in scores.index if (r := _window_return(stock[str(s)], as_of, days)) is not None
    ]
    return ReplayResult(
        as_of,
        days,
        len(train_rows),
        all_days[cutoff],
        picks,
        idx_ret,
        float(np.mean(everyone)) if everyone else None,
    )
