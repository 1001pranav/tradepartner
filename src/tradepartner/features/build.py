"""Point-in-time feature matrix (spec 5: features/build.py).

Row ``(date, symbol)`` holds information available at that date's close. Adding a feature creates
a new ``FEATURE_SET_VERSION``; an old version is never mutated (spec 10.4).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from tradepartner.backtest.view import MarketView
from tradepartner.features import market
from tradepartner.features.news import ALL_COLUMNS as NEWS_ALL
from tradepartner.features.price import PRICE_COLUMNS, RETURN_WINDOWS, price_features

FEATURE_SET_VERSION = (
    "fs2"  # fs2 drops dow/days_to_month_end: calendar effects let a model "time" instead of pick
)
MODEL_CALENDAR_COLUMNS = ["days_to_results"]
RANK_COLUMNS = ["turnover_rank"]
NEWS_FEATURE_COLUMNS = list(NEWS_ALL)
PRICE_ONLY_COLUMNS = (
    [c for c in PRICE_COLUMNS if c != "traded_value_20"]
    + RANK_COLUMNS
    + market.MARKET_COLUMNS
    + market.RELATIVE_COLUMNS
    + MODEL_CALENDAR_COLUMNS
)
ALL_FEATURE_COLUMNS = PRICE_ONLY_COLUMNS + NEWS_FEATURE_COLUMNS


@dataclass(frozen=True)
class MatrixSpec:
    version: str = FEATURE_SET_VERSION
    with_news: bool = False


def build_matrix(
    view: MarketView,
    symbols: list[str] | None = None,
    vix: pd.DataFrame | None = None,
    results: Mapping[str, list[date]] | None = None,
    news_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Features for every (date, symbol). ``news_panel`` is indexed by (date, symbol) with
    ``features.news.ALL_COLUMNS``; when absent those columns are NaN (price-only model)."""
    names = symbols or view.symbols
    per_symbol: dict[str, pd.DataFrame] = {}
    for s in names:
        bars = view.history(s, view.trading_days()[-1])
        if len(bars):
            per_symbol[s] = price_features(bars)
    if not per_symbol:
        return pd.DataFrame(columns=ALL_FEATURE_COLUMNS)
    syms = sorted(per_symbol)
    dates = pd.Index(sorted(set().union(*[f.index for f in per_symbol.values()])))

    def panel(col: str) -> pd.DataFrame:
        return pd.DataFrame({s: per_symbol[s][col].reindex(dates) for s in syms})

    rets = {n: panel(f"ret_{n}") for n in RETURN_WINDOWS}
    mk = market.market_features(view.index, vix, panel("dist_sma50"))
    rel = market.relative_features(rets, view.index, view.sectors)
    cal = market.calendar_features(dates, results, syms)
    turnover_rank = panel("traded_value_20").rank(axis=1, pct=True)

    cols: dict[str, pd.DataFrame] = {c: panel(c) for c in PRICE_COLUMNS if c != "traded_value_20"}
    cols["turnover_rank"] = turnover_rank
    for c in market.MARKET_COLUMNS:
        cols[c] = pd.DataFrame({s: mk[c] for s in syms})
    cols.update(rel)
    cols.update(cal)
    long = pd.concat({c: f.stack(future_stack=True) for c, f in cols.items()}, axis=1)
    long.index.names = ["date", "symbol"]
    long = long.reindex(columns=PRICE_ONLY_COLUMNS)
    if news_panel is not None:
        long = long.join(news_panel.reindex(columns=NEWS_FEATURE_COLUMNS), how="left")
    for c in NEWS_FEATURE_COLUMNS:
        if c not in long.columns:
            long[c] = np.nan
    return long.sort_index()


def save_matrix(matrix: pd.DataFrame, directory: Path, version: str = FEATURE_SET_VERSION) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"features_{version}.parquet"
    matrix.reset_index().to_parquet(path, index=False)
    return path
