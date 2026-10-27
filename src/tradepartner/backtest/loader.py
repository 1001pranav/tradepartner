"""Load stored candles into a ``MarketView`` (adjusted for splits/bonuses)."""

from datetime import date

import pandas as pd

from tradepartner.backtest.view import MarketView
from tradepartner.data import universe as uni
from tradepartner.data.candles import CandleStore
from tradepartner.data.corporate_actions import Action, adjust_candles
from tradepartner.data.kite_client import INDEX_TOKENS


def load_view(
    store: CandleStore,
    symbols: list[str],
    actions: dict[str, list[Action]] | None = None,
    sectors: dict[str, str] | None = None,
    members: object | None = None,
) -> MarketView:
    """Adjusted daily bars for ``symbols`` plus the Nifty 50 index (if stored)."""
    bars: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        raw = store.read("day", sym)
        # rows before a listing can carry zero prices (e.g. MAZDOCK): they are not tradeable data
        raw = raw[(raw[["open", "high", "low", "close"]] > 0).all(axis=1)]
        if raw.empty:
            continue
        bars[sym] = adjust_candles(raw, (actions or {}).get(sym, []))[
            ["date", "open", "high", "low", "close", "volume"]
        ]
    idx_raw = store.read("day", "NIFTY 50")
    index = (
        adjust_candles(idx_raw, [])[["date", "open", "high", "low", "close", "volume"]]
        if not idx_raw.empty
        else None
    )
    return MarketView(bars, index, sectors or uni.read_sectors(), members)  # type: ignore[arg-type]


def first_last(view: MarketView) -> tuple[date, date]:
    days = view.trading_days()
    return days[0], days[-1]


__all__ = ["INDEX_TOKENS", "first_last", "load_view"]
