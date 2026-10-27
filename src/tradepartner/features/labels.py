"""Targets (spec 10.5). Labels start strictly AFTER ``as_of``: entry is the next open (10.3.2)."""

from datetime import date

import pandas as pd

from tradepartner.costs import zerodha
from tradepartner.settings import CostsConfig

SWING_HORIZON = 5
POSITIONAL_HORIZON = 60
EXTRA_MARGIN = 0.005  # swing target must beat the index by round-trip cost + 0.5%


def round_trip_cost_fraction(
    costs: CostsConfig,
    notional_inr: int = 100_000,
    product: str = "CNC",
    slippage_bps: float | None = None,
) -> float:
    """Round-trip charges + 2 x slippage as a fraction of notional (from config)."""
    px = 100_000  # Rs 1,000: the fraction barely depends on the price level
    qty = max(1, notional_inr * 100 // px)
    day = date(2026, 1, 1)
    fills = [
        zerodha.FillIn(1, "b", "X", "BUY", product, qty, px, day),
        zerodha.FillIn(2, "s", "X", "SELL", product, qty, px, day),
    ]
    charges = zerodha.total_paise(zerodha.compute(fills, costs))
    bps = costs.slippage.default_bps if slippage_bps is None else slippage_bps
    return charges / (qty * px) + 2 * bps / 10_000


def forward_returns(open_: pd.Series, close: pd.Series, horizon: int) -> pd.Series:
    """Return from the NEXT open (t+1) to the close ``horizon`` days after t (t+horizon)."""
    return close.shift(-horizon) / open_.shift(-1) - 1


def swing_labels(
    stock: dict[str, pd.DataFrame], index: pd.DataFrame, horizon: int, threshold: float
) -> pd.DataFrame:
    """Columns ``fwd_ret``, ``excess``, ``y`` (1 if excess > threshold) per (date, symbol).

    The last ``horizon`` rows of each symbol have NaN labels (the future is unknown).
    """
    idx = index.set_index("date")
    idx_fwd = forward_returns(idx["open"], idx["close"], horizon)
    frames = []
    for sym, df in stock.items():
        d = df.set_index("date")
        fwd = forward_returns(d["open"], d["close"], horizon)
        excess = fwd - idx_fwd.reindex(d.index)
        y = (excess > threshold).astype("float64").where(excess.notna())
        frames.append(pd.DataFrame({"fwd_ret": fwd, "excess": excess, "y": y}).assign(symbol=sym))
    out = pd.concat(frames).reset_index().set_index(["date", "symbol"]).sort_index()
    return out


def positional_labels(
    stock: dict[str, pd.DataFrame], index: pd.DataFrame, horizon: int
) -> pd.DataFrame:
    """Cross-sectional rank (0..1) of the 60-day forward excess return, per date."""
    base = swing_labels(stock, index, horizon, 0.0)
    ranked = base.groupby(level="date")["excess"].rank(pct=True)
    base["y"] = ranked.where(base["excess"].notna())
    return base
