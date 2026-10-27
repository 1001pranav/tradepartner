"""5-minute features and labels for intraday_v1 (spec 10.4/10.5, M9).

Every value at bar ``t`` uses only bars ending at or before ``t`` (causal). The opening range is
the first three bars (09:15-09:30) and is NaN until it is complete.
"""

import numpy as np
import pandas as pd

from tradepartner.calendar import IST

OR_BARS = 3
INTRADAY_COLUMNS = [
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",
    "vwap_dist",
    "or_high_dist",
    "or_low_dist",
    "or_breakout",
    "vol_ratio",
    "atr_pct",
    "minutes_since_open",
    "day_ret",
    "range_pos",
]


def intraday_features(
    bars: pd.DataFrame, prev_close: dict[object, int] | None = None
) -> pd.DataFrame:
    """``bars``: ts (UTC end), date, open, high, low, close, volume (paise ints), sorted by ts."""
    df = bars.sort_values("ts").reset_index(drop=True)
    c, h, low, v = (df[k].astype("float64") for k in ("close", "high", "low", "volume"))
    day = df["date"]
    f = pd.DataFrame(index=df["ts"])
    for n in (1, 3, 6, 12):
        # within-day only: a return never spans the overnight gap
        f[f"ret_{n}"] = (c / c.groupby(day).shift(n) - 1).to_numpy()
    typical = (h + low + c) / 3
    pv = (typical * v).groupby(day).cumsum()
    cum_v = v.groupby(day).cumsum().replace(0, np.nan)
    vwap = pv / cum_v
    f["vwap_dist"] = (c / vwap - 1).to_numpy()
    rank = df.groupby("date").cumcount()
    or_high = h.where(rank < OR_BARS).groupby(day).transform("max")
    or_low = low.where(rank < OR_BARS).groupby(day).transform("min")
    done = rank >= OR_BARS - 1  # the range is final once the third bar has closed
    f["or_high_dist"] = (c / or_high - 1).where(done).to_numpy()
    f["or_low_dist"] = (c / or_low - 1).where(done).to_numpy()
    f["or_breakout"] = ((c > or_high) & (rank >= OR_BARS)).astype("float64").to_numpy()
    f["vol_ratio"] = (v / v.rolling(20, min_periods=5).mean().shift(1)).to_numpy()
    prev = c.shift(1)
    tr = pd.concat([h - low, (h - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    f["atr_pct"] = (tr.rolling(12, min_periods=6).mean() / c).to_numpy()
    local = df["ts"].dt.tz_convert(IST)
    f["minutes_since_open"] = ((local.dt.hour * 60 + local.dt.minute) - (9 * 60 + 15)).to_numpy(
        dtype="float64"
    )
    day_open = df["open"].astype("float64").groupby(day).transform("first")
    f["day_ret"] = (c / day_open - 1).to_numpy()
    day_hi, day_lo = h.groupby(day).cummax(), low.groupby(day).cummin()
    f["range_pos"] = ((c - day_lo) / (day_hi - day_lo).replace(0, np.nan)).fillna(0.5).to_numpy()
    return f[INTRADAY_COLUMNS]


def intraday_labels(bars: pd.DataFrame, horizon_bars: int, cost_fraction: float) -> pd.DataFrame:
    """``fwd_ret`` = close of bar t+h over the OPEN of bar t+1 (never the signal bar), same day.
    ``y`` = 1 if it beats costs. The last ``horizon_bars`` bars of each day have NaN labels.
    """
    df = bars.sort_values("ts").reset_index(drop=True)
    day = df["date"]
    entry = df["open"].astype("float64").groupby(day).shift(-1)
    exit_ = df["close"].astype("float64").groupby(day).shift(-horizon_bars)
    fwd = exit_ / entry - 1
    y = (fwd > cost_fraction).astype("float64").where(fwd.notna())
    return pd.DataFrame(
        {"fwd_ret": fwd.to_numpy(), "excess": fwd.to_numpy(), "y": y.to_numpy()}, index=df["ts"]
    )
