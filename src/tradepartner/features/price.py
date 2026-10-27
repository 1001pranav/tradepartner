"""Price/volume features (spec 10.4). Every value at row ``t`` uses only bars dated <= t, so a row
is available from the NEXT session's open (spec 10.3.1)."""

import numpy as np
import pandas as pd

RETURN_WINDOWS = (1, 5, 10, 20, 60)
PRICE_COLUMNS = (
    [f"ret_{n}" for n in RETURN_WINDOWS]
    + ["mom_12_1", "vol_20", "atr14_pct", "dist_sma20", "dist_sma50", "dist_sma200", "rsi14",
       "dist_52w_high", "dist_52w_low", "volume_z20", "gap_pct", "close_loc", "traded_value_20"]
)  # fmt: skip


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / down.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(down != 0, 100.0).where(up.notna())


def price_features(bars: pd.DataFrame) -> pd.DataFrame:
    """``bars``: adjusted daily frame with date, open, high, low, close, volume (sorted)."""
    df = bars.reset_index(drop=True)
    c, o, h, low, v = df["close"], df["open"], df["high"], df["low"], df["volume"]
    prev = c.shift(1)
    f = pd.DataFrame(index=df["date"])
    for n in RETURN_WINDOWS:
        f[f"ret_{n}"] = c.pct_change(n).to_numpy()
    f["mom_12_1"] = (c.shift(21) / c.shift(252) - 1).to_numpy()  # 12-1 month momentum
    f["vol_20"] = c.pct_change().rolling(20).std().to_numpy()
    tr = pd.concat([h - low, (h - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    f["atr14_pct"] = (tr.rolling(14).mean() / c).to_numpy()
    for n in (20, 50, 200):
        f[f"dist_sma{n}"] = (c / c.rolling(n).mean() - 1).to_numpy()
    f["rsi14"] = rsi(c).to_numpy()
    f["dist_52w_high"] = (c / c.rolling(252, min_periods=60).max() - 1).to_numpy()
    f["dist_52w_low"] = (c / c.rolling(252, min_periods=60).min() - 1).to_numpy()
    f["volume_z20"] = (
        (v - v.rolling(20).mean()) / v.rolling(20).std().replace(0, np.nan)
    ).to_numpy()
    f["gap_pct"] = (o / prev - 1).to_numpy()
    rng = (h - low).replace(0, np.nan)
    f["close_loc"] = ((c - low) / rng).fillna(0.5).to_numpy()
    f["traded_value_20"] = (c * v).rolling(20).mean().to_numpy()
    return f[PRICE_COLUMNS]
