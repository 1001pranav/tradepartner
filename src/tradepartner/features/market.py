"""Market-regime, relative and calendar features (spec 10.4)."""

from calendar import monthrange
from collections.abc import Mapping
from datetime import date

import numpy as np
import pandas as pd

MARKET_COLUMNS = ["nifty_dist_sma200", "vix_level", "vix_chg_5d", "breadth_50"]
RELATIVE_COLUMNS = ["rel_ret_5", "rel_ret_20", "rel_ret_60", "sector_rel_ret_20", "sector_mom_rank"]
CALENDAR_COLUMNS = ["dow", "days_to_month_end", "days_to_results"]


def market_features(
    index: pd.DataFrame | None, vix: pd.DataFrame | None, dist_sma50: pd.DataFrame
) -> pd.DataFrame:
    """One row per date. ``dist_sma50`` is dates x symbols (for breadth, % above the 50-day SMA)."""
    out = pd.DataFrame(index=dist_sma50.index, columns=MARKET_COLUMNS, dtype="float64")
    if index is not None and len(index):
        c = pd.Series(index["close"].to_numpy(), index=index["date"])
        out["nifty_dist_sma200"] = (c / c.rolling(200).mean() - 1).reindex(out.index)
    if vix is not None and len(vix):
        vc = pd.Series(vix["close"].to_numpy(), index=vix["date"])
        out["vix_level"] = vc.reindex(out.index)
        out["vix_chg_5d"] = (vc / vc.shift(5) - 1).reindex(out.index)
    valid = dist_sma50.notna()
    out["breadth_50"] = ((dist_sma50 > 0) & valid).sum(axis=1) / valid.sum(axis=1).replace(
        0, np.nan
    )
    return out


def relative_features(
    rets: Mapping[int, pd.DataFrame],
    index: pd.DataFrame | None,
    sectors: Mapping[str, str],
) -> dict[str, pd.DataFrame]:
    """Return windows -> dates x symbols frames, to features relative to the index and sector."""
    out: dict[str, pd.DataFrame] = {}
    idx_close = (
        pd.Series(index["close"].to_numpy(), index=index["date"])
        if index is not None and len(index)
        else None
    )
    for n in (5, 20, 60):
        r = rets[n]
        if idx_close is not None:
            idx_r = idx_close.pct_change(n).reindex(r.index)
            out[f"rel_ret_{n}"] = r.sub(idx_r, axis=0)
        else:
            out[f"rel_ret_{n}"] = r * np.nan
    r20 = rets[20]
    sector_of = pd.Series({s: sectors.get(s, f"UNMAPPED:{s}") for s in r20.columns})
    sector_mean = r20.T.groupby(sector_of).transform("mean").T
    out["sector_rel_ret_20"] = r20 - sector_mean
    out["sector_mom_rank"] = r20.T.groupby(sector_of).rank(pct=True).T
    return out


def calendar_features(
    dates: pd.Index, results: Mapping[str, list[date]] | None, symbols: list[str]
) -> dict[str, pd.DataFrame]:
    dow = pd.DataFrame(
        {s: [d.weekday() for d in dates] for s in symbols}, index=dates, dtype="float64"
    )
    dme = pd.DataFrame(
        {s: [monthrange(d.year, d.month)[1] - d.day for d in dates] for s in symbols},
        index=dates, dtype="float64",
    )  # fmt: skip
    nxt = pd.DataFrame(np.nan, index=dates, columns=symbols)
    for s in symbols:
        known = sorted(results.get(s, [])) if results else []
        if known:
            vals = []
            for d in dates:
                nxt_d = next((r for r in known if r > d), None)
                vals.append((nxt_d - d).days if nxt_d else np.nan)
            nxt[s] = vals
    return {"dow": dow, "days_to_month_end": dme, "days_to_results": nxt}
