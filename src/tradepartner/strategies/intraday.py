"""intraday_v1 (spec 16-M9): an opening-range / VWAP breakout setup, filtered by a model and by
news. LONG ONLY (intraday shorts are disabled), MIS product, no LLM in the hot path (the news
filter is precomputed). Entries only between 09:30 and 14:30 IST; the 15:10 square-off exits
whatever is still open."""

from dataclasses import replace
from datetime import datetime, time, timedelta
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from tradepartner.calendar import IST
from tradepartner.features.intraday import INTRADAY_COLUMNS, OR_BARS, intraday_features
from tradepartner.strategies.base import Candidate, round_tick

NAME = "intraday_v1"
WINDOW = (time(9, 30), time(14, 30))
MAX_BARS_LOOKBACK = 400


class IntradayScanner:
    def __init__(
        self,
        booster: lgb.Booster | None,
        columns: list[str] | None,
        params: dict[str, Any],
    ) -> None:
        self.booster = booster
        self.columns = columns or INTRADAY_COLUMNS
        self.top_k = int(params.get("top_k", 3))
        self.min_prob = float(params.get("min_prob", 0.55))
        self.min_vol_ratio = float(params.get("min_vol_ratio", 1.5))
        self.entry_buffer = float(params.get("entry_buffer_pct", 0.1)) / 100
        self.stop_atr = float(params.get("stop_atr_mult", 1.5))
        self.max_stop = float(params.get("max_stop_pct", 1.5)) / 100
        self.reward_risk = float(params.get("reward_risk", 1.6))
        self.max_bar_age = timedelta(minutes=float(params.get("max_bar_age_min", 6)))

    def in_window(self, now: datetime) -> bool:
        return WINDOW[0] <= now.astimezone(IST).time() < WINDOW[1]

    def scan(
        self, now: datetime, bars: dict[str, pd.DataFrame], blocked: frozenset[str] = frozenset()
    ) -> list[Candidate]:
        """Candidates as of ``now`` using only bars that ended at or before ``now``."""
        if not self.in_window(now):
            return []
        today = now.astimezone(IST).date()
        out: list[Candidate] = []
        for sym in sorted(bars):
            if sym in blocked:
                continue
            df = bars[sym]
            df = df[df["ts"] <= pd.Timestamp(now)].tail(MAX_BARS_LOOKBACK)
            if len(df) < OR_BARS + 3 or df["date"].iloc[-1] != today:
                continue
            if pd.Timestamp(now) - df["ts"].iloc[-1] > self.max_bar_age:
                continue  # stale data: never trade on it
            feats = intraday_features(df)
            row = feats.iloc[-1]
            if not (
                row["or_breakout"] == 1.0
                and row["vwap_dist"] > 0
                and row["vol_ratio"] >= self.min_vol_ratio
            ):
                continue
            prob: float | None = None
            contribs: tuple[tuple[str, float], ...] = ()
            if self.booster is not None:
                x = feats.iloc[[-1]].reindex(columns=self.columns)
                prob = float(self.booster.predict(x)[0])
                if prob < self.min_prob:
                    continue
                c = np.asarray(self.booster.predict(x, pred_contrib=True))[0]
                top = np.argsort(-np.abs(c[:-1]))[:5]
                contribs = tuple((self.columns[j], float(c[j])) for j in top)
            cand = self._geometry(sym, df, row, prob)
            if cand is not None:
                out.append(replace(cand, contributions=contribs))
        out.sort(key=lambda c: (-c.score, c.symbol))
        return out[: self.top_k]

    def _geometry(
        self, sym: str, df: pd.DataFrame, row: pd.Series, prob: float | None
    ) -> Candidate | None:
        today_bars = df[df["date"] == df["date"].iloc[-1]]
        close = float(today_bars["close"].iloc[-1])
        or_low = float(today_bars["low"].iloc[:OR_BARS].min())
        entry = round_tick(close * (1 + self.entry_buffer))
        atr_stop = close * (
            1 - self.stop_atr * float(row["atr_pct"] if row["atr_pct"] == row["atr_pct"] else 0.005)
        )
        stop = round_tick(max(or_low, atr_stop, entry * (1 - self.max_stop)))
        if stop <= 0 or stop >= entry:
            return None
        target = round_tick(entry + self.reward_risk * (entry - stop))
        score = prob if prob is not None else float(row["vol_ratio"])
        return Candidate(
            sym,
            entry,
            stop,
            target,
            0,
            score,
            f"ORB breakout, vol x{row['vol_ratio']:.1f}",
            "BUY",
            "MIS",
            prob,
        )
