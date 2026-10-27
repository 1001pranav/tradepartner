"""Model-backed strategies (swing_v1, positional_v1): a LightGBM model proposes candidates.

The model only ranks symbols. Entry, stop and target come from ATR geometry; quantity is decided
later by the risk engine. Strategy code never imports the router or the mode controller.
"""

from collections.abc import Callable
from datetime import date
from typing import Any, cast

import lightgbm as lgb
import numpy as np
import pandas as pd

from tradepartner.backtest.view import MarketView
from tradepartner.features import build
from tradepartner.features.news import NEWS_COLUMNS
from tradepartner.strategies.base import Candidate
from tradepartner.strategies.baseline import atr, make_candidate

NewsProvider = Callable[[date, list[str]], pd.DataFrame | None]
TOP_CONTRIBUTIONS = 5


class ModelStrategy:
    """Ranks the universe with a trained model as of a close and proposes the top few."""

    def __init__(
        self,
        name: str,
        booster: lgb.Booster,
        columns: list[str],
        params: dict[str, Any],
        objective: str,
        news_provider: NewsProvider | None = None,
    ) -> None:
        self.name = name
        self.booster = booster
        self.columns = columns
        self.objective = objective
        self.top_k = int(params.get("top_k", 5))
        self.horizon = int(params.get("horizon_days", 5))
        self.min_score = float(params.get("min_prob", 0.5 if objective == "binary" else 0.7))
        self.entry_buffer = float(params.get("entry_buffer_pct", 0.5)) / 100
        self.stop_atr = float(params.get("stop_atr_mult", 2.0))
        self.rr = float(params.get("reward_risk", 2.0))
        self.max_stop = float(params.get("max_stop_pct", 7.0)) / 100
        self.news_provider = news_provider
        self._cache: tuple[int, pd.DataFrame] | None = None

    def _matrix(self, view: MarketView) -> pd.DataFrame:
        key = id(view)
        if self._cache is None or self._cache[0] != key:
            self._cache = (key, build.build_matrix(view))
        return self._cache[1]

    def features_at(self, as_of: date, view: MarketView) -> pd.DataFrame:
        m = self._matrix(view)
        if as_of not in set(m.index.get_level_values("date")):
            return pd.DataFrame(columns=self.columns)
        rows = cast(pd.DataFrame, m.xs(as_of, level="date")).copy()
        if self.news_provider is not None and any(c in self.columns for c in NEWS_COLUMNS):
            live = self.news_provider(as_of, list(rows.index))
            if live is not None:
                for c in live.columns:
                    rows[c] = live[c].reindex(rows.index)
        return rows.reindex(columns=self.columns)

    def scores(self, as_of: date, view: MarketView) -> pd.Series:
        rows = self.features_at(as_of, view)
        rows = rows[rows.notna().any(axis=1)]
        if rows.empty:
            return pd.Series(dtype="float64")
        return pd.Series(self.booster.predict(rows), index=rows.index)

    def candidates(self, as_of: date, view: MarketView) -> list[Candidate]:
        rows = self.features_at(as_of, view)
        rows = rows[rows.notna().any(axis=1)]
        rows = rows[rows.index.isin(view.symbols_as_of(as_of))]  # point-in-time membership
        if rows.empty:
            return []
        pred = np.asarray(self.booster.predict(rows))
        contrib = np.asarray(self.booster.predict(rows, pred_contrib=True))
        order = sorted(range(len(rows)), key=lambda i: (-pred[i], str(rows.index[i])))
        out: list[Candidate] = []
        for i in order:
            if len(out) >= self.top_k or pred[i] < self.min_score:
                break
            sym = str(rows.index[i])
            h = view.history(sym, as_of, 60)
            if len(h) < 20 or h["date"].iloc[-1] != as_of:
                continue
            cand = make_candidate(
                sym, float(h["close"].iloc[-1]), self.stop_atr * atr(h), self.rr, self.horizon,
                float(pred[i]), f"model score {pred[i]:.3f}", self.entry_buffer, self.max_stop,
            )  # fmt: skip
            if cand is None:
                continue
            top = np.argsort(-np.abs(contrib[i][:-1]))[
                :TOP_CONTRIBUTIONS
            ]  # last column is the bias
            contribs = tuple((self.columns[j], float(contrib[i][j])) for j in top)
            out.append(_with(cand, float(pred[i]), contribs))
        return out


def _with(c: Candidate, prob: float, contributions: tuple[tuple[str, float], ...]) -> Candidate:
    from dataclasses import replace

    return replace(c, prob=prob, contributions=contributions)
