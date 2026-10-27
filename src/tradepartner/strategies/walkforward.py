"""A model that retrains as time passes, for historical paper trading (simulation only).

At each retrain date the model is fitted ONLY on rows whose labels were fully known by then
(label window ends before the as-of date, plus a one-day safety gap), so a replay of the past
never uses information from the future. Nothing here trades or registers a model.
"""

from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd

from tradepartner.backtest.view import MarketView
from tradepartner.features import build
from tradepartner.models import train
from tradepartner.strategies.base import Candidate
from tradepartner.strategies.baseline import Momentum12m1TopN
from tradepartner.strategies.model_strategy import ModelStrategy

MIN_TRAIN_ROWS = 2000
FIT_PARAMS = {
    "num_leaves": 15,
    "learning_rate": 0.05,
    "min_child_samples": 50,
    "bagging_freq": 1,
    "feature_fraction": 0.8,
}


class WalkForwardModel:
    def __init__(
        self,
        name: str,
        view: MarketView,
        labels: pd.DataFrame,
        params: dict[str, Any],
        objective: Literal["binary", "regression"],
        retrain_every_days: int = 21,
        rounds: int = 150,
        seed: int = 42,
    ) -> None:
        self.name = name
        self.params = params
        self.objective = objective
        self.horizon = int(params["horizon_days"])
        self.every = retrain_every_days
        self.rounds, self.seed = rounds, seed
        self.cols = build.PRICE_ONLY_COLUMNS
        self.matrix = build.build_matrix(view)
        days = view.trading_days()
        self.pos = {d: i for i, d in enumerate(days)}
        data = self.matrix[self.cols].join(labels[["y"]], how="inner")
        self.data = data[data["y"].notna()]
        self.dpos = np.array([self.pos[d] for d in self.data.index.get_level_values("date")])
        self._inner: ModelStrategy | None = None
        self._trained_at: int | None = None
        self.retrains: list[tuple[date, date]] = []  # (as_of, last training date), for audits

    def _retrain(self, as_of: date, days: list[date]) -> None:
        cutoff = self.pos[as_of] - self.horizon - 1
        rows = self.data.loc[self.dpos <= cutoff]
        self._trained_at = self.pos[as_of]
        if len(rows) < MIN_TRAIN_ROWS:
            self._inner = None
            return
        booster = train.fit(
            rows[self.cols],
            rows["y"].to_numpy(),
            {**train.base_params(self.objective, self.seed), **FIT_PARAMS},
            self.rounds,
        )
        self._inner = ModelStrategy(self.name, booster, self.cols, self.params, self.objective)
        self.retrains.append((as_of, days[cutoff]))

    def candidates(self, as_of: date, view: MarketView) -> list[Candidate]:
        days = view.trading_days()
        if self.objective == "regression" and not Momentum12m1TopN._is_first_trading_day_of_month(
            as_of, view
        ):
            return []  # positional ideas are refreshed monthly
        if self._trained_at is None or self.pos[as_of] - self._trained_at >= self.every:
            self._retrain(as_of, days)
        if self._inner is None:
            return []
        self._inner._cache = (id(view), self.matrix)  # reuse the matrix built once
        return self._inner.candidates(as_of, view)
