"""positional_v1: 1 to 12+ month holds from the cross-sectional rank model (monthly cadence)."""

from datetime import date

from tradepartner.backtest.view import MarketView
from tradepartner.strategies.base import Candidate
from tradepartner.strategies.model_strategy import ModelStrategy

NAME = "positional_v1"


class PositionalStrategy(ModelStrategy):
    def candidates(self, as_of: date, view: MarketView) -> list[Candidate]:
        from tradepartner.strategies.baseline import Momentum12m1TopN

        if not Momentum12m1TopN._is_first_trading_day_of_month(as_of, view):
            return []  # positional ideas are refreshed monthly
        return super().candidates(as_of, view)
