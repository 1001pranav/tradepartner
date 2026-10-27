"""Strategy protocol (spec 5). Strategies propose candidates; the risk engine decides size.

Strategy code must never import ``execution.router`` or ``modes.controller`` (import-linter).
"""

from dataclasses import dataclass
from datetime import date
from typing import Protocol

from tradepartner.backtest.view import MarketView


@dataclass(frozen=True)
class Candidate:
    """A trade idea as of a close. Quantity is never part of it."""

    symbol: str
    limit_price_paise: int
    stop_paise: int
    target_paise: int
    horizon_days: int  # trading days before the time stop
    score: float
    reason: str = ""
    side: str = "BUY"
    product: str = "CNC"
    prob: float | None = None  # model output (probability or rank score)
    contributions: tuple[tuple[str, float], ...] = ()  # top feature contributions (LightGBM SHAP)


class Strategy(Protocol):
    name: str

    def candidates(self, as_of: date, view: MarketView) -> list[Candidate]:
        """Ideas using only data with ``date <= as_of`` (the view enforces it)."""
        ...


def round_tick(paise: float, tick: int = 5) -> int:
    """Round to the nearest exchange tick (Rs 0.05 = 5 paise)."""
    return round(paise / tick) * tick
