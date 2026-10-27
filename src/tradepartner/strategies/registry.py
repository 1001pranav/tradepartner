"""Strategy lookup by name for the CLI, backtester and scheduler."""

from collections.abc import Callable

from tradepartner.strategies.base import Strategy
from tradepartner.strategies.baseline import Momentum12m1TopN, Momentum20TopN

REGISTRY: dict[str, Callable[[], Strategy]] = {
    Momentum20TopN.name: Momentum20TopN,
    Momentum12m1TopN.name: Momentum12m1TopN,
}


def get(name: str) -> Strategy:
    try:
        return REGISTRY[name]()
    except KeyError:
        raise KeyError(f"unknown strategy {name!r}; known: {sorted(REGISTRY)}") from None
