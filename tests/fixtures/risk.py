"""Builders for risk-engine tests: a capital-set config, a sane context and a sane proposal."""

from dataclasses import replace
from datetime import date, datetime, timedelta

from tests.fixtures.market import dt, make_cal
from tradepartner.risk.types import (
    Context,
    InstrumentMeta,
    LimitState,
    Position,
    Proposal,
    Quote,
)
from tradepartner.settings import RiskConfig, load_settings

CAPITAL_INR = 1_000_000
NOW = dt(2026, 3, 10, 10, 0)  # a Tuesday, mid-session, IST
LTP = 150_000  # Rs 1,500.00


def cfg(**changes: object) -> RiskConfig:
    base = load_settings().risk
    return replace_cfg(base.model_copy(update={"capital_base_inr": CAPITAL_INR}), **changes)


def replace_cfg(base: RiskConfig, **changes: object) -> RiskConfig:
    """Shallow update; nested sections may be given as ``per_trade={"max_risk_pct": 1}``."""
    updates: dict[str, object] = {}
    for key, value in changes.items():
        current = getattr(base, key)
        updates[key] = current.model_copy(update=value) if isinstance(value, dict) else value
    return base.model_copy(update=updates)


def quote(ltp: int = LTP, age_s: int = 0, now: datetime = NOW, **kw: object) -> Quote:
    return Quote(ltp, now - timedelta(seconds=age_s), **kw)  # type: ignore[arg-type]


def meta(
    symbol: str = "INFY", sector: str = "IT", adv_inr: int = 200_000_000, **kw: object
) -> InstrumentMeta:
    return InstrumentMeta(symbol, sector, adv_inr * 100, **kw)  # type: ignore[arg-type]


def ctx(mode: str = "MANUAL", **changes: object) -> Context:
    base = Context(
        now=NOW,
        mode=mode,  # type: ignore[arg-type]
        calendar=make_cal(),
        quotes={"INFY": quote()},
        instruments={"INFY": meta()},
        positions=(),
        limits=LimitState(equity_paise=CAPITAL_INR * 100, peak_equity_paise=CAPITAL_INR * 100),
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def limits(**changes: object) -> LimitState:
    return replace(
        LimitState(equity_paise=CAPITAL_INR * 100, peak_equity_paise=CAPITAL_INR * 100),
        **changes,  # type: ignore[arg-type]
    )


def entry(**changes: object) -> Proposal:
    base = Proposal(
        strategy="swing_v1", symbol="INFY", side="BUY", product="CNC", order_type="LIMIT",
        kind="ENTRY", signal_time=NOW, idempotency_key="k1", limit_price_paise=LTP,
        stop_loss_paise=144_000, target_paise=162_000,
    )  # fmt: skip
    return replace(base, **changes)  # type: ignore[arg-type]


def exit_(**changes: object) -> Proposal:
    base = Proposal(
        strategy="swing_v1", symbol="INFY", side="SELL", product="CNC", order_type="LIMIT",
        kind="EXIT", signal_time=NOW, idempotency_key="x1", limit_price_paise=LTP,
        exit_qty=10, exit_reason="time_stop", protective=True,
    )  # fmt: skip
    return replace(base, **changes)  # type: ignore[arg-type]


def pos(
    symbol: str = "INFY", qty: int = 10, avg: int = LTP, product: str = "CNC", **kw: object
) -> Position:
    return Position(symbol, qty, avg, product, **kw)  # type: ignore[arg-type]


__all__ = [
    "CAPITAL_INR",
    "LTP",
    "NOW",
    "cfg",
    "ctx",
    "date",
    "entry",
    "exit_",
    "limits",
    "meta",
    "pos",
    "quote",
]
