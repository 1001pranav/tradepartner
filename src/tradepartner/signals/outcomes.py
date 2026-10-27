"""Reveal outcomes once a signal's horizon has passed (spec 10.6 'hide, then reveal').

Written to ``signal_outcomes`` (append-only): the growing labelled history for the next retrain,
the shadow comparison and the drift monitors.
"""

from datetime import date, datetime

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from tradepartner.backtest.view import MarketView
from tradepartner.calendar import IST
from tradepartner.db.models import Signal, SignalOutcome
from tradepartner.features import labels
from tradepartner.settings import Settings


def forward_return(view: MarketView, symbol: str, as_of: date, horizon: int) -> float | None:
    """next open -> close ``horizon`` days after ``as_of``; None if not yet known."""
    if symbol not in view.symbols:
        return None  # no data for this symbol (delisted or outside the loaded universe)
    h = view.history(symbol, view.trading_days()[-1])
    pos = h.index[h["date"] == as_of]
    if len(pos) == 0:
        return None
    i = int(pos[0])
    if i + horizon >= len(h):
        return None
    return float(h["close"].iloc[i + horizon] / h["open"].iloc[i + 1] - 1)


def index_forward_return(view: MarketView, as_of: date, horizon: int) -> float | None:
    if view.index is None:
        return None
    idx = view.index
    pos = idx.index[idx["date"] == as_of]
    if len(pos) == 0 or int(pos[0]) + horizon >= len(idx):
        return None
    i = int(pos[0])
    return float(idx["close"].iloc[i + horizon] / idx["open"].iloc[i + 1] - 1)


def reveal_outcomes(
    session: Session, view: MarketView, settings: Settings, now: datetime, binary: bool = True
) -> int:
    done = set(session.scalars(select(SignalOutcome.signal_id)))
    threshold = labels.round_trip_cost_fraction(settings.costs) + labels.EXTRA_MARGIN
    n = 0
    for sig in session.scalars(select(Signal)):
        if sig.id in done:
            continue
        day = sig.as_of.astimezone(IST).date()
        ret = forward_return(view, sig.symbol, day, sig.horizon_days)
        if ret is None:
            continue
        idx_ret = index_forward_return(view, day, sig.horizon_days)
        excess = ret - idx_ret if idx_ret is not None else None
        label = int(excess > threshold) if (binary and excess is not None) else None
        session.add(
            SignalOutcome(
                signal_id=sig.id,
                realized_return=ret,
                excess_return=excess,
                label=label,
                revealed_at=now,
            )
        )
        n += 1
    session.commit()
    return n


def outcome_frame(session: Session, strategy: str, model_id: int | None = None) -> pd.DataFrame:
    q = (
        select(
            Signal.id,
            Signal.model_id,
            Signal.symbol,
            Signal.prob,
            Signal.created_at,
            SignalOutcome.realized_return,
            SignalOutcome.excess_return,
            SignalOutcome.label,
        )
        .join(SignalOutcome, SignalOutcome.signal_id == Signal.id)
        .where(Signal.strategy == strategy)
    )
    if model_id is not None:
        q = q.where(Signal.model_id == model_id)
    return pd.DataFrame(
        session.execute(q).all(),
        columns=[
            "signal_id",
            "model_id",
            "symbol",
            "prob",
            "created_at",
            "realized",
            "excess",
            "label",
        ],
    )
