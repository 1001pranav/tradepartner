"""Intraday signals: one per symbol per day, stored like any other signal (append-only)."""

from datetime import datetime

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from tradepartner.calendar import IST
from tradepartner.db.models import Model, Signal, SignalEvent
from tradepartner.signals.generate import GeneratedSignal, candidate_json
from tradepartner.strategies.base import Candidate
from tradepartner.strategies.intraday import NAME, IntradayScanner

FEATURE_REF = "intraday-fs1"


def generate_intraday_signals(
    session: Session,
    model: Model,
    scanner: IntradayScanner,
    bars: dict[str, pd.DataFrame],
    now: datetime,
    blocked: frozenset[str] = frozenset(),
) -> list[GeneratedSignal]:
    shadow = model.status == "shadow"
    today = now.astimezone(IST).date()
    made: list[GeneratedSignal] = []
    for cand in scanner.scan(now, bars, blocked):
        already = any(
            s.created_at.astimezone(IST).date() == today
            for s in session.scalars(
                select(Signal).where(
                    Signal.strategy == NAME,
                    Signal.model_id == model.id,
                    Signal.symbol == cand.symbol,
                )
            )
        )
        if already:
            continue  # one idea per symbol per day: no repeat entries on the same name
        sig = Signal(
            strategy=NAME,
            model_id=model.id,
            symbol=cand.symbol,
            side="BUY",
            as_of=now,
            score=cand.score,
            prob=cand.prob,
            horizon_days=0,
            explanation=cand.reason,
            features_ref=f"{FEATURE_REF}:{now.isoformat()}:{cand.symbol}",
            status="new",
            created_at=now,
        )
        session.add(sig)
        session.flush()
        session.add(
            SignalEvent(
                signal_id=sig.id,
                status="shadow" if shadow else "new",
                detail=candidate_json(cand, False),
                at=now,
                created_at=now,
            )
        )
        made.append(GeneratedSignal(sig.id, cand, cand.reason, model.id, shadow, False))
    session.commit()
    return made


__all__ = ["Candidate", "generate_intraday_signals"]
