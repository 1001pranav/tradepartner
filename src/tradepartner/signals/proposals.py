"""Signal -> OrderProposal bridge and signal lifecycle events (spec 5, 12 at 09:20).

Quantity is never decided here; the risk engine computes it. The router persists the proposal and
records the risk decision; this module maps the outcome back onto the signal as an event.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from tradepartner.db.models import Model, Signal, SignalEvent
from tradepartner.risk.types import Proposal, RiskDecision
from tradepartner.signals.generate import candidate_from_json
from tradepartner.strategies.base import Candidate


class SubmitOutcome(Protocol):
    """What signal code needs to know about a submission. The concrete router lives in
    ``execution`` and is passed in by the job layer: signal code never imports it (rule 1)."""

    @property
    def status(self) -> str: ...

    @property
    def proposal_id(self) -> int | None: ...

    @property
    def decision(self) -> RiskDecision | None: ...


class Submitter(Protocol):
    def submit(self, p: Proposal, now: datetime) -> SubmitOutcome: ...


TERMINAL = {"rejected_by_risk", "shown", "approved", "executed", "expired", "ignored"}


def record_event(
    session: Session, signal_id: int, status: str, detail: str | None, now: datetime
) -> SignalEvent:
    ev = SignalEvent(signal_id=signal_id, status=status, detail=detail, at=now, created_at=now)
    session.add(ev)
    session.commit()
    return ev


def events(session: Session, signal_id: int) -> list[SignalEvent]:
    return list(
        session.scalars(
            select(SignalEvent).where(SignalEvent.signal_id == signal_id).order_by(SignalEvent.id)
        )
    )


def current_status(session: Session, signal_id: int) -> str:
    evs = events(session, signal_id)
    return evs[-1].status if evs else "new"


def signal_candidate(session: Session, signal: Signal) -> Candidate:
    first = events(session, signal.id)[0]
    assert first.detail is not None
    return candidate_from_json(signal.symbol, first.detail)


def pending_signals(
    session: Session, strategy: str, now: datetime, max_age: timedelta
) -> list[Signal]:
    """Signals of an APPROVED model whose latest event is still 'new' and which are fresh."""
    approved_ids = set(
        session.scalars(
            select(Model.id).where(Model.strategy == strategy, Model.status == "approved")
        )
    )
    out = []
    for sig in session.scalars(
        select(Signal).where(Signal.strategy == strategy, Signal.created_at >= now - max_age)
    ):
        if sig.model_id in approved_ids and current_status(session, sig.id) == "new":
            out.append(sig)
    return sorted(out, key=lambda s: (-s.score, s.symbol))


def to_proposal(signal: Signal, cand: Candidate) -> Proposal:
    return Proposal(
        strategy=signal.strategy,
        symbol=signal.symbol,
        side="BUY",
        product=cand.product,
        order_type="LIMIT",
        kind="ENTRY",
        signal_time=signal.created_at,
        idempotency_key=f"{signal.strategy}-{signal.symbol}-{signal.as_of.date().isoformat()}-{signal.id}",
        limit_price_paise=cand.limit_price_paise,
        stop_loss_paise=cand.stop_paise,
        target_paise=cand.target_paise,
        signal_id=signal.id,
    )


@dataclass
class SubmitSummary:
    shown: int = 0
    placed: int = 0
    rejected: int = 0
    duplicates: int = 0


def submit_pending(
    session: Session,
    router: Submitter,
    strategy: str,
    now: datetime,
    max_age: timedelta = timedelta(hours=3),
) -> SubmitSummary:
    """09:20: route each pending signal through the risk engine and mode controller."""
    summary = SubmitSummary()
    for sig in pending_signals(session, strategy, now, max_age):
        result = router.submit(to_proposal(sig, signal_candidate(session, sig)), now)
        if result.status == "duplicate":
            summary.duplicates += 1
        elif result.status == "rejected":
            summary.rejected += 1
            reasons = result.decision.reasons if result.decision else []
            record_event(session, sig.id, "rejected_by_risk", json.dumps(reasons), now)
        elif result.status == "shown":
            summary.shown += 1
            record_event(
                session, sig.id, "shown", json.dumps({"proposal_id": result.proposal_id}), now
            )
        else:
            summary.placed += 1
            record_event(
                session, sig.id, "approved", json.dumps({"proposal_id": result.proposal_id}), now
            )
    return summary


def ignore(session: Session, signal_id: int, now: datetime) -> None:
    record_event(session, signal_id, "ignored", "ignored by user", now)


def expire_unfilled(session: Session, strategy: str, now: datetime) -> int:
    """15:00: shown or approved signals that never produced a fill are expired."""
    from tradepartner.db.models import Fill, Order, OrderProposal

    n = 0
    for sig in session.scalars(select(Signal).where(Signal.strategy == strategy)):
        if current_status(session, sig.id) not in ("shown", "approved"):
            continue
        prop_ids = list(
            session.scalars(select(OrderProposal.id).where(OrderProposal.signal_id == sig.id))
        )
        filled = session.scalar(
            select(Fill.id)
            .join(Order, Fill.order_id == Order.id)
            .where(Order.proposal_id.in_(prop_ids))
        )
        if filled is None and sig.created_at.date() == now.date():
            record_event(session, sig.id, "expired", "no fill by 15:00", now)
            n += 1
    return n


__all__ = ["TERMINAL", "SubmitOutcome", "SubmitSummary", "Submitter"]
