"""Gathers the evidence required to unlock AUTO for a strategy (spec 7.1 rule 2)."""

from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from tradepartner.db.models import Fill, Model
from tradepartner.modes.controller import UnlockEvidence
from tradepartner.risk.limits import BOT_ORIGINS, closed_trades, fill_strategies


def collect_unlock_evidence(session: Session, strategy: str) -> UnlockEvidence:
    """Closed paper/bot/manual-matched trades for the strategy, and the approved-model status.

    ``gate_passed`` is read from the approved model's ``metrics_json['paper_gate_passed']``,
    which the evaluation job (M6) writes after checking realized paper/manual metrics against
    the promotion gate. Absent means not passed.
    """
    model = session.scalar(
        select(Model).where(Model.strategy == strategy, Model.status == "approved")
        .order_by(Model.id.desc()).limit(1)
    )  # fmt: skip
    fills = {f.id: f for f in session.scalars(select(Fill).where(Fill.origin.in_(BOT_ORIGINS)))}
    meta = fill_strategies(session)
    trades = [
        t for t in closed_trades(session) if t.strategy == strategy and t.close_fill_id in fills
    ]
    first: date | None = min((t.close_date for t in trades), default=None)
    for f in fills.values():
        if meta.get(f.id, (None, None))[0] == strategy and (first is None or f.trade_date < first):
            first = f.trade_date
    return UnlockEvidence(
        model_approved=model is not None,
        closed_trades=len(trades),
        first_trade_date=first,
        gate_passed=bool(model and model.metrics_json.get("paper_gate_passed", False)),
    )
