"""Exchange stop-loss for intraday (MIS) positions (spec 7.3, M9): right after a MIS entry fills,
a stop-loss-market order is placed through the router as a protective exit, so it passes the same
risk engine as every other order."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from tradepartner.db.models import Fill, Lot, Order, OrderProposal
from tradepartner.risk.types import Proposal
from tradepartner.runtime import Runtime


@dataclass(frozen=True)
class MisStop:
    symbol: str
    qty: int
    stop_paise: int
    strategy: str


def mis_intents(session: Session) -> list[MisStop]:
    qty: dict[str, int] = defaultdict(int)
    info: dict[str, tuple[int, str]] = {}
    for lot in session.scalars(select(Lot).order_by(Lot.id)):
        fill = session.get(Fill, lot.open_fill_id)
        if fill is None or fill.product != "MIS" or fill.order_id is None:
            continue
        order = session.get(Order, fill.order_id)
        prop = (
            session.get(OrderProposal, order.proposal_id) if order and order.proposal_id else None
        )
        if prop is None or prop.stop_loss_paise is None:
            continue
        qty[lot.symbol] += lot.qty_remaining
        info[lot.symbol] = (prop.stop_loss_paise, prop.strategy or "intraday_v1")
    return [MisStop(s, q, info[s][0], info[s][1]) for s, q in sorted(qty.items()) if q > 0]


def place_mis_stops(rt: Runtime, now: datetime) -> list[str]:
    """Ensure every bot MIS position has an open exchange SL-M order for its full quantity."""
    placed: list[str] = []
    open_sl = {o.symbol: o for o in rt.broker.open_orders() if o.order_type in ("SL", "SL-M")}
    for it in mis_intents(rt.session):
        current = open_sl.get(it.symbol)
        if current is not None and current.qty == it.qty:
            continue
        if current is not None:  # the position changed size: replace the stop
            rt.broker.cancel_order(current.broker_order_id)
        p = Proposal(
            it.strategy,
            it.symbol,
            "SELL",
            "MIS",
            "SL-M",
            "EXIT",
            now,
            f"mis-sl-{it.symbol}-{now:%Y%m%d}-{it.qty}",
            trigger_price_paise=it.stop_paise,
            exit_qty=it.qty,
            exit_reason="stop",
            protective=True,
        )
        if rt.router.submit(p, now).status == "placed":
            placed.append(it.symbol)
    return placed
