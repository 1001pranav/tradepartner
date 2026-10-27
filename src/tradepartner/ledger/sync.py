"""Pull orders, trades, positions and holdings from Kite into the ledger (spec 9.2).

Kite's ``orders()``/``trades()`` cover today only, so this runs daily at 15:45 IST (and on demand).
Fills are inserted once, deduplicated on ``broker_trade_id``.
"""

import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from tradepartner.calendar import IST
from tradepartner.db.models import (
    Fill,
    Instrument,
    Order,
    OrderEvent,
    OrderProposal,
    PositionSnapshot,
)

TAG_RE = re.compile(r"^tp-(\d+)$")
STATUS_MAP = {"COMPLETE": "COMPLETE", "REJECTED": "REJECTED", "CANCELLED": "CANCELLED"}


class LedgerSource(Protocol):
    def orders(self) -> list[dict[str, Any]]: ...
    def trades(self) -> list[dict[str, Any]]: ...
    def positions(self) -> dict[str, list[dict[str, Any]]]: ...
    def holdings(self) -> list[dict[str, Any]]: ...


@dataclass
class SyncResult:
    orders: int = 0
    fills: int = 0
    skipped_non_equity: int = 0
    linked_proposals: int = 0
    snapshots: int = 0
    notes: list[str] = field(default_factory=list)


def _paise(value: object) -> int:
    return int((Decimal(str(value)) * 100).to_integral_value())


def _ts(value: Any) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if value.tzinfo is None:
        value = value.replace(tzinfo=IST)
    return value.astimezone(UTC)  # type: ignore[no-any-return]


def _is_nse_equity(row: dict[str, Any]) -> bool:
    return row.get("exchange") == "NSE" and row.get("product") in ("CNC", "MIS")


def origin_for(session: Session, tag: str | None) -> tuple[str, int | None]:
    """(origin, proposal_id). Bot orders carry ``tag = 'tp-<proposal_id>'``; all else is manual."""
    if tag and tag.startswith("tpk-"):  # kill-switch exits are bot actions without a proposal
        return "bot_auto", None
    m = TAG_RE.match(tag or "")
    if not m:
        return "manual_external", None
    proposal = session.get(OrderProposal, int(m.group(1)))
    if proposal is None:
        return "manual_external", None
    return ("bot_auto" if proposal.mode_at_creation == "AUTO" else "bot_manual_api"), proposal.id


def _upsert_order(session: Session, row: dict[str, Any], result: SyncResult) -> None:
    bid = str(row["order_id"])
    origin, proposal_id = origin_for(session, row.get("tag"))
    order = session.scalar(select(Order).where(Order.broker_order_id == bid))
    if order is None:
        order = Order(
            proposal_id=proposal_id,
            broker_order_id=bid,
            origin=origin,
            symbol=row["tradingsymbol"],
            side=row["transaction_type"],
            product=row["product"],
            order_type=row.get("order_type", "LIMIT"),
            qty=int(row["quantity"]),
            price_paise=_paise(row.get("price") or 0) or None,
            trigger_paise=_paise(row.get("trigger_price") or 0) or None,
            tag=row.get("tag"),
            placed_at=_ts(row.get("order_timestamp") or datetime.now(UTC)),
            status=row["status"],
        )
        session.add(order)
        session.flush()
        result.orders += 1
    status = row["status"]
    filled = int(row.get("filled_quantity") or 0)
    last = session.scalar(
        select(OrderEvent).where(OrderEvent.order_id == order.id).order_by(OrderEvent.id.desc())
    )
    if last is None or (last.status, last.filled_qty) != (status, filled):
        session.add(
            OrderEvent(
                order_id=order.id,
                status=status,
                filled_qty=filled,
                avg_price_paise=_paise(row.get("average_price") or 0) or None,
                message=row.get("status_message"),
                at=_ts(row.get("exchange_update_timestamp") or datetime.now(UTC)),
            )
        )
    order.status = status


def _link_manual_proposals(
    session: Session, trade_date: date, band_pct: float, result: SyncResult
) -> None:
    """MANUAL ``kite_app`` mode: match untagged completed orders to that day's shown proposals."""
    orders = session.scalars(
        select(Order).where(
            Order.origin == "manual_external",
            Order.proposal_id.is_(None),
            Order.status == "COMPLETE",
        )
    )
    taken = set(session.scalars(select(Order.proposal_id).where(Order.proposal_id.is_not(None))))
    for order in orders:
        if order.placed_at.astimezone(IST).date() != trade_date or order.price_paise is None:
            continue
        for prop in session.scalars(
            select(OrderProposal).where(
                OrderProposal.symbol == order.symbol,
                OrderProposal.side == order.side,
                OrderProposal.product == order.product,
                OrderProposal.qty == order.qty,
                OrderProposal.risk_decision == "approved",
            )
        ):
            if prop.id in taken or prop.limit_price_paise is None:
                continue
            drift = abs(order.price_paise - prop.limit_price_paise) * 10_000
            if drift <= int(band_pct * 100) * prop.limit_price_paise:
                order.proposal_id = prop.id  # origin stays manual_external, with a link
                taken.add(prop.id)
                result.linked_proposals += 1
                break


def sync(
    session: Session,
    source: LedgerSource,
    today: date,
    account: str,
    band_pct: float = 1.5,
) -> SyncResult:
    result = SyncResult()
    for row in source.orders():
        if not _is_nse_equity(row):
            result.skipped_non_equity += 1
            continue
        _upsert_order(session, row, result)
    session.flush()

    isins = {
        sym: isin
        for sym, isin in session.execute(select(Instrument.tradingsymbol, Instrument.isin))
    }
    existing = set(session.scalars(select(Fill.broker_trade_id)))
    orders_by_bid = {
        o.broker_order_id: o for o in session.scalars(select(Order)) if o.broker_order_id
    }
    for t in source.trades():
        if not _is_nse_equity(t):
            result.skipped_non_equity += 1
            continue
        tid = str(t["trade_id"])
        if tid in existing:
            continue
        existing.add(tid)
        order = orders_by_bid.get(str(t.get("order_id")))
        filled_at = _ts(t.get("fill_timestamp") or t.get("exchange_timestamp"))
        session.add(
            Fill(
                broker_trade_id=tid,
                broker_order_id=str(t.get("order_id")) if t.get("order_id") else None,
                order_id=order.id if order else None,
                origin=order.origin if order else "manual_external",
                account=account,
                symbol=t["tradingsymbol"],
                isin=isins.get(t["tradingsymbol"]),
                exchange="NSE",
                side=t["transaction_type"],
                product=t["product"],
                qty=int(t["quantity"]),
                price_paise=_paise(t["average_price"]),
                filled_at=filled_at,
                trade_date=filled_at.astimezone(IST).date(),
                source="kite_api",
            )
        )
        result.fills += 1
    session.flush()
    _link_manual_proposals(session, today, band_pct, result)
    result.snapshots = snapshot_positions(session, source, today)
    session.commit()
    return result


def snapshot_positions(session: Session, source: LedgerSource, day: date) -> int:
    n = 0
    for p in source.positions().get("net", []):
        if p.get("exchange") != "NSE" or not int(p.get("quantity") or 0):
            continue
        session.add(
            PositionSnapshot(
                snapshot_date=day, kind="position", symbol=p["tradingsymbol"],
                isin=None, product=p.get("product"), qty=int(p["quantity"]),
                avg_price_paise=_paise(p.get("average_price") or 0),
            )
        )  # fmt: skip
        n += 1
    for h in source.holdings():
        qty = int(h.get("quantity") or 0) + int(h.get("t1_quantity") or 0)
        if h.get("exchange") != "NSE" or not qty:
            continue
        session.add(
            PositionSnapshot(
                snapshot_date=day, kind="holding", symbol=h["tradingsymbol"],
                isin=h.get("isin"), product="CNC", qty=qty,
                avg_price_paise=_paise(h.get("average_price") or 0),
            )
        )  # fmt: skip
        n += 1
    return n
