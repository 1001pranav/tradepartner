from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from tradepartner.db.models import (
    Fill,
    Instrument,
    Order,
    OrderEvent,
    OrderProposal,
    PositionSnapshot,
)
from tradepartner.ledger import sync

DAY = date(2026, 3, 2)
IST_NAIVE = datetime(2026, 3, 2, 10, 15, 0)  # Kite returns naive IST datetimes


class FakeLedgerKite:
    def __init__(self) -> None:
        self._orders: list[dict[str, Any]] = []
        self._trades: list[dict[str, Any]] = []
        self._positions: dict[str, list[dict[str, Any]]] = {"net": []}
        self._holdings: list[dict[str, Any]] = []

    def orders(self) -> list[dict[str, Any]]:
        return self._orders

    def trades(self) -> list[dict[str, Any]]:
        return self._trades

    def positions(self) -> dict[str, list[dict[str, Any]]]:
        return self._positions

    def holdings(self) -> list[dict[str, Any]]:
        return self._holdings


def order(oid: str, tag: str | None = None, status: str = "COMPLETE", filled: int = 10,
          sym: str = "INFY", price: float = 1500.0, product: str = "CNC",
          exchange: str = "NSE", qty: int = 10) -> dict[str, Any]:  # fmt: skip
    return {
        "order_id": oid, "tradingsymbol": sym, "exchange": exchange, "transaction_type": "BUY",
        "product": product, "order_type": "LIMIT", "quantity": qty, "filled_quantity": filled,
        "price": price, "average_price": price, "tag": tag, "status": status,
        "order_timestamp": IST_NAIVE, "exchange_update_timestamp": IST_NAIVE,
    }  # fmt: skip


def trade(tid: str, oid: str, sym: str = "INFY", exchange: str = "NSE", price: float = 1500.0,
          qty: int = 10, product: str = "CNC") -> dict[str, Any]:  # fmt: skip
    return {
        "trade_id": tid, "order_id": oid, "tradingsymbol": sym, "exchange": exchange,
        "transaction_type": "BUY", "product": product, "quantity": qty, "average_price": price,
        "fill_timestamp": IST_NAIVE,
    }  # fmt: skip


def proposal(db: Session, pid: int = 7, mode: str = "AUTO", price: int = 150000) -> OrderProposal:
    p = OrderProposal(
        id=pid, symbol="INFY", side="BUY", product="CNC", order_type="LIMIT", qty=10,
        limit_price_paise=price, mode_at_creation=mode, risk_decision="approved",
        idempotency_key=f"k{pid}",
    )  # fmt: skip
    db.add(p)
    db.commit()
    return p


def test_origin_from_tag(db: Session) -> None:
    proposal(db, 7, "AUTO")
    proposal(db, 8, "MANUAL")
    assert sync.origin_for(db, "tp-7") == ("bot_auto", 7)
    assert sync.origin_for(db, "tp-8") == ("bot_manual_api", 8)
    assert sync.origin_for(db, "tp-999") == ("manual_external", None)
    assert sync.origin_for(db, "tpk-abc123") == ("bot_auto", None)
    assert sync.origin_for(db, "random") == ("manual_external", None)
    assert sync.origin_for(db, None) == ("manual_external", None)


def test_sync_inserts_orders_fills_events_and_is_idempotent(db: Session) -> None:
    proposal(db, 7)
    db.add(
        Instrument(
            instrument_token=1,
            exchange="NSE",
            tradingsymbol="INFY",
            isin="INE009A01021",
            segment="NSE",
        )
    )
    db.commit()
    k = FakeLedgerKite()
    k._orders = [order("A1", "tp-7"), order("A2"), order("F1", exchange="NFO", sym="NIFTYFUT")]
    k._trades = [
        trade("T1", "A1"),
        trade("T2", "A2", price=1501.25),
        trade("TF", "F1", exchange="NFO"),
    ]
    r = sync.sync(db, k, DAY, "acct")
    assert (r.orders, r.fills, r.skipped_non_equity) == (2, 2, 2)
    f1 = db.scalars(select(Fill).where(Fill.broker_trade_id == "T1")).one()
    assert f1.origin == "bot_auto" and f1.isin == "INE009A01021" and f1.price_paise == 150000
    assert f1.filled_at == datetime(2026, 3, 2, 4, 45, tzinfo=UTC) and f1.trade_date == DAY
    assert (
        db.scalars(select(Fill).where(Fill.broker_trade_id == "T2")).one().origin
        == "manual_external"
    )
    again = sync.sync(db, k, DAY, "acct")
    assert (again.orders, again.fills) == (0, 0)
    assert db.scalar(select(func.count()).select_from(Fill)) == 2
    assert db.scalar(select(func.count()).select_from(OrderEvent)) == 2  # no duplicate events


def test_order_status_progress_appends_events(db: Session) -> None:
    k = FakeLedgerKite()
    k._orders = [order("A1", status="OPEN", filled=0)]
    sync.sync(db, k, DAY, "a")
    k._orders = [
        order("A1", status="OPEN", filled=4),
    ]
    sync.sync(db, k, DAY, "a")
    k._orders = [order("A1", status="COMPLETE", filled=10)]
    sync.sync(db, k, DAY, "a")
    events = db.scalars(select(OrderEvent).order_by(OrderEvent.id)).all()
    assert [(e.status, e.filled_qty) for e in events] == [
        ("OPEN", 0),
        ("OPEN", 4),
        ("COMPLETE", 10),
    ]
    assert db.scalars(select(Order)).one().status == "COMPLETE"


def test_trade_without_known_order_is_manual(db: Session) -> None:
    k = FakeLedgerKite()
    k._trades = [trade("T9", "ghost")]
    sync.sync(db, k, DAY, "a")
    f = db.scalars(select(Fill)).one()
    assert f.origin == "manual_external" and f.order_id is None


def test_manual_fill_matched_to_shown_proposal_within_band(db: Session) -> None:
    proposal(db, 7, price=150000)
    k = FakeLedgerKite()
    k._orders = [order("M1", price=1510.0)]  # +0.67%: inside the 1.5% band
    r = sync.sync(db, k, DAY, "a")
    o = db.scalars(select(Order)).one()
    assert r.linked_proposals == 1 and o.proposal_id == 7 and o.origin == "manual_external"
    # a second identical order cannot claim the same proposal
    k._orders.append(order("M2", price=1510.0))
    sync.sync(db, k, DAY, "a")
    assert db.scalars(select(Order).where(Order.broker_order_id == "M2")).one().proposal_id is None


def test_manual_fill_outside_band_or_wrong_day_or_wrong_qty_not_matched(db: Session) -> None:
    proposal(db, 7, price=150000)
    k = FakeLedgerKite()
    k._orders = [order("far", price=1600.0), order("qty", price=1500.0, qty=11, filled=11)]
    sync.sync(db, k, DAY, "a")
    assert (
        db.scalar(select(func.count()).select_from(Order).where(Order.proposal_id.is_not(None)))
        == 0
    )
    k._orders = [order("day", price=1500.0)]
    sync.sync(db, k, date(2026, 3, 3), "a")  # proposal day != order day
    assert db.scalars(select(Order).where(Order.broker_order_id == "day")).one().proposal_id is None


def test_proposal_without_limit_price_not_matched(db: Session) -> None:
    p = proposal(db, 7)
    p.limit_price_paise = None
    db.commit()
    k = FakeLedgerKite()
    k._orders = [order("M1")]
    assert sync.sync(db, k, DAY, "a").linked_proposals == 0


def test_market_order_without_price_not_linked(db: Session) -> None:
    proposal(db, 7)
    k = FakeLedgerKite()
    k._orders = [order("M1", price=0.0)]
    assert sync.sync(db, k, DAY, "a").linked_proposals == 0


def test_snapshots_positions_and_holdings(db: Session) -> None:
    k = FakeLedgerKite()
    k._positions = {"net": [
        {"exchange": "NSE", "tradingsymbol": "INFY", "quantity": 5, "product": "CNC", "average_price": 1500.5},
        {"exchange": "NSE", "tradingsymbol": "TCS", "quantity": 0, "product": "MIS", "average_price": 1.0},
        {"exchange": "NFO", "tradingsymbol": "X", "quantity": 1, "product": "NRML", "average_price": 1.0},
    ]}  # fmt: skip
    k._holdings = [
        {"exchange": "NSE", "tradingsymbol": "WIPRO", "isin": "INE075A01022", "quantity": 8, "t1_quantity": 2, "average_price": 250.0},
        {"exchange": "NSE", "tradingsymbol": "ZERO", "quantity": 0, "average_price": 1.0},
        {"exchange": "BSE", "tradingsymbol": "BSEONLY", "quantity": 1, "average_price": 1.0},
    ]  # fmt: skip
    r = sync.sync(db, k, DAY, "a")
    assert r.snapshots == 2
    snaps = {s.symbol: s for s in db.scalars(select(PositionSnapshot))}
    assert snaps["INFY"].avg_price_paise == 150050 and snaps["INFY"].kind == "position"
    assert snaps["WIPRO"].qty == 10 and snaps["WIPRO"].kind == "holding"


def test_timestamp_parsing_accepts_iso_strings_and_aware_values() -> None:
    assert sync._ts("2026-03-02T10:15:00") == datetime(2026, 3, 2, 4, 45, tzinfo=UTC)
    assert sync._ts(datetime(2026, 3, 2, 4, 45, tzinfo=UTC)) == datetime(
        2026, 3, 2, 4, 45, tzinfo=UTC
    )
