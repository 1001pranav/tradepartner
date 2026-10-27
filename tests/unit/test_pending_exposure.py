"""Finding 1: resting and not-yet-synced entries must count against portfolio limits."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from tests.fixtures.risk import replace_cfg
from tests.fixtures.world import World
from tradepartner.db.models import Order
from tradepartner.execution.broker import BrokerOrder
from tradepartner.risk.engine import RiskEngine
from tradepartner.risk.limits import compute_limit_state


def _resting(w: World, symbol: str, key: str):  # type: ignore[no-untyped-def]
    px = w.quotes[symbol].ltp_paise
    return w.submit_entry(symbol, key, limit_price_paise=px * 99 // 100)  # below LTP: rests


def test_resting_entry_counts_toward_max_open_positions(db: Session) -> None:
    w = World(db)
    w.risk_cfg = replace_cfg(w.risk_cfg, portfolio={"max_open_positions": 1})
    w.engine = RiskEngine(w.risk_cfg)
    w.router.engine = w.engine
    w.set_price("TCS", 400_000)
    w.instruments["TCS"] = w.instruments["INFY"].__class__("TCS", "IT2", 200_000_000 * 100)
    w.go_auto()
    first = _resting(w, "INFY", "a")
    assert first.status == "placed" and first.order and first.order.status == "OPEN"
    second = _resting(w, "TCS", "b")
    assert second.status == "rejected"
    assert any("open_positions" in r for r in second.decision.reasons)  # type: ignore[union-attr]


def test_router_records_order_when_broker_does_not(db: Session) -> None:
    """KiteBroker writes no Order row at placement; the router must, so the daily cap sees it."""
    w = World(db)

    class Silent:
        def place_order(self, order):  # type: ignore[no-untyped-def]
            return BrokerOrder(
                "K1", order.symbol, order.side, order.product, order.order_type, order.qty,
                0, order.limit_price_paise, None, None, "OPEN", order.tag,
            )  # fmt: skip

    w.router.broker = Silent()  # type: ignore[assignment]
    w.go_auto()
    res = _resting(w, "INFY", "a")
    assert res.status == "placed"
    row = db.scalar(select(Order).where(Order.broker_order_id == "K1"))
    assert row is not None and row.origin == "bot_auto"
    state = compute_limit_state(db, w.risk_cfg, w.quotes, w.now)
    assert state.orders_today == 1


def test_pending_entry_is_the_unfilled_remainder_only(db: Session) -> None:
    from tradepartner.db.models import Fill
    from tradepartner.risk.context import pending_entries

    w = World(db)
    w.go_auto()
    _resting(w, "INFY", "a")
    order = db.scalars(select(Order)).one()
    assert [p.qty for p in pending_entries(db)] == [order.qty]

    def fill(qty: int, n: int) -> None:
        db.add(
            Fill(
                broker_trade_id=f"t{n}", broker_order_id=order.broker_order_id, order_id=order.id,
                origin="paper", account="paper", symbol="INFY", isin=None, exchange="NSE",
                side="BUY", product=order.product, qty=qty, price_paise=1, filled_at=w.now,
                trade_date=w.now.date(), source="paper",
            )
        )  # fmt: skip
        db.flush()

    fill(order.qty - 1, 1)  # partly filled: only the remainder is still pending
    assert [p.qty for p in pending_entries(db)] == [1]
    fill(1, 2)  # fully filled but status not yet synced: nothing pending
    assert pending_entries(db) == ()
