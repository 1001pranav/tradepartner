"""PaperBroker: a simulated exchange that fills against supplied quotes (spec 16-M3).

It owns persistence of its orders, order events and fills (``origin = 'paper'``,
``source = 'paper'``) so a paper day produces a real ledger. GTT OCO protection is simulated and
fires against quotes like the real thing.
"""

from collections.abc import Callable
from datetime import UTC, date, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from tradepartner.calendar import IST
from tradepartner.db.models import Fill, Order, OrderEvent
from tradepartner.execution.broker import BrokerOrder, GttInfo, UnsupportedOrderError
from tradepartner.risk.approved import ApprovedOrder
from tradepartner.risk.types import Quote

QuoteFn = Callable[[str], Quote | None]


def _slip_up(price: int, bps: float) -> int:
    return price + -(-price * int(bps * 100) // 1_000_000)  # ceil


def _slip_down(price: int, bps: float) -> int:
    return price - price * int(bps * 100) // 1_000_000  # floor


class PaperBroker:
    def __init__(
        self,
        session: Session,
        quotes: QuoteFn,
        slippage_bps: float = 10.0,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.session = session
        self.quotes = quotes
        self.slippage_bps = slippage_bps
        self.clock = clock
        self._gtts: dict[str, dict[str, int | str]] = {}
        self._gtt_seq = 0

    # ---- helpers ----

    def _next_id(self) -> str:
        n = len(list(self.session.scalars(select(Order.id).where(Order.origin == "paper"))))
        return f"paper-{n + 1}"

    def _view(self, o: Order) -> BrokerOrder:
        filled = sum(f.qty for f in self.session.scalars(select(Fill).where(Fill.order_id == o.id)))
        value = sum(
            f.qty * f.price_paise
            for f in self.session.scalars(select(Fill).where(Fill.order_id == o.id))
        )
        return BrokerOrder(
            o.broker_order_id or "", o.symbol, o.side, o.product, o.order_type, o.qty, filled,
            o.price_paise, o.trigger_paise, value // filled if filled else None, o.status, o.tag,
        )  # fmt: skip

    def _event(self, o: Order, status: str, filled: int, avg: int | None, msg: str) -> None:
        self.session.add(
            OrderEvent(order_id=o.id, status=status, filled_qty=filled, avg_price_paise=avg,
                       message=msg, at=self.clock())
        )  # fmt: skip

    def _fill(self, o: Order, price: int, why: str) -> None:
        now = self.clock()
        n = len(list(self.session.scalars(select(Fill.id).where(Fill.order_id == o.id))))
        self.session.add(
            Fill(
                broker_trade_id=f"{o.broker_order_id}-t{n + 1}",
                broker_order_id=o.broker_order_id,
                order_id=o.id,
                origin="paper",
                account="paper",
                symbol=o.symbol,
                isin=None,
                exchange="NSE",
                side=o.side,
                product=o.product,
                qty=o.qty,
                price_paise=price,
                filled_at=now,
                trade_date=now.astimezone(IST).date(),
                source="paper",
            )
        )
        o.status = "COMPLETE"
        self._event(o, "COMPLETE", o.qty, price, why)

    def _marketable_price(self, o: Order, ltp: int) -> int | None:
        """Fill price if the order can trade at ``ltp`` now, else None."""
        if o.order_type == "MARKET":
            return (
                _slip_up(ltp, self.slippage_bps)
                if o.side == "BUY"
                else _slip_down(ltp, self.slippage_bps)
            )
        if o.order_type == "SL-M":
            hit = (
                ltp <= (o.trigger_paise or 0) if o.side == "SELL" else ltp >= (o.trigger_paise or 0)
            )
            if not hit:
                return None
            return (
                _slip_down(ltp, self.slippage_bps)
                if o.side == "SELL"
                else _slip_up(ltp, self.slippage_bps)
            )
        limit = o.price_paise or 0
        if o.order_type == "SL":
            hit = (
                ltp <= (o.trigger_paise or 0) if o.side == "SELL" else ltp >= (o.trigger_paise or 0)
            )
            if not hit:
                return None
        if o.side == "BUY":
            return min(limit, _slip_up(ltp, self.slippage_bps)) if ltp <= limit else None
        return max(limit, _slip_down(ltp, self.slippage_bps)) if ltp >= limit else None

    def _try_fill(self, o: Order) -> None:
        q = self.quotes(o.symbol)
        if q is None or o.status != "OPEN":
            return
        price = self._marketable_price(o, q.ltp_paise)
        if price is not None:
            self._fill(o, price, "paper fill")

    # ---- Broker protocol ----

    def place_order(self, order: ApprovedOrder) -> BrokerOrder:
        if not isinstance(order, ApprovedOrder):
            raise TypeError("only a RiskEngine-approved order can be placed")
        if order.product not in ("CNC", "MIS"):
            raise UnsupportedOrderError(f"unsupported product {order.product}")
        existing = (
            self.session.scalar(
                select(Order).where(Order.tag == order.tag, Order.origin == "paper")
            )
            if order.tag
            else None
        )
        if existing is not None:
            return self._view(existing)  # idempotent: the same proposal is never placed twice
        o = Order(
            proposal_id=order.proposal_id,
            broker_order_id=self._next_id(),
            origin="paper",
            symbol=order.symbol,
            side=order.side,
            product=order.product,
            order_type=order.order_type,
            qty=order.qty,
            price_paise=order.limit_price_paise,
            trigger_paise=order.trigger_price_paise,
            tag=order.tag,
            placed_at=self.clock(),
            status="OPEN",
        )
        self.session.add(o)
        self.session.flush()
        self._event(o, "OPEN", 0, None, "accepted")
        self._try_fill(o)
        self.session.commit()
        return self._view(o)

    def cancel_order(self, broker_order_id: str) -> None:
        o = self.session.scalar(select(Order).where(Order.broker_order_id == broker_order_id))
        if o is not None and o.status == "OPEN":
            o.status = "CANCELLED"
            self._event(o, "CANCELLED", 0, None, "cancelled")
            self.session.commit()

    def open_orders(self) -> list[BrokerOrder]:
        rows = self.session.scalars(
            select(Order).where(Order.origin == "paper", Order.status == "OPEN").order_by(Order.id)
        )
        return [self._view(o) for o in rows]

    def get_order(self, broker_order_id: str) -> BrokerOrder | None:
        o = self.session.scalar(select(Order).where(Order.broker_order_id == broker_order_id))
        return self._view(o) if o else None

    def on_quote(self, symbol: str) -> None:
        """Advance the simulation: fill resting orders and fire GTT stops/targets."""
        for o in list(
            self.session.scalars(
                select(Order).where(
                    Order.origin == "paper", Order.status == "OPEN", Order.symbol == symbol
                )
            )
        ):
            self._try_fill(o)
        q = self.quotes(symbol)
        if q is not None:
            for gid, g in list(self._gtts.items()):
                hit = q.ltp_paise <= int(g["stop"]) or q.ltp_paise >= int(g["target"])
                if g["symbol"] == symbol and g["status"] == "active" and hit:
                    self._trigger_gtt(gid, q.ltp_paise)
        self.session.commit()

    # ---- simulated GTT OCO ----

    def place_gtt_oco(
        self, symbol: str, qty: int, last_price_paise: int, stop_paise: int, target_paise: int
    ) -> str:
        if not (0 < stop_paise < last_price_paise < target_paise):
            raise ValueError("GTT needs stop < last price < target")
        self._gtt_seq += 1
        gid = f"gtt-{self._gtt_seq}"
        self._gtts[gid] = {"symbol": symbol, "qty": qty, "stop": stop_paise,
                           "target": target_paise, "status": "active"}  # fmt: skip
        return gid

    def modify_gtt_oco(
        self, gtt_id: str, qty: int, last_price_paise: int, stop_paise: int, target_paise: int
    ) -> None:
        g = self._gtts[gtt_id]
        g.update(qty=qty, stop=stop_paise, target=target_paise)

    def list_gtts(self) -> list[GttInfo]:
        return [
            GttInfo(
                gid,
                str(g["symbol"]),
                int(g["qty"]),
                int(g["stop"]),
                int(g["target"]),
                str(g["status"]),
            )
            for gid, g in self._gtts.items()
        ]

    def delete_gtt(self, gtt_id: str) -> None:
        if gtt_id in self._gtts:
            self._gtts[gtt_id]["status"] = "cancelled"

    def _trigger_gtt(self, gid: str, ltp: int) -> None:
        g = self._gtts[gid]
        g["status"] = "triggered"
        o = Order(
            proposal_id=None, broker_order_id=self._next_id(), origin="paper",
            symbol=str(g["symbol"]), side="SELL", product="CNC", order_type="MARKET",
            qty=int(g["qty"]), price_paise=None, trigger_paise=None, tag=f"gtt-{gid}"[:20],
            placed_at=self.clock(), status="OPEN",
        )  # fmt: skip
        self.session.add(o)
        self.session.flush()
        self._event(o, "OPEN", 0, None, f"GTT {gid} triggered")
        self._fill(o, _slip_down(ltp, self.slippage_bps), f"GTT {gid} triggered")


def today_ist(clock: Callable[[], datetime]) -> date:
    return clock().astimezone(IST).date()
