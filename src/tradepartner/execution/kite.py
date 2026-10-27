"""KiteBroker: the live Zerodha adapter (spec 2, 7.3, 16-M8).

Safety properties, all enforced here and tested:
* ``place_order`` only accepts an ``ApprovedOrder`` (minted by the risk engine) and is only ever
  called by ``execution.router``.
* Only NSE cash equity (CNC / MIS); anything else raises ``UnsupportedOrderError``.
* Limit orders by default; market orders must carry a non-zero market protection.
* One order per second (token bucket), tags <= 20 characters, no duplicate placement for a tag.
* Protective orders: GTT OCO for CNC. A GTT can only SELL, only up to the position.

Live-order paths are exercised only by Pranav, with 1-share orders (CLAUDE.md rule 10).
"""

from collections.abc import Callable
from decimal import Decimal
from typing import Any, Protocol

from tradepartner.data.candles import RateLimiter
from tradepartner.execution.broker import BrokerOrder, GttInfo, UnsupportedOrderError
from tradepartner.risk.approved import ApprovedOrder

VARIETY = "regular"
EXCHANGE = "NSE"
VALIDITY = "DAY"
GTT_OCO = "two-leg"
OPEN_STATUSES = {
    "OPEN",
    "TRIGGER PENDING",
    "PUT ORDER REQ RECEIVED",
    "VALIDATION PENDING",
    "OPEN PENDING",
}
STOP_LEG_BUFFER_PCT = Decimal("1.0")  # stop leg sells as a limit this far below its trigger


class BrokerError(Exception):
    pass


class StaticIpError(BrokerError):
    """The API rejected the call because the source IP is not registered (SEBI algo rules)."""


class KiteOrdersApi(Protocol):
    def place_order(self, **kwargs: Any) -> str: ...
    def cancel_order(
        self, variety: str, order_id: str, parent_order_id: str | None = None
    ) -> str: ...
    def orders(self) -> list[dict[str, Any]]: ...
    def order_history(self, order_id: str) -> list[dict[str, Any]]: ...
    def trades(self) -> list[dict[str, Any]]: ...
    def place_gtt(self, **kwargs: Any) -> dict[str, Any]: ...
    def modify_gtt(self, **kwargs: Any) -> dict[str, Any]: ...
    def get_gtts(self) -> list[dict[str, Any]]: ...
    def delete_gtt(self, trigger_id: int) -> dict[str, Any]: ...


def rupees(paise: int) -> float:
    return float(Decimal(paise) / 100)


def tick_down(paise: int, tick: int = 5) -> int:
    return paise // tick * tick


def _view(o: dict[str, Any]) -> BrokerOrder:
    price, trig = o.get("price") or 0, o.get("trigger_price") or 0
    avg = o.get("average_price") or 0
    return BrokerOrder(
        broker_order_id=str(o["order_id"]),
        symbol=o["tradingsymbol"],
        side=o["transaction_type"],
        product=o["product"],
        order_type=o["order_type"],
        qty=int(o["quantity"]),
        filled_qty=int(o.get("filled_quantity") or 0),
        price_paise=round(price * 100) or None,
        trigger_paise=round(trig * 100) or None,
        avg_price_paise=round(avg * 100) or None,
        status="OPEN" if str(o["status"]).upper() in OPEN_STATUSES else str(o["status"]).upper(),
        tag=o.get("tag"),
    )


class KiteBroker:
    def __init__(
        self,
        api: KiteOrdersApi,
        limiter: RateLimiter | None = None,
        on_error: Callable[[str], None] = lambda message: None,
    ) -> None:
        self.api = api
        self._limiter = limiter or RateLimiter(1.0)  # hard cap: 1 order per second (spec 2.1)
        self._on_error = on_error

    # ---- error mapping ----

    def _call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            text = str(exc)
            if "ip" in text.lower() and (
                "not allowed" in text.lower()
                or "whitelist" in text.lower()
                or "static" in text.lower()
                or "registered" in text.lower()
            ):
                self._on_error(
                    "Order rejected: source IP is not registered with Zerodha (static IP)."
                )
                raise StaticIpError(text) from exc
            self._on_error(f"Broker call failed: {text}")
            raise BrokerError(text) from exc

    # ---- Broker protocol ----

    def place_order(self, order: ApprovedOrder) -> BrokerOrder:
        if not isinstance(order, ApprovedOrder):
            raise TypeError("only a RiskEngine-approved order can be placed")
        if order.product not in ("CNC", "MIS") or order.side not in ("BUY", "SELL"):
            raise UnsupportedOrderError(f"unsupported {order.product}/{order.side}")
        if order.order_type not in ("LIMIT", "SL", "SL-M", "MARKET"):
            raise UnsupportedOrderError(f"unsupported order type {order.order_type}")
        if order.tag is not None:
            existing = self._find_tag(order.tag)
            if existing is not None:
                return existing  # never place the same proposal twice, even after a crash
        self._limiter.acquire()
        kwargs: dict[str, Any] = {
            "variety": VARIETY,
            "exchange": EXCHANGE,
            "tradingsymbol": order.symbol,
            "transaction_type": order.side,
            "quantity": order.qty,
            "product": order.product,
            "order_type": order.order_type,
            "validity": VALIDITY,
            "tag": order.tag,
        }
        if order.limit_price_paise is not None:
            kwargs["price"] = rupees(order.limit_price_paise)
        if order.trigger_price_paise is not None:
            kwargs["trigger_price"] = rupees(order.trigger_price_paise)
        if order.order_type == "MARKET":
            kwargs["market_protection"] = order.market_protection_pct
        order_id = self._call(self.api.place_order, **kwargs)
        found = self.get_order(str(order_id))
        if found is not None:
            return found
        return BrokerOrder(
            str(order_id),
            order.symbol,
            order.side,
            order.product,
            order.order_type,
            order.qty,
            0,
            order.limit_price_paise,
            order.trigger_price_paise,
            None,
            "OPEN",
            order.tag,
        )

    def _find_tag(self, tag: str) -> BrokerOrder | None:
        for o in self._call(self.api.orders):
            if o.get("tag") == tag and str(o["status"]).upper() not in ("REJECTED", "CANCELLED"):
                return _view(o)
        return None

    def cancel_order(self, broker_order_id: str) -> None:
        self._call(self.api.cancel_order, VARIETY, broker_order_id)

    def open_orders(self) -> list[BrokerOrder]:
        return [
            _view(o)
            for o in self._call(self.api.orders)
            if str(o["status"]).upper() in OPEN_STATUSES
        ]

    def get_order(self, broker_order_id: str) -> BrokerOrder | None:
        for o in self._call(self.api.orders):
            if str(o["order_id"]) == broker_order_id:
                return _view(o)
        return None

    # ---- GTT OCO (laptop-off protection, spec 7.3) ----

    def _gtt_orders(self, symbol: str, qty: int, stop: int, target: int) -> list[dict[str, Any]]:
        if qty < 1:
            raise UnsupportedOrderError("GTT quantity must be at least 1")
        stop_leg = tick_down(int(Decimal(stop) * (100 - STOP_LEG_BUFFER_PCT) / 100))
        base = {
            "exchange": EXCHANGE,
            "tradingsymbol": symbol,
            "transaction_type": "SELL",
            "quantity": qty,
            "order_type": "LIMIT",
            "product": "CNC",
        }
        return [{**base, "price": rupees(stop_leg)}, {**base, "price": rupees(tick_down(target))}]

    def place_gtt_oco(
        self, symbol: str, qty: int, last_price_paise: int, stop_paise: int, target_paise: int
    ) -> str:
        if not (0 < stop_paise < last_price_paise < target_paise):
            raise ValueError("GTT needs stop < last price < target")
        res = self._call(
            self.api.place_gtt,
            trigger_type=GTT_OCO,
            tradingsymbol=symbol,
            exchange=EXCHANGE,
            trigger_values=[rupees(stop_paise), rupees(target_paise)],
            last_price=rupees(last_price_paise),
            orders=self._gtt_orders(symbol, qty, stop_paise, target_paise),
        )
        return str(res["trigger_id"])

    def modify_gtt_oco(
        self, gtt_id: str, qty: int, last_price_paise: int, stop_paise: int, target_paise: int
    ) -> None:
        current = next((g for g in self._call(self.api.get_gtts) if str(g["id"]) == gtt_id), None)
        if current is None:
            raise BrokerError(f"GTT {gtt_id} not found")
        symbol = current["condition"]["tradingsymbol"]
        self._call(
            self.api.modify_gtt,
            trigger_id=int(gtt_id),
            trigger_type=GTT_OCO,
            tradingsymbol=symbol,
            exchange=EXCHANGE,
            trigger_values=[rupees(stop_paise), rupees(target_paise)],
            last_price=rupees(last_price_paise),
            orders=self._gtt_orders(symbol, qty, stop_paise, target_paise),
        )

    def list_gtts(self) -> list[GttInfo]:
        out = []
        for g in self._call(self.api.get_gtts):
            tv = g["condition"]["trigger_values"]
            if len(tv) != 2:
                continue
            out.append(
                GttInfo(
                    str(g["id"]),
                    g["condition"]["tradingsymbol"],
                    int(g["orders"][0]["quantity"]),
                    round(min(tv) * 100),
                    round(max(tv) * 100),
                    str(g["status"]),
                )
            )
        return out

    def delete_gtt(self, gtt_id: str) -> None:
        self._call(self.api.delete_gtt, int(gtt_id))
