"""A fake Kite orders/GTT API that RECORDS every call. Tests never touch a real broker."""

from typing import Any


class FakeKiteApi:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._orders: list[dict[str, Any]] = []
        self._gtts: list[dict[str, Any]] = []
        self.fail_with: Exception | None = None
        self._next = 1000

    def _record(self, name: str, **kw: Any) -> None:
        self.calls.append((name, kw))
        if self.fail_with is not None:
            raise self.fail_with

    def place_order(self, **kw: Any) -> str:
        self._record("place_order", **kw)
        self._next += 1
        self._orders.append(
            {
                "order_id": str(self._next),
                "tradingsymbol": kw["tradingsymbol"],
                "transaction_type": kw["transaction_type"],
                "product": kw["product"],
                "order_type": kw["order_type"],
                "quantity": kw["quantity"],
                "filled_quantity": 0,
                "price": kw.get("price"),
                "trigger_price": kw.get("trigger_price"),
                "average_price": 0,
                "status": "OPEN",
                "tag": kw.get("tag"),
            }
        )
        return str(self._next)

    def cancel_order(self, variety: str, order_id: str, parent_order_id: str | None = None) -> str:
        self._record("cancel_order", variety=variety, order_id=order_id)
        for o in self._orders:
            if o["order_id"] == order_id:
                o["status"] = "CANCELLED"
        return order_id

    def orders(self) -> list[dict[str, Any]]:
        return list(self._orders)

    def order_history(self, order_id: str) -> list[dict[str, Any]]:
        return [o for o in self._orders if o["order_id"] == order_id]

    def trades(self) -> list[dict[str, Any]]:
        return []

    def place_gtt(self, **kw: Any) -> dict[str, Any]:
        self._record("place_gtt", **kw)
        gid = 5000 + len(self._gtts)
        self._gtts.append(
            {
                "id": gid,
                "status": "active",
                "condition": {
                    "tradingsymbol": kw["tradingsymbol"],
                    "trigger_values": kw["trigger_values"],
                    "last_price": kw["last_price"],
                },
                "orders": kw["orders"],
            }
        )
        return {"trigger_id": gid}

    def modify_gtt(self, **kw: Any) -> dict[str, Any]:
        self._record("modify_gtt", **kw)
        for g in self._gtts:
            if g["id"] == kw["trigger_id"]:
                g["condition"]["trigger_values"] = kw["trigger_values"]
                g["orders"] = kw["orders"]
        return {"trigger_id": kw["trigger_id"]}

    def get_gtts(self) -> list[dict[str, Any]]:
        return list(self._gtts)

    def delete_gtt(self, trigger_id: int) -> dict[str, Any]:
        self._record("delete_gtt", trigger_id=trigger_id)
        for g in self._gtts:
            if g["id"] == trigger_id:
                g["status"] = "deleted"
        return {"trigger_id": trigger_id}
