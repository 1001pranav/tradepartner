"""Broker protocol. Only ``execution.router`` may call ``place_order`` (CLAUDE.md rule 1)."""

from dataclasses import dataclass
from typing import Protocol

from tradepartner.risk.approved import ApprovedOrder


@dataclass(frozen=True)
class BrokerOrder:
    broker_order_id: str
    symbol: str
    side: str
    product: str
    order_type: str
    qty: int
    filled_qty: int
    price_paise: int | None
    trigger_paise: int | None
    avg_price_paise: int | None
    status: str  # OPEN | COMPLETE | CANCELLED | REJECTED
    tag: str | None


@dataclass(frozen=True)
class GttInfo:
    gtt_id: str
    symbol: str
    qty: int
    stop_paise: int
    target_paise: int
    status: str  # active | triggered | cancelled | expired


class UnsupportedOrderError(Exception):
    """Raised when asked to trade anything but NSE cash equity (CNC/MIS)."""


class Broker(Protocol):
    def place_order(self, order: ApprovedOrder) -> BrokerOrder: ...
    def cancel_order(self, broker_order_id: str) -> None: ...
    def open_orders(self) -> list[BrokerOrder]: ...
    def get_order(self, broker_order_id: str) -> BrokerOrder | None: ...
    def place_gtt_oco(
        self, symbol: str, qty: int, last_price_paise: int, stop_paise: int, target_paise: int
    ) -> str: ...
    def modify_gtt_oco(
        self, gtt_id: str, qty: int, last_price_paise: int, stop_paise: int, target_paise: int
    ) -> None: ...
    def list_gtts(self) -> list[GttInfo]: ...
    def delete_gtt(self, gtt_id: str) -> None: ...
