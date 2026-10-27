"""``ApprovedOrder``: the only thing ``Broker.place_order`` accepts."""

from dataclasses import dataclass, field
from datetime import date

from tradepartner.risk import _mint


@dataclass(frozen=True)
class ApprovedOrder:
    """Created only by ``RiskEngine`` (via ``risk.engine``). Direct construction raises."""

    strategy: str
    symbol: str
    side: str  # BUY | SELL
    product: str  # CNC | MIS
    order_type: str  # LIMIT | SL | SL-M | MARKET
    qty: int
    limit_price_paise: int | None
    trigger_price_paise: int | None
    stop_loss_paise: int | None
    target_paise: int | None
    time_stop_date: date | None
    risk_amount_paise: int
    market_protection_pct: float | None
    kind: str  # ENTRY | EXIT
    idempotency_key: str
    tag: str | None
    proposal_id: int | None
    _token: object = field(repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if self._token is not _mint.TOKEN:
            raise TypeError("ApprovedOrder can only be created by the RiskEngine")
        if self.qty < 1:
            raise ValueError("an approved order must have qty >= 1")
        if self.order_type == "MARKET" and not self.market_protection_pct:
            raise ValueError("market orders must carry market protection")
        if self.tag is not None and len(self.tag) > 20:
            raise ValueError("broker tags are limited to 20 characters")
