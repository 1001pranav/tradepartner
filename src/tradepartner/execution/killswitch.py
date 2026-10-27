"""Kill switch procedure (spec 7.1 rule 4).

1. global mode -> OFF            2. cancel open bot ENTRY orders
3. square off bot MIS positions with limit orders at LTP -/+ market protection, falling back to
   market orders (with protection) after 60 s     4. leave CNC holdings and their GTTs in place
5. notify Pranav.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

from sqlalchemy.orm import Session

from tradepartner.db import audit
from tradepartner.db.models import RiskEvent
from tradepartner.execution.broker import Broker, BrokerOrder
from tradepartner.execution.router import Router
from tradepartner.risk import kill
from tradepartner.risk.engine import RiskEngine
from tradepartner.risk.types import Context, Proposal

FALLBACK_AFTER = timedelta(seconds=60)


@dataclass
class Pending:
    broker_order_id: str
    symbol: str
    qty: int
    deadline: datetime
    escalated: bool = False


@dataclass
class KillReport:
    cancelled: list[str] = field(default_factory=list)
    squared_off: list[str] = field(default_factory=list)
    escalated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


class KillSwitch:
    def __init__(
        self,
        session: Session,
        router: Router,
        broker: Broker,
        engine: RiskEngine,
        context_factory: Callable[[], Context],
        set_global_off: Callable[[str, datetime], object],
        notify: Callable[[str, str], None],
        kill_path: Path | None = None,
        pending: list[Pending] | None = None,
    ) -> None:
        self.session = session
        self.router = router
        self.broker = broker
        self.engine = engine
        self.context_factory = context_factory
        self.set_global_off = set_global_off
        self.notify = notify
        self.kill_path = kill_path
        self.pending: list[Pending] = pending if pending is not None else []

    def _emergency(self, symbol: str, qty: int, now: datetime, order_type: str, price: int | None,
                   key: str) -> BrokerOrder | None:  # fmt: skip
        ctx = self.context_factory()
        p = Proposal(
            strategy="kill_switch", symbol=symbol, side="SELL", product="MIS",
            order_type=order_type,
            kind="EXIT", signal_time=now, idempotency_key=key, limit_price_paise=price,
            exit_qty=qty, exit_reason="kill", protective=True, emergency=True,
        )  # fmt: skip
        decision = self.engine.approve_emergency_exit(p, ctx)
        if not decision.approved or decision.order is None:
            return None
        return self.router.place_emergency(decision.order)

    def trigger(self, now: datetime, reason: str) -> KillReport:
        report = KillReport()
        kill.arm(self.kill_path)
        self.set_global_off(f"KILL: {reason}", now)
        for o in self.broker.open_orders():
            if o.tag and o.tag.startswith("tp-") and o.side == "BUY":
                self.broker.cancel_order(o.broker_order_id)
                report.cancelled.append(o.broker_order_id)
        ctx = self.context_factory()
        protection = Decimal(str(self.engine.cfg.market_protection_pct))
        for pos in ctx.positions:
            if pos.product != "MIS" or not pos.bot_created or pos.qty <= 0:
                continue
            q = ctx.quotes.get(pos.symbol)
            if q is None:
                report.skipped.append(pos.symbol)
                continue
            limit = int(Decimal(q.ltp_paise) * (100 - protection) / 100)
            placed = self._emergency(
                pos.symbol, pos.qty, now, "LIMIT", limit, f"kill-{now.isoformat()}-{pos.symbol}"
            )
            if placed is None:
                report.skipped.append(pos.symbol)
                continue
            report.squared_off.append(pos.symbol)
            self.pending.append(
                Pending(placed.broker_order_id, pos.symbol, pos.qty, now + FALLBACK_AFTER)
            )
        self.session.add(
            RiskEvent(at=now, rule="kill_switch", severity="halt", scope="global",
                      details_json={"reason": reason, **report.__dict__})
        )  # fmt: skip
        audit.log(self.session, "user", "kill_switch", {"reason": reason, **report.__dict__}, now)
        self.session.commit()
        self.notify(
            "TradePartner KILL SWITCH",
            f"{reason}. Cancelled {len(report.cancelled)} entries, squaring off "
            f"{len(report.squared_off)} MIS positions. CNC holdings and GTTs untouched.",
        )
        return report

    def tick(self, now: datetime) -> list[str]:
        """After 60 s, replace unfilled square-off limits with protected market orders."""
        escalated: list[str] = []
        for pend in self.pending:
            if pend.escalated or now < pend.deadline:
                continue
            current = self.broker.get_order(pend.broker_order_id)
            pend.escalated = True
            if current is None or current.status != "OPEN":
                continue
            self.broker.cancel_order(pend.broker_order_id)
            remaining = pend.qty - current.filled_qty
            key = f"kill-mkt-{now.isoformat()}-{pend.symbol}"
            if self._emergency(pend.symbol, remaining, now, "MARKET", None, key) is not None:
                escalated.append(pend.symbol)
        return escalated
