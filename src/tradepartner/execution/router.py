"""The ONLY caller of ``Broker.place_order`` (spec 3, CLAUDE.md rule 1).

Flow: persist the proposal (idempotent) -> ``RiskEngine.check`` -> record the decision -> apply
halts -> by mode: OFF logs, MANUAL shows, AUTO places. Protective exits run in every mode.
Strategy, model, LLM and signal code must never import this module (import-linter).
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from tradepartner.db import audit
from tradepartner.db.models import Order, OrderProposal, Signal
from tradepartner.execution.broker import Broker, BrokerOrder
from tradepartner.ledger.sync import origin_for
from tradepartner.modes.state import ModeReader
from tradepartner.risk.approved import ApprovedOrder
from tradepartner.risk.engine import RiskEngine, record_blocks
from tradepartner.risk.types import Context, Proposal, RiskDecision, RuleResult

Status = Literal["duplicate", "rejected", "logged_off", "shown", "placed"]
ContextFactory = Callable[[str], Context]  # strategy -> fresh context
HaltEnforcer = Callable[[Iterable[RuleResult], datetime], list[str]]


@dataclass
class SubmitResult:
    status: Status
    proposal_id: int | None = None
    decision: RiskDecision | None = None
    order: BrokerOrder | None = None
    halted: list[str] = field(default_factory=list)


class Router:
    def __init__(
        self,
        session: Session,
        engine: RiskEngine,
        modes: ModeReader,
        broker: Broker,
        context_factory: ContextFactory,
        halt_enforcer: HaltEnforcer,
        manual_execution: str = "kite_app",
    ) -> None:
        self.session = session
        self.engine = engine
        self.modes = modes
        self.broker = broker
        self.context_factory = context_factory
        self.halt_enforcer = halt_enforcer
        self.manual_execution = manual_execution

    # The one and only call site of Broker.place_order.
    def _place(self, order: ApprovedOrder, now: datetime | None = None) -> BrokerOrder:
        if not isinstance(order, ApprovedOrder):
            raise TypeError("router only places ApprovedOrder")
        placed = self.broker.place_order(order)
        self._record_order(order, placed, now or datetime.now(UTC))
        return placed

    def _record_order(self, order: ApprovedOrder, placed: BrokerOrder, now: datetime) -> None:
        """Write the Order row at placement so limits see it before the 30 s broker sync."""
        if self.session.scalar(
            select(Order.id).where(Order.broker_order_id == placed.broker_order_id)
        ):
            return
        origin, proposal_id = origin_for(self.session, order.tag)
        self.session.add(
            Order(
                proposal_id=proposal_id or order.proposal_id,
                broker_order_id=placed.broker_order_id,
                origin=origin,
                symbol=placed.symbol,
                side=placed.side,
                product=placed.product,
                order_type=placed.order_type,
                qty=placed.qty,
                price_paise=placed.price_paise,
                trigger_paise=placed.trigger_paise,
                tag=placed.tag,
                placed_at=now,
                status=placed.status,
            )
        )
        self.session.commit()

    def place_emergency(self, order: ApprovedOrder) -> BrokerOrder:
        """For the kill switch: the order came from ``approve_emergency_exit``."""
        audit.log(
            self.session,
            "system",
            "emergency_order",
            {"symbol": order.symbol, "qty": order.qty, "type": order.order_type},
            None,
        )
        self.session.commit()
        return self._place(order)

    def submit(self, p: Proposal, now: datetime) -> SubmitResult:
        ctx = self.context_factory(p.strategy)
        duplicate = self.session.scalar(
            select(OrderProposal.id).where(OrderProposal.idempotency_key == p.idempotency_key)
        )
        if duplicate is not None:
            record_blocks(
                self.session,
                [RuleResult("duplicate", False, "block", "idempotency key already submitted")],
                f"strategy:{p.strategy}",
                now,
            )
            return SubmitResult("duplicate", duplicate)

        row = OrderProposal(
            signal_id=p.signal_id,
            symbol=p.symbol,
            side=p.side,
            product=p.product,
            order_type=p.order_type,
            qty=p.exit_qty or 0,
            limit_price_paise=p.limit_price_paise,
            trigger_price_paise=p.trigger_price_paise,
            stop_loss_paise=p.stop_loss_paise,
            target_paise=p.target_paise,
            time_stop_date=p.time_stop_date,
            mode_at_creation=ctx.mode,
            idempotency_key=p.idempotency_key,
            strategy=p.strategy,
            kind=p.kind,
            exit_reason=p.exit_reason,
            max_qty=p.max_qty,
            created_at=now,
        )
        self.session.add(row)
        self.session.flush()
        p = replace(p, proposal_id=row.id)

        decision = self.engine.check(p, ctx)
        row.qty = decision.sizing.qty
        row.risk_amount_paise = decision.sizing.risk_amount_paise
        row.risk_decision = "approved" if decision.approved else "rejected"
        row.risk_reasons_json = list(decision.reasons)
        self.session.commit()

        record_blocks(
            self.session,
            [r for r in decision.failures if r.severity == "block"],
            f"strategy:{p.strategy}",
            now,
        )
        halted = self.halt_enforcer(decision.halts, now)
        if not decision.approved or decision.order is None:
            return SubmitResult("rejected", row.id, decision, None, halted)

        protective_exit = p.kind == "EXIT" and p.protective
        mode = self.modes.effective(p.strategy)
        if protective_exit or mode == "AUTO":
            placed = self._place(decision.order, now)
            return SubmitResult("placed", row.id, decision, placed, halted)
        if mode == "MANUAL":
            return SubmitResult("shown", row.id, decision, None, halted)
        return SubmitResult("logged_off", row.id, decision, None, halted)  # unreachable via rules

    def approve_and_place(self, proposal_id: int, now: datetime) -> SubmitResult:
        """MANUAL ``api_approve`` mode: the user's click sends a risk-approved proposal via the API.

        The proposal is re-checked against fresh data (so a stale or no-longer-safe order is
        refused) and is never placed twice. Requires ``manual_execution: api_approve`` and a
        registered static IP (the broker rejects the call otherwise).
        """
        if self.manual_execution != "api_approve":
            raise PermissionError("manual_execution is 'kite_app': place the order in the Kite app")
        row = self.session.get(OrderProposal, proposal_id)
        if row is None or row.risk_decision != "approved":
            raise ValueError("only a risk-approved proposal can be approved")
        if (
            self.session.scalar(select(Order.id).where(Order.proposal_id == proposal_id))
            is not None
        ):
            raise ValueError("this proposal has already been placed")
        signal = self.session.get(Signal, row.signal_id) if row.signal_id else None
        p = Proposal(
            strategy=row.strategy or "manual",
            symbol=row.symbol,
            side=row.side,
            product=row.product,
            order_type=row.order_type,
            kind="ENTRY" if row.kind in (None, "ENTRY") else "EXIT",
            signal_time=signal.created_at if signal else row.created_at,
            idempotency_key=f"{row.idempotency_key}#approve",
            limit_price_paise=row.limit_price_paise,
            trigger_price_paise=row.trigger_price_paise,
            stop_loss_paise=row.stop_loss_paise,
            target_paise=row.target_paise,
            exit_qty=row.qty if row.kind == "EXIT" else None,
            signal_id=row.signal_id,
            proposal_id=row.id,
            max_qty=row.max_qty,
        )
        ctx = self.context_factory(p.strategy)
        decision = self.engine.check(p, ctx)
        row.risk_decision = "approved" if decision.approved else "rejected"
        row.risk_reasons_json = list(decision.reasons)
        self.session.commit()
        halted = self.halt_enforcer(decision.halts, now)
        if not decision.approved or decision.order is None:
            record_blocks(self.session, decision.failures, f"strategy:{p.strategy}", now)
            return SubmitResult("rejected", row.id, decision, None, halted)
        return SubmitResult("placed", row.id, decision, self._place(decision.order, now), halted)
