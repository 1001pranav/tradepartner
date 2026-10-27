"""RiskEngine: the single gate every order passes through (spec 7.2).

``check`` is a pure function of ``(proposal, context)``. It runs *all* rules and collects every
failure. It is also the only place an ``ApprovedOrder`` is minted. Applying halts (which mutate
mode state) is a separate, explicit step: ``enforce_halts``.
"""

import hashlib
from collections.abc import Callable, Iterable
from datetime import datetime

from sqlalchemy.orm import Session

from tradepartner.db import audit
from tradepartner.db.models import RiskEvent
from tradepartner.modes.controller import ModeController
from tradepartner.risk import _mint, rules
from tradepartner.risk.approved import ApprovedOrder
from tradepartner.risk.sizing import compute_sizing
from tradepartner.risk.types import (
    Context,
    Proposal,
    RiskDecision,
    RuleResult,
    Sizing,
)
from tradepartner.settings import RiskConfig

Rule = Callable[[Proposal, Context, RiskConfig, Sizing], RuleResult]


def _tag(p: Proposal) -> str | None:
    """Broker tag (max 20 chars). Proposals use ``tp-<id>``; emergency exits ``tpk-<hash>``."""
    if p.proposal_id is not None:
        return f"tp-{p.proposal_id}"
    if p.emergency:
        return "tpk-" + hashlib.sha256(p.idempotency_key.encode()).hexdigest()[:12]
    return None


def _mint_order(p: Proposal, sz: Sizing, cfg: RiskConfig) -> ApprovedOrder:
    is_market = p.order_type == "MARKET"
    return ApprovedOrder(
        strategy=p.strategy,
        symbol=p.symbol,
        side=p.side,
        product=p.product,
        order_type=p.order_type,
        qty=sz.qty,
        limit_price_paise=p.limit_price_paise,
        trigger_price_paise=p.trigger_price_paise,
        stop_loss_paise=p.stop_loss_paise,
        target_paise=p.target_paise,
        time_stop_date=p.time_stop_date,
        risk_amount_paise=sz.risk_amount_paise,
        market_protection_pct=cfg.market_protection_pct if is_market else None,
        kind=p.kind,
        idempotency_key=p.idempotency_key,
        tag=_tag(p),
        proposal_id=p.proposal_id,
        _token=_mint.TOKEN,
    )


class RiskEngine:
    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg

    def check(self, p: Proposal, ctx: Context) -> RiskDecision:
        sizing = compute_sizing(p, ctx, self.cfg)
        rule_set: Iterable[Rule] = rules.ENTRY_RULES if p.kind == "ENTRY" else rules.EXIT_RULES
        results = tuple(rule(p, ctx, self.cfg, sizing) for rule in rule_set)
        approved = all(r.passed for r in results)
        order = _mint_order(p, sizing, self.cfg) if approved else None
        return RiskDecision(approved, results, sizing, order)

    def check_halts(self, ctx: Context, strategies: Iterable[str]) -> list[RuleResult]:
        """Evaluated by the 30-second tracker: loss limits independent of any proposal."""
        probe = Proposal("-", "-", "BUY", "CNC", "LIMIT", "ENTRY", ctx.now, "-")
        out = [
            r
            for r in (rule(probe, ctx, self.cfg, Sizing()) for rule in rules.HALT_RULES[:3])
            if not r.passed
        ]
        for name in strategies:
            p = Proposal(name, "-", "BUY", "CNC", "LIMIT", "ENTRY", ctx.now, "-")
            r = rules.rule_consecutive_losses(p, ctx, self.cfg, Sizing())
            if not r.passed:
                out.append(r)
        return out

    def approve_emergency_exit(self, p: Proposal, ctx: Context) -> RiskDecision:
        """Kill-switch exits: only structural checks apply; mode/kill/time rules do not block."""
        if not (p.emergency and p.kind == "EXIT"):
            raise ValueError("only emergency exits use this path")
        sizing = compute_sizing(p, ctx, self.cfg)
        results = tuple(rule(p, ctx, self.cfg, sizing) for rule in rules.EMERGENCY_RULES)
        approved = all(r.passed for r in results)
        order = _mint_order(p, sizing, self.cfg) if approved else None
        return RiskDecision(approved, results, sizing, order)


def enforce_halts(
    session: Session,
    controller: ModeController,
    results: Iterable[RuleResult],
    now: datetime,
) -> list[str]:
    """Record every ``halt`` result and force the affected scope OFF (spec 7.1 rule 3).

    Returns the scopes that were actually switched off.
    """
    switched: list[str] = []
    for r in results:
        if r.passed or r.severity != "halt":
            continue
        if controller.halt(r.scope, f"{r.rule}: {r.message}", now):
            session.add(
                RiskEvent(
                    at=now,
                    rule=r.rule,
                    severity="halt",
                    scope=r.scope,
                    details_json={"message": r.message},
                )
            )
            audit.log(session, "system", "risk_halt", {"rule": r.rule, "scope": r.scope}, now)
            session.commit()
            switched.append(r.scope)
    return switched


def record_blocks(
    session: Session, results: Iterable[RuleResult], scope: str, now: datetime
) -> int:
    """Persist ``block`` failures as risk events (every halt is recorded by ``enforce_halts``)."""
    n = 0
    for r in results:
        if not r.passed and r.severity == "block":
            session.add(
                RiskEvent(at=now, rule=r.rule, severity="block", scope=scope,
                          details_json={"message": r.message})
            )  # fmt: skip
            n += 1
    session.commit()
    return n
