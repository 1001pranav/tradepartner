"""M8 live acceptance helpers: 1-SHARE orders only, exercised by Pranav, never by tests or agents
(CLAUDE.md rule 10). Everything goes through the normal risk engine, mode state and router.
"""

from datetime import datetime

from tradepartner.data.quotes import poll
from tradepartner.db.models import OrderProposal
from tradepartner.execution.protective import ProtectiveManager
from tradepartner.execution.router import SubmitResult
from tradepartner.risk.types import Proposal
from tradepartner.runtime import Runtime
from tradepartner.strategies.base import round_tick

STRATEGY = "live_test"
MAX_QTY = 1  # hard-coded: a live test never trades more than one share
STOP_PCT, TARGET_PCT = 0.03, 0.05


class LiveTestError(Exception):
    pass


def require_live(rt: Runtime) -> None:
    s = rt.settings
    if s.app.broker != "kite":
        raise LiveTestError("config/app.yaml: broker must be 'kite' for a live test")
    if s.app.manual_execution != "api_approve":
        raise LiveTestError(
            "config/app.yaml: manual_execution must be 'api_approve' (needs a static IP)"
        )
    if s.risk.capital_base_inr <= 0:
        raise LiveTestError("capital_base_inr is 0")


def prepare_entry(rt: Runtime, kite: object, symbol: str, now: datetime) -> SubmitResult:
    """Build a 1-share limit entry at the last price, run it through the risk engine."""
    require_live(rt)
    if poll(kite, [symbol], rt.quotes, now) == 0:  # type: ignore[arg-type]
        raise LiveTestError(f"no quote for {symbol}")
    ltp = rt.quotes.quotes[symbol].ltp_paise
    limit = round_tick(ltp)
    p = Proposal(
        strategy=STRATEGY,
        symbol=symbol,
        side="BUY",
        product="CNC",
        order_type="LIMIT",
        kind="ENTRY",
        signal_time=now,
        idempotency_key=f"live-test-{symbol}-{now:%Y%m%d%H%M%S}",
        limit_price_paise=limit,
        stop_loss_paise=round_tick(limit * (1 - STOP_PCT)),
        target_paise=round_tick(limit * (1 + TARGET_PCT)),
        max_qty=MAX_QTY,
    )
    return rt.router.submit(p, now)


def place_entry(rt: Runtime, proposal_id: int, now: datetime) -> SubmitResult:
    require_live(rt)
    row = rt.session.get(OrderProposal, proposal_id)
    if row is None or row.max_qty != MAX_QTY or row.qty != MAX_QTY:
        raise LiveTestError("refusing: this proposal is not a capped 1-share live-test proposal")
    result = rt.router.approve_and_place(proposal_id, now)
    if result.order is not None and result.order.qty != MAX_QTY:
        rt.broker.cancel_order(
            result.order.broker_order_id
        )  # defence in depth; cannot normally happen
        raise LiveTestError(f"order quantity {result.order.qty} is not {MAX_QTY}: cancelled")
    return result


def protect(rt: Runtime, symbol: str, now: datetime) -> str:
    require_live(rt)
    pm = ProtectiveManager(
        rt.session,
        rt.broker,
        lambda s: q.ltp_paise if (q := rt.quotes.get(s)) else None,
        lambda t, m: None,
        lambda: now,
    )
    return pm.place_for(symbol)
