"""ApprovedOrder guard, engine paths, halt enforcement and property tests (spec 15)."""

from dataclasses import replace
from datetime import timedelta

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from tests.fixtures.market import dt
from tests.fixtures.risk import (
    CAPITAL_INR,
    LTP,
    NOW,
    cfg,
    ctx,
    entry,
    exit_,
    limits,
    meta,
    pos,
    quote,
)
from tradepartner.db.models import AuditLog, RiskEvent
from tradepartner.modes.controller import ModeController
from tradepartner.modes.state import GLOBAL
from tradepartner.risk import _mint
from tradepartner.risk import engine as eng_mod
from tradepartner.risk.approved import ApprovedOrder
from tradepartner.risk.engine import RiskEngine, enforce_halts, record_blocks
from tradepartner.risk.types import RuleResult

C = cfg()


def _kwargs(**over: object) -> dict[str, object]:
    base: dict[str, object] = dict(
        strategy="s", symbol="X", side="BUY", product="CNC", order_type="LIMIT", qty=1,
        limit_price_paise=100, trigger_price_paise=None, stop_loss_paise=90, target_paise=120,
        time_stop_date=None, risk_amount_paise=10, market_protection_pct=None, kind="ENTRY",
        idempotency_key="k", tag=None, proposal_id=None,
    )  # fmt: skip
    base.update(over)
    return base


def test_approved_order_cannot_be_constructed_directly() -> None:
    with pytest.raises(TypeError, match="RiskEngine"):
        ApprovedOrder(**_kwargs())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="RiskEngine"):
        ApprovedOrder(**_kwargs(), _token=object())  # type: ignore[arg-type]
    assert ApprovedOrder(**_kwargs(), _token=_mint.TOKEN)  # type: ignore[arg-type]
    assert repr(_mint.TOKEN) == "<mint-token>"


def test_approved_order_invariants() -> None:
    with pytest.raises(ValueError, match="qty"):
        ApprovedOrder(**_kwargs(qty=0), _token=_mint.TOKEN)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="protection"):
        ApprovedOrder(**_kwargs(order_type="MARKET"), _token=_mint.TOKEN)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="20"):
        ApprovedOrder(**_kwargs(tag="x" * 21), _token=_mint.TOKEN)  # type: ignore[arg-type]
    ok = ApprovedOrder(
        **_kwargs(order_type="MARKET", market_protection_pct=1.0), _token=_mint.TOKEN
    )  # type: ignore[arg-type]
    assert ok.market_protection_pct == 1.0


def test_engine_mints_tag_from_proposal_id() -> None:
    d = RiskEngine(C).check(replace(entry(), proposal_id=42), ctx())
    assert d.order is not None and d.order.tag == "tp-42" and d.order.proposal_id == 42


def test_emergency_exit_path() -> None:
    held = ctx("OFF", positions=(pos("INFY", 10, product="MIS"),), limits=limits(kill_active=True))
    p = exit_(
        product="MIS", order_type="MARKET", limit_price_paise=None, emergency=True, exit_qty=10
    )
    d = RiskEngine(C).approve_emergency_exit(p, held)
    assert d.approved and d.order is not None
    assert (
        d.order.order_type == "MARKET" and d.order.market_protection_pct == C.market_protection_pct
    )
    too_many = RiskEngine(C).approve_emergency_exit(replace(p, exit_qty=11), held)
    assert not too_many.approved and too_many.order is None
    with pytest.raises(ValueError, match="emergency"):
        RiskEngine(C).approve_emergency_exit(exit_(), held)
    with pytest.raises(ValueError, match="emergency"):
        RiskEngine(C).approve_emergency_exit(entry(emergency=True), held)


def test_enforce_halts_switches_scopes_off_and_records(db: Session) -> None:
    ctl = ModeController(db)
    now = dt(2026, 3, 10, 10, 0)
    ctl.set_mode(GLOBAL, "MANUAL", "user", "go", now)
    ctl.set_mode("strategy:swing_v1", "MANUAL", "user", "go", now)
    halts = [
        RuleResult("daily_loss", False, "halt", "limit"),
        RuleResult("consecutive_losses", False, "halt", "6 losses", "strategy:swing_v1"),
        RuleResult("fine", True),
        RuleResult("soft", False, "block", "just a block"),
    ]
    assert enforce_halts(db, ctl, halts, now) == [GLOBAL, "strategy:swing_v1"]
    assert ctl.current(GLOBAL).mode == "OFF" and ctl.current("strategy:swing_v1").mode == "OFF"
    assert (
        db.scalar(select(func.count()).select_from(RiskEvent).where(RiskEvent.severity == "halt"))
        == 2
    )
    assert (
        db.scalar(select(func.count()).select_from(AuditLog).where(AuditLog.action == "risk_halt"))
        == 2
    )
    # already OFF: no new event spam from the 30 s tracker
    assert enforce_halts(db, ctl, halts, now) == []
    assert (
        db.scalar(select(func.count()).select_from(RiskEvent).where(RiskEvent.severity == "halt"))
        == 2
    )


def test_record_blocks_only_writes_blocks(db: Session) -> None:
    rs = [
        RuleResult("a", False, "block", "x"),
        RuleResult("b", True),
        RuleResult("c", False, "halt", "y"),
    ]
    assert record_blocks(db, rs, "global", NOW) == 1
    assert db.scalar(select(func.count()).select_from(RiskEvent)) == 1


def test_engine_module_exposes_rule_alias() -> None:
    assert eng_mod.Rule is not None


# ---------- property tests: the invariants that protect money ----------

prices = st.integers(min_value=100, max_value=5_000_000)
APPROVED_SEEN: list[int] = []  # proves the property test actually reaches the approved branch


@settings(max_examples=800, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    ltp=st.integers(5_000, 1_000_000),
    limit_off_bp=st.integers(-200, 200),  # +/-2% around LTP (band is 1.5%)
    stop_bp=st.integers(-100, 1_000),  # stop 12% below .. 1% above entry (max allowed 8%)
    target_bp=st.integers(-100, 2_500),
    has_stop=st.sampled_from([True] * 9 + [False]),
    mode=st.sampled_from(["MANUAL"] * 4 + ["AUTO"] * 4 + ["OFF"]),
    product=st.sampled_from(["CNC", "MIS"]),
    side=st.sampled_from(["BUY"] * 9 + ["SELL"]),
    adv_inr=st.integers(10_000_000, 5_000_000_000),
    daily=st.integers(-4_000_000, 1_000_000),
    orders_today=st.integers(0, 40),
    minute=st.integers(9 * 60, 15 * 60 + 30),
)
def test_engine_never_approves_unsafe_orders(
    ltp: int, limit_off_bp: int, stop_bp: int, target_bp: int, has_stop: bool, mode: str,
    product: str, side: str, adv_inr: int, daily: int, orders_today: int, minute: int,
) -> None:  # fmt: skip
    limit = max(1, ltp + ltp * limit_off_bp // 10_000)
    stop = limit - limit * stop_bp // 10_000 if side == "BUY" else limit + limit * stop_bp // 10_000
    target = (
        limit + limit * target_bp // 10_000
        if side == "BUY"
        else limit - limit * target_bp // 10_000
    )
    now = dt(2026, 3, 10, minute // 60, minute % 60)
    c = ctx(
        mode, now=now, quotes={"INFY": quote(ltp, now=now)}, instruments={"INFY": meta(adv_inr=adv_inr)},
        limits=limits(daily_pnl_paise=daily, orders_today=orders_today),
    )  # fmt: skip
    p = entry(
        side=side, product=product, limit_price_paise=limit, signal_time=now,
        stop_loss_paise=stop if has_stop else None, target_paise=target,
    )  # fmt: skip
    d = RiskEngine(C).check(p, c)
    if not d.approved:
        assert d.order is None and d.failures
        return
    o = d.order
    assert o is not None
    APPROVED_SEEN.append(o.qty)
    cap = C.capital_base_paise
    pos_pct = 5 if mode == "AUTO" else 10
    assert o.qty >= 1  # never qty < 1
    assert o.qty * limit * 100 <= cap * pos_pct  # never above the notional cap
    assert o.stop_loss_paise is not None and o.stop_loss_paise > 0  # never without a stop
    assert o.side == "BUY"  # shorts are disabled by default
    assert o.risk_amount_paise * 10_000 <= cap * 75  # never above 0.75% risk
    assert abs(limit - ltp) * 10_000 <= ltp * 150  # inside the 1.5% price band
    assert o.order_type == "LIMIT" and mode != "OFF"
    assert -daily * 100 < cap * (1 if mode == "AUTO" else 2)  # not past the daily loss limit
    assert orders_today < (10 if mode == "AUTO" else 30)
    assert o.qty * limit * 1000 <= adv_inr * 100 * 5  # ADV cap: 0.5%
    assert 9 * 60 + 20 <= minute < 15 * 60  # inside the entry window
    # reward:risk at least 1.5
    assert (target - limit) * 1000 >= (limit - stop) * 1500


@settings(max_examples=100, deadline=None)
@given(
    held=st.integers(0, 50),
    want=st.integers(0, 80),
    mode=st.sampled_from(["OFF", "MANUAL", "AUTO"]),
)
def test_exit_never_exceeds_position(held: int, want: int, mode: str) -> None:
    c = ctx(mode, positions=(pos("INFY", held),) if held else ())
    d = RiskEngine(C).check(exit_(exit_qty=want), c)
    if d.approved:
        assert d.order is not None and 1 <= d.order.qty <= held


def test_property_constants_are_consistent() -> None:
    assert (
        C.capital_base_paise == CAPITAL_INR * 100 and LTP == 150_000 and timedelta(0) == timedelta()
    )


def test_zz_property_test_reached_approvals() -> None:
    assert len(APPROVED_SEEN) >= 5, "generator never produced an approvable order"
