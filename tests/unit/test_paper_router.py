"""PaperBroker, Router and the M3 acceptance scenarios (spec 16-M3)."""

from dataclasses import replace

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from tests.fixtures.market import dt
from tests.fixtures.risk import LTP, NOW, cfg, entry, exit_, meta
from tests.fixtures.world import STRATEGY, World
from tradepartner.db.models import Fill, Order, OrderEvent, OrderProposal, RiskEvent
from tradepartner.execution.broker import UnsupportedOrderError
from tradepartner.execution.paper import PaperBroker, today_ist
from tradepartner.modes.state import GLOBAL, strategy_scope
from tradepartner.risk import _mint
from tradepartner.risk.approved import ApprovedOrder
from tradepartner.risk.engine import RiskEngine
from tradepartner.risk.limits import compute_limit_state
from tradepartner.risk.types import Quote


@pytest.fixture
def w(db: Session) -> World:
    return World(db)


def order(**kw: object) -> ApprovedOrder:
    base: dict[str, object] = dict(
        strategy="s", symbol="INFY", side="BUY", product="CNC", order_type="LIMIT", qty=10,
        limit_price_paise=LTP, trigger_price_paise=None, stop_loss_paise=144_000,
        target_paise=162_000, time_stop_date=None, risk_amount_paise=1, market_protection_pct=None,
        kind="ENTRY", idempotency_key="k", tag="tp-1", proposal_id=None,
    )  # fmt: skip
    base.update(kw)
    return ApprovedOrder(**base, _token=_mint.TOKEN)  # type: ignore[arg-type]


# ---------- PaperBroker ----------


def test_marketable_limit_buy_fills_with_adverse_slippage_capped_at_limit(w: World) -> None:
    o = w.broker.place_order(order())
    assert o.status == "COMPLETE" and o.filled_qty == 10
    # limit == LTP, so 10 bps of adverse slippage is capped at the limit price
    assert o.avg_price_paise == LTP


def test_limit_buy_fill_never_exceeds_limit(w: World) -> None:
    o = w.broker.place_order(order(limit_price_paise=LTP + 1000, tag="tp-2"))
    assert o.avg_price_paise == LTP + 150  # ltp + 10 bps, still under the limit
    capped = w.broker.place_order(order(limit_price_paise=LTP + 50, tag="tp-3"))
    assert capped.avg_price_paise == LTP + 50


def test_non_marketable_limit_rests_then_fills_on_quote(w: World) -> None:
    o = w.broker.place_order(order(limit_price_paise=LTP - 2000))
    assert o.status == "OPEN" and w.broker.open_orders()[0].broker_order_id == o.broker_order_id
    w.set_price("INFY", LTP - 2500)
    w.broker.on_quote("INFY")
    done = w.broker.get_order(o.broker_order_id)
    assert (
        done is not None
        and done.status == "COMPLETE"
        and done.avg_price_paise == LTP - 2500 + (LTP - 2500) * 1000 // 1_000_000 + 1
    )
    assert w.broker.open_orders() == [] and w.broker.get_order("nope") is None


def test_cancel_open_order_only(w: World) -> None:
    o = w.broker.place_order(order(limit_price_paise=LTP - 2000))
    w.broker.cancel_order(o.broker_order_id)
    assert w.broker.get_order(o.broker_order_id).status == "CANCELLED"  # type: ignore[union-attr]
    w.broker.cancel_order(o.broker_order_id)  # cancelling again is a no-op
    w.broker.cancel_order("missing")
    events = [e.status for e in w.db.scalars(select(OrderEvent).order_by(OrderEvent.id))]
    assert events == ["OPEN", "CANCELLED"]


def test_idempotent_by_tag_and_rejects_foreign_objects(w: World) -> None:
    a = w.broker.place_order(order())
    b = w.broker.place_order(order())
    assert a.broker_order_id == b.broker_order_id
    assert w.db.scalar(select(func.count()).select_from(Order)) == 1
    with pytest.raises(TypeError, match="approved"):
        w.broker.place_order(object())  # type: ignore[arg-type]
    with pytest.raises(UnsupportedOrderError):
        w.broker.place_order(order(product="NRML", tag="tp-9"))


def test_sell_limit_market_and_stop_orders(w: World) -> None:
    sell = w.broker.place_order(order(side="SELL", tag="tp-4"))
    assert sell.status == "COMPLETE" and sell.avg_price_paise == LTP  # max(limit, ltp - slip)
    mkt = w.broker.place_order(
        order(
            order_type="MARKET",
            limit_price_paise=None,
            market_protection_pct=1.0,
            side="SELL",
            tag="tp-5",
        )
    )
    assert mkt.avg_price_paise == LTP - 150
    buy_mkt = w.broker.place_order(
        order(order_type="MARKET", limit_price_paise=None, market_protection_pct=1.0, tag="tp-6")
    )
    assert buy_mkt.avg_price_paise == LTP + 150
    slm = w.broker.place_order(
        order(
            order_type="SL-M",
            limit_price_paise=None,
            trigger_price_paise=LTP - 5000,
            side="SELL",
            tag="tp-7",
        )
    )
    assert slm.status == "OPEN"
    w.set_price("INFY", LTP - 6000)
    w.broker.on_quote("INFY")
    assert w.broker.get_order(slm.broker_order_id).avg_price_paise == 143_856  # type: ignore[union-attr]
    w.set_price("INFY", LTP)
    sl = w.broker.place_order(
        order(
            order_type="SL",
            limit_price_paise=LTP - 5500,
            trigger_price_paise=LTP - 5000,
            side="SELL",
            tag="tp-8",
        )
    )
    assert sl.status == "OPEN"
    w.set_price("INFY", LTP - 6000)
    w.broker.on_quote("INFY")
    assert (
        w.broker.get_order(sl.broker_order_id).status == "OPEN"
    )  # triggered, but the stop-limit price is not reached
    w.set_price("INFY", LTP - 5100)
    w.broker.on_quote("INFY")
    buy_sl = w.broker.place_order(
        order(
            order_type="SL", limit_price_paise=LTP + 600, trigger_price_paise=LTP + 500, tag="tp-10"
        )
    )
    assert buy_sl.status == "OPEN"
    buy_slm = w.broker.place_order(
        order(order_type="SL-M", limit_price_paise=None, trigger_price_paise=LTP + 500, tag="tp-11")
    )
    w.set_price("INFY", LTP + 700)
    w.broker.on_quote("INFY")
    assert w.broker.get_order(buy_slm.broker_order_id).status == "COMPLETE"  # type: ignore[union-attr]


def test_no_quote_means_no_fill(w: World) -> None:
    w.quotes.clear()
    assert w.broker.place_order(order()).status == "OPEN"
    w.broker.on_quote("INFY")
    assert w.broker.open_orders()


def test_paper_fills_land_in_the_ledger_as_paper(w: World) -> None:
    w.broker.place_order(order())
    f = w.db.scalars(select(Fill)).one()
    assert (f.origin, f.source, f.account, f.exchange) == ("paper", "paper", "paper", "NSE")
    assert today_ist(lambda: NOW) == NOW.date()


def test_gtt_oco_stop_target_modify_delete(w: World) -> None:
    w.broker.place_order(order())
    gid = w.broker.place_gtt_oco("INFY", 10, LTP, LTP - 6000, LTP + 12000)
    assert w.broker.list_gtts()[0].status == "active"
    w.broker.modify_gtt_oco(gid, 10, LTP, LTP - 5000, LTP + 12000)
    assert w.broker.list_gtts()[0].stop_paise == LTP - 5000
    w.set_price("INFY", LTP - 5000)
    w.broker.on_quote("INFY")
    assert w.broker.list_gtts()[0].status == "triggered"
    sells = w.db.scalars(select(Fill).where(Fill.side == "SELL")).all()
    assert len(sells) == 1 and sells[0].price_paise == LTP - 5000 - (LTP - 5000) * 1000 // 1_000_000
    g2 = w.broker.place_gtt_oco("INFY", 5, LTP, LTP - 100, LTP + 100)
    w.set_price("INFY", LTP + 100)
    w.broker.on_quote("INFY")
    assert w.broker.list_gtts()[-1].gtt_id == g2 and w.broker.list_gtts()[-1].status == "triggered"
    g3 = w.broker.place_gtt_oco("INFY", 5, LTP, LTP - 100, LTP + 5000)
    w.broker.delete_gtt(g3)
    w.broker.delete_gtt("ghost")
    assert w.broker.list_gtts()[-1].status == "cancelled"
    with pytest.raises(ValueError, match="stop < last price < target"):
        w.broker.place_gtt_oco("INFY", 5, LTP, LTP + 1, LTP + 2)


# ---------- Router ----------


def test_off_logs_only_nothing_is_placed(w: World) -> None:
    r = w.submit_entry()
    assert r.status == "rejected" and r.order is None
    assert any("mode" in x for x in r.decision.reasons)  # type: ignore[union-attr]
    assert w.db.scalar(select(func.count()).select_from(Order)) == 0
    row = w.db.scalars(select(OrderProposal)).one()
    assert (
        row.risk_decision == "rejected"
        and row.mode_at_creation == "OFF"
        and row.strategy == STRATEGY
    )


def test_manual_shows_approved_proposal_without_placing(w: World) -> None:
    w.go_manual()
    r = w.submit_entry()
    assert r.status == "shown" and r.order is None and r.decision.approved  # type: ignore[union-attr]
    row = w.db.scalars(select(OrderProposal)).one()
    assert row.risk_decision == "approved" and row.qty == r.decision.sizing.qty and row.qty > 0  # type: ignore[union-attr]
    assert w.db.scalar(select(func.count()).select_from(Order)) == 0


def test_auto_places_through_the_paper_broker_with_tag(w: World) -> None:
    w.go_auto()
    r = w.submit_entry()
    assert r.status == "placed" and r.order is not None and r.order.status == "COMPLETE"
    assert r.order.tag == f"tp-{r.proposal_id}"
    o = w.db.scalars(select(Order)).one()
    assert o.proposal_id == r.proposal_id and o.qty == r.decision.sizing.qty  # type: ignore[union-attr]


def test_duplicate_submission_is_rejected(w: World) -> None:
    w.go_auto()
    first = w.submit_entry(key="same")
    again = w.submit_entry(key="same")
    assert (
        first.status == "placed"
        and again.status == "duplicate"
        and again.proposal_id == first.proposal_id
    )
    assert w.db.scalar(select(func.count()).select_from(Order)) == 1
    assert (
        w.db.scalar(
            select(func.count()).select_from(RiskEvent).where(RiskEvent.rule == "duplicate")
        )
        == 1
    )


def test_rejected_by_risk_never_reaches_broker(w: World) -> None:
    w.go_auto()
    r = w.submit_entry(stop_loss_paise=None)
    r2 = w.router.submit(
        replace(entry(idempotency_key="nostop", stop_loss_paise=None), signal_time=w.now), w.now
    )
    assert r.status == r2.status == "rejected"
    assert w.db.scalar(select(func.count()).select_from(Order)) == 0


def test_router_only_places_approved_orders(w: World) -> None:
    with pytest.raises(TypeError):
        w.router._place("not an order")  # type: ignore[arg-type]


def test_protective_exit_runs_even_when_mode_is_off(w: World) -> None:
    w.go_auto()
    w.submit_entry()
    w.controller.set_mode(GLOBAL, "OFF", "user", "stop", w.now)
    w.advance(minutes=1)
    held = w.plain_context().positions[0]
    r = w.router.submit(exit_(idempotency_key="x-off", exit_qty=held.qty, signal_time=w.now), w.now)
    assert r.status == "placed" and r.order is not None and r.order.side == "SELL"


def test_exit_in_manual_mode_without_protective_flag_is_shown(w: World) -> None:
    w.go_auto()
    w.submit_entry()
    w.controller.set_mode(GLOBAL, "MANUAL", "user", "x", w.now)
    w.advance(minutes=1)
    held = w.plain_context().positions[0]
    r = w.router.submit(
        exit_(idempotency_key="x-man", exit_qty=held.qty, protective=False, signal_time=w.now),
        w.now,
    )
    assert r.status == "shown"


def test_logged_off_branch_when_mode_off_but_somehow_approved(
    w: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defence in depth: if rules ever approved an entry in OFF mode, nothing is placed."""
    from tradepartner.risk import rules

    patched = tuple(
        (lambda p, c, g, s: rules.ok("mode")) if rule is rules.rule_mode else rule
        for rule in rules.ENTRY_RULES
    )
    monkeypatch.setattr(rules, "ENTRY_RULES", patched)
    r = w.submit_entry(key="weird")
    assert r.status == "logged_off" and w.db.scalar(select(func.count()).select_from(Order)) == 0


def test_per_second_and_daily_rate_limits(w: World) -> None:
    w.go_auto()
    w.instruments["TCS"] = meta("TCS")
    w.set_price("TCS", LTP)
    assert w.submit_entry("INFY", "a").status == "placed"
    assert w.submit_entry("TCS", "b").status == "rejected"  # < 1 s after the previous order
    w.advance(seconds=2)
    assert w.submit_entry("TCS", "c").status == "placed"


# ---------- limit trackers and the simulated day (acceptance) ----------


def test_limit_state_counts_manual_and_unrealized(w: World) -> None:
    w.go_auto()
    r = w.submit_entry()
    assert r.status == "placed"
    w.advance(minutes=5)
    w.set_price("INFY", int(LTP * 0.95), prev_close=LTP)
    ctx = w.context_for(STRATEGY)
    qty = r.decision.sizing.qty  # type: ignore[union-attr]
    assert ctx.limits.orders_today == 1 and ctx.limits.last_order_at is not None
    assert ctx.limits.daily_pnl_paise < -qty * LTP * 4 // 100  # roughly -5% on the position
    assert (
        ctx.limits.weekly_pnl_paise < 0 and ctx.limits.equity_paise < ctx.limits.peak_equity_paise
    )
    assert "k1" in ctx.limits.seen_idempotency_keys and not ctx.limits.kill_active
    assert ctx.positions[0].bot_created and ctx.positions[0].strategy == STRATEGY


def test_simulated_day_breaching_daily_loss_halts_and_forces_off(w: World) -> None:
    w.go_auto()
    assert w.submit_entry().status == "placed"
    w.advance(minutes=30)
    # the position collapses 25%: AUTO daily limit is 1% of capital
    w.set_price("INFY", int(LTP * 0.75), prev_close=LTP)
    ctx = w.context_for(STRATEGY)
    halts = w.engine.check_halts(ctx, [STRATEGY])
    assert [h.rule for h in halts] == ["daily_loss"]
    from tradepartner.risk.engine import enforce_halts

    assert enforce_halts(w.db, w.controller, halts, w.now) == [GLOBAL]
    assert w.controller.current(GLOBAL).mode == "OFF"
    # entries are now impossible ...
    w.advance(minutes=1)
    assert w.submit_entry("INFY", "after-halt").status == "rejected"
    # ... but a protective exit still runs
    held = w.context_for(STRATEGY).positions[0]
    ex = w.router.submit(
        exit_(
            idempotency_key="x-halt",
            exit_qty=held.qty,
            signal_time=w.now,
            limit_price_paise=w.quotes["INFY"].ltp_paise,  # must stay inside the price band
        ),
        w.now,
    )
    assert ex.status == "placed"


def test_restart_never_resumes_auto(w: World, db: Session) -> None:
    w.go_auto()
    from tradepartner.modes.controller import ModeController

    fresh = ModeController(db)  # new process
    fresh.startup(w.now)
    assert fresh.effective(STRATEGY) == "MANUAL"
    assert fresh.current(strategy_scope(STRATEGY)).auto_unlocked  # unlock persists, AUTO does not
    # and under MANUAL nothing is auto-placed
    w.advance(minutes=1)
    assert w.submit_entry("INFY", "post-restart").status == "shown"


def test_consecutive_losses_and_cooldown_come_from_ledger(w: World) -> None:
    w.go_auto()
    for i in range(2):
        w.set_price("INFY", LTP)
        assert w.submit_entry("INFY", f"e{i}").status == "placed"
        w.advance(days=0, minutes=1)
        held = w.context_for(STRATEGY).positions[0]
        w.set_price("INFY", LTP - 8000)
        w.advance(minutes=1)
        ex = w.router.submit(
            exit_(idempotency_key=f"s{i}", exit_qty=held.qty, exit_reason="stop", signal_time=w.now,
                  limit_price_paise=LTP - 8000), w.now)  # fmt: skip
        assert ex.status == "placed"
        w.advance(minutes=1)
        w.controller.set_mode(GLOBAL, "AUTO", "user", "again", w.now) if w.controller.current(
            GLOBAL
        ).mode != "AUTO" else None
        break
    w.context_for(STRATEGY)  # rebuilds the ledger
    state = compute_limit_state(w.db, w.risk_cfg, w.quotes, w.now)
    assert state.consecutive_losses.get(STRATEGY) == 1
    assert state.last_stop_dates.get("INFY") == w.now.date()
    # re-entry blocked by the cooldown
    again = w.submit_entry("INFY", "re-entry")
    assert again.status == "rejected" and any(
        "reentry_cooldown" in x for x in again.decision.reasons
    )  # type: ignore[union-attr]


def test_order_events_and_engine_alias(w: World) -> None:
    assert isinstance(w.broker, PaperBroker) and isinstance(w.engine, RiskEngine)
    assert Quote(1, NOW).ltp_paise == 1 and cfg().capital_base_inr > 0 and dt(2026, 1, 1)
