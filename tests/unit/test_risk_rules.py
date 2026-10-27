"""Every risk rule: pass, fail and boundary cases (spec 15)."""

from datetime import date, timedelta

import pytest

from tests.fixtures.market import dt
from tests.fixtures.risk import (
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
    replace_cfg,
)
from tradepartner.risk import rules, sizing
from tradepartner.risk.engine import RiskEngine
from tradepartner.risk.types import Context, InstrumentMeta, Proposal, Sizing

C = cfg()
SZ = Sizing(66, 66 * LTP, 66 * 6000)


def run(rule, p: Proposal | None = None, c: Context | None = None, conf=C, sz: Sizing = SZ):  # type: ignore[no-untyped-def]
    return rule(p or entry(), c or ctx(), conf, sz)


def test_baseline_proposal_is_approved_with_expected_size() -> None:
    d = RiskEngine(C).check(entry(), ctx())
    assert d.approved, d.reasons
    # min(risk 7,500/60=125, notional 100,000/1,500=66, adv cap 1,000,000/1,500=666) = 66
    assert d.sizing.qty == 66 and d.sizing.risk_amount_paise == 66 * 6000
    assert d.order is not None and d.order.qty == 66 and d.order.tag is None


def test_engine_collects_every_failure_not_just_the_first() -> None:
    d = RiskEngine(C).check(
        entry(stop_loss_paise=None, target_paise=None, product="NRML"), ctx("OFF")
    )
    rules_failed = {r.rule for r in d.failures}
    assert {"product", "mode", "stop_loss", "reward_risk", "quantity"} <= rules_failed
    assert not d.approved and d.order is None and len(d.reasons) == len(d.failures)


# ---------- sizing ----------


def test_sizing_binding_constraints() -> None:
    c = ctx()
    assert sizing.compute_sizing(entry(), c, C).qty == 66  # position cap binds
    tight = replace_cfg(C, per_trade={"max_risk_pct": 0.1})
    assert sizing.compute_sizing(entry(), c, tight).qty == 16  # risk binds: 1,000 / 60
    thin = ctx(instruments={"INFY": meta(adv_inr=1_800_000)})
    assert (
        sizing.compute_sizing(entry(), thin, C).qty == 6
    )  # ADV cap binds: 0.5% of Rs 18 lakh = 9,000 / 1,500


def test_sizing_returns_zero_when_inputs_unusable() -> None:
    c = ctx()
    for p in (entry(stop_loss_paise=None), entry(stop_loss_paise=LTP), entry(order_type="SL"),
              entry(limit_price_paise=None), entry(stop_loss_paise=0)):  # fmt: skip
        assert sizing.compute_sizing(p, c, C).qty == 0
    assert sizing.compute_sizing(entry(), ctx(instruments={}), C).qty == 0
    assert sizing.compute_sizing(entry(), c, replace_cfg(C, capital_base_inr=0)).qty == 0


def test_short_sizing_uses_stop_above_entry() -> None:
    p = entry(side="SELL", stop_loss_paise=156_000, target_paise=138_000)
    assert sizing.compute_sizing(p, ctx(), C).qty == 66


def test_exit_sizing_uses_exit_qty_and_price_fallbacks() -> None:
    assert sizing.compute_sizing(exit_(exit_qty=7), ctx(), C).qty == 7
    assert (
        sizing.compute_sizing(
            exit_(limit_price_paise=None, order_type="MARKET"), ctx(), C
        ).notional_paise
        == 10 * LTP
    )
    assert (
        sizing.compute_sizing(exit_(limit_price_paise=None), ctx(quotes={}), C).notional_paise == 0
    )
    assert sizing.compute_sizing(exit_(exit_qty=None), ctx(), C).qty == 0


def test_auto_mode_uses_tighter_limits() -> None:
    assert (
        sizing.effective_position_pct(C, "AUTO") == 5
        and sizing.effective_position_pct(C, "MANUAL") == 10
    )
    assert (
        sizing.effective_daily_loss_pct(C, "AUTO") == 1.0
        and sizing.effective_daily_loss_pct(C, "MANUAL") == 2.0
    )
    assert (
        sizing.effective_orders_per_day(C, "AUTO") == 10
        and sizing.effective_orders_per_day(C, "OFF") == 30
    )
    assert sizing.compute_sizing(entry(), ctx("AUTO"), C).qty == 33


# ---------- structural ----------


def test_capital_set() -> None:
    assert run(rules.rule_capital_set).passed
    assert not run(rules.rule_capital_set, conf=replace_cfg(C, capital_base_inr=0)).passed


@pytest.mark.parametrize(
    ("kw", "passed"), [({}, True), ({"exchange": "NFO"}, False), ({"segment": "NSE_FO"}, False)]
)
def test_segment(kw: dict[str, str], passed: bool) -> None:
    assert run(rules.rule_segment, entry(**kw)).passed is passed


def test_product() -> None:
    assert run(rules.rule_product, entry(product="MIS")).passed
    assert not run(rules.rule_product, entry(product="NRML")).passed
    assert not run(
        rules.rule_product, conf=replace_cfg(C, allowed_products=["CNC"]), p=entry(product="MIS")
    ).passed


def test_short_selling() -> None:
    assert run(rules.rule_short_selling).passed
    assert not run(rules.rule_short_selling, entry(side="SELL", product="CNC")).passed
    assert not run(
        rules.rule_short_selling, entry(side="SELL", product="MIS")
    ).passed  # disabled by default
    allow = replace_cfg(C, allow_intraday_short=True)
    assert run(rules.rule_short_selling, entry(side="SELL", product="MIS"), conf=allow).passed
    assert not run(rules.rule_short_selling, entry(side="SELL", product="CNC"), conf=allow).passed
    assert not run(rules.rule_short_selling, entry(side="HOLD")).passed
    assert run(rules.rule_short_selling, exit_()).passed


def test_order_type() -> None:
    assert run(rules.rule_order_type).passed
    assert not run(rules.rule_order_type, entry(order_type="WEIRD")).passed
    assert not run(
        rules.rule_order_type, entry(order_type="SL", trigger_price_paise=1)
    ).passed  # entries: LIMIT only
    assert not run(rules.rule_order_type, exit_(order_type="MARKET", limit_price_paise=None)).passed
    assert run(
        rules.rule_order_type, exit_(order_type="MARKET", limit_price_paise=None, emergency=True)
    ).passed
    allow = replace_cfg(C, allow_market_orders=True)
    assert run(
        rules.rule_order_type, exit_(order_type="MARKET", limit_price_paise=None), conf=allow
    ).passed
    assert not run(rules.rule_order_type, entry(limit_price_paise=None)).passed
    assert not run(rules.rule_order_type, exit_(order_type="SL")).passed  # SL needs a trigger
    assert run(
        rules.rule_order_type,
        exit_(order_type="SL-M", limit_price_paise=None, trigger_price_paise=140_000),
    ).passed
    assert not run(rules.rule_order_type, exit_(order_type="SL-M", limit_price_paise=None)).passed


def test_mode() -> None:
    assert (
        run(rules.rule_mode, c=ctx("MANUAL")).passed and run(rules.rule_mode, c=ctx("AUTO")).passed
    )
    assert not run(rules.rule_mode, c=ctx("OFF")).passed
    assert run(rules.rule_mode, exit_(), ctx("OFF")).passed  # protective exits still run
    assert not run(rules.rule_mode, exit_(protective=False), ctx("OFF")).passed


def test_kill_switch() -> None:
    assert run(rules.rule_kill_switch).passed
    killed = ctx(limits=limits(kill_active=True))
    assert not run(rules.rule_kill_switch, c=killed).passed
    assert run(rules.rule_kill_switch, exit_(emergency=True), killed).passed


def test_trading_day_and_time_window() -> None:
    sunday = ctx(now=dt(2026, 3, 8, 10, 0))
    assert not run(rules.rule_trading_day, c=sunday).passed and run(rules.rule_trading_day).passed
    for hh, mm, ok in [
        (9, 19, False),
        (9, 20, True),
        (14, 59, True),
        (15, 0, False),
        (15, 29, False),
    ]:
        assert run(rules.rule_time_window, c=ctx(now=dt(2026, 3, 10, hh, mm))).passed is ok, (
            hh,
            mm,
        )


def test_duplicate_and_signal_age() -> None:
    assert run(rules.rule_duplicate).passed
    assert not run(
        rules.rule_duplicate, c=ctx(limits=limits(seen_idempotency_keys=frozenset({"k1"})))
    ).passed
    assert run(rules.rule_signal_age, entry(signal_time=NOW - timedelta(minutes=120))).passed
    assert not run(rules.rule_signal_age, entry(signal_time=NOW - timedelta(minutes=121))).passed
    assert not run(rules.rule_signal_age, entry(signal_time=NOW + timedelta(minutes=1))).passed


def test_reconciliation_blocks_only_auto() -> None:
    blocked = limits(reconciliation_blocked=True)
    assert not run(rules.rule_reconciliation, c=ctx("AUTO", limits=blocked)).passed
    assert run(rules.rule_reconciliation, c=ctx("MANUAL", limits=blocked)).passed
    assert run(rules.rule_reconciliation, c=ctx("AUTO")).passed


# ---------- market data ----------


def test_quote_freshness() -> None:
    assert run(rules.rule_quote_fresh, c=ctx(quotes={"INFY": quote(age_s=10)})).passed
    assert not run(rules.rule_quote_fresh, c=ctx(quotes={"INFY": quote(age_s=11)})).passed
    assert not run(
        rules.rule_quote_fresh, c=ctx(quotes={"INFY": quote(age_s=-5)})
    ).passed  # future-dated
    assert not run(rules.rule_quote_fresh, c=ctx(quotes={})).passed


def test_price_band() -> None:
    for px, ok in [
        (LTP, True),
        (152_250, True),
        (152_251, False),
        (147_750, True),
        (147_749, False),
    ]:
        assert run(rules.rule_price_band, entry(limit_price_paise=px)).passed is ok, px
    assert not run(rules.rule_price_band, c=ctx(quotes={})).passed
    assert not run(rules.rule_price_band, c=ctx(quotes={"INFY": quote(0)})).passed
    assert run(
        rules.rule_price_band, exit_(order_type="MARKET", limit_price_paise=None, emergency=True)
    ).passed


def test_circuit_asm_skip_days_min_price_liquidity() -> None:
    assert not run(rules.rule_circuit, c=ctx(quotes={"INFY": quote(upper_circuit_hit=True)})).passed
    assert not run(rules.rule_circuit, c=ctx(quotes={"INFY": quote(lower_circuit_hit=True)})).passed
    assert run(
        rules.rule_circuit,
        c=ctx(quotes={"INFY": quote(upper_circuit_hit=True)}),
        conf=replace_cfg(C, data_rules={"block_if_circuit_hit": False}),
    ).passed
    assert not run(rules.rule_asm_gsm, c=ctx(instruments={"INFY": meta(asm_gsm=True)})).passed
    assert run(rules.rule_asm_gsm).passed and run(rules.rule_asm_gsm, c=ctx(instruments={})).passed
    today = date(2026, 3, 10)
    assert not run(
        rules.rule_skip_days, c=ctx(instruments={"INFY": meta(result_date=today)})
    ).passed
    assert not run(rules.rule_skip_days, c=ctx(instruments={"INFY": meta(ex_date=today)})).passed
    assert run(
        rules.rule_skip_days,
        c=ctx(instruments={"INFY": meta(result_date=today + timedelta(days=1))}),
    ).passed
    assert run(rules.rule_skip_days, c=ctx(instruments={})).passed
    off = replace_cfg(C, time_rules={"skip_days": []})
    assert run(
        rules.rule_skip_days, c=ctx(instruments={"INFY": meta(result_date=today)}), conf=off
    ).passed
    assert run(rules.rule_min_price, c=ctx(quotes={"INFY": quote(5_000)})).passed
    assert not run(rules.rule_min_price, c=ctx(quotes={"INFY": quote(4_999)})).passed
    assert not run(rules.rule_min_price, c=ctx(quotes={})).passed
    assert run(rules.rule_liquidity, c=ctx(instruments={"INFY": meta(adv_inr=50_000_000)})).passed
    assert not run(
        rules.rule_liquidity, c=ctx(instruments={"INFY": meta(adv_inr=49_999_999)})
    ).passed
    assert not run(rules.rule_liquidity, c=ctx(instruments={})).passed


# ---------- per trade ----------


def test_stop_loss_rules() -> None:
    assert run(rules.rule_stop_loss).passed
    assert not run(rules.rule_stop_loss, entry(stop_loss_paise=None)).passed
    assert not run(rules.rule_stop_loss, entry(stop_loss_paise=0)).passed
    assert not run(rules.rule_stop_loss, entry(stop_loss_paise=LTP)).passed  # not below entry
    assert not run(rules.rule_stop_loss, entry(side="SELL", stop_loss_paise=LTP - 1)).passed
    optional = replace_cfg(C, per_trade={"require_stop_loss": False})
    assert run(rules.rule_stop_loss, entry(stop_loss_paise=None), conf=optional).passed
    assert run(
        rules.rule_stop_loss, entry(order_type="MARKET", limit_price_paise=None)
    ).passed  # entry unknown


def test_stop_distance_boundary() -> None:
    assert run(rules.rule_stop_distance, entry(stop_loss_paise=138_000)).passed  # exactly 8%
    assert not run(rules.rule_stop_distance, entry(stop_loss_paise=137_999)).passed
    assert run(rules.rule_stop_distance, entry(stop_loss_paise=None)).passed
    assert run(rules.rule_stop_distance, entry(order_type="MARKET", limit_price_paise=None)).passed


def test_reward_risk_boundary() -> None:
    assert run(rules.rule_reward_risk, entry(target_paise=159_000)).passed  # 9,000 / 6,000 = 1.5
    assert not run(rules.rule_reward_risk, entry(target_paise=158_999)).passed
    assert not run(rules.rule_reward_risk, entry(target_paise=None)).passed
    assert not run(rules.rule_reward_risk, entry(target_paise=LTP - 1)).passed
    assert not run(rules.rule_reward_risk, entry(stop_loss_paise=LTP + 1)).passed
    assert run(
        rules.rule_reward_risk, entry(side="SELL", stop_loss_paise=156_000, target_paise=141_000)
    ).passed


def test_quantity_and_derived_limits() -> None:
    assert run(rules.rule_quantity).passed and not run(rules.rule_quantity, sz=Sizing()).passed
    assert run(rules.rule_max_risk).passed
    assert not run(rules.rule_max_risk, sz=Sizing(1, 1, 750_001)).passed
    assert run(rules.rule_max_risk, sz=Sizing(1, 1, 750_000)).passed
    assert not run(rules.rule_position_size, sz=Sizing(67, 67 * LTP, 0)).passed
    assert not run(rules.rule_position_size, c=ctx("AUTO"), sz=Sizing(34, 34 * LTP, 0)).passed
    assert not run(rules.rule_adv_cap, sz=Sizing(667, 667 * LTP, 0)).passed
    assert run(rules.rule_adv_cap, sz=Sizing(666, 666 * LTP, 0)).passed
    assert not run(rules.rule_adv_cap, c=ctx(instruments={})).passed


# ---------- portfolio ----------


def test_open_positions_limit() -> None:
    held = tuple(pos(f"S{i}") for i in range(8))
    assert not run(rules.rule_open_positions, c=ctx(positions=held)).passed
    assert run(rules.rule_open_positions, c=ctx(positions=held[:7])).passed
    assert run(
        rules.rule_open_positions, entry(symbol="S3"), ctx(positions=held)
    ).passed  # adding to a held name


def test_sector_gross_and_mis_exposure() -> None:
    it = {"S1": meta("S1", "IT"), "S2": meta("S2", "IT"), "B1": meta("B1", "BANK"), "INFY": meta()}
    quotes = {s: quote() for s in it}
    big = (pos("S1", 100), pos("S2", 100))  # 2 x Rs 1.5 lakh in IT = Rs 3 lakh = exactly 30%
    c = ctx(positions=big, instruments=it, quotes=quotes)
    assert not run(rules.rule_sector_exposure, c=c, sz=Sizing(1, LTP, 0)).passed
    assert run(
        rules.rule_sector_exposure,
        c=ctx(positions=(pos("B1", 100),), instruments=it, quotes=quotes),
        sz=Sizing(66, 66 * LTP, 0),
    ).passed
    assert run(
        rules.rule_sector_exposure,
        entry(symbol="NEW"),
        ctx(instruments={}),
        sz=Sizing(66, 66 * LTP, 0),
    ).passed
    book = (pos("S0", 650),)  # Rs 9.75 lakh; +Rs 0.99 lakh breaches the 100% gross limit
    c2 = ctx(positions=book, quotes={"S0": quote(), "INFY": quote()})
    assert run(rules.rule_gross_exposure, c=c2, sz=Sizing(66, 66 * LTP, 0)).passed is False
    assert run(rules.rule_gross_exposure, c=ctx(), sz=Sizing(66, 66 * LTP, 0)).passed
    mis = ctx(positions=(pos("S1", 190, product="MIS"),), quotes={"S1": quote(), "INFY": quote()})
    assert not run(
        rules.rule_mis_exposure, entry(product="MIS"), mis, sz=Sizing(66, 66 * LTP, 0)
    ).passed
    assert run(rules.rule_mis_exposure, entry(product="CNC"), mis).passed
    assert run(
        rules.rule_mis_exposure, entry(product="MIS"), ctx(), sz=Sizing(10, 10 * LTP, 0)
    ).passed
    # positions without a live quote fall back to average price
    assert run(
        rules.rule_gross_exposure, c=ctx(positions=(pos("ZZZ", 10),)), sz=Sizing(1, LTP, 0)
    ).passed


def test_no_averaging_down() -> None:
    under = ctx(positions=(pos("INFY", 10, avg=LTP + 100),))
    assert not run(rules.rule_no_averaging_down, c=under).passed
    assert run(
        rules.rule_no_averaging_down, c=ctx(positions=(pos("INFY", 10, avg=LTP - 100),))
    ).passed
    assert run(rules.rule_no_averaging_down, c=ctx()).passed
    assert run(
        rules.rule_no_averaging_down,
        c=under,
        conf=replace_cfg(C, portfolio={"no_averaging_down": False}),
    ).passed
    assert run(rules.rule_no_averaging_down, entry(side="SELL"), under).passed


def test_reentry_cooldown_counts_trading_days() -> None:
    stopped = date(2026, 3, 6)  # Friday: Mon 9 and Tue 10 = 2 trading days elapsed (< 3)
    assert not run(
        rules.rule_reentry_cooldown, c=ctx(limits=limits(last_stop_dates={"INFY": stopped}))
    ).passed
    assert run(
        rules.rule_reentry_cooldown,
        c=ctx(now=dt(2026, 3, 11, 10), limits=limits(last_stop_dates={"INFY": stopped})),
    ).passed
    assert run(rules.rule_reentry_cooldown, c=ctx()).passed


# ---------- loss limits ----------


def test_daily_loss_halts_at_the_limit_and_is_tighter_in_auto() -> None:
    cap = lim = 1_000_000 * 100
    at_limit = limits(daily_pnl_paise=-int(lim * 0.02))
    r = run(rules.rule_daily_loss, c=ctx(limits=at_limit))
    assert not r.passed and r.severity == "halt"
    assert run(
        rules.rule_daily_loss, c=ctx(limits=limits(daily_pnl_paise=-int(lim * 0.02) + 1))
    ).passed
    auto = limits(daily_pnl_paise=-int(cap * 0.01))
    assert not run(rules.rule_daily_loss, c=ctx("AUTO", limits=auto)).passed  # AUTO halts at 1%
    assert run(rules.rule_daily_loss, c=ctx("MANUAL", limits=auto)).passed


def test_weekly_loss_and_drawdown_halt() -> None:
    cap = 1_000_000 * 100
    assert not run(
        rules.rule_weekly_loss, c=ctx(limits=limits(weekly_pnl_paise=-int(cap * 0.04)))
    ).passed
    assert run(
        rules.rule_weekly_loss, c=ctx(limits=limits(weekly_pnl_paise=-int(cap * 0.04) + 1))
    ).passed
    peak = 2 * cap
    r = run(
        rules.rule_drawdown,
        c=ctx(limits=limits(peak_equity_paise=peak, equity_paise=int(peak * 0.9))),
    )
    assert not r.passed and r.severity == "halt"
    assert run(
        rules.rule_drawdown,
        c=ctx(limits=limits(peak_equity_paise=peak, equity_paise=int(peak * 0.9) + 1)),
    ).passed
    assert run(
        rules.rule_drawdown, c=ctx(limits=limits(peak_equity_paise=0, equity_paise=0))
    ).passed


def test_consecutive_losses_halts_only_that_strategy() -> None:
    r = run(rules.rule_consecutive_losses, c=ctx(limits=limits(consecutive_losses={"swing_v1": 6})))
    assert not r.passed and r.severity == "halt" and r.scope == "strategy:swing_v1"
    assert run(
        rules.rule_consecutive_losses, c=ctx(limits=limits(consecutive_losses={"swing_v1": 5}))
    ).passed
    assert run(
        rules.rule_consecutive_losses, c=ctx(limits=limits(consecutive_losses={"other": 9}))
    ).passed


# ---------- rate limits ----------


def test_order_rate_and_daily_count() -> None:
    assert not run(
        rules.rule_order_rate, c=ctx(limits=limits(last_order_at=NOW - timedelta(milliseconds=500)))
    ).passed
    assert run(
        rules.rule_order_rate, c=ctx(limits=limits(last_order_at=NOW - timedelta(seconds=1)))
    ).passed
    assert run(rules.rule_order_rate).passed
    assert not run(rules.rule_orders_per_day, c=ctx(limits=limits(orders_today=30))).passed
    assert run(rules.rule_orders_per_day, c=ctx(limits=limits(orders_today=29))).passed
    assert not run(rules.rule_orders_per_day, c=ctx("AUTO", limits=limits(orders_today=10))).passed
    assert run(
        rules.rule_orders_per_day, exit_(), ctx(limits=limits(orders_today=99))
    ).passed  # never trap a position
    assert not run(
        rules.rule_orders_per_day, exit_(protective=False), ctx(limits=limits(orders_today=99))
    ).passed


# ---------- exits ----------


def test_exit_position_rule() -> None:
    held = ctx(positions=(pos("INFY", 10),))
    assert run(rules.rule_exit_position, exit_(exit_qty=10), held, sz=Sizing(10, 0, 0)).passed
    assert not run(rules.rule_exit_position, exit_(exit_qty=11), held, sz=Sizing(11, 0, 0)).passed
    assert not run(rules.rule_exit_position, exit_(), ctx(), sz=Sizing(10, 0, 0)).passed
    assert not run(rules.rule_exit_position, exit_(), held, sz=Sizing(0, 0, 0)).passed
    assert not run(rules.rule_exit_position, exit_(side="BUY"), held, sz=Sizing(5, 0, 0)).passed
    assert not run(rules.rule_exit_position, exit_(product="MIS"), held, sz=Sizing(5, 0, 0)).passed


def test_exit_approved_through_engine_even_when_off_and_halted() -> None:
    held = ctx("OFF", positions=(pos("INFY", 10),), limits=limits(daily_pnl_paise=-(10**9)))
    d = RiskEngine(C).check(exit_(), held)
    assert d.approved and d.order is not None and d.order.kind == "EXIT" and d.order.qty == 10


def test_entry_blocked_when_halted_but_halt_reported() -> None:
    d = RiskEngine(C).check(entry(), ctx(limits=limits(daily_pnl_paise=-(10**9))))
    assert not d.approved and [r.rule for r in d.halts] == ["daily_loss"]


def test_check_halts_for_tracker() -> None:
    eng = RiskEngine(C)
    assert eng.check_halts(ctx(), ["swing_v1"]) == []
    bad = ctx(
        limits=limits(
            daily_pnl_paise=-(10**9),
            weekly_pnl_paise=-(10**9),
            equity_paise=1,
            peak_equity_paise=10**9,
            consecutive_losses={"swing_v1": 6},
        )
    )
    got = {r.rule for r in eng.check_halts(bad, ["swing_v1", "positional_v1"])}
    assert got == {"daily_loss", "weekly_loss", "drawdown", "consecutive_losses"}
    assert InstrumentMeta("X").sector == "UNKNOWN"
