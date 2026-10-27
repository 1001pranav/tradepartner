"""One function per risk rule, each returning a ``RuleResult`` (spec 7.2, CLAUDE.md style).

All rules are pure: ``rule_x(proposal, ctx, cfg, sizing) -> RuleResult``.
"""

from datetime import datetime, time

from tradepartner.calendar import IST
from tradepartner.risk.sizing import (
    effective_daily_loss_pct,
    effective_orders_per_day,
    effective_position_pct,
    entry_price,
    pct,
)
from tradepartner.risk.types import Context, Position, Proposal, RuleResult, Severity, Sizing
from tradepartner.settings import RiskConfig


def ok(rule: str) -> RuleResult:
    return RuleResult(rule, True)


def fail(
    rule: str, message: str, severity: Severity = "block", scope: str = "global"
) -> RuleResult:
    return RuleResult(rule, False, severity, message, scope)


def _hm(text: str) -> time:
    h, m = text.split(":")
    return time(int(h), int(m))


def _local(now: datetime) -> datetime:
    return now.astimezone(IST)


def _notional(pos: Position, ctx: Context) -> int:
    q = ctx.quotes.get(pos.symbol)
    return pos.qty * (q.ltp_paise if q else pos.avg_price_paise)


# ---------------- structural ----------------


def rule_capital_set(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    if cfg.capital_base_paise <= 0:
        return fail("capital_set", "capital_base_inr is 0")
    return ok("capital_set")


def rule_segment(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    if p.segment not in cfg.allowed_segments or p.exchange != "NSE":
        return fail("segment", f"{p.exchange}/{p.segment} is not NSE cash equity")
    return ok("segment")


def rule_product(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    if p.product not in cfg.allowed_products:
        return fail("product", f"product {p.product} not allowed")
    return ok("product")


def rule_short_selling(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    if p.kind == "ENTRY" and p.side == "SELL":
        if p.product == "CNC":
            return fail("short_selling", "delivery short selling is never allowed")
        if not cfg.allow_intraday_short:
            return fail("short_selling", "intraday short selling is disabled")
    if p.kind == "ENTRY" and p.side not in ("BUY", "SELL"):
        return fail("short_selling", f"unknown side {p.side}")
    return ok("short_selling")


def rule_order_type(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    if p.order_type not in ("LIMIT", "SL", "SL-M", "MARKET"):
        return fail("order_type", f"unknown order type {p.order_type}")
    if p.kind == "ENTRY" and p.order_type != "LIMIT":
        return fail("order_type", "entries must be limit orders")
    if p.order_type == "MARKET" and not (p.emergency or cfg.allow_market_orders):
        return fail("order_type", "market orders are disabled (emergency exits only)")
    if p.order_type in ("LIMIT", "SL") and not p.limit_price_paise:
        return fail("order_type", "limit price required")
    if p.order_type in ("SL", "SL-M") and not p.trigger_price_paise:
        return fail("order_type", "trigger price required")
    return ok("order_type")


def rule_mode(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    if ctx.mode == "OFF" and not (p.kind == "EXIT" and p.protective):
        return fail("mode", "mode is OFF: only protective exits run")
    return ok("mode")


def rule_kill_switch(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    if ctx.limits.kill_active and not p.emergency:
        return fail("kill_switch", "kill switch is active")
    return ok("kill_switch")


def rule_trading_day(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    if not ctx.calendar.is_trading_day(_local(ctx.now).date()):
        return fail("trading_day", "not a trading day")
    return ok("trading_day")


def rule_time_window(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    t = _local(ctx.now).time()
    lo, hi = _hm(cfg.time_rules.no_new_entries_before), _hm(cfg.time_rules.no_new_entries_after)
    if not (lo <= t < hi):
        return fail("time_window", f"new entries only between {lo:%H:%M} and {hi:%H:%M} IST")
    return ok("time_window")


def rule_duplicate(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    if p.idempotency_key in ctx.limits.seen_idempotency_keys:
        return fail("duplicate", "idempotency key already submitted")
    return ok("duplicate")


def rule_signal_age(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    age_min = (ctx.now - p.signal_time).total_seconds() / 60
    if age_min > cfg.data_rules.max_signal_age_minutes:
        return fail("signal_age", f"signal is {age_min:.0f} min old")
    if age_min < 0:
        return fail("signal_age", "signal is dated in the future")
    return ok("signal_age")


def rule_reconciliation(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    if ctx.mode == "AUTO" and ctx.limits.reconciliation_blocked:
        return fail("reconciliation", "unresolved ledger/broker mismatch blocks AUTO entries")
    return ok("reconciliation")


def rule_intraday_gate(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    """Intraday is built last and gated hardest (spec 1, M9): with the real broker, a MIS entry
    needs an approved model, at least 3 months of passing paper trading and enough closed trades.
    Paper trading itself is always allowed."""
    if (
        p.kind == "ENTRY"
        and p.product == "MIS"
        and ctx.live_broker
        and not ctx.limits.intraday_live_ok
    ):
        return fail("intraday_gate", "live intraday needs 3 months of passing paper trading first")
    return ok("intraday_gate")


# ---------------- market data ----------------


def rule_quote_fresh(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    q = ctx.quotes.get(p.symbol)
    if q is None:
        return fail("quote_fresh", "no quote")
    age = (ctx.now - q.as_of).total_seconds()
    if age > cfg.data_rules.max_quote_staleness_sec or age < 0:
        return fail("quote_fresh", f"quote is {age:.0f}s old")
    return ok("quote_fresh")


def rule_price_band(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    q = ctx.quotes.get(p.symbol)
    if p.order_type not in ("LIMIT", "SL") or not p.limit_price_paise:
        return ok("price_band")  # market-type exits carry market protection instead
    if q is None or q.ltp_paise <= 0:
        return fail("price_band", "no quote to check the limit price against")
    drift = abs(p.limit_price_paise - q.ltp_paise) * 10_000
    if drift > q.ltp_paise * int(cfg.per_trade.limit_price_band_pct * 100):
        return fail(
            "price_band", f"limit is outside +/-{cfg.per_trade.limit_price_band_pct}% of LTP"
        )
    return ok("price_band")


def rule_circuit(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    q = ctx.quotes.get(p.symbol)
    if cfg.data_rules.block_if_circuit_hit and q and (q.upper_circuit_hit or q.lower_circuit_hit):
        return fail("circuit", "circuit limit hit")
    return ok("circuit")


def rule_asm_gsm(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    meta = ctx.instruments.get(p.symbol)
    if cfg.data_rules.block_asm_gsm and meta and meta.asm_gsm:
        return fail("asm_gsm", "symbol is on an exchange surveillance list")
    return ok("asm_gsm")


def rule_skip_days(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    meta = ctx.instruments.get(p.symbol)
    today = _local(ctx.now).date()
    if meta is None:
        return ok("skip_days")
    if "result_day_of_symbol" in cfg.time_rules.skip_days and meta.result_date == today:
        return fail("skip_days", "results day")
    if "ex_date_of_symbol" in cfg.time_rules.skip_days and meta.ex_date == today:
        return fail("skip_days", "ex-date")
    return ok("skip_days")


def rule_min_price(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    q = ctx.quotes.get(p.symbol)
    if q is None or q.ltp_paise < int(cfg.per_trade.min_price_inr * 100):
        return fail("min_price", f"price below INR {cfg.per_trade.min_price_inr}")
    return ok("min_price")


def rule_liquidity(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    meta = ctx.instruments.get(p.symbol)
    if meta is None or meta.avg_daily_value_paise < cfg.per_trade.min_avg_daily_value_inr * 100:
        return fail("liquidity", "20-day average traded value below minimum")
    return ok("liquidity")


# ---------------- per-trade ----------------


def rule_stop_loss(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    stop, entry = p.stop_loss_paise, entry_price(p)
    if stop is None or stop <= 0:
        return (
            fail("stop_loss", "a stop loss is required")
            if cfg.per_trade.require_stop_loss
            else ok("stop_loss")
        )
    if entry is not None and (
        (p.side == "BUY" and stop >= entry) or (p.side == "SELL" and stop <= entry)
    ):
        return fail("stop_loss", "stop is on the wrong side of entry")
    return ok("stop_loss")


def rule_stop_distance(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    stop, entry = p.stop_loss_paise, entry_price(p)
    if stop is None or entry is None or entry <= 0:
        return ok("stop_distance")  # rule_stop_loss reports the missing pieces
    dist = abs(entry - stop) * 10_000
    if dist > entry * int(cfg.per_trade.max_stop_distance_pct * 100):
        return fail("stop_distance", f"stop wider than {cfg.per_trade.max_stop_distance_pct}%")
    return ok("stop_distance")


def rule_reward_risk(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    entry, stop, target = entry_price(p), p.stop_loss_paise, p.target_paise
    if entry is None or stop is None or target is None:
        return fail("reward_risk", "entry, stop and target are required")
    risk = entry - stop if p.side == "BUY" else stop - entry
    reward = target - entry if p.side == "BUY" else entry - target
    if risk <= 0 or reward <= 0:
        return fail("reward_risk", "target/stop are on the wrong side of entry")
    if reward * 1000 < risk * int(cfg.per_trade.min_reward_risk * 1000):
        return fail("reward_risk", f"reward:risk below {cfg.per_trade.min_reward_risk}")
    return ok("reward_risk")


def rule_quantity(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    if sz.qty < 1:
        return fail("quantity", "computed quantity is below 1 share")
    return ok("quantity")


def rule_max_risk(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    if sz.risk_amount_paise > cfg.capital_base_paise * pct(cfg.per_trade.max_risk_pct):
        return fail("max_risk", "risk exceeds per-trade limit")
    return ok("max_risk")


def rule_position_size(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    if sz.notional_paise > cfg.capital_base_paise * pct(effective_position_pct(cfg, ctx.mode)):
        return fail("position_size", "notional exceeds the per-symbol position limit")
    return ok("position_size")


def rule_adv_cap(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    meta = ctx.instruments.get(p.symbol)
    adv = meta.avg_daily_value_paise if meta else 0
    if sz.notional_paise > adv * pct(cfg.per_trade.max_pct_of_adv):
        return fail(
            "adv_cap", f"order exceeds {cfg.per_trade.max_pct_of_adv}% of average daily value"
        )
    return ok("adv_cap")


# ---------------- portfolio ----------------


def rule_open_positions(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    held = {pos.symbol for pos in ctx.positions if pos.qty > 0}
    after = len(held | {p.symbol})
    if after > cfg.portfolio.max_open_positions:
        return fail("open_positions", f"would exceed {cfg.portfolio.max_open_positions} positions")
    return ok("open_positions")


def _sector(symbol: str, ctx: Context) -> str:
    meta = ctx.instruments.get(symbol)
    return meta.sector if meta else "UNKNOWN"


def rule_sector_exposure(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    sector = _sector(p.symbol, ctx)
    current = sum(
        _notional(pos, ctx) for pos in ctx.positions if _sector(pos.symbol, ctx) == sector
    )
    limit = cfg.capital_base_paise * pct(cfg.portfolio.max_sector_exposure_pct)
    if current + sz.notional_paise > limit:
        return fail("sector_exposure", f"sector {sector} would exceed its limit")
    return ok("sector_exposure")


def rule_gross_exposure(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    gross = sum(_notional(pos, ctx) for pos in ctx.positions)
    if gross + sz.notional_paise > cfg.capital_base_paise * pct(
        cfg.portfolio.max_gross_exposure_pct
    ):
        return fail("gross_exposure", "gross exposure limit (no leverage on CNC)")
    return ok("gross_exposure")


def rule_mis_exposure(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    if p.product != "MIS":
        return ok("mis_exposure")
    mis = sum(_notional(pos, ctx) for pos in ctx.positions if pos.product == "MIS")
    if mis + sz.notional_paise > cfg.capital_base_paise * pct(cfg.portfolio.max_mis_exposure_pct):
        return fail("mis_exposure", "intraday exposure limit")
    return ok("mis_exposure")


def rule_no_averaging_down(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    if not cfg.portfolio.no_averaging_down or p.side != "BUY":
        return ok("averaging_down")
    q = ctx.quotes.get(p.symbol)
    for pos in ctx.positions:
        if pos.symbol == p.symbol and pos.qty > 0 and q and q.ltp_paise < pos.avg_price_paise:
            return fail("averaging_down", "position is under water; no averaging down")
    return ok("averaging_down")


def rule_reentry_cooldown(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    stopped = ctx.limits.last_stop_dates.get(p.symbol)
    if stopped is None:
        return ok("reentry_cooldown")
    today = _local(ctx.now).date()
    elapsed = len(ctx.calendar.trading_days(stopped, today)) - 1
    if elapsed < cfg.portfolio.reentry_cooldown_days_after_stop:
        return fail("reentry_cooldown", f"stopped out {stopped}; cooling off")
    return ok("reentry_cooldown")


# ---------------- loss limits (halt severity) ----------------


def rule_daily_loss(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    limit = cfg.capital_base_paise * pct(effective_daily_loss_pct(cfg, ctx.mode))
    if -ctx.limits.daily_pnl_paise >= limit:
        return fail("daily_loss", "daily loss limit reached", "halt")
    return ok("daily_loss")


def rule_weekly_loss(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    if -ctx.limits.weekly_pnl_paise >= cfg.capital_base_paise * pct(
        cfg.loss_limits.weekly_loss_pct
    ):
        return fail("weekly_loss", "weekly loss limit reached", "halt")
    return ok("weekly_loss")


def rule_drawdown(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    peak, eq = ctx.limits.peak_equity_paise, ctx.limits.equity_paise
    if peak > 0 and (peak - eq) >= peak * pct(cfg.loss_limits.max_drawdown_pct):
        return fail("drawdown", "maximum drawdown from equity peak reached", "halt")
    return ok("drawdown")


def rule_consecutive_losses(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    n = ctx.limits.consecutive_losses.get(p.strategy, 0)
    if n >= cfg.loss_limits.consecutive_losses_halt:
        return fail(
            "consecutive_losses", f"{n} consecutive losses: strategy goes OFF", "halt",
            f"strategy:{p.strategy}",
        )  # fmt: skip
    return ok("consecutive_losses")


# ---------------- rate limits ----------------


def rule_order_rate(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    last = ctx.limits.last_order_at
    if last is not None:
        gap = (ctx.now - last).total_seconds()
        if gap < 1.0 / cfg.rate_limits.max_orders_per_second:
            return fail("order_rate", "orders per second limit")
    return ok("order_rate")


def rule_orders_per_day(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    # SPEC-QUESTION: protective exits are exempt from the daily order count so a capped-out day
    # can never trap an open position. The per-second limit still applies to them.
    if p.kind == "EXIT" and p.protective:
        return ok("orders_per_day")
    if ctx.limits.orders_today >= effective_orders_per_day(cfg, ctx.mode):
        return fail("orders_per_day", "daily order limit reached")
    return ok("orders_per_day")


# ---------------- exits ----------------


def rule_exit_position(p: Proposal, ctx: Context, cfg: RiskConfig, sz: Sizing) -> RuleResult:
    held = sum(
        pos.qty
        for pos in ctx.positions
        if pos.symbol == p.symbol and pos.product == p.product and pos.qty > 0
    )
    if p.side != "SELL":
        return fail("exit_position", "exits sell existing long positions")
    if sz.qty < 1 or sz.qty > held:
        return fail("exit_position", f"exit qty {sz.qty} exceeds position {held}")
    return ok("exit_position")


ENTRY_RULES = (
    rule_capital_set, rule_segment, rule_product, rule_short_selling, rule_order_type, rule_mode,
    rule_kill_switch, rule_intraday_gate, rule_trading_day, rule_time_window, rule_duplicate,
    rule_signal_age,
    rule_reconciliation, rule_quote_fresh, rule_price_band, rule_circuit, rule_asm_gsm,
    rule_skip_days, rule_min_price, rule_liquidity, rule_stop_loss, rule_stop_distance,
    rule_reward_risk, rule_quantity, rule_max_risk, rule_position_size, rule_adv_cap,
    rule_open_positions, rule_sector_exposure, rule_gross_exposure, rule_mis_exposure,
    rule_no_averaging_down, rule_reentry_cooldown, rule_daily_loss, rule_weekly_loss,
    rule_drawdown, rule_consecutive_losses, rule_order_rate, rule_orders_per_day,
)  # fmt: skip

EXIT_RULES = (
    rule_capital_set, rule_segment, rule_product, rule_order_type, rule_mode, rule_kill_switch,
    rule_trading_day, rule_duplicate, rule_quote_fresh, rule_price_band, rule_exit_position,
    rule_order_rate, rule_orders_per_day,
)  # fmt: skip

# Rules that need only a subset for kill-switch emergency exits (never blocked by mode/kill).
EMERGENCY_RULES = (rule_capital_set, rule_segment, rule_product, rule_exit_position)

# Evaluated by the 30-second tracker, independent of any proposal.
HALT_RULES = (rule_daily_loss, rule_weekly_loss, rule_drawdown, rule_consecutive_losses)
