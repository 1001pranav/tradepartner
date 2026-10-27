"""Event-driven daily backtester (spec 13).

Runs the SAME ``RiskEngine`` as live trading; it only builds ``Context`` from simulated state.
Signals at close T become orders for the open of T+1. Full cost model (spec 8) and slippage.
Deterministic: no randomness anywhere; ties are broken by symbol.
"""

from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

import pandas as pd

from tradepartner.backtest import fills
from tradepartner.backtest.metrics import Metrics, compute
from tradepartner.backtest.view import MarketView
from tradepartner.calendar import IST, TradingCalendar
from tradepartner.costs import zerodha
from tradepartner.risk.engine import RiskEngine
from tradepartner.risk.types import (
    Context,
    InstrumentMeta,
    LimitState,
    Mode,
    Position,
    Proposal,
    Quote,
)
from tradepartner.settings import Settings
from tradepartner.strategies.base import Candidate, Strategy, round_tick

SURVIVORSHIP_BANNER = (
    "WARNING: universe membership is the CURRENT list applied to all history (survivorship bias)."
)


@dataclass
class SimPos:
    symbol: str
    qty: int
    entry_price: int
    entry_date: date
    stop: int
    target: int
    time_stop_day: date | None
    strategy: str
    entry_charges: int
    entry_slippage: int


@dataclass(frozen=True)
class Trade:
    symbol: str
    strategy: str
    entry_date: date
    exit_date: date
    qty: int
    entry_price_paise: int
    exit_price_paise: int
    gross_pnl_paise: int
    charges_paise: int
    slippage_paise: int
    pnl_paise: int  # net of charges (slippage is already inside the fill prices)
    reason: str


@dataclass
class BacktestResult:
    strategy: str
    trades: list[Trade]
    equity: pd.Series  # date -> paise
    invested: pd.Series  # date -> fraction of equity invested at the close
    metrics: Metrics
    rejections: Counter[str]
    unfilled: int
    skipped_cash: int
    params: dict[str, object]
    warnings: list[str] = field(default_factory=list)

    def trade_frame(self) -> pd.DataFrame:
        return pd.DataFrame([t.__dict__ for t in self.trades])


def _px(x: float) -> int:
    return round(x * 100)


def _ist(day: date, hh: int, mm: int, ss: int = 0) -> datetime:
    return datetime.combine(day, time(hh, mm, ss), tzinfo=IST)


def _charge(
    settings: Settings, symbol: str, side: str, qty: int, price: int, day: date, n: int
) -> int:
    fill = zerodha.FillIn(n, f"sim-{n}", symbol, side, "CNC", qty, price, day)
    return zerodha.total_paise(zerodha.compute([fill], settings.costs))


def run_backtest(
    strategy: Strategy,
    view: MarketView,
    settings: Settings,
    start: date,
    end: date,
    *,
    mode: Mode = "MANUAL",
    slippage_bps: float | None = None,
    survivorship_warning: bool = False,
    halt_cooldown_days: int = 20,
) -> BacktestResult:
    cfg = settings.risk
    bps = settings.costs.slippage.default_bps if slippage_bps is None else slippage_bps
    days = [d for d in view.trading_days() if start <= d <= end]
    calendar = TradingCalendar(extra_trading_days=frozenset(days))
    engine = RiskEngine(cfg)
    capital = cfg.capital_base_paise

    cash = capital
    positions: dict[str, SimPos] = {}
    trades: list[Trade] = []
    equity: dict[date, int] = {}
    invested: dict[date, float] = {}
    rejections: Counter[str] = Counter()
    seen_keys: set[str] = set()
    pending: list[Candidate] = []
    unfilled = skipped_cash = fill_counter = 0
    prev_close_equity = capital
    peak = capital
    week_start_equity = capital
    last_week: tuple[int, int] | None = None
    last_close: dict[str, int] = {}
    prev_bar_close: dict[str, int] = {}
    last_stop: dict[str, date] = {}
    # Live, a halt stays until Pranav resets it. The simulation models that as a cooling-off
    # period after which the loss streak and the equity peak restart (otherwise one bad run would
    # silence the strategy for the rest of history).
    halted_until: int | None = None
    streak_base = 0

    def streaks() -> dict[str, int]:
        out: dict[str, int] = {}
        for t in trades[streak_base:]:
            out[t.strategy] = out.get(t.strategy, 0) + 1 if t.pnl_paise < 0 else 0
        return out

    def day_index(day: date) -> int:
        return days.index(day)

    def mark(day: date, use_open: bool) -> int:
        total = cash
        for p in positions.values():
            bar = view.bar(p.symbol, day)
            px = (
                _px(float(bar["open" if use_open else "close"]))
                if bar is not None
                else last_close.get(p.symbol, p.entry_price)
            )
            total += p.qty * px
        return total

    def close_position(p: SimPos, day: date, price: int, reason: str) -> None:
        nonlocal cash, fill_counter
        fill_counter += 1
        charges = _charge(settings, p.symbol, "SELL", p.qty, price, day, fill_counter)
        gross = p.qty * (price - p.entry_price)
        cash += p.qty * price - charges
        trades.append(
            Trade(
                p.symbol,
                p.strategy,
                p.entry_date,
                day,
                p.qty,
                p.entry_price,
                price,
                gross,
                p.entry_charges + charges,
                p.entry_slippage,
                gross - p.entry_charges - charges,
                reason,
            )
        )
        if reason == "stop":
            last_stop[p.symbol] = day
        del positions[p.symbol]

    def build_context(
        day: date, now: datetime, orders_today: int, last_order: datetime | None
    ) -> Context:
        nonlocal week_start_equity
        eq_now = mark(day, use_open=True)
        quotes: dict[str, Quote] = {}
        metas: dict[str, InstrumentMeta] = {}
        prev_day = days[days.index(day) - 1] if days.index(day) > 0 else day
        for sym in set(positions) | {c.symbol for c in pending}:
            bar = view.bar(sym, day)
            if bar is not None:
                quotes[sym] = Quote(_px(float(bar["open"])), now, prev_bar_close.get(sym))
            metas[sym] = InstrumentMeta(
                sym,
                view.sector(sym),
                _px(view.avg_daily_value(sym, prev_day)),
            )
        return Context(
            now=now,
            mode=mode,
            calendar=calendar,
            quotes=quotes,
            instruments=metas,
            positions=tuple(
                Position(p.symbol, p.qty, p.entry_price, "CNC", p.strategy, True)
                for p in sorted(positions.values(), key=lambda p: p.symbol)
            ),
            limits=LimitState(
                daily_pnl_paise=eq_now - prev_close_equity,
                weekly_pnl_paise=eq_now - week_start_equity,
                equity_paise=eq_now,
                peak_equity_paise=max(peak, eq_now),
                consecutive_losses=streaks(),
                orders_today=orders_today,
                last_order_at=last_order,
                last_stop_dates=dict(last_stop),
                seen_idempotency_keys=frozenset(seen_keys),
            ),
        )

    for day in days:
        iso = day.isocalendar()
        week = (iso.year, iso.week)
        if week != last_week:
            week_start_equity = prev_close_equity
            last_week = week

        orders_today = 0
        last_order: datetime | None = None
        clock = 0

        def next_now(day: date = day) -> datetime:
            nonlocal clock
            clock += 2  # >= 1 s apart so the per-second rate rule is not what limits a backtest
            return _ist(day, 9, 20) + timedelta(seconds=clock)

        # ---- 1. exits for positions opened before today ----
        for sym in sorted(positions):
            p = positions[sym]
            bar = view.bar(sym, day)
            if bar is None:
                continue
            o, h, low = _px(float(bar["open"])), _px(float(bar["high"])), _px(float(bar["low"]))
            if p.time_stop_day is not None and day >= p.time_stop_day:
                limit = round_tick(o * 0.998)
                ctx = build_context(day, next_now(), orders_today, last_order)
                prop = Proposal(
                    p.strategy,
                    sym,
                    "SELL",
                    "CNC",
                    "LIMIT",
                    "EXIT",
                    _ist(day, 8, 50),
                    f"{p.strategy}-{sym}-{day}-timestop",
                    limit_price_paise=limit,
                    exit_qty=p.qty,
                    exit_reason="time_stop",
                    protective=True,
                )
                decision = engine.check(prop, ctx)
                seen_keys.add(prop.idempotency_key)
                if decision.approved:
                    orders_today += 1
                    last_order = ctx.now
                    close_position(p, day, max(limit, fills.slip(o, bps, up=False)), "time_stop")
                    continue
                rejections.update(r.rule for r in decision.failures)
            hit = fills.exit_fill(False, o, h, low, p.stop, p.target, bps, prev_bar_close.get(sym))
            if hit is not None:
                close_position(p, day, hit[0], hit[1])

        # ---- 2. entries from yesterday's candidates ----
        if halted_until is not None and day_index(day) >= halted_until:
            halted_until, streak_base, peak = (
                None,
                len(trades),
                prev_close_equity,
            )  # the manual reset
        for cand in sorted(pending, key=lambda c: (-c.score, c.symbol)):
            bar = view.bar(cand.symbol, day)
            if bar is None or cand.symbol in positions:
                continue
            if halted_until is not None:
                rejections.update(["halt_cooldown"])
                continue
            now = next_now()
            ctx = build_context(day, now, orders_today, last_order)
            key = f"{strategy.name}-{cand.symbol}-{day}"
            prop = Proposal(
                strategy.name,
                cand.symbol,
                cand.side,
                cand.product,
                "LIMIT",
                "ENTRY",
                _ist(day, 8, 50),
                key,
                limit_price_paise=cand.limit_price_paise,
                stop_loss_paise=cand.stop_paise,
                target_paise=cand.target_paise,
            )
            decision = engine.check(prop, ctx)
            seen_keys.add(key)
            if not decision.approved or decision.order is None:
                rejections.update(r.rule for r in decision.failures)
                if decision.halts:
                    halted_until = day_index(day) + halt_cooldown_days
                continue
            o, h, low = _px(float(bar["open"])), _px(float(bar["high"])), _px(float(bar["low"]))
            px = fills.buy_fill(
                o, h, low, cand.limit_price_paise, bps, prev_bar_close.get(cand.symbol)
            )
            if px is None:
                unfilled += 1
                continue
            qty = decision.order.qty
            fill_counter += 1
            charges = _charge(settings, cand.symbol, "BUY", qty, px, day, fill_counter)
            if qty * px + charges > cash:
                skipped_cash += 1
                continue
            cash -= qty * px + charges
            orders_today += 1
            last_order = now
            base = min(cand.limit_price_paise, o) if o <= cand.limit_price_paise else px
            pos = SimPos(
                cand.symbol,
                qty,
                px,
                day,
                cand.stop_paise,
                cand.target_paise,
                _nth_trading_day(days, day, cand.horizon_days),
                strategy.name,
                charges,
                qty * (px - base),
            )
            positions[cand.symbol] = pos
            hit = fills.exit_fill(
                True, o, h, low, pos.stop, pos.target, bps, prev_bar_close.get(cand.symbol)
            )
            if hit is not None:  # stop is hit first when both occur inside the entry bar
                close_position(pos, day, hit[0], hit[1])
        pending = []

        # ---- 3. mark to market at the close ----
        for sym in view.symbols:
            bar = view.bar(sym, day)
            if bar is not None:
                last_close[sym] = _px(float(bar["close"]))
        eq_close = mark(day, use_open=False)
        equity[day] = eq_close
        invested[day] = (eq_close - cash) / eq_close if eq_close > 0 else 0.0
        peak = max(peak, eq_close)
        prev_close_equity = eq_close
        for sym in view.symbols:
            bar = view.bar(sym, day)
            if bar is not None:
                prev_bar_close[sym] = _px(float(bar["close"]))

        # ---- 4. signals at this close become orders for the next open ----
        if day != days[-1]:
            pending = strategy.candidates(day, view)

    # positions still open at the end stay open: marked in the equity curve, not in the trade list
    eq_series = pd.Series(equity, dtype="int64")
    inv_series = pd.Series(invested, dtype="float64")
    gross_profit = sum(t.gross_pnl_paise for t in trades if t.gross_pnl_paise > 0)
    cost = sum(t.charges_paise + abs(t.slippage_paise) for t in trades)
    notional = sum(t.qty * (t.entry_price_paise + t.exit_price_paise) for t in trades)
    metrics = compute(
        eq_series, [t.pnl_paise for t in trades], notional, cost, gross_profit, inv_series
    )
    warnings = [SURVIVORSHIP_BANNER] if survivorship_warning else []
    return BacktestResult(
        strategy.name,
        trades,
        eq_series,
        inv_series,
        metrics,
        rejections,
        unfilled,
        skipped_cash,
        {
            "mode": mode,
            "slippage_bps": bps,
            "start": start,
            "end": end,
            "capital_paise": capital,
            "costs": "config/costs.yaml",
            "halt_cooldown_days": halt_cooldown_days,
        },
        warnings,
    )


def _nth_trading_day(days: list[date], day: date, n: int) -> date | None:
    i = days.index(day) + n
    return days[i] if i < len(days) else None
