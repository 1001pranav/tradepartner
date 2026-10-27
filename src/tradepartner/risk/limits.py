"""Loss-limit trackers and counters, recomputed from the ledger (spec 7.2).

They include manual trades in the same account: losses are losses regardless of who placed them.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from tradepartner.calendar import IST
from tradepartner.db.models import Fill, Lot, Order, OrderProposal, Realized
from tradepartner.ledger.reconcile import has_reconciliation_block
from tradepartner.risk.types import LimitState, Quote
from tradepartner.settings import RiskConfig

BOT_ORIGINS = ("bot_auto", "bot_manual_api", "paper")


@dataclass(frozen=True)
class _Closed:
    close_fill_id: int
    close_date: date
    close_time: datetime
    pnl: int
    strategy: str | None
    symbol: str
    stopped: bool


def fill_strategies(session: Session) -> dict[int, tuple[str | None, str | None]]:
    """fill id -> (strategy, exit_reason) through order -> proposal."""
    rows = session.execute(
        select(Fill.id, OrderProposal.strategy, OrderProposal.exit_reason)
        .join(Order, Fill.order_id == Order.id)
        .join(OrderProposal, Order.proposal_id == OrderProposal.id)
    )
    return {fid: (strat, reason) for fid, strat, reason in rows}


def closed_trades(session: Session) -> list[_Closed]:
    """Realized rows grouped per closing fill (= one trade), in time order."""
    meta = fill_strategies(session)
    fills = {f.id: f for f in session.scalars(select(Fill))}
    pnl: dict[int, int] = defaultdict(int)
    for row in session.scalars(select(Realized)):
        pnl[row.close_fill_id] += row.pnl_paise
    out: list[_Closed] = []
    for fid, total in pnl.items():
        f = fills[fid]
        strat, reason = meta.get(fid, (None, None))
        out.append(
            _Closed(fid, f.trade_date, f.filled_at, total, strat, f.symbol, reason == "stop")
        )
    out.sort(key=lambda c: (c.close_time, c.close_fill_id))
    return out


def consecutive_losses(trades: list[_Closed]) -> dict[str, int]:
    streak: dict[str, int] = defaultdict(int)
    for t in trades:
        if t.strategy is None:
            continue
        streak[t.strategy] = streak[t.strategy] + 1 if t.pnl < 0 else 0
    return dict(streak)


INTRADAY = "intraday_v1"
INTRADAY_MIN_TRADES = 40
INTRADAY_MIN_MONTHS = 3


def intraday_live_ok(session: Session, now: datetime) -> bool:
    """Approved intraday model + passing paper gate + >= 40 closed paper trades over >= 3 months."""
    from tradepartner.calendar import add_months
    from tradepartner.db.models import Model

    model = session.scalar(
        select(Model)
        .where(Model.strategy == INTRADAY, Model.status == "approved")
        .order_by(Model.id.desc())
        .limit(1)
    )
    if model is None or not model.metrics_json.get("paper_gate_passed", False):
        return False
    paper = {f.id for f in session.scalars(select(Fill).where(Fill.origin == "paper"))}
    trades = [
        t for t in closed_trades(session) if t.strategy == INTRADAY and t.close_fill_id in paper
    ]
    if len(trades) < INTRADAY_MIN_TRADES:
        return False
    return now.astimezone(IST).date() >= add_months(
        min(t.close_date for t in trades), INTRADAY_MIN_MONTHS
    )


def compute_limit_state(
    session: Session,
    cfg: RiskConfig,
    quotes: dict[str, Quote],
    now: datetime,
    kill_active: bool = False,
) -> LimitState:
    capital = cfg.capital_base_paise
    today = now.astimezone(IST).date()
    week_start = today - timedelta(days=today.weekday())
    trades = closed_trades(session)

    realized_today = sum(t.pnl for t in trades if t.close_date == today)
    realized_week = sum(t.pnl for t in trades if t.close_date >= week_start)
    cumulative = 0
    peak_closed = capital
    by_date: dict[date, int] = defaultdict(int)
    for t in trades:
        by_date[t.close_date] += t.pnl
    for d in sorted(by_date):
        cumulative += by_date[d]
        peak_closed = max(peak_closed, capital + cumulative)

    unreal_total = 0
    unreal_day = 0
    for lot in session.scalars(select(Lot)):
        q = quotes.get(lot.symbol)
        if q is None or lot.qty_remaining <= 0:
            continue
        unreal_total += lot.qty_remaining * (q.ltp_paise - lot.cost_paise_per_share)
        ref = (
            lot.cost_paise_per_share
            if lot.open_date >= today or q.prev_close_paise is None
            else q.prev_close_paise
        )
        unreal_day += lot.qty_remaining * (q.ltp_paise - ref)

    equity = capital + cumulative + unreal_total
    bot_orders = list(session.scalars(select(Order).where(Order.origin.in_(BOT_ORIGINS))))
    today_orders = [o for o in bot_orders if o.placed_at.astimezone(IST).date() == today]
    stops: dict[str, date] = {}
    for t in trades:
        if t.stopped:
            stops[t.symbol] = t.close_date
    cutoff = now - timedelta(days=7)
    keys = frozenset(
        session.scalars(
            select(OrderProposal.idempotency_key).where(OrderProposal.created_at >= cutoff)
        )
    )
    return LimitState(
        daily_pnl_paise=realized_today + unreal_day,
        weekly_pnl_paise=realized_week + unreal_total,
        equity_paise=equity,
        peak_equity_paise=max(peak_closed, equity),
        consecutive_losses=consecutive_losses(trades),
        orders_today=len(today_orders),
        last_order_at=max((o.placed_at for o in today_orders), default=None),
        last_stop_dates=stops,
        seen_idempotency_keys=keys,
        reconciliation_blocked=has_reconciliation_block(session),
        kill_active=kill_active,
        intraday_live_ok=intraday_live_ok(session, now),
    )
