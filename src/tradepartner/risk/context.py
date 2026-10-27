"""Builds the immutable ``Context`` the risk engine needs from the database (spec 7.2)."""

from collections import defaultdict
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from tradepartner.calendar import TradingCalendar
from tradepartner.db.models import Fill, Lot, Order
from tradepartner.risk.limits import BOT_ORIGINS, compute_limit_state, fill_strategies
from tradepartner.risk.types import Context, InstrumentMeta, Mode, Position, Quote
from tradepartner.settings import RiskConfig


def positions_from_lots(session: Session) -> tuple[Position, ...]:
    fills = {f.id: f for f in session.scalars(select(Fill))}
    strategies = fill_strategies(session)
    qty: dict[tuple[str, str], int] = defaultdict(int)
    cost: dict[tuple[str, str], int] = defaultdict(int)
    strat: dict[tuple[str, str], str | None] = {}
    bot: dict[tuple[str, str], bool] = defaultdict(bool)
    for lot in session.scalars(select(Lot)):
        f = fills[lot.open_fill_id]
        key = (lot.symbol, f.product)
        qty[key] += lot.qty_remaining
        cost[key] += lot.qty_remaining * lot.cost_paise_per_share
        strat.setdefault(key, strategies.get(f.id, (None, None))[0])
        bot[key] = bot[key] or f.origin in BOT_ORIGINS
    return tuple(
        Position(sym, q, cost[(sym, prod)] // q, prod, strat.get((sym, prod)), bot[(sym, prod)])
        for (sym, prod), q in sorted(qty.items())
    )


TERMINAL_ORDER_STATUSES = ("COMPLETE", "CANCELLED", "REJECTED")


def pending_entries(session: Session) -> tuple[Position, ...]:
    """The unfilled remainder of open bot BUY orders, so resting entries count against limits."""
    filled = {
        oid: int(total or 0)
        for oid, total in session.execute(
            select(Fill.order_id, func.sum(Fill.qty)).group_by(Fill.order_id)
        )
    }
    out: list[Position] = []
    for o in session.scalars(
        select(Order).where(
            Order.side == "BUY",
            Order.origin.in_(BOT_ORIGINS),
            Order.status.not_in(TERMINAL_ORDER_STATUSES),
        )
    ):
        remaining = o.qty - filled.get(o.id, 0)
        if remaining > 0:
            out.append(Position(o.symbol, remaining, o.price_paise or 0, o.product, None, True))
    return tuple(out)


def build_context(
    session: Session,
    cfg: RiskConfig,
    now: datetime,
    calendar: TradingCalendar,
    quotes: dict[str, Quote],
    instruments: dict[str, InstrumentMeta],
    mode: Mode,
    kill_active: bool = False,
    live_broker: bool = False,
) -> Context:
    return Context(
        now=now,
        mode=mode,
        calendar=calendar,
        quotes=quotes,
        instruments=instruments,
        positions=positions_from_lots(session),
        pending_entries=pending_entries(session),
        limits=compute_limit_state(session, cfg, quotes, now, kill_active),
        live_broker=live_broker,
    )
