"""Builds the immutable ``Context`` the risk engine needs from the database (spec 7.2)."""

from collections import defaultdict
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from tradepartner.calendar import TradingCalendar
from tradepartner.db.models import Fill, Lot
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
        limits=compute_limit_state(session, cfg, quotes, now, kill_active),
        live_broker=live_broker,
    )
