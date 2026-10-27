"""Daily report (spec 12 at 17:00): what happened today, in one page and one notification."""

from dataclasses import dataclass, field
from datetime import date
from html import escape

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from tradepartner.calendar import IST, fy_bounds, indian_fy
from tradepartner.db.models import (
    Fill,
    JobRun,
    Lot,
    NewsRaw,
    NewsScore,
    OrderProposal,
    Realized,
    RiskEvent,
)


@dataclass
class DailyReport:
    day: date
    realized_by_category: dict[str, int] = field(default_factory=dict)  # paise
    fills: int = 0
    proposals: dict[str, int] = field(default_factory=dict)  # approved / rejected
    risk_events: dict[str, int] = field(default_factory=dict)  # severity -> n
    open_lots: int = 0
    open_cost_paise: int = 0
    failed_jobs: list[str] = field(default_factory=list)
    news_stored: int = 0
    news_scored: int = 0
    fy_realized: dict[str, int] = field(default_factory=dict)

    @property
    def realized_total(self) -> int:
        return sum(self.realized_by_category.values())


def build(session: Session, day: date) -> DailyReport:
    rep = DailyReport(day)
    for cat, total in session.execute(
        select(Realized.category, func.sum(Realized.pnl_paise))
        .join(Fill, Fill.id == Realized.close_fill_id)
        .where(Fill.trade_date == day)
        .group_by(Realized.category)
    ):
        rep.realized_by_category[str(cat)] = int(total)
    rep.fills = (
        session.scalar(select(func.count()).select_from(Fill).where(Fill.trade_date == day)) or 0
    )
    for p in session.scalars(select(OrderProposal)):
        if p.created_at.astimezone(IST).date() == day and p.risk_decision:
            rep.proposals[p.risk_decision] = rep.proposals.get(p.risk_decision, 0) + 1
    for e in session.scalars(select(RiskEvent)):
        if e.at.astimezone(IST).date() == day:
            rep.risk_events[e.severity] = rep.risk_events.get(e.severity, 0) + 1
    lots = list(session.scalars(select(Lot)))
    rep.open_lots = len(lots)
    rep.open_cost_paise = sum(lot.qty_remaining * lot.cost_paise_per_share for lot in lots)
    rep.failed_jobs = sorted(
        {
            j.job
            for j in session.scalars(select(JobRun).where(JobRun.status == "error"))
            if j.started_at.astimezone(IST).date() == day
        }
    )
    rep.news_stored = sum(
        1 for n in session.scalars(select(NewsRaw)) if n.fetched_at.astimezone(IST).date() == day
    )
    rep.news_scored = sum(
        1 for s in session.scalars(select(NewsScore)) if s.scored_at.astimezone(IST).date() == day
    )
    start, end = fy_bounds(indian_fy(day))
    for cat, total in session.execute(
        select(Realized.category, func.sum(Realized.pnl_paise))
        .join(Fill, Fill.id == Realized.close_fill_id)
        .where(Fill.trade_date >= start, Fill.trade_date <= end)
        .group_by(Realized.category)
    ):
        rep.fy_realized[str(cat)] = int(total)
    return rep


def _inr(paise: int) -> str:
    return f"{'-' if paise < 0 else ''}INR {abs(paise) / 100:,.2f}"


def render_text(rep: DailyReport) -> str:
    parts = [f"{rep.day}: realized {_inr(rep.realized_total)}, {rep.fills} fills"]
    if rep.proposals:
        parts.append("proposals " + ", ".join(f"{k} {v}" for k, v in sorted(rep.proposals.items())))
    if rep.risk_events:
        parts.append(
            "risk events " + ", ".join(f"{k} {v}" for k, v in sorted(rep.risk_events.items()))
        )
    if rep.failed_jobs:
        parts.append("FAILED JOBS: " + ", ".join(rep.failed_jobs))
    return "; ".join(parts)


def render_html(rep: DailyReport) -> str:
    def rows(d: dict[str, int], money: bool = True) -> str:
        return (
            "".join(
                f"<tr><td>{escape(k)}</td><td>{_inr(v) if money else v}</td></tr>"
                for k, v in sorted(d.items())
            )
            or "<tr><td colspan=2>none</td></tr>"
        )

    return (
        f"<!doctype html><meta charset=utf-8><title>Daily report {rep.day}</title>"
        f"<h1>Daily report {rep.day}</h1><p>{escape(render_text(rep))}</p>"
        f"<h2>Realized today</h2><table>{rows(rep.realized_by_category)}</table>"
        "<h2>Financial year to date (advance-tax helper)</h2>"
        f"<table>{rows(rep.fy_realized)}</table>"
        f"<h2>Proposals</h2><table>{rows(rep.proposals, False)}</table>"
        f"<h2>Risk events</h2><table>{rows(rep.risk_events, False)}</table>"
        f"<p>Open lots: {rep.open_lots} (cost {_inr(rep.open_cost_paise)}). "
        f"News stored {rep.news_stored}, scored {rep.news_scored}.</p>"
        + (
            f"<p><b>Failed jobs:</b> {escape(', '.join(rep.failed_jobs))}</p>"
            if rep.failed_jobs
            else ""
        )
    )
