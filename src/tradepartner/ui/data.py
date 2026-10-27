"""Streamlit-free data helpers for the UI pages, so they can be unit tested."""

from dataclasses import dataclass
from datetime import date

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from tradepartner.calendar import IST, TradingCalendar
from tradepartner.data.candles import CandleStore
from tradepartner.data.corporate_actions import Action
from tradepartner.db.models import AuditLog, Fill, NewsEntity, NewsRaw, NewsScore

ORIGINS = ["bot_auto", "bot_manual_api", "manual_external", "paper"]


def ledger_frame(
    session: Session,
    origins: list[str] | None = None,
    products: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    symbol: str = "",
) -> pd.DataFrame:
    stmt = select(Fill).order_by(Fill.filled_at.desc(), Fill.id.desc())
    rows = []
    for f in session.scalars(stmt):
        if origins and f.origin not in origins:
            continue
        if products and f.product not in products:
            continue
        if start and f.trade_date < start:
            continue
        if end and f.trade_date > end:
            continue
        if symbol and symbol.upper() not in f.symbol.upper():
            continue
        rows.append(
            {
                "id": f.id,
                "time (IST)": f.filled_at.astimezone(IST).strftime("%Y-%m-%d %H:%M"),
                "symbol": f.symbol,
                "side": f.side,
                "product": f.product,
                "qty": f.qty,
                "price": f.price_paise / 100,
                "origin": f.origin,
                "source": f.source,
                "trade_id": f.broker_trade_id,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "id",
            "time (IST)",
            "symbol",
            "side",
            "product",
            "qty",
            "price",
            "origin",
            "source",
            "trade_id",
        ],
    )


def rupees(paise: int) -> str:
    """Rupee string with Indian digit grouping (12,34,567.89)."""
    sign = "-" if paise < 0 else ""
    whole, frac = divmod(abs(paise), 100)
    digits = str(whole)
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups: list[str] = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        digits = ",".join([*groups, tail])
    return f"{sign}₹{digits}.{frac:02d}"


# ---------------- Today / Positions / System (M7) ----------------


def today_rows(session: Session, day: date) -> list[dict[str, object]]:
    """One row per signal created on ``day`` with its latest status and proposal/risk detail."""
    from sqlalchemy import select as _select

    from tradepartner.db.models import OrderProposal, Signal
    from tradepartner.signals.proposals import current_status, signal_candidate

    rows: list[dict[str, object]] = []
    for sig in session.scalars(_select(Signal).order_by(Signal.id.desc())):
        if sig.created_at.astimezone(IST).date() != day:
            continue
        cand = signal_candidate(session, sig)
        prop = session.scalar(
            _select(OrderProposal)
            .where(OrderProposal.signal_id == sig.id)
            .order_by(OrderProposal.id.desc())
        )
        rows.append(
            {
                "signal_id": sig.id,
                "strategy": sig.strategy,
                "symbol": sig.symbol,
                "status": current_status(session, sig.id),
                "model_id": sig.model_id,
                "entry": cand.limit_price_paise,
                "stop": cand.stop_paise,
                "target": cand.target_paise,
                "horizon": cand.horizon_days,
                "prob": sig.prob,
                "explanation": sig.explanation,
                "qty": prop.qty if prop else None,
                "risk": prop.risk_amount_paise if prop else None,
                "decision": prop.risk_decision if prop else None,
                "reasons": prop.risk_reasons_json if prop else None,
                "proposal_id": prop.id if prop else None,
            }
        )
    return rows


def kite_order_text(row: dict[str, object]) -> str:
    """The exact order and GTT to place by hand in the Kite app (MANUAL kite_app mode)."""
    entry, stop, target = int(row["entry"]), int(row["stop"]), int(row["target"])  # type: ignore[call-overload]
    qty = int(row["qty"] or 0)  # type: ignore[call-overload]
    return (
        f"BUY {qty} x {row['symbol']}  (NSE, CNC, LIMIT) @ {entry / 100:.2f}\n"
        f"After the fill, set a GTT OCO for {qty} x {row['symbol']}:\n"
        f"  stop-loss trigger {stop / 100:.2f}   target trigger {target / 100:.2f}\n"
        f"Time stop: exit by the close of trading day {row['horizon']} if neither is hit."
    )


def protective_status(session: Session) -> dict[str, str]:
    """Latest protective-order check per symbol (written by the scheduler, M8)."""
    from sqlalchemy import select as _select

    from tradepartner.db.models import AuditLog

    out: dict[str, str] = {}
    for row in session.scalars(
        _select(AuditLog).where(AuditLog.action == "protective_check").order_by(AuditLog.id)
    ):
        payload = row.payload_json
        for sym, status in dict(payload.get("symbols", {})).items():
            out[str(sym)] = str(status)
    return out


# ---------------- Data page helpers ----------------


def quality_frame(
    store: CandleStore, calendar: TradingCalendar, actions: dict[str, list[Action]] | None = None
) -> pd.DataFrame:
    """Every quality issue as a table (symbol, kind, severity, detail)."""
    from tradepartner.data import quality

    issues = quality.report(store, store.symbols("day"), calendar, actions)
    return pd.DataFrame(
        [
            {"symbol": i.symbol, "kind": i.kind, "severity": i.severity, "detail": i.detail}
            for i in issues
        ],
        columns=["symbol", "kind", "severity", "detail"],
    )


def symbol_frame(store: CandleStore, symbol: str) -> pd.DataFrame:
    """One symbol's daily bars in rupees (loader rules: zero-price rows dropped)."""
    df = store.read("day", symbol)
    if df.empty:
        return df
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)].copy()
    for c in ("open", "high", "low", "close"):
        df[c] = df[c] / 100
    return df.set_index("date")[["open", "high", "low", "close", "volume"]]


def coverage(store: CandleStore) -> pd.DataFrame:
    rows = []
    for sym in store.symbols("day"):
        df = store.read("day", sym)
        rows.append(
            {
                "symbol": sym,
                "bars": len(df),
                "first": df["date"].min() if len(df) else None,
                "last": df["date"].max() if len(df) else None,
            }
        )
    return pd.DataFrame(rows, columns=["symbol", "bars", "first", "last"])


@dataclass(frozen=True)
class UniverseSummary:
    members: int
    without_data: list[str]
    excluded: dict[str, str]
    static_bias: bool


def universe_summary(
    session: Session, store: CandleStore, name: str = "nifty100"
) -> UniverseSummary:
    from tradepartner.data import universe as uni

    members = uni.all_symbols(session, name)
    have = set(store.symbols("day"))
    return UniverseSummary(
        len(members),
        [s for s in members if s not in have],
        uni.read_excluded(),
        uni.has_survivorship_bias(session, name),
    )


@dataclass(frozen=True)
class NewsSummary:
    articles: int
    scored: int
    discarded: int
    by_source: dict[str, int]
    valid_rate: float | None
    per_hour: float | None


def news_summary(session: Session) -> NewsSummary:
    by_source = dict(
        session.execute(select(NewsRaw.source, func.count()).group_by(NewsRaw.source)).all()
    )
    raws = [s.raw_json for s in session.scalars(select(NewsScore))]
    bad = sum(1 for r in raws if "error" in r)
    last = session.scalars(
        select(AuditLog).where(AuditLog.action == "news_score_run").order_by(AuditLog.id.desc())
    ).first()
    return NewsSummary(
        articles=sum(by_source.values()),
        scored=len(raws),
        discarded=bad,
        by_source=by_source,
        valid_rate=(len(raws) - bad) / len(raws) if raws else None,
        per_hour=float(last.payload_json["per_hour"]) if last else None,
    )


def recent_news(session: Session, limit: int = 100) -> pd.DataFrame:
    """Latest articles with linked symbols and (if scored) sentiment, for the News page."""
    rows = []
    for n in session.scalars(select(NewsRaw).order_by(NewsRaw.published_at.desc()).limit(limit)):
        syms = sorted(session.scalars(select(NewsEntity.symbol).where(NewsEntity.news_id == n.id)))
        sc = list(session.scalars(select(NewsScore).where(NewsScore.news_id == n.id)))
        rows.append(
            {
                "published": n.published_at,
                "source": n.source,
                "symbols": ", ".join(syms),
                "title": n.title,
                "sentiment": round(sum(x.sentiment for x in sc) / len(sc), 2) if sc else None,
                "event": ", ".join(sorted({x.event_type for x in sc})),
                "summary": sc[0].summary if sc else "",
            }
        )
    return pd.DataFrame(rows)


def missing_days(store: CandleStore, calendar: TradingCalendar) -> pd.DataFrame:
    """Calendar trading days with no candle, with how many symbols miss each. A day missing for
    (almost) every symbol is a market holiday the calendar doesn't know yet; a day missing for a
    few is a genuine data gap."""
    counts: dict[date, int] = {}
    symbols = store.symbols("day")
    for sym in symbols:
        df = store.read("day", sym)
        if df.empty:
            continue
        df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]  # skip pre-listing junk
        if df.empty:
            continue
        have = set(df["date"])
        for d in calendar.trading_days(min(have), max(have)):
            if d not in have:
                counts[d] = counts.get(d, 0) + 1
    total = len(symbols)
    rows = [
        {
            "date": d,
            "weekday": d.strftime("%A"),
            "symbols_missing": n,
            "of": total,
            "likely": "market holiday" if total and n >= 0.9 * total else "data gap",
        }
        for d, n in sorted(counts.items())
    ]
    return pd.DataFrame(rows, columns=["date", "weekday", "symbols_missing", "of", "likely"])


def jump_reasons_frame() -> pd.DataFrame:
    """Explained jumps (latest reason per symbol/day) for the Data page."""
    from tradepartner.data.jump_reasons import load_reasons

    rows = [
        {
            "symbol": r.symbol,
            "date": r.day,
            "move": f"{r.move:+.1%}",
            "reason": r.category,
            "explanation": r.explanation,
            "confidence": r.confidence,
            "evidence": " | ".join(r.evidence[:4]),
        }
        for r in load_reasons().values()
    ]
    cols = ["symbol", "date", "move", "reason", "explanation", "confidence", "evidence"]
    return pd.DataFrame(rows, columns=cols).sort_values(["symbol", "date"], ignore_index=True)
