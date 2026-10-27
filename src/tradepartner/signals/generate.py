"""Run strategies -> Signal rows with explanations (spec 5, 12 at 08:50).

``signals`` is append-only, so a signal never changes after insert. Its trade geometry and
lifecycle live in ``signal_events`` (append-only): the first event ('new' for the approved model,
'shadow' for a shadow model) carries the candidate as JSON.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from tradepartner.backtest.view import MarketView
from tradepartner.calendar import session_close_utc
from tradepartner.db.models import Model, Signal, SignalEvent
from tradepartner.features.build import FEATURE_SET_VERSION
from tradepartner.signals.explain import Explainer, Facts
from tradepartner.strategies.base import Candidate
from tradepartner.strategies.model_strategy import ModelStrategy

NewsLines = Callable[[str, datetime], list[str]]


@dataclass(frozen=True)
class GeneratedSignal:
    signal_id: int
    candidate: Candidate
    explanation: str | None
    model_id: int
    shadow: bool
    used_llm: bool


def candidate_json(c: Candidate, used_llm: bool) -> str:
    return json.dumps(
        {
            "limit": c.limit_price_paise,
            "stop": c.stop_paise,
            "target": c.target_paise,
            "horizon": c.horizon_days,
            "prob": c.prob,
            "contrib": [list(x) for x in c.contributions],
            "reason": c.reason,
            "explanation_llm": used_llm,
            "product": c.product,
        }
    )


def candidate_from_json(symbol: str, text: str) -> Candidate:
    d = json.loads(text)
    return Candidate(
        symbol,
        d["limit"],
        d["stop"],
        d["target"],
        d["horizon"],
        d.get("prob") or 0.0,
        d.get("reason", ""),
        "BUY",
        d.get("product", "CNC"),
        d.get("prob"),
        tuple((a, b) for a, b in d.get("contrib", [])),
    )


def _facts(c: Candidate, view: MarketView, as_of: date, news: list[str]) -> Facts:
    h = view.history(c.symbol, as_of, 25)
    close = float(h["close"].iloc[-1])
    ret20 = (close / float(h["close"].iloc[-21]) - 1) * 100 if len(h) >= 21 else None
    return Facts(
        c.symbol,
        None if c.prob is None else c.prob * 100,
        c.contributions,
        tuple(news),
        close,
        ret20,
        c.limit_price_paise / 100,
        c.stop_paise / 100,
        c.target_paise / 100,
        c.horizon_days,
    )


def generate_signals(
    session: Session,
    model: Model,
    strategy: ModelStrategy,
    view: MarketView,
    as_of: date,
    now: datetime,
    explainer: Explainer | None,
    news_lines: NewsLines | None = None,
) -> list[GeneratedSignal]:
    shadow = model.status == "shadow"
    as_of_ts = session_close_utc(as_of)
    out: list[GeneratedSignal] = []
    for cand in strategy.candidates(as_of, view):
        exists = session.scalar(
            select(Signal.id).where(
                Signal.strategy == strategy.name,
                Signal.model_id == model.id,
                Signal.symbol == cand.symbol,
                Signal.as_of == as_of_ts,
            )
        )
        if exists is not None:
            continue  # idempotent: a re-run after a crash never duplicates
        text: str | None = None
        used_llm = False
        if explainer is not None and not shadow:  # shadow signals are only logged
            news = news_lines(cand.symbol, now) if news_lines else []
            exp = explainer.explain(_facts(cand, view, as_of, news))
            text, used_llm = exp.text, exp.used_llm
        sig = Signal(
            strategy=strategy.name,
            model_id=model.id,
            symbol=cand.symbol,
            side="BUY",
            as_of=as_of_ts,
            score=cand.score,
            prob=cand.prob,
            expected_return=None,
            horizon_days=cand.horizon_days,
            explanation=text,
            features_ref=f"{FEATURE_SET_VERSION}:{as_of.isoformat()}:{cand.symbol}",
            status="new",
            created_at=now,
        )
        session.add(sig)
        session.flush()
        session.add(
            SignalEvent(
                signal_id=sig.id,
                status="shadow" if shadow else "new",
                detail=candidate_json(cand, used_llm),
                at=now,
                created_at=now,
            )
        )
        out.append(GeneratedSignal(sig.id, cand, text, model.id, shadow, used_llm))
    session.commit()
    return out


def recent_news_lines(session: Session, symbol: str, now: datetime, limit: int = 3) -> list[str]:
    """Summaries of the latest relevant scored news (point-in-time: only already-fetched items)."""
    from tradepartner.db.models import NewsRaw, NewsScore

    rows = session.execute(
        select(NewsScore.summary, NewsRaw.published_at)
        .join(NewsRaw, NewsRaw.id == NewsScore.news_id)
        .where(NewsScore.symbol == symbol, NewsScore.relevance >= 0.5, NewsRaw.fetched_at <= now)
        .order_by(NewsRaw.published_at.desc())
        .limit(limit)
    ).all()
    return [str(r[0]) for r in rows]
