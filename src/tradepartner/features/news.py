"""Point-in-time news features from ``news_scores`` (spec 10.4) with the leakage rules of 10.3.

* An article is usable only once ``max(published_at, fetched_at) <= as_of`` (live) or
  ``published_at + delay <= as_of`` (backtest, where fetch time reflects a later backfill).
* LLM knowledge-cutoff rule: features dated before ``knowledge_cutoff + cutoff_buffer_days`` are
  NaN (``llm_features_valid = 0``), because the model may "remember" how those events played out.
"""

from collections.abc import Sequence
from datetime import date, datetime, timedelta
from typing import Literal, get_args

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from tradepartner.calendar import IST
from tradepartner.db.models import NewsRaw, NewsScore, ResultDate
from tradepartner.llm.schemas import EventType
from tradepartner.settings import LLMConfig

EVENT_TYPES: tuple[str, ...] = get_args(EventType)
RELEVANCE_MIN = 0.5
MATERIAL_MIN = 0.5
LOOKBACK_DAYS = 30
RESULTS_HORIZON_DAYS = 7
DEFAULT_DELAY = timedelta(minutes=15)

NEWS_COLUMNS = [
    "n_rel_1d",
    "n_rel_3d",
    "n_rel_7d",
    "sent_wmean_7d",
    "sent_maxabs_7d",
    "days_since_material",
    "results_in_next_7d",
] + [f"ev_{e}_7d" for e in EVENT_TYPES]
ALL_COLUMNS = [*NEWS_COLUMNS, "llm_features_valid"]


def llm_features_start(cfg: LLMConfig) -> date:
    """First date for which LLM-derived features may be used in training/evaluation."""
    return date.fromisoformat(cfg.knowledge_cutoff) + timedelta(days=cfg.cutoff_buffer_days)


def _blank(symbols: Sequence[str], valid: float) -> pd.DataFrame:
    df = pd.DataFrame(np.nan, index=list(symbols), columns=ALL_COLUMNS, dtype="float64")
    df["llm_features_valid"] = valid
    return df


def _load_rows(
    session: Session,
    symbols: Sequence[str],
    since: datetime,
    mode: str,
    delay: timedelta,
) -> pd.DataFrame:
    q = (
        select(NewsScore, NewsRaw)
        .join(NewsRaw, NewsRaw.id == NewsScore.news_id)
        .where(NewsScore.symbol.in_(list(symbols)), NewsRaw.published_at >= since)
    )
    rows = []
    for score, news in session.execute(q):
        if score.confidence == 0 and score.relevance == 0:
            continue  # discarded output: zero weight
        avail = (
            max(news.published_at, news.fetched_at) if mode == "live" else news.published_at + delay
        )
        rows.append(
            {
                "symbol": score.symbol,
                "avail": avail,
                "rel": score.relevance,
                "sent": score.sentiment,
                "mat": score.materiality,
                "event": score.event_type,
            }
        )
    df = pd.DataFrame(rows, columns=["symbol", "avail", "rel", "sent", "mat", "event"])
    return df[df["rel"] >= RELEVANCE_MIN]


def _symbol_row(rows: pd.DataFrame, as_of: datetime) -> dict[str, float]:
    """Features for one symbol from its rows. Rows not yet available at ``as_of`` are ignored:
    this single filter is the lookahead guard for every news feature."""
    s = rows[rows["avail"] <= as_of]
    out: dict[str, float] = {}
    age = (as_of - s["avail"]).dt.total_seconds() / 86400 if len(s) else pd.Series(dtype=float)
    for w in (1, 3, 7):
        out[f"n_rel_{w}d"] = float((age <= w).sum()) if len(s) else 0.0
    wk = s[age <= 7] if len(s) else s
    weights = (wk["rel"] * wk["mat"]) if len(wk) else pd.Series(dtype=float)
    out["sent_wmean_7d"] = (
        float((weights * wk["sent"]).sum() / weights.sum())
        if len(wk) and weights.sum() > 0
        else np.nan
    )
    out["sent_maxabs_7d"] = float((wk["sent"].abs() * weights).max()) if len(wk) else 0.0
    material = s[s["mat"] >= MATERIAL_MIN] if len(s) else s
    out["days_since_material"] = (
        float(((as_of - material["avail"]).dt.total_seconds() / 86400).min())
        if len(material)
        else np.nan
    )
    for e in EVENT_TYPES:
        out[f"ev_{e}_7d"] = float((wk["event"] == e).sum()) if len(wk) else 0.0
    return out


def _upcoming_results(
    session: Session, symbols: Sequence[str], today: date
) -> list[tuple[str, date, datetime]]:
    q = select(ResultDate.symbol, ResultDate.result_date, ResultDate.created_at).where(
        ResultDate.symbol.in_(list(symbols)),
        ResultDate.result_date > today,
        ResultDate.result_date <= today + timedelta(days=RESULTS_HORIZON_DAYS),
    )
    return list(session.execute(q))


def news_features(
    session: Session,
    symbols: Sequence[str],
    as_of: datetime,
    llm_cfg: LLMConfig,
    *,
    mode: Literal["live", "backtest"] = "live",
    delay: timedelta = DEFAULT_DELAY,
    enforce_cutoff: bool = True,
) -> pd.DataFrame:
    """One row per symbol, using only information available at ``as_of``."""
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    if enforce_cutoff and as_of.astimezone(IST).date() < llm_features_start(llm_cfg):
        return _blank(symbols, 0.0)
    out = _blank(symbols, 1.0)
    if not symbols:
        return out
    df = _load_rows(session, symbols, as_of - timedelta(days=LOOKBACK_DAYS + 1), mode, delay)
    for sym in symbols:
        for col, val in _symbol_row(df[df["symbol"] == sym], as_of).items():
            out.loc[sym, col] = val
    today = as_of.astimezone(IST).date()
    upcoming = {
        sym for sym, _, created in _upcoming_results(session, symbols, today) if created <= as_of
    }
    out["results_in_next_7d"] = [1.0 if s in upcoming else 0.0 for s in symbols]
    return out


def news_panel(
    session: Session,
    symbols: Sequence[str],
    days: Sequence[date],
    llm_cfg: LLMConfig,
    *,
    mode: Literal["live", "backtest"] = "backtest",
    delay: timedelta = DEFAULT_DELAY,
    enforce_cutoff: bool = True,
) -> pd.DataFrame:
    """News features for every (date, symbol), as of each date's session close (15:30 IST).

    Uses exactly the same per-symbol function as ``news_features``, so training rows and live
    rows cannot disagree. Dates before the LLM-cutoff window are NaN with ``llm_features_valid=0``.
    """
    from tradepartner.calendar import session_close_utc

    first = session_close_utc(min(days)) - timedelta(days=LOOKBACK_DAYS + 1) if days else None
    rows = (
        _load_rows(session, symbols, first, mode, delay)
        if days and first
        else pd.DataFrame(columns=["symbol", "avail", "rel", "sent", "mat", "event"])
    )
    by_symbol = {s: rows[rows["symbol"] == s] for s in symbols}

    records = []
    for day in days:
        as_of = session_close_utc(day)
        blocked = enforce_cutoff and day < llm_features_start(llm_cfg)
        upcoming: set[str] = (
            set()
            if blocked
            else {
                sym
                for sym, _, created in _upcoming_results(session, symbols, day)
                if created <= as_of
            }
        )
        for sym in symbols:
            if blocked:
                row = {c: np.nan for c in NEWS_COLUMNS}
                row["llm_features_valid"] = 0.0
            else:
                row = _symbol_row(by_symbol[sym], as_of)
                row["results_in_next_7d"] = 1.0 if sym in upcoming else 0.0
                row["llm_features_valid"] = 1.0
            records.append({"date": day, "symbol": sym, **row})
    if not records:
        return pd.DataFrame(columns=ALL_COLUMNS)
    return pd.DataFrame(records).set_index(["date", "symbol"])[ALL_COLUMNS].sort_index()
