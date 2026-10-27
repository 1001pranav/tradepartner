"""Explain unexplained price jumps from news (research annotation only).

For each jump: gather headlines (Google News, date-windowed) and the company's NSE announcements
around the day, add deterministic market context (Nifty move, how many other stocks jumped), and
ask the local LLM to name the reason using ONLY that evidence. Results are appended to a JSONL
file (latest row for a symbol/day wins). Nothing here changes prices, features or any decision.
"""

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from tradepartner.calendar import IST
from tradepartner.data.candles import CandleStore
from tradepartner.data.quality import JUMP_THRESHOLD, find_jumps
from tradepartner.llm import prompts
from tradepartner.llm.client import LLMUnavailableError, OllamaClient
from tradepartner.news.sources.base import RawArticle
from tradepartner.paths import home_dir

PROMPT = "jump_reason"
PROMPT_VERSION = "v2"
LOOK_BACK_DAYS = 3  # news before the move can be the cause (results the evening before)
MAX_EVIDENCE = 12
MARKET_MOVE_PCT = 1.5  # Nifty move that can explain a stock's jump on its own
MARKET_BROAD_COUNT = 5  # or this many universe stocks moving >10% the same day
Category = Literal[
    "results", "guidance", "order_win", "regulatory", "litigation", "m&a", "management_change",
    "capital_raise", "rating_change", "short_report_or_fraud", "market_wide", "other", "unknown",
]  # fmt: skip


class JumpReasonOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    explanation: str  # first, so the model states the event before it picks a label
    category: Category
    evidence_ids: list[int] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1, description="between 0 and 1, for example 0.7")

    @field_validator("confidence", mode="before")
    @classmethod
    def _scale(cls, v: object) -> object:
        # small models often answer on a 0-100 scale
        return float(v) / 100 if isinstance(v, int | float) and 1 < v <= 100 else v

    @field_validator("explanation")
    @classmethod
    def _short(cls, v: str) -> str:
        if len(v.split()) > 60:
            raise ValueError("explanation too long")
        return v


@dataclass(frozen=True)
class JumpReason:
    symbol: str
    day: str
    move: float
    category: str
    explanation: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    context: str = ""
    model: str = ""
    created_at: str = ""


class Searcher(Protocol):
    """What the explainer needs from the web; faked in tests."""

    def headlines(self, name: str, start: date, end: date) -> list[RawArticle]: ...
    def announcements(self, symbol: str, start: date, end: date) -> list[RawArticle]: ...


class WebSearcher:
    """Google News + NSE announcements, politely rate-limited."""

    def __init__(self, acquire: Callable[[], None]) -> None:
        from tradepartner.news.sources.gnews import GoogleNewsSearch
        from tradepartner.news.sources.nse import NseAnnouncements

        self._g = GoogleNewsSearch(acquire=acquire)
        self._n = NseAnnouncements(acquire=acquire)

    def headlines(self, name: str, start: date, end: date) -> list[RawArticle]:
        return self._g.search(f'"{name}"' if name else "", start, end) if name else []

    def announcements(self, symbol: str, start: date, end: date) -> list[RawArticle]:
        return self._n.fetch_for_symbol(symbol, start, end)


def reasons_path() -> Path:
    return home_dir() / "jump_reasons.jsonl"


def load_reasons(path: Path | None = None) -> dict[tuple[str, str], JumpReason]:
    p = path or reasons_path()
    out: dict[tuple[str, str], JumpReason] = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = JumpReason(**json.loads(line))
                out[(r.symbol, r.day)] = r  # later rows are corrections
    return out


def _append(r: JumpReason, path: Path | None = None) -> None:
    p = path or reasons_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        fh.write(json.dumps(asdict(r)) + "\n")


def market_context(store: CandleStore, symbols: list[str], day: date) -> str:
    """Nifty's move that day and how many universe stocks moved more than 10% the same way."""
    parts = []
    nifty = store.read("day", "NIFTY 50")
    if not nifty.empty:
        ret = nifty.set_index("date")["close"].astype("float64").pct_change()
        if day in ret.index and pd.notna(ret[day]):
            parts.append(f"Nifty 50 moved {ret[day]:+.1%}")
    big = 0
    for s in symbols:
        df = store.read("day", s)
        r = df.set_index("date")["close"].astype("float64").pct_change() if not df.empty else None
        if r is not None and day in r.index and pd.notna(r[day]) and abs(r[day]) > 0.10:
            big += 1
    parts.append(f"{big} stocks in the universe moved more than 10% that day")
    return "; ".join(parts)


def collect_evidence(
    searcher: Searcher, symbol: str, name: str, day: date
) -> list[tuple[str, RawArticle]]:
    start, end = day - timedelta(days=LOOK_BACK_DAYS), day + timedelta(days=1)
    found: dict[str, tuple[str, RawArticle]] = {}
    for art in searcher.announcements(symbol, start, end):
        found.setdefault(art.title, ("NSE announcement", art))
    for art in searcher.headlines(name, start, end):
        found.setdefault(art.title, ("News", art))

    def key(item: tuple[str, RawArticle]) -> tuple[int, bool]:
        pub = item[1].published_at
        dist = abs((pub.astimezone(IST).date() - day).days) if pub else 99
        return (dist, item[0] != "NSE announcement")  # nearest the move first, exchange first

    ranked = sorted(found.values(), key=key)
    return ranked[:MAX_EVIDENCE]


def _supported(
    out: JumpReasonOut,
    evidence: list[tuple[str, RawArticle]],
    symbol: str,
    name: str,
    context: str,
) -> JumpReasonOut:
    """Deterministic guard on the model's answer: a reason must cite evidence that actually names
    the company, and 'market_wide' needs the market data to show a broad move."""
    if out.category == "unknown":
        return out
    if out.category == "market_wide":
        nifty = re.search(r"Nifty 50 moved ([+-][\d.]+)%", context)
        broad = re.search(r"(\d+) stocks", context)
        if (nifty and abs(float(nifty.group(1))) >= MARKET_MOVE_PCT) or (
            broad and int(broad.group(1)) >= MARKET_BROAD_COUNT
        ):
            return out
        return JumpReasonOut(
            explanation="Market data does not show a broad move, and no company event was found.",
            category="unknown",
            confidence=0,
        )
    tokens = {symbol.lower(), *(w.lower() for w in name.split()[:1] if len(w) > 3)}
    cited = [evidence[i - 1][1].title.lower() for i in out.evidence_ids if 1 <= i <= len(evidence)]
    if any(t in title for title in cited for t in tokens):
        return out
    return JumpReasonOut(
        explanation="No cited evidence names the company; the model's reason is unsupported.",
        category="unknown",
        confidence=0,
    )


def explain_jump(
    llm: OllamaClient,
    searcher: Searcher,
    store: CandleStore,
    universe: list[str],
    symbol: str,
    name: str,
    day: date,
    move: float,
    now: datetime,
) -> JumpReason | None:
    evidence = collect_evidence(searcher, symbol, name, day)
    context = market_context(store, universe, day)
    lines = [
        f"{i}. [{kind}, {(a.published_at.date().isoformat() if a.published_at else '?')}] {a.title}"
        for i, (kind, a) in enumerate(evidence, start=1)
    ]
    prompt = prompts.load(PROMPT, PROMPT_VERSION).render(
        symbol=symbol,
        name=name or symbol,
        day=day.isoformat(),
        move=f"{move:+.1%}",
        context=context,
        evidence="\n".join(lines) or "(no articles or announcements found)",
    )
    try:
        res = llm.generate_with_retry(prompt, JumpReasonOut)
    except LLMUnavailableError:
        return None
    if not res.valid or not isinstance(res.parsed, JumpReasonOut):
        return JumpReason(symbol, day.isoformat(), move, "unknown", "model output invalid", 0.0,
                          lines, context, llm.model, now.isoformat())  # fmt: skip
    out = res.parsed
    out = _supported(out, evidence, symbol, name, context)
    return JumpReason(symbol, day.isoformat(), move, out.category, out.explanation, out.confidence,
                      lines, context, llm.model, now.isoformat())  # fmt: skip


def explain_all(
    llm: OllamaClient,
    searcher: Searcher,
    store: CandleStore,
    jumps: dict[str, list[tuple[date, float]]],
    names: dict[str, str],
    universe: list[str],
    now: datetime,
    limit: int | None = None,
    redo: bool = False,
    path: Path | None = None,
    progress: Callable[[str], None] = lambda s: None,
) -> tuple[int, int]:
    """Returns (explained, skipped_already_done). Resumable: finished jumps are skipped."""
    done = load_reasons(path)
    explained = skipped = 0
    for symbol in sorted(jumps):
        for day, move in jumps[symbol]:
            if (symbol, day.isoformat()) in done and not redo:
                skipped += 1
                continue
            if limit is not None and explained >= limit:
                return explained, skipped
            r = explain_jump(
                llm, searcher, store, universe, symbol, names.get(symbol, ""), day, move, now
            )
            if r is None:
                progress("Ollama unavailable: stopping")
                return explained, skipped
            _append(r, path)
            explained += 1
            progress(f"{symbol} {day} {move:+.1%}: {r.category}")
    return explained, skipped


__all__ = ["JUMP_THRESHOLD", "JumpReason", "explain_all", "find_jumps", "load_reasons"]
