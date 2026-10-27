"""LLM news analyst (spec 10.2). Scores articles and resolves ambiguous entity links.

The LLM is an analyst only: it scores news and writes explanations. It never decides quantity,
prices or whether to trade.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from tradepartner.calendar import IST
from tradepartner.db.models import NewsEntity, NewsRaw, NewsScore
from tradepartner.llm import prompts
from tradepartner.llm.client import LLMUnavailableError, OllamaClient
from tradepartner.llm.schemas import NewsScoreOut, SymbolChoice
from tradepartner.news.entity_link import Ambiguous

log = structlog.get_logger()
MAX_ARTICLE_CHARS = 10_000  # ~2,500 tokens: title + lede + first paragraphs
EXCHANGE_SOURCES = ("nse", "bse")


def truncate(text: str, max_chars: int = MAX_ARTICLE_CHARS) -> str:
    return text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0] + " ..."


@dataclass
class ScoreRun:
    scored: int = 0
    discarded: int = 0
    seconds: float = 0.0
    llm_down: bool = False

    @property
    def valid_rate(self) -> float:
        total = self.scored + self.discarded
        return self.scored / total if total else float("nan")

    @property
    def per_hour(self) -> float:
        return (self.scored + self.discarded) / self.seconds * 3600 if self.seconds else 0.0


class NewsAnalyst:
    def __init__(
        self,
        session: Session,
        llm: OllamaClient,
        prompt_version: str = prompts.NEWS_SCORE_VERSION,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.session = session
        self.llm = llm
        self.prompt = prompts.load("news_score", prompt_version)
        self.clock = clock
        self._digest: str | None = None
        self._digest_loaded = False

    def digest(self) -> str | None:
        if not self._digest_loaded:
            self._digest = self.llm.model_digest()
            self._digest_loaded = True
        return self._digest

    # ---- selecting work ----

    def pending(
        self, limit: int, priority_symbols: frozenset[str] = frozenset()
    ) -> list[tuple[NewsRaw, str]]:
        """Unscored (article, symbol) pairs for this model and prompt version.

        Order: priority (universe/held/proposal) symbols and exchange announcements first, then
        newest first."""
        done = set(
            self.session.execute(
                select(NewsScore.news_id, NewsScore.symbol).where(
                    NewsScore.prompt_version == self.prompt.version,
                    NewsScore.llm_model == self.llm.model,
                )
            ).all()
        )
        pairs = [
            (news, ent.symbol)
            for ent, news in self.session.execute(
                select(NewsEntity, NewsRaw).join(NewsRaw, NewsRaw.id == NewsEntity.news_id)
            ).all()
            if (news.id, ent.symbol) not in done
        ]
        pairs.sort(
            key=lambda p: (
                p[1] not in priority_symbols,
                p[0].source not in EXCHANGE_SOURCES,
                -p[0].published_at.timestamp(),
                p[1],
            )
        )
        return pairs[:limit]

    # ---- scoring ----

    def _store(
        self, news: NewsRaw, symbol: str, out: NewsScoreOut | None, raw: str, error: str | None
    ) -> None:
        if out is not None:
            row = NewsScore(
                news_id=news.id,
                symbol=symbol,
                llm_model=self.llm.model,
                llm_model_digest=self.digest(),
                prompt_version=self.prompt.version,
                relevance=out.relevance if out.relevant else min(out.relevance, 0.2),
                sentiment=out.sentiment,
                event_type=out.event_type,
                materiality=out.materiality,
                horizon=out.expected_horizon,
                confidence=out.confidence,
                summary=out.summary,
                raw_json=out.model_dump(),
                scored_at=self.clock(),
            )
        else:
            # Spec 10.2: invalid twice -> discard and log. A neutral zero-weight row records the
            # attempt so the same article is not retried forever; features ignore it.
            row = NewsScore(
                news_id=news.id,
                symbol=symbol,
                llm_model=self.llm.model,
                llm_model_digest=self.digest(),
                prompt_version=self.prompt.version,
                relevance=0.0,
                sentiment=0.0,
                event_type="other",
                materiality=0.0,
                horizon="days",
                confidence=0.0,
                summary="[discarded: invalid model output]",
                raw_json={"error": error, "raw": raw[:500]},
                scored_at=self.clock(),
            )
        self.session.add(row)

    def score_one(self, news: NewsRaw, symbol: str) -> bool:
        """True if a valid score was stored, False if the output was discarded."""
        published = news.published_at.astimezone(IST).strftime("%Y-%m-%d %H:%M IST")
        prompt = self.prompt.render(
            symbol=symbol, published=published, title=news.title, body=truncate(news.body or "")
        )
        result = self.llm.generate_with_retry(prompt, NewsScoreOut)
        if result.valid and isinstance(result.parsed, NewsScoreOut):
            self._store(news, symbol, result.parsed, result.raw, None)
            return True
        log.warning("news_score_discarded", news_id=news.id, symbol=symbol, error=result.error)
        self._store(news, symbol, None, result.raw, result.error)
        return False

    def run(
        self,
        limit: int = 200,
        priority_symbols: frozenset[str] = frozenset(),
        max_seconds: float | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> ScoreRun:
        """Score pending pairs until done, ``limit`` or ``max_seconds`` (the 08:00-09:00 window)."""
        out = ScoreRun()
        t0 = monotonic()
        for news, symbol in self.pending(limit, priority_symbols):
            if max_seconds is not None and monotonic() - t0 >= max_seconds:
                break
            try:
                if self.score_one(news, symbol):
                    out.scored += 1
                else:
                    out.discarded += 1
            except LLMUnavailableError:
                out.llm_down = True
                break
            self.session.commit()
        self.session.commit()
        out.seconds = monotonic() - t0
        return out

    # ---- ambiguous entity links ----

    def resolve_ambiguous(self, news_id: int, ambiguous: tuple[Ambiguous, ...]) -> list[str]:
        """Ask the LLM to choose among candidates. A symbol outside the list is rejected."""
        news = self.session.get(NewsRaw, news_id)
        if news is None:
            return []
        chosen: list[str] = []
        prompt_tpl = prompts.load("disambiguate", "v1")
        for amb in ambiguous:
            prompt = prompt_tpl.render(
                candidates=", ".join(amb.candidates),
                title=news.title,
                body=truncate(news.body or "", 4000),
            )
            res = self.llm.generate_with_retry(prompt, SymbolChoice)
            pick = res.parsed.symbol if res.valid and isinstance(res.parsed, SymbolChoice) else None
            if pick in amb.candidates and pick not in chosen:
                self.session.add(
                    NewsEntity(news_id=news.id, symbol=pick, match_method="llm", confidence=0.7)
                )
                chosen.append(pick)
        self.session.commit()
        return chosen
