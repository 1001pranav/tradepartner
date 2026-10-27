"""Fetch, dedupe and store raw news (spec 10.1). ``news_raw`` is append-only.

Point-in-time rules: ``published_at`` comes from the source. If it is missing (or impossibly in
the future) we use ``fetched_at`` and set ``published_at_estimated``. Features may only use an
article once ``max(published_at, fetched_at) <= as_of``.
"""

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from difflib import SequenceMatcher

from sqlalchemy import select
from sqlalchemy.orm import Session

from tradepartner.db.models import NewsEntity, NewsRaw
from tradepartner.news.entity_link import AliasTable, Ambiguous
from tradepartner.news.sources.base import RawArticle

NEAR_DUP_RATIO = 0.85
NEAR_DUP_WINDOW = timedelta(hours=24)
FUTURE_TOLERANCE = timedelta(minutes=5)
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", _NON_ALNUM.sub(" ", text.lower())).strip()


def content_hash(title: str, body: str) -> str:
    return hashlib.sha256(f"{normalize(title)}\n{normalize(body)[:2000]}".encode()).hexdigest()


@dataclass
class IngestResult:
    stored: int = 0
    dup_url: int = 0
    dup_hash: int = 0
    near_dup: int = 0
    skipped_empty: int = 0
    linked: int = 0
    ambiguous: list[tuple[int, tuple[Ambiguous, ...]]] = field(default_factory=list)


def _near_duplicate(
    session: Session, title: str, published: datetime, symbol_hint: str | None = None
) -> bool:
    """Exchange announcements for DIFFERENT companies share boilerplate titles ("ABB: Board
    meeting intimation" vs "TCS: ..."); they are only duplicates of the same company's."""
    norm = normalize(title)
    q = select(NewsRaw).where(
        NewsRaw.published_at >= published - NEAR_DUP_WINDOW,
        NewsRaw.published_at <= published + NEAR_DUP_WINDOW,
    )
    if symbol_hint:
        q = q.join(NewsEntity, NewsEntity.news_id == NewsRaw.id).where(
            NewsEntity.symbol == symbol_hint
        )
    window = session.scalars(q)
    for existing in window:
        # keep the earliest: only an existing article at least as early counts as the original
        if existing.published_at <= published and (
            SequenceMatcher(None, norm, normalize(existing.title)).ratio() >= NEAR_DUP_RATIO
        ):
            return True
    return False


def ingest(
    session: Session,
    articles: Iterable[RawArticle],
    now: datetime,
    aliases: AliasTable | None = None,
) -> IngestResult:
    res = IngestResult()
    ordered = sorted(articles, key=lambda a: (a.published_at or now, a.url))
    for art in ordered:
        title = art.title.strip()
        if not title or not art.url:
            res.skipped_empty += 1
            continue
        published, estimated = art.published_at, False
        if published is None or published > now + FUTURE_TOLERANCE:
            published, estimated = now, True
        if session.scalar(select(NewsRaw.id).where(NewsRaw.url == art.url)) is not None:
            res.dup_url += 1
            continue
        chash = content_hash(title, art.body)
        if session.scalar(select(NewsRaw.id).where(NewsRaw.content_hash == chash)) is not None:
            res.dup_hash += 1
            continue
        if _near_duplicate(session, title, published, art.symbol_hint):
            res.near_dup += 1
            continue
        row = NewsRaw(
            source=art.source,
            url=art.url,
            title=title,
            body=art.body or None,
            published_at=published,
            published_at_estimated=estimated,
            fetched_at=now,
            content_hash=chash,
            lang=art.lang,
        )
        session.add(row)
        session.flush()
        res.stored += 1
        linked: set[str] = set()
        if art.symbol_hint:
            session.add(
                NewsEntity(
                    news_id=row.id,
                    symbol=art.symbol_hint,
                    match_method="isin" if art.isin_hint else "alias",
                    confidence=1.0,
                )
            )
            linked.add(art.symbol_hint)
        if aliases is not None:
            matches, amb = aliases.match(title, art.body or "")
            for m in matches:
                if m.symbol not in linked:
                    session.add(
                        NewsEntity(
                            news_id=row.id,
                            symbol=m.symbol,
                            match_method=m.method,
                            confidence=m.confidence,
                        )
                    )
                    linked.add(m.symbol)
            if amb and not linked:
                res.ambiguous.append((row.id, tuple(amb)))
        res.linked += len(linked)
    session.commit()
    return res
