"""Google News RSS search with a date window (``after:``/``before:``), for explaining past moves.

Headlines and snippets only; used for research annotation, never stored in ``news_raw``.
"""

from collections.abc import Callable
from datetime import date, timedelta

import httpx
import structlog

from tradepartner.news.sources.base import RawArticle
from tradepartner.news.sources.rss import parse_feed

log = structlog.get_logger()
URL = "https://news.google.com/rss/search"


class GoogleNewsSearch:
    name = "gnews"

    def __init__(
        self, client: httpx.Client | None = None, acquire: Callable[[], None] = lambda: None
    ) -> None:
        self._client = client or httpx.Client(timeout=20, follow_redirects=True)
        self._acquire = acquire

    def search(self, query: str, start: date, end: date) -> list[RawArticle]:
        """Articles matching ``query`` published from ``start`` to ``end`` (inclusive)."""
        before = (end + timedelta(days=1)).isoformat()
        params = {
            "q": f"{query} after:{start.isoformat()} before:{before}",
            "hl": "en-IN",
            "gl": "IN",
            "ceid": "IN:en",
        }
        for attempt in range(2):  # Google drops the odd connection: one retry
            try:
                self._acquire()
                r = self._client.get(URL, params=params)
                r.raise_for_status()
                return parse_feed(r.text, "gnews")
            except Exception as exc:  # network, XML: one bad query must not stop the run
                log.warning("gnews_failed", query=query, attempt=attempt, error=type(exc).__name__)
        return []
