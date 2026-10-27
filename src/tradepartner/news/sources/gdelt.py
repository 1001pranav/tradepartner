"""GDELT DOC 2.0 for historical backfill (RSS only gives recent items). Titles only."""

from collections.abc import Callable
from datetime import UTC, datetime

import httpx
import structlog

from tradepartner.news.sources.base import RawArticle, clean_text

log = structlog.get_logger()
URL = "https://api.gdeltproject.org/api/v2/doc/doc"


class GdeltSource:
    name = "gdelt"

    def __init__(
        self,
        queries: dict[str, str] | None = None,
        client: httpx.Client | None = None,
        acquire: Callable[[], None] = lambda: None,
        max_records: int = 250,
    ) -> None:
        """``queries`` maps a query string to the symbol it is about (e.g. '"Infosys"' -> INFY)."""
        self.queries = queries or {}
        self._client = client or httpx.Client(timeout=30)
        self._acquire = acquire
        self.max_records = max_records

    def fetch_range(self, start: datetime, end: datetime) -> list[RawArticle]:
        out: list[RawArticle] = []
        for query, symbol in sorted(self.queries.items()):
            params = {
                "query": f"{query} sourcecountry:IN",
                "mode": "artlist",
                "format": "json",
                "maxrecords": str(self.max_records),
                "sort": "datedesc",
                "startdatetime": start.astimezone(UTC).strftime("%Y%m%d%H%M%S"),
                "enddatetime": end.astimezone(UTC).strftime("%Y%m%d%H%M%S"),
            }
            try:
                self._acquire()
                r = self._client.get(URL, params=params)
                r.raise_for_status()
                arts = r.json().get("articles", [])
            except (httpx.HTTPError, ValueError) as exc:
                log.warning("gdelt_failed", query=query, error=type(exc).__name__)
                continue
            for a in arts:
                try:
                    seen = datetime.strptime(a["seendate"], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
                except (KeyError, ValueError):
                    continue
                title = clean_text(a.get("title"))
                if title and a.get("url"):
                    out.append(
                        RawArticle(
                            "gdelt",
                            a["url"],
                            title,
                            "",
                            seen,
                            a.get("language"),
                            symbol_hint=symbol,
                        )
                    )
        return out

    def fetch(self, since: datetime) -> list[RawArticle]:
        return self.fetch_range(since, datetime.now(UTC))
