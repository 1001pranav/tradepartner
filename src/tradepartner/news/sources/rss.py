"""RSS/Atom feeds from Indian financial news outlets (URLs configured in ``app.yaml``)."""

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import defusedxml.ElementTree as ET  # noqa: N817
import httpx
import structlog

from tradepartner.news.sources.base import RawArticle, clean_text

log = structlog.get_logger()
ATOM = "{http://www.w3.org/2005/Atom}"


def _parse_date(text: str | None) -> datetime | None:
    if not text:
        return None
    text = text.strip()
    try:
        dt = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def parse_feed(xml: str, source: str) -> list[RawArticle]:
    root = ET.fromstring(xml)
    out: list[RawArticle] = []
    for item in root.iter("item"):  # RSS 2.0
        link = (item.findtext("link") or "").strip()
        title = clean_text(item.findtext("title"))
        if link and title:
            out.append(
                RawArticle(
                    source,
                    link,
                    title,
                    clean_text(item.findtext("description")),
                    _parse_date(item.findtext("pubDate")),
                )
            )
    for entry in root.iter(f"{ATOM}entry"):  # Atom
        link_el = entry.find(f"{ATOM}link")
        link = (link_el.get("href") if link_el is not None else "") or ""
        title = clean_text(entry.findtext(f"{ATOM}title"))
        if link and title:
            body = clean_text(entry.findtext(f"{ATOM}summary") or entry.findtext(f"{ATOM}content"))
            when = entry.findtext(f"{ATOM}published") or entry.findtext(f"{ATOM}updated")
            out.append(RawArticle(source, link.strip(), title, body, _parse_date(when)))
    return out


class RssSource:
    name = "rss"

    def __init__(self, feeds: list[str], client: httpx.Client | None = None) -> None:
        self.feeds = feeds
        self._client = client or httpx.Client(timeout=20, follow_redirects=True)

    def fetch(self, since: datetime) -> list[RawArticle]:
        out: list[RawArticle] = []
        for url in self.feeds:
            try:
                r = self._client.get(url)
                r.raise_for_status()
                items = parse_feed(r.text, f"rss:{httpx.URL(url).host}")
            except (
                httpx.HTTPError,
                ET.ParseError,
                ValueError,
            ) as exc:  # ValueError: forbidden XML entities
                log.warning("rss_feed_failed", url=url, error=type(exc).__name__)
                continue
            # keep undated items (ingest estimates them); drop clearly old dated ones
            out.extend(a for a in items if a.published_at is None or a.published_at >= since)
        return out
