"""BSE corporate announcements. Endpoint and fields are [VERIFY]-class; be polite."""

from collections.abc import Callable
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import httpx
import structlog

from tradepartner.news.sources.base import RawArticle, clean_text

log = structlog.get_logger()
IST = ZoneInfo("Asia/Kolkata")
URL = "https://api.bseindia.com/BseIndiaAPI/api/AnnGetData/w"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; personal research use)",
    "Referer": "https://www.bseindia.com/",
}


class BseAnnouncements:
    name = "bse"

    def __init__(
        self, client: httpx.Client | None = None, acquire: Callable[[], None] = lambda: None
    ) -> None:
        self._client = client or httpx.Client(timeout=20, headers=HEADERS)
        self._acquire = acquire

    def fetch(self, since: datetime) -> list[RawArticle]:
        today = datetime.now(UTC).astimezone(IST)
        params = {
            "strCat": "-1",
            "strType": "C",
            "strScrip": "",
            "strSearch": "P",
            "strPrevDate": since.astimezone(IST).strftime("%Y%m%d"),
            "strToDate": today.strftime("%Y%m%d"),
        }
        try:
            self._acquire()
            r = self._client.get(URL, params=params, headers=HEADERS)
            r.raise_for_status()
            payload = r.json()
            if not isinstance(payload, dict):  # e.g. the bare string "No Record Found!"
                log.info("bse_no_records", response=str(payload)[:60])
                return []
            rows = payload.get("Table") or []
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("bse_fetch_failed", error=type(exc).__name__)
            return []
        out: list[RawArticle] = []
        for row in rows:
            title = clean_text(row.get("HEADLINE") or row.get("NEWSSUB"))
            company = clean_text(row.get("SLONGNAME"))
            when_raw = str(row.get("NEWS_DT") or "")
            try:
                when = datetime.fromisoformat(when_raw).replace(tzinfo=IST).astimezone(UTC)
            except ValueError:
                when = None
            url = str(row.get("NSURL") or f"https://www.bseindia.com/#{row.get('NEWSID', title)}")
            if title:
                out.append(
                    RawArticle(
                        "bse",
                        url,
                        f"{company}: {title}" if company else title,
                        clean_text(row.get("NEWSSUB")),
                        when,
                        "en",
                    )
                )
        return [a for a in out if a.published_at is None or a.published_at >= since]
