"""NSE corporate announcements and the results/board-meeting calendar.

Official exchange feeds: respect the terms of use and use polite rate limits. NSE's website API
needs browser-like headers and a session cookie; the endpoints and field names below are
[VERIFY]-class and may change. A rate limiter is mandatory.
"""

from collections.abc import Callable
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

import httpx
import structlog

from tradepartner.news.sources.base import RawArticle, clean_text

log = structlog.get_logger()
IST = ZoneInfo("Asia/Kolkata")
BASE = "https://www.nseindia.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; personal research use)",
    "Accept": "application/json,text/plain,*/*",
    "Referer": f"{BASE}/companies-listing/corporate-filings-announcements",
}
RESULT_WORDS = ("financial result", "results")


def _dt(text: str) -> datetime | None:
    for fmt in ("%d-%b-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d-%b-%Y"):
        try:
            return datetime.strptime(text.strip(), fmt).replace(tzinfo=IST).astimezone(UTC)
        except ValueError:
            continue
    return None


class NseAnnouncements:
    name = "nse"

    def __init__(
        self,
        client: httpx.Client | None = None,
        acquire: Callable[[], None] = lambda: None,
    ) -> None:
        self._client = client or httpx.Client(timeout=20, headers=HEADERS, follow_redirects=True)
        self._acquire = acquire  # rate limiter hook
        self._warm = False

    def _get(self, path: str, params: dict[str, str]) -> object:
        if not self._warm:  # NSE sets cookies on the landing page
            self._acquire()
            self._client.get(BASE, headers=HEADERS)
            self._warm = True
        self._acquire()
        r = self._client.get(f"{BASE}{path}", params=params, headers=HEADERS)
        r.raise_for_status()
        return r.json()

    def fetch(self, since: datetime) -> list[RawArticle]:
        today = datetime.now(UTC).astimezone(IST).date()
        try:
            data = self._get(
                "/api/corporate-announcements",
                {
                    "index": "equities",
                    "from_date": since.astimezone(IST).strftime("%d-%m-%Y"),
                    "to_date": today.strftime("%d-%m-%Y"),
                },
            )
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("nse_fetch_failed", error=type(exc).__name__)
            return []
        out = self._articles(data)
        return [a for a in out if a.published_at is None or a.published_at >= since]

    @staticmethod
    def _articles(data: object) -> list[RawArticle]:
        out: list[RawArticle] = []
        for row in data if isinstance(data, list) else []:
            sym = str(row.get("symbol") or "").strip()
            when = _dt(str(row.get("an_dt") or row.get("sort_date") or ""))
            title = clean_text(row.get("desc"))
            if not (sym and title):
                continue
            url = str(row.get("attchmntFile") or f"{BASE}/ann/{sym}/{row.get('an_dt', '')}")
            out.append(
                RawArticle(
                    "nse",
                    url,
                    f"{sym}: {title}",
                    clean_text(row.get("attchmntText")),
                    when,
                    "en",
                    symbol_hint=sym,
                    isin_hint=row.get("sm_isin"),
                )
            )
        return out

    def fetch_for_symbol(self, symbol: str, start: date, end: date) -> list[RawArticle]:
        """One company's announcements in a date range (used to explain a past price move)."""
        try:
            data = self._get(
                "/api/corporate-announcements",
                {
                    "index": "equities",
                    "symbol": symbol,
                    "from_date": start.strftime("%d-%m-%Y"),
                    "to_date": end.strftime("%d-%m-%Y"),
                },
            )
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("nse_symbol_fetch_failed", symbol=symbol, error=type(exc).__name__)
            return []
        return self._articles(data)

    def fetch_result_calendar(self, start: date, end: date) -> list[tuple[str, date, str]]:
        """(symbol, date, purpose) for upcoming financial-results board meetings."""
        try:
            data = self._get(
                "/api/event-calendar",
                {
                    "index": "equities",
                    "from_date": start.strftime("%d-%m-%Y"),
                    "to_date": end.strftime("%d-%m-%Y"),
                },
            )
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("nse_calendar_failed", error=type(exc).__name__)
            return []
        out: list[tuple[str, date, str]] = []
        for row in data if isinstance(data, list) else []:
            purpose = str(row.get("purpose") or "")
            when = _dt(str(row.get("date") or ""))
            if when and any(w in purpose.lower() for w in RESULT_WORDS):
                out.append((str(row.get("symbol")), when.astimezone(IST).date(), purpose))
        return out
