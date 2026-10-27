"""Composition root for the real application: Keychain-backed Kite, local Ollama, macOS
notifications and the broker chosen in ``config/app.yaml``. Used by the CLI and the UI."""

from collections.abc import Callable
from typing import Any

import httpx
from sqlalchemy.orm import Session

from tradepartner import notify as notify_mod
from tradepartner.calendar import TradingCalendar, utc_now
from tradepartner.data import candles, kite_client
from tradepartner.data.quotes import QuoteBook
from tradepartner.db.migrate import upgrade
from tradepartner.db.session import make_engine, make_session_factory
from tradepartner.execution.broker import Broker
from tradepartner.execution.kite import KiteBroker
from tradepartner.execution.paper import PaperBroker
from tradepartner.jobs import JobContext
from tradepartner.llm.client import OllamaClient
from tradepartner.news.sources.bse import BseAnnouncements
from tradepartner.news.sources.nse import NseAnnouncements
from tradepartner.news.sources.rss import RssSource
from tradepartner.paths import home_dir
from tradepartner.settings import Settings


def kite_or_none() -> Any | None:
    try:
        return kite_client.get_kite()
    except kite_client.NotLoggedInError:
        return None


def internet_ok() -> bool:
    try:
        return httpx.get("https://1.1.1.1", timeout=5).status_code < 500
    except httpx.HTTPError:
        return False


def broker_factory(settings: Settings) -> Callable[[Session, QuoteBook], Broker]:
    """``broker: paper`` is the default and the only one used until you change the config."""

    def make(session: Session, quotes: QuoteBook) -> Broker:
        if settings.app.broker == "kite":
            api = kite_client.get_kite()  # raises NotLoggedInError: never silently falls back
            return KiteBroker(api, on_error=lambda m: notify_mod.notify("TradePartner broker", m))
        return PaperBroker(session, quotes.get, settings.costs.slippage.default_bps, utc_now)

    return make


def build_job_context(
    settings: Settings, llm: Callable[[], OllamaClient] | None = None
) -> JobContext:
    home_dir().mkdir(parents=True, exist_ok=True)
    engine = make_engine()
    upgrade(engine.url.render_as_string(hide_password=False))
    limiter = candles.RateLimiter(0.5)
    make_llm = llm or (lambda: OllamaClient(settings.app.llm.model, settings.app.llm.num_ctx))
    return JobContext(
        settings=settings,
        session_factory=make_session_factory(engine),
        store=candles.CandleStore(),
        quotes=QuoteBook(),
        kite=kite_or_none,
        llm=make_llm,
        clock=utc_now,
        notifier=notify_mod.notify,
        calendar=TradingCalendar.from_config(),
        news_sources=lambda: [
            RssSource(settings.app.rss_feeds),
            NseAnnouncements(acquire=limiter.acquire),
            BseAnnouncements(acquire=limiter.acquire),
        ],
        broker_factory=broker_factory(settings),
        internet_check=internet_ok,
    )
