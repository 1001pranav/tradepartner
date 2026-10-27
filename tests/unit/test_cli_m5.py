from datetime import UTC, date, datetime, timedelta

import httpx
import pytest
from sqlalchemy import func, select
from typer.testing import CliRunner

from tests.fixtures.llm import FakeOllama
from tests.unit.test_news_ingest import RSS
from tradepartner import cli
from tradepartner.db.models import (
    Instrument,
    NewsEntity,
    NewsRaw,
    NewsScore,
    ResultDate,
    UniverseMember,
)
from tradepartner.llm.client import OllamaClient

runner = CliRunner()
_REAL = httpx.Client


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No real HTTP: every source gets a MockTransport, and sleeps are skipped."""

    def handler(req: httpx.Request) -> httpx.Response:
        if "api/corporate-announcements" in req.url.path:
            return httpx.Response(
                200,
                json=[
                    {
                        "symbol": "INFY",
                        "desc": "Board meeting",
                        "an_dt": datetime.now(UTC).strftime("%d-%b-%Y %H:%M:%S"),
                        "attchmntFile": "https://nse/x",
                    }
                ],
            )
        if "event-calendar" in req.url.path:
            return httpx.Response(
                200,
                json=[
                    {
                        "symbol": "TCS",
                        "purpose": "Financial Results",
                        "date": (date.today() + timedelta(days=2)).strftime("%d-%b-%Y"),
                    }
                ],
            )
        if "AnnGetData" in req.url.path:
            return httpx.Response(200, json={"Table": []})
        if "gdeltproject" in str(req.url):
            return httpx.Response(
                200,
                json={
                    "articles": [
                        {
                            "url": "https://g/1",
                            "title": "Infosys wins",
                            "seendate": "20260310T043000Z",
                        }
                    ]
                },
            )
        return httpx.Response(200, text=RSS)

    transport = httpx.MockTransport(handler)
    real = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda *a, **k: real(*a, **{"transport": transport, **k})
    )  # an explicit transport (e.g. FakeOllama's) still wins

    class Free:
        def acquire(self) -> None: ...

    monkeypatch.setattr(cli.candles, "RateLimiter", lambda rate: Free())


def seed_instruments() -> None:
    with cli._session() as s:
        s.add(
            Instrument(
                instrument_token=1,
                exchange="NSE",
                tradingsymbol="INFY",
                name="Infosys Limited",
                segment="NSE",
            )
        )
        s.add(UniverseMember(universe_name="nifty100", symbol="INFY", valid_from=date(2000, 1, 1)))
        s.commit()


def test_news_fetch_stores_and_links() -> None:
    seed_instruments()
    r = runner.invoke(cli.app, ["news", "fetch", "--hours", "100000"])
    assert r.exit_code == 0, r.output
    assert "rss: stored" in r.output and "nse: stored 1" in r.output
    with cli._session() as s:
        assert s.scalar(select(func.count()).select_from(NewsRaw)) >= 2
        assert (
            s.scalar(
                select(func.count()).select_from(NewsEntity).where(NewsEntity.symbol == "INFY")
            )
            >= 2
        )
    again = runner.invoke(cli.app, ["news", "fetch", "--sources", "nse"])
    assert "stored 0" in again.output


def test_news_score_and_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    seed_instruments()
    runner.invoke(cli.app, ["news", "fetch", "--sources", "nse"])
    fake = FakeOllama()
    monkeypatch.setattr(cli, "_llm", lambda settings: fake.client())
    r = runner.invoke(cli.app, ["news", "score", "--limit", "10"])
    assert (
        r.exit_code == 0 and "scored 1, discarded 0" in r.output and "valid rate 100.0%" in r.output
    )
    stats = runner.invoke(cli.app, ["news", "stats"])
    assert "schema-valid rate 100.0%" in stats.output and "articles/hour" in stats.output
    with cli._session() as s:
        assert s.scalar(select(func.count()).select_from(NewsScore)) == 1


def test_news_score_requires_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeOllama()
    fake.up = False
    monkeypatch.setattr(cli, "_llm", lambda settings: fake.client())
    r = runner.invoke(cli.app, ["news", "score"])
    assert r.exit_code == 1 and "not reachable" in r.output


def test_news_score_stops_when_ollama_dies_mid_run(monkeypatch: pytest.MonkeyPatch) -> None:
    seed_instruments()
    runner.invoke(cli.app, ["news", "fetch", "--sources", "nse"])
    fake = FakeOllama([500, 500])
    monkeypatch.setattr(cli, "_llm", lambda settings: fake.client(retries=1))
    r = runner.invoke(cli.app, ["news", "score"])
    assert r.exit_code == 1 and "Ollama became unavailable" in r.output


def test_news_stats_empty_and_results_calendar_and_backfill() -> None:
    assert "articles 0, scores 0" in runner.invoke(cli.app, ["news", "stats"]).output
    cal = runner.invoke(cli.app, ["news", "results-calendar"])
    assert "1 result dates added" in cal.output
    assert "0 result dates added" in runner.invoke(cli.app, ["news", "results-calendar"]).output
    with cli._session() as s:
        assert s.scalars(select(ResultDate.symbol)).one() == "TCS"
    seed_instruments()
    bf = runner.invoke(
        cli.app,
        ["news", "backfill", "--from", "2026-03-01", "--to", "2026-03-10", "--window-days", "10"],
    )
    assert bf.exit_code == 0 and "historical articles stored" in bf.output


def test_fetch_resolves_ambiguous_with_the_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    with cli._session() as s:
        for i, (sym, name) in enumerate(
            [("TATAMOTORS", "Tata Motors Limited"), ("TATASTEEL", "Tata Steel Limited")]
        ):
            s.add(
                Instrument(
                    instrument_token=i, exchange="NSE", tradingsymbol=sym, name=name, segment="NSE"
                )
            )
        s.commit()
    amb_rss = RSS.replace("Infosys &amp; partners", "Tata group plans spending").replace(
        "<item><title>No date</title><link>https://a/2</link></item>", ""
    )
    llm = FakeOllama(default={"symbol": "TATASTEEL"}).client()  # built before httpx is patched
    monkeypatch.setattr(cli, "_llm", lambda settings: llm)
    transport = httpx.MockTransport(lambda r: httpx.Response(200, text=amb_rss))
    monkeypatch.setattr(httpx, "Client", lambda *a, **k: _REAL(transport=transport))
    r = runner.invoke(cli.app, ["news", "fetch", "--sources", "rss", "--hours", "100000"])
    assert r.exit_code == 0 and "needed disambiguation" in r.output
    with cli._session() as s:
        assert s.scalars(
            select(NewsEntity.symbol).where(NewsEntity.match_method == "llm")
        ).all() == ["TATASTEEL"]
    assert isinstance(cli._llm(cli.load_settings()), OllamaClient)
