"""Explaining unexplained jumps from web evidence (no network: fake searcher + fake Ollama)."""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd

from tests.fixtures.llm import FakeOllama
from tradepartner.data import jump_reasons as jr
from tradepartner.data.candles import CandleStore, normalize_rows
from tradepartner.data.quality import find_jumps
from tradepartner.news.sources.base import RawArticle
from tradepartner.news.sources.gnews import GoogleNewsSearch

NOW = datetime(2026, 10, 27, tzinfo=UTC)
DAY = date(2023, 2, 1)


class FakeSearch:
    def __init__(self, ann: list[RawArticle] | None = None, news: list[RawArticle] | None = None):
        self.ann, self.news = ann or [], news or []
        self.queries: list[tuple[str, date, date]] = []

    def headlines(self, name: str, start: date, end: date) -> list[RawArticle]:
        self.queries.append((name, start, end))
        return self.news

    def announcements(self, symbol: str, start: date, end: date) -> list[RawArticle]:
        return self.ann


def art(title: str, when: datetime = NOW) -> RawArticle:
    return RawArticle("x", f"https://x/{title}", title, "", when)


def reply(
    category: str = "short_report_or_fraud", conf: float = 0.8, ids: list[int] | None = None
) -> dict[str, object]:
    return {"category": category, "explanation": "Short seller report.",
            "evidence_ids": [1] if ids is None else ids, "confidence": conf}  # fmt: skip


def store_with_jump(tmp_path: Path) -> CandleStore:
    store = CandleStore(tmp_path)
    days = [date(2023, 1, 30) + timedelta(days=i) for i in range(5)]
    closes = [100, 100, 70, 71, 72]
    rows = [
        {"date": datetime(d.year, d.month, d.day, tzinfo=UTC), "open": c, "high": c, "low": c,
         "close": c, "volume": 1}
        for d, c in zip(days, closes, strict=True)
    ]  # fmt: skip
    store.write("day", "ABC", normalize_rows(rows, "day"))
    return store


def test_find_jumps_ignores_those_near_corporate_actions(tmp_path: Path) -> None:
    df = store_with_jump(tmp_path).read("day", "ABC")
    jumps = find_jumps(df, [])
    assert [(d, round(r, 2)) for d, r in jumps] == [(date(2023, 2, 1), -0.30)]

    class A:
        ex_date = date(2023, 2, 2)

    assert find_jumps(df, [A()]) == []  # type: ignore[list-item]


def test_explain_all_writes_reason_and_resumes(tmp_path: Path) -> None:
    store = store_with_jump(tmp_path / "c")
    out = tmp_path / "r.jsonl"
    fake = FakeOllama(default=reply())
    search = FakeSearch(
        news=[art("Short seller report hits ABC", datetime(2023, 1, 31, tzinfo=UTC))]
    )
    jumps = {"ABC": find_jumps(store.read("day", "ABC"), [])}
    done, skipped = jr.explain_all(
        fake.client(), search, store, jumps, {"ABC": "ABC Ltd"}, ["ABC"], NOW, path=out
    )
    assert (done, skipped) == (1, 0)
    r = jr.load_reasons(out)[("ABC", "2023-02-01")]
    assert r.category == "short_report_or_fraud" and "Short seller" in r.evidence[0]
    assert search.queries[0][0] == "ABC Ltd"
    prompt = fake.requests[0]["messages"][0]["content"]  # type: ignore[index]
    assert "Short seller report hits ABC" in prompt and "-30.0%" in prompt
    # resume: nothing re-asked unless redo
    done2, skipped2 = jr.explain_all(
        fake.client(), search, store, jumps, {}, ["ABC"], NOW, path=out
    )
    assert (done2, skipped2) == (0, 1)
    jr.explain_all(fake.client(), search, store, jumps, {}, ["ABC"], NOW, redo=True, path=out)
    assert len(out.read_text().splitlines()) == 2  # a correction is a new row, never an edit


def run_one(tmp_path: Path, reply_: object, news: list[RawArticle], nifty: list[int] | None = None):  # type: ignore[no-untyped-def]
    store = store_with_jump(tmp_path / "c")
    if nifty:
        n = store.read("day", "ABC").copy()
        n["close"] = nifty
        store.write("day", "NIFTY 50", n)
    jumps = {"ABC": find_jumps(store.read("day", "ABC"), [])}
    out = tmp_path / "r.jsonl"
    jr.explain_all(FakeOllama(default=reply_).client(), FakeSearch(news=news), store, jumps,
                   {"ABC": "Abcco Ltd"}, ["ABC"], NOW, path=out)  # fmt: skip
    return jr.load_reasons(out)[("ABC", "2023-02-01")]


def test_reason_must_cite_evidence_naming_the_company(tmp_path: Path) -> None:
    ok = run_one(tmp_path / "a", reply("results"), [art("Abcco Ltd Q3 profit jumps")])
    assert ok.category == "results"
    other = run_one(tmp_path / "b", reply("results"), [art("Welspun India Q4 profit falls")])
    assert other.category == "unknown" and "unsupported" in other.explanation
    nocite = run_one(tmp_path / "c", reply("results", ids=[]), [art("Abcco Ltd Q3 profit")])
    assert nocite.category == "unknown"
    badid = run_one(tmp_path / "d", reply("results", ids=[9]), [art("Abcco Ltd Q3 profit")])
    assert badid.category == "unknown"


def test_market_wide_needs_market_data_and_confidence_scale_is_normalised(tmp_path: Path) -> None:
    news = [art("Unrelated")]
    no = run_one(tmp_path / "a", reply("market_wide"), news)
    assert no.category == "unknown"
    yes = run_one(tmp_path / "b", reply("market_wide", conf=85), news, nifty=[100, 100, 95, 95, 95])
    assert yes.category == "market_wide" and yes.confidence == 0.85


def test_no_evidence_forces_unknown_and_limit_and_llm_down(tmp_path: Path) -> None:
    store = store_with_jump(tmp_path / "c")
    jumps = {"ABC": find_jumps(store.read("day", "ABC"), [])}
    out = tmp_path / "r.jsonl"
    fake = FakeOllama(default=reply("results"))  # model invents a reason with no evidence
    jr.explain_all(fake.client(), FakeSearch(), store, jumps, {}, ["ABC"], NOW, path=out)
    r = jr.load_reasons(out)[("ABC", "2023-02-01")]
    assert r.category == "unknown" and r.confidence == 0
    assert (
        jr.explain_all(fake.client(), FakeSearch(), store, jumps, {}, ["ABC"], NOW, limit=0,
                       redo=True, path=out)[0] == 0
    )  # fmt: skip
    down = FakeOllama()
    down.up = False
    msgs: list[str] = []
    assert jr.explain_all(down.client(), FakeSearch(), store, jumps, {}, ["ABC"], NOW,
                          redo=True, path=out, progress=msgs.append) == (0, 0)  # fmt: skip
    assert msgs == ["Ollama unavailable: stopping"]


def test_invalid_model_output_is_recorded_as_unknown(tmp_path: Path) -> None:
    store = store_with_jump(tmp_path / "c")
    jumps = {"ABC": find_jumps(store.read("day", "ABC"), [])}
    out = tmp_path / "r.jsonl"
    fake = FakeOllama(default={"category": "banana"})
    jr.explain_all(fake.client(), FakeSearch(news=[art("x")]), store, jumps, {}, ["ABC"], NOW,
                   path=out)  # fmt: skip
    assert jr.load_reasons(out)[("ABC", "2023-02-01")].explanation == "model output invalid"


def test_market_context_counts_nifty_and_broad_moves(tmp_path: Path) -> None:
    store = store_with_jump(tmp_path)
    nifty = store.read("day", "ABC").copy()
    nifty["close"] = [100, 100, 95, 95, 95]
    store.write("day", "NIFTY 50", nifty)
    text = jr.market_context(store, ["ABC"], DAY)
    assert "Nifty 50 moved -5.0%" in text and "1 stocks" in text
    assert "0 stocks" in jr.market_context(store, ["ABC"], date(2023, 2, 3))


def test_evidence_prefers_exchange_announcements_and_is_capped() -> None:
    near = datetime(2023, 2, 1, 6, tzinfo=UTC)
    ann = [art("ABC: Outcome of board meeting", near)]
    news = [art(f"headline {i}", near - timedelta(hours=i)) for i in range(30)] + [art("nodate")]
    news[-1] = RawArticle("x", "u", "nodate", "", None)
    ev = jr.collect_evidence(FakeSearch(ann, news), "ABC", "ABC Ltd", DAY)
    assert ev[0][0] == "NSE announcement" and len(ev) == jr.MAX_EVIDENCE


def test_google_news_search_builds_date_window_and_survives_errors() -> None:
    seen: list[str] = []

    def ok(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.params["q"])
        return httpx.Response(
            200,
            text="<rss><channel><item><title>T</title><link>http://a</link></item></channel></rss>",
        )

    g = GoogleNewsSearch(httpx.Client(transport=httpx.MockTransport(ok)))
    assert [a.title for a in g.search('"ABC"', date(2023, 1, 29), date(2023, 2, 1))] == ["T"]
    assert seen == ['"ABC" after:2023-01-29 before:2023-02-02']
    bad = GoogleNewsSearch(
        httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    )
    assert bad.search("x", DAY, DAY) == []


def test_jump_reasons_frame_empty_and_filled(tmp_path: Path) -> None:
    from tradepartner.ui import data

    assert data.jump_reasons_frame().empty
    jr._append(jr.JumpReason("ABC", "2023-02-01", -0.3, "unknown", "none", 0.0, ["a"]))
    assert data.jump_reasons_frame().iloc[0]["move"] == "-30.0%"
    assert isinstance(pd.DataFrame(), pd.DataFrame)
