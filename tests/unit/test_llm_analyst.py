"""Ollama client, schema validation, scoring, discard rules and disambiguation (spec 10.2)."""

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from tests.fixtures.llm import GOOD, FakeOllama
from tradepartner.db.models import NewsEntity, NewsRaw, NewsScore
from tradepartner.llm import prompts
from tradepartner.llm.client import LLMUnavailableError, OllamaClient
from tradepartner.llm.schemas import NewsScoreOut, SymbolChoice, json_schema
from tradepartner.news.analyst import NewsAnalyst, ScoreRun, truncate
from tradepartner.news.entity_link import Ambiguous

NOW = datetime(2026, 3, 10, 5, 0, tzinfo=UTC)


# ---------- schema ----------


def test_schema_accepts_good_and_rejects_bad() -> None:
    assert NewsScoreOut.model_validate(GOOD).event_type == "earnings"
    for bad in (
        {**GOOD, "sentiment": 1.5},
        {**GOOD, "relevance": -0.1},
        {**GOOD, "event_type": "rumour"},
        {**GOOD, "expected_horizon": "years"},
        {**GOOD, "summary": " ".join(["word"] * 41)},
        {**GOOD, "extra": 1},
        {k: v for k, v in GOOD.items() if k != "confidence"},
    ):
        with pytest.raises(ValidationError):
            NewsScoreOut.model_validate(bad)
    assert NewsScoreOut.model_validate({**GOOD, "summary": " ".join(["w"] * 40)})
    assert "event_type" in json_schema(NewsScoreOut)["properties"]  # type: ignore[operator]
    assert SymbolChoice(symbol=None).symbol is None


# ---------- client ----------


def test_request_is_deterministic_and_schema_constrained() -> None:
    fake = FakeOllama()
    c = fake.client()
    res = c.generate("hello", NewsScoreOut)
    assert res.valid and isinstance(res.parsed, NewsScoreOut)
    body = fake.requests[0]
    assert body["options"] == {"temperature": 0, "seed": 42, "num_ctx": 4096}
    assert body["stream"] is False and body["keep_alive"] == "5m"
    assert body["format"]["properties"]["sentiment"]["maximum"] == 1  # type: ignore[index]
    assert body["messages"][0]["content"] == "hello"  # type: ignore[index]


def test_invalid_output_is_reported_then_retried_once() -> None:
    fake = FakeOllama(["not json", {**GOOD, "sentiment": 9}])
    res = fake.client().generate_with_retry("p", NewsScoreOut)
    assert not res.valid and res.parsed is None and res.error and len(fake.requests) == 2
    ok = FakeOllama(["not json", GOOD]).client().generate_with_retry("p", NewsScoreOut)
    assert ok.valid
    one_shot = FakeOllama([GOOD])
    assert (
        one_shot.client().generate_with_retry("p", NewsScoreOut).valid
        and len(one_shot.requests) == 1
    )


def test_transport_errors_retry_with_backoff_then_raise() -> None:
    sleeps: list[float] = []
    fake = FakeOllama([500, GOOD])
    assert fake.client(retries=1, sleep=sleeps.append).generate(
        "p", NewsScoreOut
    ).valid and sleeps == [1.0]
    dead = FakeOllama([500, 500, 500])
    with pytest.raises(LLMUnavailableError):
        dead.client(retries=2, sleep=lambda s: None).generate("p", NewsScoreOut)
    dead.up = False
    with pytest.raises(LLMUnavailableError):
        dead.client(retries=0).generate("p", NewsScoreOut)


def test_health_and_digest() -> None:
    fake = FakeOllama()
    c = fake.client("m")
    assert c.health() and c.model_digest() == "sha256:abc123"
    assert fake.client("other").model_digest() is None
    fake.up = False
    assert not c.health() and c.model_digest() is None
    bad = httpx.Client(
        base_url="http://t",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, text="not json")),
    )
    assert OllamaClient("m", client=bad).model_digest() is None


# ---------- prompts ----------


def test_prompts_are_versioned_files() -> None:
    p = prompts.load("news_score", "v1")
    assert (
        p.id == "news_score_v1"
        and "{symbol}" in p.template
        and "never give investment advice" in p.template
    )
    assert "INFY" in p.render(symbol="INFY", published="x", title="t", body="b")
    with pytest.raises(FileNotFoundError):
        prompts.load("news_score", "v99")


def test_truncate_cuts_on_a_word_boundary() -> None:
    assert truncate("short") == "short"
    cut = truncate("word " * 5000, 100)
    assert len(cut) <= 104 and cut.endswith(" ...")


# ---------- analyst ----------


def seed(db: Session, n: int = 3, source: str = "rss:x") -> list[NewsRaw]:
    rows = []
    for i in range(n):
        r = NewsRaw(source=source, url=f"https://x/{source}/{i}", title=f"Infosys story {i}", body="body",
                    published_at=NOW - timedelta(hours=i), fetched_at=NOW, content_hash=f"{source}{i}")  # fmt: skip
        db.add(r)
        db.flush()
        db.add(NewsEntity(news_id=r.id, symbol="INFY", match_method="alias", confidence=0.9))
        rows.append(r)
    db.commit()
    return rows


def test_scoring_stores_everything_the_spec_requires(db: Session) -> None:
    seed(db, 1)
    fake = FakeOllama()
    a = NewsAnalyst(db, fake.client(), clock=lambda: NOW)
    run = a.run()
    assert (run.scored, run.discarded) == (1, 0) and run.valid_rate == 1.0
    s = db.scalars(select(NewsScore)).one()
    assert (s.llm_model, s.llm_model_digest, s.prompt_version) == ("m", "sha256:abc123", "v1")
    assert s.raw_json["event_type"] == "earnings" and s.scored_at == NOW and s.horizon == "days"
    assert "INFY" in fake.requests[0]["messages"][0]["content"]  # type: ignore[index]
    assert a.run().scored == 0  # nothing left for this model + prompt version


def test_not_relevant_caps_relevance(db: Session) -> None:
    seed(db, 1)
    NewsAnalyst(
        db, FakeOllama(default={**GOOD, "relevant": False, "relevance": 0.9}).client()
    ).run()
    assert db.scalars(select(NewsScore)).one().relevance == 0.2


def test_invalid_twice_is_discarded_with_a_zero_weight_marker(db: Session) -> None:
    seed(db, 2)
    fake = FakeOllama(["junk", "junk", GOOD])
    run = NewsAnalyst(db, fake.client()).run()
    assert (run.scored, run.discarded) == (1, 1) and run.valid_rate == 0.5
    bad = [s for s in db.scalars(select(NewsScore)) if "error" in s.raw_json]
    assert (
        len(bad) == 1
        and bad[0].confidence == 0
        and bad[0].relevance == 0
        and "discarded" in bad[0].summary
    )
    assert len(fake.requests) == 3  # one retry for the bad article, none for the good one


def test_ordering_priority_symbols_and_exchange_first(db: Session) -> None:
    seed(db, 2, "rss:x")
    seed(db, 1, "nse")
    other = NewsRaw(
        source="rss:y",
        url="https://y/1",
        title="Other",
        published_at=NOW + timedelta(hours=5),
        fetched_at=NOW,
        content_hash="o",
    )
    db.add(other)
    db.flush()
    db.add(NewsEntity(news_id=other.id, symbol="TCS", match_method="alias", confidence=0.9))
    db.commit()
    a = NewsAnalyst(db, FakeOllama().client())
    order = [(n.source, s) for n, s in a.pending(10, frozenset({"TCS"}))]
    assert order[0] == ("rss:y", "TCS")  # priority symbol first even though not an exchange feed
    assert order[1][0] == "nse"
    assert len(a.pending(2)) == 2


def test_run_respects_time_budget_and_llm_outage(db: Session) -> None:
    seed(db, 4)
    ticks = iter([0.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0])
    run = NewsAnalyst(db, FakeOllama().client()).run(max_seconds=50, monotonic=lambda: next(ticks))
    assert run.scored == 2 and run.per_hour > 0  # stopped by the budget (the 08:00-09:00 window)
    down = FakeOllama([500] * 10)
    r2 = NewsAnalyst(db, down.client(retries=0)).run()
    assert r2.llm_down and r2.scored == 0


def test_score_run_rates_for_empty_runs() -> None:
    empty = ScoreRun()
    assert empty.per_hour == 0.0 and empty.valid_rate != empty.valid_rate  # NaN


def test_digest_is_looked_up_once(db: Session) -> None:
    seed(db, 3)
    fake = FakeOllama()
    tag_calls = {"n": 0}
    inner = fake.handler

    def counting(req: httpx.Request) -> httpx.Response:
        tag_calls["n"] += req.url.path == "/api/tags"
        return inner(req)

    http = httpx.Client(base_url="http://t", transport=httpx.MockTransport(counting))
    NewsAnalyst(db, OllamaClient("m", client=http)).run()
    assert tag_calls["n"] == 1


def test_resolve_ambiguous_only_accepts_listed_candidates(db: Session) -> None:
    row = seed(db, 1)[0]
    amb = (Ambiguous("tata", ("TATAMOTORS", "TATASTEEL")),)
    a = NewsAnalyst(db, FakeOllama(default={"symbol": "TATASTEEL"}).client())
    assert a.resolve_ambiguous(row.id, amb) == ["TATASTEEL"]
    link = db.scalars(select(NewsEntity).where(NewsEntity.symbol == "TATASTEEL")).one()
    assert link.match_method == "llm"
    # a symbol outside the candidate list is rejected; so is null
    assert (
        NewsAnalyst(db, FakeOllama(default={"symbol": "RELIANCE"}).client()).resolve_ambiguous(
            row.id, amb
        )
        == []
    )
    assert (
        NewsAnalyst(db, FakeOllama(default={"symbol": None}).client()).resolve_ambiguous(
            row.id, amb
        )
        == []
    )
    assert NewsAnalyst(db, FakeOllama(default="junk").client()).resolve_ambiguous(row.id, amb) == []
    assert a.resolve_ambiguous(99999, amb) == []
    assert (
        db.scalar(
            select(func.count()).select_from(NewsEntity).where(NewsEntity.match_method == "llm")
        )
        == 1
    )
    assert json.dumps({}) == "{}"
