"""Leakage tests (spec 10.3, CLAUDE.md rule 8): future-dated items never appear in features."""

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest
from sqlalchemy.orm import Session

from tradepartner.db.models import NewsRaw, NewsScore, ResultDate
from tradepartner.features import news as nf
from tradepartner.settings import LLMConfig, load_settings

LLM = load_settings().app.llm
# A cutoff far in the past so the cutoff rule is not what these tests exercise
OLD = LLMConfig(
    model="m", knowledge_cutoff="2020-01-01", cutoff_buffer_days=0, num_ctx=4096, keep_alive="5m"
)
AS_OF = datetime(2026, 3, 10, 4, 0, tzinfo=UTC)  # 09:30 IST


def add(db: Session, symbol: str, published: datetime, fetched: datetime | None = None, *, sent: float = 0.5,
        rel: float = 0.9, mat: float = 0.8, event: str = "earnings", conf: float = 0.9, key: str = "") -> None:  # fmt: skip
    n = NewsRaw(source="t", url=f"u{symbol}{published.isoformat()}{key}", title="t", published_at=published,
                fetched_at=fetched or published, content_hash=f"h{symbol}{published.isoformat()}{key}")  # fmt: skip
    db.add(n)
    db.flush()
    db.add(NewsScore(news_id=n.id, symbol=symbol, llm_model="m", prompt_version="v1", relevance=rel, sentiment=sent,
                     event_type=event, materiality=mat, horizon="days", confidence=conf, summary="s", raw_json={},
                     scored_at=fetched or published))  # fmt: skip
    db.commit()


def feats(db: Session, symbols: list[str] | None = None, as_of: datetime = AS_OF, **kw: object):  # type: ignore[no-untyped-def]
    symbols = ["INFY"] if symbols is None else symbols
    return nf.news_features(db, symbols, as_of, kw.pop("cfg", OLD), **kw)  # type: ignore[arg-type]


def test_future_dated_articles_never_enter_features(db: Session) -> None:
    add(db, "INFY", AS_OF - timedelta(hours=2), sent=0.4)
    before = feats(db).loc["INFY"].copy()
    # add every kind of future-looking item and prove the features do not move
    add(db, "INFY", AS_OF + timedelta(minutes=1), sent=-1.0, key="a")  # published after as_of
    add(
        db,
        "INFY",
        AS_OF - timedelta(hours=1),
        fetched=AS_OF + timedelta(hours=1),
        sent=-1.0,
        key="b",
    )  # fetched later
    add(db, "INFY", AS_OF + timedelta(days=30), sent=-1.0, event="litigation", key="c")
    after = feats(db).loc["INFY"]
    assert (before.fillna(-999) == after.fillna(-999)).all()
    assert after["n_rel_1d"] == 1 and after["ev_litigation_7d"] == 0


def test_availability_is_the_later_of_published_and_fetched_in_live_mode(db: Session) -> None:
    add(db, "INFY", AS_OF - timedelta(hours=5), fetched=AS_OF - timedelta(minutes=1))
    assert feats(db).loc["INFY", "n_rel_1d"] == 1
    assert (
        feats(db, as_of=AS_OF - timedelta(minutes=2)).loc["INFY", "n_rel_1d"] == 0
    )  # published but not yet fetched


def test_backtest_mode_uses_published_plus_delay_not_fetch_time(db: Session) -> None:
    # historical backfill: fetched long after publication must not hide the article in a backtest...
    add(db, "INFY", AS_OF - timedelta(hours=3), fetched=AS_OF + timedelta(days=400))
    assert feats(db, mode="backtest").loc["INFY", "n_rel_1d"] == 1
    # ...but the 15-minute delay still applies
    add(db, "TCS", AS_OF - timedelta(minutes=10), sent=0.5)
    assert feats(db, ["TCS"], mode="backtest").loc["TCS", "n_rel_1d"] == 0
    assert (
        feats(db, ["TCS"], mode="backtest", delay=timedelta(minutes=5)).loc["TCS", "n_rel_1d"] == 1
    )
    assert (
        feats(db, ["TCS"], mode="backtest", as_of=AS_OF + timedelta(minutes=6)).loc[
            "TCS", "n_rel_1d"
        ]
        == 1
    )


def test_llm_knowledge_cutoff_blocks_features_before_cutoff_plus_buffer(db: Session) -> None:
    cfg = LLMConfig(
        model="m",
        knowledge_cutoff="2026-01-01",
        cutoff_buffer_days=90,
        num_ctx=4096,
        keep_alive="5m",
    )
    assert nf.llm_features_start(cfg) == date(2026, 4, 1)
    add(db, "INFY", AS_OF - timedelta(hours=1))
    blocked = feats(db, cfg=cfg)  # as_of is 2026-03-10, before 2026-04-01
    assert (
        blocked.loc["INFY", "llm_features_valid"] == 0
        and blocked[nf.NEWS_COLUMNS].isna().all().all()
    )
    allowed = feats(db, cfg=cfg, as_of=datetime(2026, 4, 1, 4, 0, tzinfo=UTC))
    assert allowed.loc["INFY", "llm_features_valid"] == 1 and allowed.loc["INFY", "n_rel_1d"] == 0
    override = feats(db, cfg=cfg, enforce_cutoff=False)
    assert (
        override.loc["INFY", "n_rel_1d"] == 1
    )  # explicit opt-out (live trading), never for evaluation


def test_default_config_cutoff_is_applied() -> None:
    assert nf.llm_features_start(LLM) == date(2024, 12, 31) + timedelta(days=90)


def test_results_calendar_respects_when_the_date_was_known(db: Session) -> None:
    known = AS_OF - timedelta(days=3)
    db.add(ResultDate(symbol="INFY", result_date=date(2026, 3, 12), source="nse", created_at=known))
    db.add(
        ResultDate(
            symbol="TCS",
            result_date=date(2026, 3, 12),
            source="nse",
            created_at=AS_OF + timedelta(days=1),
        )
    )
    db.add(ResultDate(symbol="ITC", result_date=date(2026, 3, 25), source="nse", created_at=known))
    db.commit()
    f = feats(db, ["INFY", "TCS", "ITC"])
    assert f["results_in_next_7d"].to_dict() == {
        "INFY": 1.0,
        "TCS": 0.0,
        "ITC": 0.0,
    }  # TCS date not yet announced


def test_feature_values_weighting_counts_and_event_onehots(db: Session) -> None:
    add(
        db,
        "INFY",
        AS_OF - timedelta(hours=2),
        sent=0.8,
        rel=1.0,
        mat=1.0,
        event="earnings",
        key="1",
    )
    add(
        db,
        "INFY",
        AS_OF - timedelta(days=2),
        sent=-0.4,
        rel=0.5,
        mat=0.5,
        event="litigation",
        key="2",
    )
    add(db, "INFY", AS_OF - timedelta(days=5), sent=0.2, rel=1.0, mat=0.2, event="macro", key="3")
    add(
        db,
        "INFY",
        AS_OF - timedelta(days=20),
        sent=0.9,
        rel=1.0,
        mat=0.9,
        event="order_win",
        key="4",
    )
    add(db, "INFY", AS_OF - timedelta(hours=1), rel=0.3, key="5")  # below the relevance bar
    add(db, "INFY", AS_OF - timedelta(hours=1), rel=0.0, conf=0.0, key="6")  # discarded output
    f = feats(db).loc["INFY"]
    assert (f["n_rel_1d"], f["n_rel_3d"], f["n_rel_7d"]) == (1, 2, 3)
    w = np.array([1.0, 0.25, 0.2])
    s = np.array([0.8, -0.4, 0.2])
    assert f["sent_wmean_7d"] == pytest.approx((w * s).sum() / w.sum())
    assert f["sent_maxabs_7d"] == pytest.approx(0.8)
    assert f["days_since_material"] == pytest.approx(2 / 24)  # the earnings item, 2 h old
    assert (f["ev_earnings_7d"], f["ev_litigation_7d"], f["ev_macro_7d"], f["ev_order_win_7d"]) == (
        1,
        1,
        1,
        0,
    )
    assert f["llm_features_valid"] == 1


def test_no_news_gives_zero_counts_and_nan_sentiment(db: Session) -> None:
    f = feats(db, ["INFY", "TCS"])
    assert (
        (f["n_rel_7d"] == 0).all()
        and f["sent_wmean_7d"].isna().all()
        and f["days_since_material"].isna().all()
    )
    assert (f["sent_maxabs_7d"] == 0).all() and list(f.columns) == nf.ALL_COLUMNS
    assert feats(db, []).empty


def test_naive_as_of_is_rejected(db: Session) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        nf.news_features(db, ["INFY"], datetime(2026, 3, 10), OLD)
