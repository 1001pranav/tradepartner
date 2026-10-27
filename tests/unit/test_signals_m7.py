"""Explanations + number check, model strategies, signal generation, proposals, outcomes, shadow."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import DatabaseError
from sqlalchemy.orm import Session

from tests.fixtures.llm import FakeOllama
from tests.fixtures.pipeline import (
    NOW0,
    TrainedArtifact,
    add_model,
    fast_settings,
    make_env,
    train_once,
)
from tests.fixtures.synth import synth_view
from tradepartner.backtest.view import MarketView
from tradepartner.db.models import Model, OrderProposal, Signal, SignalEvent, SignalOutcome
from tradepartner.models import registry
from tradepartner.modes.state import GLOBAL, strategy_scope
from tradepartner.signals import explain, generate, outcomes, proposals, shadow
from tradepartner.strategies.factory import build_model_strategy

NOW = NOW0.astimezone(UTC)


@pytest.fixture(scope="module")
def art(tmp_path_factory: pytest.TempPathFactory) -> TrainedArtifact:
    return train_once(tmp_path_factory.mktemp("trained"))


# ---------------- explanation number check ----------------


def test_number_canonicalisation_and_check() -> None:
    assert explain.canonical("1,500.00") == "1500" and explain.canonical("62.50") == "62.5"
    assert explain.numbers_in("Price 1,500.00, up 3.5% in 20 days.") == {"1500", "3.5", "20"}
    facts = "Last close (rupees): 1500.00\nModel score (percent): 62.50"
    assert explain.unsupported_numbers("Close was 1,500 and the score 62.5 percent.", facts) == []
    assert explain.unsupported_numbers("It will reach 1800 and score 70.", facts) == ["1800", "70"]
    assert explain.numbers_in("no digits here") == set() and explain.f2(1.005) in ("1.00", "1.01")
    assert explain.canonical("1..") == "1"  # trailing dots tolerated
    assert explain.canonical("abc") == "abc"


def _facts(news: tuple[str, ...] = ("Company wins an order",)) -> explain.Facts:
    return explain.Facts(
        "INFY",
        62.5,
        (("ret_20", 0.31), ("rsi14", -0.12)),
        news,
        1500.0,
        4.2,
        1507.5,
        1440.0,
        1620.0,
        5,
    )


def test_template_explanation_always_passes_its_own_check() -> None:
    f = _facts()
    text_ = explain.template_explanation(f)
    assert (
        explain.unsupported_numbers(text_, f.text()) == []
        and "INFY" in text_
        and "wrong if" in text_
    )
    bare = explain.Facts("TCS", None, (), (), 3000.0, None, 3010.0, 2900.0, 3200.0, 3)
    t2 = explain.template_explanation(bare)
    assert explain.unsupported_numbers(t2, bare.text()) == [] and "no single dominant driver" in t2


def test_explainer_accepts_a_faithful_llm_answer() -> None:
    good = "INFY scores 62.5 percent mainly on ret_20 (0.31). It is wrong if price falls to 1440."
    llm = FakeOllama(default={"explanation": good}).client()
    exp = explain.Explainer(llm).explain(_facts())
    assert exp.used_llm and exp.text == good and exp.rejected_numbers == ()


def test_explainer_rejects_invented_numbers_then_falls_back_to_the_template() -> None:
    fake = FakeOllama(default={"explanation": "Expect a 25 percent rally to 2000 rupees."})
    exp = explain.Explainer(fake.client()).explain(_facts())
    assert (
        not exp.used_llm and set(exp.rejected_numbers) == {"25", "2000"} and len(fake.requests) == 2
    )
    assert (
        explain.unsupported_numbers(exp.text, _facts().text()) == []
    )  # what is shown is always clean
    retry = FakeOllama(
        [{"explanation": "Invented 99."}, {"explanation": "Plain words only, no figures."}]
    )
    assert explain.Explainer(retry.client()).explain(_facts()).used_llm
    bad_json = FakeOllama(default="not json")
    assert not explain.Explainer(bad_json.client()).explain(_facts()).used_llm


def test_explainer_without_or_with_a_dead_llm() -> None:
    assert not explain.Explainer(None).explain(_facts()).used_llm
    dead = FakeOllama()
    dead.up = False
    assert not explain.Explainer(dead.client(retries=0)).explain(_facts()).used_llm
    with pytest.raises(ValueError):
        explain.ExplanationOut(explanation="x" * 1000)
    with pytest.raises(ValueError):
        explain.ExplanationOut(explanation="   ")


# ---------------- model strategy ----------------


def _view() -> MarketView:
    return synth_view(n_symbols=8, n_days=760, seed=21, momentum=0.25)


def test_model_strategy_proposes_valid_geometry_and_contributions(
    db: Session, art: TrainedArtifact
) -> None:
    model = add_model(db, art)
    strat = build_model_strategy(model, fast_settings())
    view = _view()
    day = view.trading_days()[-1]
    cands = strat.candidates(day, view)
    assert 0 < len(cands) <= strat.top_k
    for c in cands:
        assert c.stop_paise < c.limit_price_paise < c.target_paise and c.limit_price_paise % 5 == 0
        assert c.prob is not None and 0 <= c.prob <= 1 and 1 <= len(c.contributions) <= 5
        assert c.horizon_days == 5 and all(name in strat.columns for name, _ in c.contributions)
    assert [c.prob for c in cands] == sorted((c.prob for c in cands), reverse=True)
    assert strat.candidates(day + timedelta(days=500), view) == []  # unknown date -> nothing
    assert not strat.scores(day, view).empty and strat.scores(day + timedelta(days=500), view).empty


def test_model_strategy_has_no_lookahead(db: Session, art: TrainedArtifact) -> None:
    model = add_model(db, art)
    view = _view()
    days = view.trading_days()
    cut = days[600]
    scrambled = {}
    for s in view.symbols:
        df = view.history(s, days[-1]).copy()
        df.loc[df["date"] > cut, ["open", "high", "low", "close"]] = 1e6
        scrambled[s] = df
    idx = view.index.copy()  # type: ignore[union-attr]
    idx.loc[idx["date"] > cut, ["open", "high", "low", "close"]] = 1e6
    other = MarketView(scrambled, idx, view.sectors)
    s1 = build_model_strategy(model, fast_settings())
    s2 = build_model_strategy(model, fast_settings())
    for d in (days[590], days[595], cut):
        assert s1.candidates(d, view) == s2.candidates(d, other)


def test_live_news_provider_fills_news_columns(db: Session, art: TrainedArtifact) -> None:
    import pandas as pd

    from tradepartner.features.build import ALL_FEATURE_COLUMNS

    model = add_model(db, art)
    model.metrics_json = {**model.metrics_json, "feature_columns": ALL_FEATURE_COLUMNS}
    seen: list[int] = []

    def provider(day, symbols):  # type: ignore[no-untyped-def]
        seen.append(len(symbols))
        return pd.DataFrame(0.0, index=symbols, columns=["n_rel_1d", "llm_features_valid"])

    strat = build_model_strategy(model, fast_settings(), provider)
    rows = strat.features_at(_view().trading_days()[-1], _view())
    assert (
        seen
        and (rows["llm_features_valid"] == 0.0).all()
        and list(rows.columns) == ALL_FEATURE_COLUMNS
    )


# ---------------- signal generation ----------------


def test_generate_is_idempotent_append_only_and_shadow_is_not_explained(
    db: Session, art: TrainedArtifact
) -> None:
    approved, shadow_m = add_model(db, art, "approved"), add_model(db, art, "shadow")
    view = _view()
    day = view.trading_days()[-1]
    fake = FakeOllama(default={"explanation": "Momentum drives it; wrong if the stop is hit."})
    ex = explain.Explainer(fake.client())
    a = generate.generate_signals(
        db, approved, build_model_strategy(approved, fast_settings()), view, day, NOW, ex
    )
    s = generate.generate_signals(
        db, shadow_m, build_model_strategy(shadow_m, fast_settings()), view, day, NOW, ex
    )
    assert a and s and all(g.explanation and g.used_llm and not g.shadow for g in a)
    assert all(
        g.explanation is None and g.shadow for g in s
    )  # shadow signals are logged, never explained
    assert (
        generate.generate_signals(
            db, approved, build_model_strategy(approved, fast_settings()), view, day, NOW, ex
        )
        == []
    )
    first = db.scalars(select(SignalEvent).where(SignalEvent.signal_id == a[0].signal_id)).one()
    assert (
        first.status == "new"
        and json.loads(first.detail)["limit"] == a[0].candidate.limit_price_paise
    )  # type: ignore[arg-type]
    shadow_ev = db.scalars(select(SignalEvent).where(SignalEvent.signal_id == s[0].signal_id)).one()
    assert shadow_ev.status == "shadow"
    sig = db.get(Signal, a[0].signal_id)
    assert (
        sig is not None
        and sig.created_at == NOW
        and sig.features_ref
        and sig.features_ref.startswith("fs2:")
    )
    with pytest.raises(DatabaseError, match="append-only"):
        db.execute(text("UPDATE signals SET score = 0"))
    db.rollback()
    with pytest.raises(DatabaseError, match="append-only"):
        db.execute(text("DELETE FROM signal_events"))
    db.rollback()
    assert proposals.candidate_from_json("X", first.detail).symbol == "X"  # type: ignore[arg-type]


def test_recent_news_lines_are_point_in_time(db: Session) -> None:
    from tradepartner.db.models import NewsRaw, NewsScore

    for i, fetched in enumerate((NOW - timedelta(hours=2), NOW + timedelta(hours=2))):
        n = NewsRaw(
            source="t",
            url=f"u{i}",
            title="t",
            published_at=NOW - timedelta(hours=3),
            fetched_at=fetched,
            content_hash=f"h{i}",
        )
        db.add(n)
        db.flush()
        db.add(NewsScore(news_id=n.id, symbol="INFY", llm_model="m", prompt_version="v1", relevance=0.9, sentiment=0.1,
                         event_type="other", materiality=0.5, horizon="days", confidence=0.9, summary=f"item {i}", raw_json={}, scored_at=NOW))  # fmt: skip
    db.commit()
    assert generate.recent_news_lines(db, "INFY", NOW) == [
        "item 0"
    ]  # the not-yet-fetched item is excluded


# ---------------- proposals and lifecycle ----------------


def _signals(db: Session, art: TrainedArtifact, tmp_path: Path, mode: str = "MANUAL"):  # type: ignore[no-untyped-def]
    from tests.fixtures.pipeline import Env

    env: Env = make_env(db, tmp_path)
    add_model(db, art, "approved")
    env.set_clock(8, 50)
    day = env.view.trading_days()[-1]
    model = db.scalars(select(Model)).one()
    gen = generate.generate_signals(
        db,
        model,
        build_model_strategy(model, env.ctx.settings),
        env.view,
        day,
        env.ctx.clock(),
        None,
    )
    env.set_clock(9, 20)
    rt = env.ctx.runtime(db)
    for sym in env.view.symbols:
        env.ctx.quotes.quotes[sym] = rt.quotes.quotes.get(sym) or __import__("tradepartner.data.quotes", fromlist=["to_quote"]).to_quote(
            env.kite.quote([f"NSE:{sym}"])[f"NSE:{sym}"], env.ctx.clock())  # fmt: skip
    rt.controller.set_mode(GLOBAL, mode, "user", "t", env.ctx.clock())  # type: ignore[arg-type]
    rt.controller.set_mode(strategy_scope("swing_v1"), "MANUAL", "user", "t", env.ctx.clock())
    if mode == "AUTO":
        from tradepartner.modes.controller import UnlockEvidence

        rt.controller.unlock_auto(
            "swing_v1",
            UnlockEvidence(True, 40, NOW.date() - timedelta(days=200), True),
            "swing_v1",
            env.ctx.clock(),
        )
        rt.controller.set_mode(strategy_scope("swing_v1"), "AUTO", "user", "t", env.ctx.clock())
    return env, rt, gen


def test_pending_signals_flow_to_shown_in_manual_mode(
    db: Session, art: TrainedArtifact, tmp_path: Path
) -> None:
    env, rt, gen = _signals(db, art, tmp_path)
    assert (
        len(proposals.pending_signals(db, "swing_v1", env.ctx.clock(), timedelta(hours=3)))
        == len(gen)
        > 0
    )
    summary = proposals.submit_pending(db, rt.router, "swing_v1", env.ctx.clock())
    assert (
        summary.shown + summary.rejected == len(gen) and summary.shown > 0 and summary.placed == 0
    )
    statuses = {proposals.current_status(db, g.signal_id) for g in gen}
    assert statuses <= {"shown", "rejected_by_risk"} and "shown" in statuses
    shown = db.scalars(select(OrderProposal).where(OrderProposal.risk_decision == "approved")).all()
    assert shown and all(
        p.qty >= 1 and p.signal_id is not None and p.mode_at_creation == "MANUAL" for p in shown
    )
    assert (
        proposals.pending_signals(db, "swing_v1", env.ctx.clock(), timedelta(hours=3)) == []
    )  # nothing re-submitted
    again = proposals.submit_pending(db, rt.router, "swing_v1", env.ctx.clock())
    assert again.shown == again.placed == again.rejected == 0


def test_auto_mode_places_orders_for_approved_proposals(
    db: Session, art: TrainedArtifact, tmp_path: Path
) -> None:
    env, rt, gen = _signals(db, art, tmp_path, "AUTO")
    summary = proposals.submit_pending(db, rt.router, "swing_v1", env.ctx.clock())
    assert summary.placed >= 1 and summary.shown == 0
    from tradepartner.db.models import Order

    orders = db.scalars(select(Order)).all()
    assert len(orders) == summary.placed and all(o.tag and o.tag.startswith("tp-") for o in orders)
    assert any(proposals.current_status(db, g.signal_id) == "approved" for g in gen)


def test_stale_signals_are_not_submitted_and_ignore_and_expire(
    db: Session, art: TrainedArtifact, tmp_path: Path
) -> None:
    env, rt, gen = _signals(db, art, tmp_path)
    late = env.ctx.clock() + timedelta(hours=5)
    assert (
        proposals.pending_signals(db, "swing_v1", late, timedelta(hours=3)) == []
    )  # max signal age
    proposals.ignore(db, gen[0].signal_id, env.ctx.clock())
    assert proposals.current_status(db, gen[0].signal_id) == "ignored"
    proposals.submit_pending(db, rt.router, "swing_v1", env.ctx.clock())
    expired = proposals.expire_unfilled(db, "swing_v1", env.ctx.clock())
    assert expired >= 1  # shown but never filled
    assert proposals.current_status(db, gen[1].signal_id) in ("expired", "rejected_by_risk")


def test_proposal_idempotency_key_is_stable(
    db: Session, art: TrainedArtifact, tmp_path: Path
) -> None:
    _, _, gen = _signals(db, art, tmp_path)
    sig = db.get(Signal, gen[0].signal_id)
    assert sig is not None
    a = proposals.to_proposal(sig, gen[0].candidate)
    assert (
        a == proposals.to_proposal(sig, gen[0].candidate)
        and a.kind == "ENTRY"
        and a.signal_id == sig.id
    )
    assert a.signal_time == sig.created_at and a.order_type == "LIMIT"


# ---------------- outcomes and shadow ----------------


def test_outcomes_match_hand_computed_forward_returns(db: Session, art: TrainedArtifact) -> None:
    approved = add_model(db, art)
    view = _view()
    days = view.trading_days()
    as_of = days[-10]  # 9 later bars exist, horizon is 5
    sig = Signal(strategy="swing_v1", model_id=approved.id, symbol="SYM00", side="BUY", as_of=__import__("tradepartner.calendar", fromlist=["x"]).session_close_utc(as_of),
                 score=0.6, prob=0.6, horizon_days=5, status="new", created_at=NOW)  # fmt: skip
    future = Signal(strategy="swing_v1", model_id=approved.id, symbol="SYM00", side="BUY", as_of=__import__("tradepartner.calendar", fromlist=["x"]).session_close_utc(days[-2]),
                    score=0.6, prob=0.6, horizon_days=5, status="new", created_at=NOW)  # fmt: skip
    db.add_all([sig, future])
    db.commit()
    n = outcomes.reveal_outcomes(db, view, fast_settings(), NOW)
    assert n == 1  # the recent signal's horizon has not passed: it stays hidden
    out = db.scalars(select(SignalOutcome)).one()
    h = view.history("SYM00", days[-1]).reset_index(drop=True)
    i = int(h.index[h["date"] == as_of][0])
    expected = h["close"].iloc[i + 5] / h["open"].iloc[i + 1] - 1
    assert out.realized_return == pytest.approx(float(expected)) and out.excess_return is not None
    assert out.label in (0, 1)
    assert outcomes.reveal_outcomes(db, view, fast_settings(), NOW) == 0  # idempotent
    assert outcomes.forward_return(view, "SYM00", as_of, 5) == pytest.approx(float(expected))
    assert outcomes.forward_return(view, "SYM00", days[-2], 5) is None
    assert outcomes.forward_return(view, "NOPE", as_of, 5) is None
    noidx = MarketView({s: view.history(s, days[-1]) for s in view.symbols})
    assert outcomes.index_forward_return(noidx, as_of, 5) is None
    assert outcomes.index_forward_return(view, days[-2], 5) is None
    assert outcomes.index_forward_return(view, date_before(days[0]), 5) is None
    with pytest.raises(DatabaseError, match="append-only"):
        db.execute(text("DELETE FROM signal_outcomes"))
    db.rollback()


def date_before(d):  # type: ignore[no-untyped-def]
    return d - timedelta(days=4000)


def _outcome(
    db: Session, model: Model, excess: float, created: datetime, sym: str = "SYM00"
) -> None:
    s = Signal(strategy="swing_v1", model_id=model.id, symbol=sym, side="BUY", as_of=created, score=0.5, prob=0.6,
               horizon_days=5, status="new", created_at=created)  # fmt: skip
    db.add(s)
    db.flush()
    db.add(
        SignalOutcome(
            signal_id=s.id,
            realized_return=excess,
            excess_return=excess,
            label=int(excess > 0.005),
            revealed_at=NOW,
        )
    )
    db.commit()


def test_shadow_comparison_passes_only_with_enough_outcomes_and_no_regression(
    db: Session, art: TrainedArtifact
) -> None:
    approved, cand = add_model(db, art, "approved"), add_model(db, art, "candidate")
    registry.start_shadow(db, cand.id, NOW - timedelta(days=15))
    for i in range(25):
        _outcome(db, cand, 0.01 + 0.0001 * i, NOW - timedelta(days=14, hours=-i))
        _outcome(db, approved, 0.002, NOW - timedelta(days=14, hours=-i))
    res = shadow.shadow_comparison(db, db.get(Model, cand.id), approved, NOW)  # type: ignore[arg-type]
    assert res["passed"] and res["candidate"]["n"] == 25 and res["approved"]["n"] == 25
    assert db.get(Model, cand.id).metrics_json["shadow_comparison"]["passed"]  # type: ignore[union-attr]
    worse = add_model(db, art, "candidate")
    registry.start_shadow(db, worse.id, NOW - timedelta(days=15))
    for i in range(25):
        _outcome(db, worse, -0.01, NOW - timedelta(days=14, hours=-i))
    assert not shadow.shadow_comparison(db, db.get(Model, worse.id), approved, NOW)["passed"]  # type: ignore[arg-type]
    few = add_model(db, art, "candidate")
    registry.start_shadow(db, few.id, NOW - timedelta(days=15))
    for i in range(3):
        _outcome(db, few, 0.05, NOW - timedelta(days=14, hours=-i))
    assert not shadow.shadow_comparison(db, db.get(Model, few.id), approved, NOW)["passed"]  # type: ignore[arg-type]
    assert shadow.shadow_comparison(db, db.get(Model, cand.id), None, NOW)[
        "passed"
    ]  # no approved model yet
    assert db.scalar(select(func.count()).select_from(Model)) == 4 and datetime.now(UTC)
