"""Walk-forward splits, training, the promotion gate, registry, approval and drift (spec 10.5-10.6)."""

from datetime import UTC, date, datetime, timedelta
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from tests.fixtures.synth import bdays
from tradepartner.db.models import AuditLog, Model
from tradepartner.models import approval, drift, evaluate, registry, train
from tradepartner.models.cv import assert_no_overlap, inner_split, walk_forward_splits
from tradepartner.models.registry import ModelError

NOW = datetime(2026, 6, 1, 5, 0, tzinfo=UTC)


# ---------- walk-forward splits ----------


@given(horizon=st.integers(1, 70), n=st.integers(600, 1100), months=st.sampled_from([1, 3, 6]))
@settings(max_examples=40, deadline=None)
def test_splits_never_let_a_label_window_touch_the_test_period(
    horizon: int, n: int, months: int
) -> None:
    dates = bdays(date(2020, 1, 1), n)
    splits = walk_forward_splits(dates, horizon, months, 300)
    assert splits, "expected at least one window"
    prev_end = None
    for sp in splits:
        assert_no_overlap(sp, horizon)
        assert sp.train_idx.max() + horizon < sp.test_idx.min()  # embargo == horizon (purge)
        assert sp.train_idx[0] == 0 and len(sp.train_idx) >= 300  # expanding from the start
        assert dates[sp.train_idx.max()] == sp.train_end and sp.train_end < sp.test_start
        assert prev_end is None or sp.test_start > prev_end  # windows roll forward, never overlap
        prev_end = sp.test_end


def test_split_edge_cases() -> None:
    assert walk_forward_splits([], 5) == []
    with pytest.raises(ValueError, match="sorted"):
        walk_forward_splits([date(2024, 1, 2), date(2024, 1, 1)], 5)
    short = bdays(date(2024, 1, 1), 200)
    assert walk_forward_splits(short, 5, 3, 504) == []  # not enough history to train
    dates = bdays(date(2020, 1, 1), 700)
    gap = [
        d for d in dates if not (date(2021, 6, 1) <= d < date(2021, 9, 1))
    ]  # a whole empty test block
    assert all(sp.test_idx.size for sp in walk_forward_splits(gap, 5, 3, 300))
    bad = walk_forward_splits(dates, 5, 3, 300)[0]
    object.__setattr__(bad, "train_idx", np.arange(0, bad.test_idx.min() + 1))
    with pytest.raises(AssertionError):
        assert_no_overlap(bad, 5)
    tr, va = inner_split(1000, 5)
    assert tr.max() + 5 < va.min() and len(va) == 200


# ---------- training ----------


def _toy(n_dates: int = 700, n_sym: int = 6, signal: float = 1.0, seed: int = 0):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    dates = bdays(date(2020, 1, 1), n_dates)
    idx = pd.MultiIndex.from_product(
        [dates, [f"S{i}" for i in range(n_sym)]], names=["date", "symbol"]
    )
    x = pd.DataFrame(
        {
            "f1": rng.normal(size=len(idx)),
            "f2": rng.normal(size=len(idx)),
            "f3": rng.normal(size=len(idx)),
        },
        index=idx,
    )
    fwd = signal * 0.02 * x["f1"] + rng.normal(0, 0.02, len(idx))
    lab = pd.DataFrame({"fwd_ret": fwd, "excess": fwd, "y": (fwd > 0).astype(float)}, index=idx)
    return x, lab


def test_walk_forward_trains_only_on_the_past_and_finds_the_signal() -> None:
    x, lab = _toy()
    wf = train.walk_forward(x, lab, ["f1", "f2", "f3"], objective="binary", horizon=5, n_trials=2,
                            test_months=3, min_train_days=300)  # fmt: skip
    assert (
        len(wf.splits) >= 3
        and len(wf.oos) > 1000
        and set(wf.oos.columns) >= {"pred", "y", "window"}
    )
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(wf.oos["y"], wf.oos["pred"]) > 0.65  # the planted signal is learnt
    for sp in wf.splits:  # OOS predictions only ever fall inside their own test window
        w = wf.oos[wf.oos["window"] == sp.window].index.get_level_values("date")
        assert w.min() >= sp.test_start and w.max() <= sp.test_end
    booster, t0, t1 = train.fit_final(x, lab, ["f1", "f2", "f3"], wf, "binary")
    share = train.gain_share(booster)
    assert share["f1"] > 0.5 and sum(share.values()) == pytest.approx(1.0)
    assert t0 == min(lab.index.get_level_values("date")) and t1 == max(
        lab.index.get_level_values("date")
    )


def test_walk_forward_is_deterministic_and_regression_works() -> None:
    x, lab = _toy(500)
    kw = dict(objective="regression", horizon=5, n_trials=1, test_months=3, min_train_days=250)
    a = train.walk_forward(x, lab, ["f1", "f2", "f3"], **kw)  # type: ignore[arg-type]
    b = train.walk_forward(x, lab, ["f1", "f2", "f3"], **kw)  # type: ignore[arg-type]
    pd.testing.assert_series_equal(a.oos["pred"], b.oos["pred"])
    assert a.window_params and "num_leaves" in a.params


def test_tuning_is_bounded_and_skips_tiny_folds() -> None:
    x, lab = _toy(400)
    xs, ys = x.iloc[:2000], lab["y"].to_numpy()[:2000]
    params, rounds = train.tune(
        xs, ys, 5, "binary", 1, n_trials=999
    )  # capped at 50 -> keep it fast
    assert rounds >= 1 and "learning_rate" in params
    assert train.tune(x.iloc[:30], lab["y"].to_numpy()[:30], 5, "binary", 1, 5)[1] == 100
    assert train.tune(xs, ys, 5, "binary", 1, 0)[1] == 100
    assert train.MAX_TRIALS == 50


def test_walk_forward_with_no_usable_windows_returns_empty() -> None:
    x, lab = _toy(120)
    wf = train.walk_forward(
        x, lab, ["f1"], objective="binary", horizon=5, n_trials=1, min_train_days=504
    )
    assert wf.oos.empty and wf.splits == []


# ---------- promotion gate ----------


def _oos(
    n_days: int = 320, n_sym: int = 8, skill: float = 1.0, seed: int = 1
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    dates = bdays(date(2024, 1, 1), n_days)
    idx = pd.MultiIndex.from_product(
        [dates, [f"S{i}" for i in range(n_sym)]], names=["date", "symbol"]
    )
    fwd = rng.normal(0.0, 0.03, len(idx))
    pred = skill * fwd + (1 - skill) * rng.normal(0, 0.03, len(idx))
    window = np.array([i // (63 * n_sym) for i in range(len(idx))])
    oos = pd.DataFrame(
        {
            "fwd_ret": fwd,
            "excess": fwd,
            "y": (fwd > 0).astype(float),
            "pred": pred,
            "window": window,
        },
        index=idx,
    )
    base = pd.Series(rng.normal(size=len(idx)), index=idx)  # an uninformative baseline
    return oos, base


def test_gate_passes_a_skilled_model_but_approval_stays_pending() -> None:
    oos, base = _oos(skill=1.0)
    rep = evaluate.evaluate_gate(
        oos, base, {"f": 0.2}, horizon=5, cost_rt=0.004, slippage_bps=10, top_k=3
    )
    assert rep.ready_for_review and rep.failed == [] and rep.approved is False
    names = [c.name for c in rep.checks]
    assert names[-1] == "human_approval" and not rep.checks[-1].passed  # never automatic
    assert len(rep.checks) == 7 and rep.to_dict()["failed"] == []


def test_gate_fails_each_condition() -> None:
    oos, base = _oos(skill=0.0)
    rep = evaluate.evaluate_gate(
        oos, base, {"f": 0.9}, horizon=5, cost_rt=0.004, slippage_bps=10, top_k=3
    )
    assert not rep.ready_for_review
    assert "feature_concentration" in rep.failed
    assert set(rep.failed) - {"feature_concentration"}  # a skill-less model fails at least one more
    few, b2 = _oos(n_days=40, skill=1.0)
    small = evaluate.evaluate_gate(
        few, b2, {"f": 0.2}, horizon=5, cost_rt=0.004, slippage_bps=10, top_k=1
    )
    assert "min_trades" in small.failed
    costly = evaluate.evaluate_gate(
        oos, base, {"f": 0.2}, horizon=5, cost_rt=0.2, slippage_bps=10, top_k=3
    )
    assert "double_slippage_positive" in costly.failed
    empty = evaluate.evaluate_gate(
        oos.iloc[0:0], base.iloc[0:0], {}, horizon=5, cost_rt=0.0, slippage_bps=1
    )
    assert not empty.ready_for_review


def test_gate_double_slippage_check_can_fail_alone() -> None:
    oos, base = _oos(skill=1.0)
    thin = oos.assign(
        fwd_ret=oos["fwd_ret"].clip(-0.001, 0.001) + 0.0015
    )  # edge of ~0.15% per trade
    rep = evaluate.evaluate_gate(
        thin, base, {"f": 0.2}, horizon=5, cost_rt=0.001, slippage_bps=60, top_k=3
    )
    assert "double_slippage_positive" in rep.failed


def test_simulate_one_position_per_symbol_and_floor() -> None:
    oos, _ = _oos(n_days=60, n_sym=4)
    sim = evaluate.simulate(oos, oos["pred"], horizon=5, top_k=4, cost_rt=0.0)
    held = sim.trades.groupby("symbol")["date"].apply(lambda s: sorted(s))
    dates = sorted(oos.index.get_level_values("date").unique())
    pos = {d: i for i, d in enumerate(dates)}
    for ds in held:
        assert all(pos[b] - pos[a] >= 6 for a, b in pairwise(ds))  # entry next open + 5 days
    none = evaluate.simulate(oos, oos["pred"], 5, 4, 0.0, floor=99.0)
    assert none.trades.empty and (none.daily == 0).all()


def test_metric_helpers() -> None:
    assert evaluate.max_drawdown(pd.Series([0.1, -0.5, 0.0])) == pytest.approx(-0.5)
    assert evaluate.max_drawdown(pd.Series(dtype=float)) == 0.0
    assert np.isnan(evaluate.sharpe(pd.Series([0.0, 0.0, 0.0]))) and np.isnan(
        evaluate.sharpe(pd.Series([1.0]))
    )
    assert np.isnan(
        evaluate.positive_window_fraction(
            evaluate.Sim(pd.DataFrame(columns=["net", "window"]), pd.Series(dtype=float)), []
        )
    )
    q = evaluate.prediction_quality(
        pd.DataFrame({"y": [0, 1, 0, 1.0], "pred": [0.1, 0.9, 0.4, 0.6]}), True
    )
    assert q["auc"] == 1.0 and 0 <= q["brier"] < 0.2 and q["base_rate"] == 0.5
    r = evaluate.prediction_quality(
        pd.DataFrame({"y": [1.0, 2.0, 3.0], "pred": [1.0, 2.0, 3.0]}), False
    )
    assert (
        r["spearman"] == pytest.approx(1.0)
        and evaluate.prediction_quality(pd.DataFrame(columns=["y", "pred"]), True) == {}
    )


# ---------- registry and approval ----------


def _booster():  # type: ignore[no-untyped-def]
    x, lab = _toy(120, 4)
    return train.fit(x, lab["y"].to_numpy(), {**train.base_params("binary", 1), "num_leaves": 4}, 5)


GATE_OK = {"gate": {"ready_for_review": True, "failed": []}}


def _candidate(db: Session, tmp_path, metrics=None, strategy="swing_v1") -> Model:  # type: ignore[no-untyped-def]
    return registry.register_candidate(db, strategy, _booster(), algo="lightgbm", feature_set_version="fs1",
                                       train_start=date(2020, 1, 1), train_end=date(2025, 1, 1), params={},
                                       metrics=metrics if metrics is not None else dict(GATE_OK), now=NOW, root=tmp_path)  # fmt: skip


def test_register_candidate_saves_a_loadable_artifact(db: Session, tmp_path) -> None:  # type: ignore[no-untyped-def]
    m = _candidate(db, tmp_path)
    assert m.status == "candidate" and m.approved_by is None and m.version.endswith("-1")
    assert registry.load_booster(m).num_trees() == 5
    assert _candidate(db, tmp_path).version.endswith("-2")
    assert (
        registry.by_status(db, "swing_v1", "candidate")
        and registry.current_approved(db, "swing_v1") is None
    )
    with pytest.raises(ModelError, match="not found"):
        registry.get(db, 999)


def test_nothing_gets_approved_without_the_full_process(db: Session, tmp_path) -> None:  # type: ignore[no-untyped-def]
    m = _candidate(db, tmp_path)
    with pytest.raises(ModelError, match="only the user"):
        approval.approve(db, m.id, "system", NOW)
    with pytest.raises(ModelError, match="shadow first"):
        approval.approve(db, m.id, "user", NOW)
    registry.start_shadow(db, m.id, NOW)
    with pytest.raises(ModelError, match="14 days"):
        approval.approve(db, m.id, "user", NOW + timedelta(days=13))
    with pytest.raises(ModelError, match="shadow comparison"):
        approval.approve(db, m.id, "user", NOW + timedelta(days=14))
    registry.record_shadow_comparison(db, m.id, {"passed": True}, NOW + timedelta(days=14))
    assert approval.approve(db, m.id, "user", NOW + timedelta(days=14)).status == "approved"
    assert db.get(Model, m.id).approved_by == "user"  # type: ignore[union-attr]
    with pytest.raises(ModelError, match="candidate can enter shadow"):
        registry.start_shadow(db, m.id, NOW)


def test_a_model_that_failed_the_gate_cannot_be_approved(db: Session, tmp_path) -> None:  # type: ignore[no-untyped-def]
    m = _candidate(db, tmp_path, {"gate": {"ready_for_review": False, "failed": ["min_trades"]}})
    registry.start_shadow(db, m.id, NOW)
    registry.record_shadow_comparison(db, m.id, {"passed": True}, NOW)
    with pytest.raises(ModelError, match="promotion gate not passed: min_trades"):
        approval.approve(db, m.id, "user", NOW + timedelta(days=30))
    bare = _candidate(db, tmp_path, {})
    assert "no gate report" in " ".join(approval.approval_blockers(bare, NOW))


def test_rollback_restores_the_previous_model(db: Session, tmp_path) -> None:  # type: ignore[no-untyped-def]
    later = NOW + timedelta(days=20)
    a, b = _candidate(db, tmp_path), _candidate(db, tmp_path)
    for m in (a, b):
        registry.start_shadow(db, m.id, NOW)
        registry.record_shadow_comparison(db, m.id, {"passed": True}, NOW)
    approval.approve(db, a.id, "user", later)
    approval.approve(db, b.id, "user", later)
    assert (
        registry.current_approved(db, "swing_v1").id == b.id
        and db.get(Model, a.id).status == "retired"
    )  # type: ignore[union-attr]
    with pytest.raises(ModelError, match="only the user"):
        approval.rollback(db, "swing_v1", "system", later)
    back = approval.rollback(db, "swing_v1", "user", later)
    assert back.id == a.id and registry.current_approved(db, "swing_v1").id == a.id  # type: ignore[union-attr]
    assert db.get(Model, b.id).status == "retired"  # type: ignore[union-attr]
    with pytest.raises(ModelError, match="no previous model"):
        approval.rollback(db, "swing_v1", "user", later)
    with pytest.raises(ModelError, match="no approved model"):
        approval.rollback(db, "positional_v1", "user", later)
    assert approval.retire(db, a.id, "user", later).status == "retired"
    with pytest.raises(ModelError, match="only the user"):
        approval.retire(db, a.id, "x", later)
    assert (
        db.scalar(
            select(func.count()).select_from(AuditLog).where(AuditLog.action == "model_approved")
        )
        == 2
    )


# ---------- drift ----------


def test_psi_detects_shift_and_ignores_noise() -> None:
    rng = np.random.default_rng(0)
    train_vals = pd.Series(rng.normal(0, 1, 5000))
    b = drift.feature_bins(train_vals)
    assert len(b["edges"]) == len(b["expected"]) + 1 and sum(b["expected"]) == pytest.approx(1.0)
    same = drift.psi(b["edges"], b["expected"], pd.Series(rng.normal(0, 1, 2000)))
    shifted = drift.psi(b["edges"], b["expected"], pd.Series(rng.normal(1.5, 1, 2000)))
    assert same < 0.05 and shifted > 0.25
    assert np.isnan(drift.psi(b["edges"], b["expected"], pd.Series([np.nan])))
    const = drift.feature_bins(pd.Series([3.0] * 50))
    assert len(const["edges"]) >= 3


def test_drift_report_flags_psi_and_brier() -> None:
    rng = np.random.default_rng(1)
    bins = {"f": drift.feature_bins(pd.Series(rng.normal(0, 1, 3000)))}
    calm = drift.check_drift(
        bins,
        pd.DataFrame({"f": rng.normal(0, 1, 500)}),
        0.2,
        rng.uniform(0.3, 0.7, 80),
        rng.integers(0, 2, 80).astype(float),
        brier_margin=1.0,
    )
    assert not calm.flagged and calm.n_outcomes == 60 and calm.brier is not None
    moved = drift.check_drift(
        bins, pd.DataFrame({"f": rng.normal(2, 1, 500)}), None, np.array([]), np.array([])
    )
    assert moved.flagged and "PSI" in moved.reasons[0]
    probs = np.full(60, 0.95)
    bad = drift.check_drift({}, pd.DataFrame(), 0.20, probs, np.zeros(60))
    assert bad.flagged and "Brier" in bad.reasons[0] and bad.brier == pytest.approx(0.9025)
    few = drift.check_drift({}, pd.DataFrame(), 0.2, probs[:10], np.zeros(10))
    assert not few.flagged and few.brier is None
    assert (
        drift.check_drift(
            {"missing": bins["f"]}, pd.DataFrame({"x": [1.0]}), None, np.array([]), np.array([])
        ).psi
        == {}
    )
