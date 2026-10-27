"""M9: intraday model + gate, signals, MIS stops, the live-intraday risk gate, unlock, jobs, CLI."""

from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from tests.fixtures.intraday import breakout_day, synth_bars
from tests.fixtures.market import dt
from tests.fixtures.pipeline import NOW0, add_model, make_env, train_once
from tests.fixtures.risk import LTP, cfg, ctx, entry, limits
from tests.fixtures.world import STRATEGY, World
from tradepartner import cli, jobs
from tradepartner.data import candles
from tradepartner.db.models import Model, Order, Signal, SignalEvent, UniverseMember
from tradepartner.execution.mis_protect import mis_intents, place_mis_stops
from tradepartner.models import intraday as im
from tradepartner.modes.controller import MIN_MONTHS_BY_STRATEGY, UnlockEvidence, unlock_blockers
from tradepartner.risk import limits as risk_limits
from tradepartner.risk import rules
from tradepartner.risk.engine import RiskEngine
from tradepartner.settings import load_settings
from tradepartner.signals.intraday import generate_intraday_signals
from tradepartner.strategies import intraday as strat

runner = CliRunner()
NOW = datetime(2026, 3, 10, 5, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def bars():  # type: ignore[no-untyped-def]
    return {f"IDX{i}": synth_bars(i, n_days=130) for i in range(4)}


# ---------------- model + gate ----------------


def test_intraday_training_registers_a_paper_only_candidate(db: Session, bars, tmp_path) -> None:  # type: ignore[no-untyped-def]
    rep = im.train_intraday(
        db, bars, load_settings(), NOW, n_trials=1, test_months=1, min_train_days=40
    )
    m = db.scalars(select(Model)).one()
    assert m.strategy == "intraday_v1" and m.status == "candidate" and m.approved_by is None
    gate = rep["gate"]
    assert gate["approved"] is False and gate["checks"][-1]["name"] == "human_approval"  # type: ignore[index]
    assert m.metrics_json["horizon_bars"] == 6 and m.metrics_json["paper_gate_passed"] is False
    assert any("paper-only" in w for w in m.metrics_json["warnings"])
    assert m.metrics_json["feature_columns"] and 0 < rep["cost_fraction"] < 0.01  # type: ignore[operator]
    assert not im.intraday_cost_fraction(load_settings()) <= 0


def test_intraday_matrix_has_no_cross_day_labels_and_rejects_empty(bars) -> None:  # type: ignore[no-untyped-def]
    fm, lm = im.build_matrix(bars, 0.001)
    assert fm.index.names == ["date", "symbol", "ts"] and fm.index.equals(lm.index)
    day = lm.xs(sorted(set(lm.index.get_level_values("date")))[3], level="date").xs(
        "IDX0", level="symbol"
    )
    assert day["fwd_ret"].iloc[-6:].isna().all() and day["fwd_ret"].iloc[:-6].notna().all()
    with pytest.raises(im.ModelError, match="no usable"):
        im.build_matrix({"X": synth_bars(1, n_days=0)}, 0.001)


def test_simulate_intraday_holds_each_name_once_and_counts_daily() -> None:
    from tradepartner.models import evaluate

    fm, lm = im.build_matrix(
        {"IDX0": synth_bars(0, n_days=20), "IDX1": synth_bars(1, n_days=20)}, 0.0
    )
    oos = lm.assign(pred=fm["vol_ratio"].fillna(0), window=0)
    sim = evaluate.simulate_intraday(oos, oos["pred"], 6, 1, 0.0, floor=1.5)
    assert len(sim.trades) > 0 and len(sim.daily) == 20
    assert evaluate.simulate_intraday(oos, oos["pred"], 6, 1, 0.0, floor=1e9).trades.empty


# ---------------- signals ----------------


def test_one_intraday_signal_per_symbol_per_day_and_shadow_logged(db: Session) -> None:
    from tests.fixtures.pipeline import TrainedArtifact

    art = TrainedArtifact("x", {"feature_columns": ["f"]}, {}, date(2020, 1, 1), date(2025, 1, 1))
    model = add_model(db, art, "approved", "intraday_v1")
    shadow = add_model(db, art, "shadow", "intraday_v1")
    b = breakout_day()
    sc = strat.IntradayScanner(None, None, {"min_vol_ratio": 1.5})
    t = b["ts"].iloc[7].to_pydatetime()
    made = generate_intraday_signals(db, model, sc, {"ORB": b}, t)
    assert len(made) == 1 and made[0].candidate.product == "MIS" and not made[0].shadow
    assert (
        generate_intraday_signals(db, model, sc, {"ORB": b}, t + timedelta(minutes=5)) == []
    )  # same day, same name
    s2 = generate_intraday_signals(db, shadow, sc, {"ORB": b}, t)
    assert len(s2) == 1 and s2[0].shadow
    statuses = [e.status for e in db.scalars(select(SignalEvent).order_by(SignalEvent.id))]
    assert statuses == ["new", "shadow"]
    sig = db.scalars(select(Signal).where(Signal.model_id == model.id)).one()
    assert (
        sig.strategy == "intraday_v1"
        and sig.horizon_days == 0
        and sig.features_ref
        and sig.features_ref.startswith("intraday-fs1")
    )


# ---------------- MIS stops ----------------


def test_mis_positions_get_an_exchange_stop_through_the_router(db: Session) -> None:
    w = World(db)
    w.go_auto()
    r = w.submit_entry(product="MIS")
    assert r.status == "placed"
    w.advance(minutes=1)
    w.context_for(STRATEGY)  # rebuild lots
    (it,) = mis_intents(db)
    assert it.symbol == "INFY" and it.qty == r.decision.sizing.qty and it.stop_paise < LTP  # type: ignore[union-attr]
    rt = SimpleNamespace(session=db, broker=w.broker, router=w.router)
    assert place_mis_stops(rt, w.now) == ["INFY"]  # type: ignore[arg-type]
    (sl,) = [o for o in w.broker.open_orders() if o.order_type == "SL-M"]
    assert (
        sl.side == "SELL"
        and sl.product == "MIS"
        and sl.trigger_paise == it.stop_paise
        and sl.qty == it.qty
    )
    w.advance(seconds=5)
    assert place_mis_stops(rt, w.now) == []  # already protected: nothing new
    # the stop fires when price falls through it (paper exchange)
    w.set_price("INFY", it.stop_paise - 100)
    w.broker.on_quote("INFY")
    assert w.broker.get_order(sl.broker_order_id).status == "COMPLETE"  # type: ignore[union-attr]


def test_mis_stop_is_replaced_when_the_position_changes_size(db: Session) -> None:
    w = World(db)
    w.go_auto()
    w.submit_entry(product="MIS")
    w.advance(minutes=1)
    w.context_for(STRATEGY)
    rt = SimpleNamespace(session=db, broker=w.broker, router=w.router)
    place_mis_stops(rt, w.now)  # type: ignore[arg-type]
    old = next(o for o in w.broker.open_orders() if o.order_type == "SL-M")
    from tradepartner.db.models import Lot

    lot = db.scalars(select(Lot)).one()
    lot.qty_remaining -= 1
    db.commit()
    w.advance(seconds=5)
    assert place_mis_stops(rt, w.now) == ["INFY"]  # type: ignore[arg-type]
    assert w.broker.get_order(old.broker_order_id).status == "CANCELLED"  # type: ignore[union-attr]
    assert db.scalar(select(func.count()).select_from(Order).where(Order.order_type == "SL-M")) == 2


# ---------------- the live-intraday gate (risk engine) ----------------


def test_live_mis_entries_are_blocked_until_paper_evidence_exists() -> None:
    c = cfg()
    p = entry(product="MIS", strategy="intraday_v1")
    paper = RiskEngine(c).check(p, ctx())
    assert paper.approved, paper.reasons  # paper trading is always allowed
    live = RiskEngine(c).check(p, ctx(live_broker=True))
    assert not live.approved and any(r.startswith("intraday_gate") for r in live.reasons)
    cleared = RiskEngine(c).check(p, ctx(live_broker=True, limits=limits(intraday_live_ok=True)))
    assert cleared.approved, cleared.reasons
    cnc = RiskEngine(c).check(entry(product="CNC"), ctx(live_broker=True))
    assert cnc.approved  # swing/positional are not held back by the intraday gate
    exit_p = RiskEngine(c).check
    assert exit_p is not None and rules.rule_intraday_gate(p, ctx(), c, rules.Sizing()).passed


def test_intraday_live_ok_needs_model_gate_trades_and_three_months(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.fixtures.pipeline import TrainedArtifact

    now = datetime(2026, 9, 1, tzinfo=UTC)
    assert not risk_limits.intraday_live_ok(db, now)  # no model at all
    art = TrainedArtifact(
        "x",
        {"feature_columns": ["f"], "paper_gate_passed": False},
        {},
        date(2020, 1, 1),
        date(2025, 1, 1),
    )
    m = add_model(db, art, "approved", "intraday_v1")
    assert not risk_limits.intraday_live_ok(db, now)  # gate flag not set
    m.metrics_json = {**m.metrics_json, "paper_gate_passed": True}
    db.commit()
    assert not risk_limits.intraday_live_ok(db, now)  # no paper trades yet

    def fake_trades(n: int, first: date) -> list[risk_limits._Closed]:
        return [
            risk_limits._Closed(
                i, first + timedelta(days=i), dt(2026, 3, 1, 10, 0), 1, "intraday_v1", "X", False
            )
            for i in range(n)
        ]

    from tradepartner.db.models import Fill

    for i in range(45):
        db.add(Fill(id=i, broker_trade_id=f"t{i}", origin="paper", account="p", symbol="X", exchange="NSE", side="BUY", product="MIS", qty=1, price_paise=100,
                    filled_at=now, trade_date=date(2026, 3, 1), source="paper"))  # fmt: skip
    db.commit()
    monkeypatch.setattr(risk_limits, "closed_trades", lambda s: fake_trades(10, date(2026, 3, 1)))
    assert not risk_limits.intraday_live_ok(db, now)  # only 10 closed trades
    monkeypatch.setattr(risk_limits, "closed_trades", lambda s: fake_trades(45, date(2026, 8, 1)))
    assert not risk_limits.intraday_live_ok(db, now)  # tracking started 1 month ago
    monkeypatch.setattr(risk_limits, "closed_trades", lambda s: fake_trades(45, date(2026, 3, 1)))
    assert risk_limits.intraday_live_ok(
        db, now
    )  # > 3 months, 45 paper trades, gate passed, approved


# ---------------- separate, stricter unlock ----------------


def test_intraday_auto_unlock_needs_three_months_not_two() -> None:
    ev = UnlockEvidence(True, 40, date(2026, 3, 1), True)
    today = date(2026, 5, 15)  # 2.5 months of tracking
    assert unlock_blockers(ev, today, "swing_v1") == []
    blockers = unlock_blockers(ev, today, "intraday_v1")
    assert len(blockers) == 1 and "3 calendar months" in blockers[0]
    assert (
        unlock_blockers(ev, date(2026, 6, 1), "intraday_v1") == []
        and MIN_MONTHS_BY_STRATEGY["intraday_v1"] == 3
    )


# ---------------- jobs ----------------


def test_intraday_jobs_are_registered_and_disabled_by_default(db: Session, tmp_path) -> None:  # type: ignore[no-untyped-def]
    env = make_env(db, tmp_path)
    assert {"intraday_scan", "mis_protect"} <= set(jobs.BY_NAME)
    assert "disabled" in jobs.job_intraday_scan(env.ctx)
    assert jobs.job_mis_protect(env.ctx) == "ok"


def test_intraday_scan_job_end_to_end_in_paper(db: Session, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from tests.fixtures.pipeline import fast_settings
    from tradepartner.models import registry as reg
    from tradepartner.models import train as tr
    from tradepartner.settings import StrategyConfig

    base = fast_settings()
    enabled = {**base.strategies["intraday_v1"].model_dump(), "enabled": True, "min_prob": 0.0}
    settings = base.model_copy(
        update={"strategies": {**base.strategies, "intraday_v1": StrategyConfig(**enabled)}}
    )
    env = make_env(db, tmp_path, settings=settings)
    # a real, tiny intraday model trained on synthetic bars, then approved directly for the test
    fm, lm = im.build_matrix({f"SYM0{i}": synth_bars(i, n_days=130) for i in range(3)}, 0.001)
    data = fm.join(lm[["y"]]).dropna(subset=["y"])
    booster = tr.fit(
        data[im.INTRADAY_COLUMNS],
        data["y"].to_numpy(),
        {**tr.base_params("binary", 1), "num_leaves": 4},
        5,
    )
    model = reg.register_candidate(db, "intraday_v1", booster, algo="lightgbm", feature_set_version="intraday-fs1",
                                   train_start=date(2025, 1, 1), train_end=date(2025, 6, 1), params={}, root=tmp_path, now=NOW,
                                   metrics={"feature_columns": im.INTRADAY_COLUMNS, "objective": "binary", "paper_gate_passed": False})  # fmt: skip
    model.status = "approved"
    db.commit()
    day = breakout_day(day=NOW0.date())
    env.ctx.store.write("5minute", "SYM00", day)
    env.clock_now = day["ts"].iloc[7].to_pydatetime().astimezone(NOW0.tzinfo)
    out = jobs.job_intraday_scan(env.ctx)
    assert (
        "1 signals" in out
    )  # the breakout is found, scored by the model and routed through the risk engine
    sig = db.scalars(select(Signal).where(Signal.strategy == "intraday_v1")).one()
    assert sig.symbol == "SYM00" and sig.horizon_days == 0 and sig.model_id == model.id
    assert jobs.job_intraday_scan(env.ctx).startswith("0 signals")  # one idea per symbol per day


# ---------------- CLI ----------------


def test_train_intraday_command(monkeypatch: pytest.MonkeyPatch) -> None:
    assert (
        runner.invoke(cli.app, ["train", "--strategy", "intraday_v1", "--no-news"]).exit_code == 1
    )  # no bars
    with cli._session() as s:
        for i in range(3):
            s.add(
                UniverseMember(
                    universe_name="nifty100", symbol=f"IDX{i}", valid_from=date(2000, 1, 1)
                )
            )
        s.commit()
    store = candles.CandleStore()
    for i in range(3):
        store.write("5minute", f"IDX{i}", synth_bars(i, n_days=130))
    real = im.train_intraday
    monkeypatch.setattr(
        im, "train_intraday", lambda *a, **k: real(*a, **{**k, "min_train_days": 40})
    )
    r = runner.invoke(cli.app, ["train", "--strategy", "intraday_v1", "--trials", "1", "--no-news"])
    assert r.exit_code == 0, r.output
    assert "PAPER-ONLY" in r.output and "nothing was approved" in r.output
    assert "intraday_v1" in runner.invoke(cli.app, ["models", "list"]).output


def test_poll_quotes_feeds_the_bar_builder(db: Session, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from tradepartner.data.live import Tick
    from tradepartner.data.quotes import QuoteBook, poll

    class Src:
        def quote(self, instruments: list[str]) -> dict[str, dict[str, object]]:
            return {
                i: {
                    "last_price": 100.0,
                    "volume": 5000,
                    "ohlc": {"close": 99.0},
                    "timestamp": datetime(2025, 6, 2, 9, 16),
                }
                for i in instruments
            }

    ticks: list[Tick] = []
    n = poll(Src(), ["A", "B"], QuoteBook(), NOW, on_tick=ticks.append)  # type: ignore[arg-type]
    assert (
        n == 2
        and [t.symbol for t in ticks] == ["A", "B"]
        and ticks[0].ltp_paise == 10000
        and ticks[0].day_volume == 5000
    )
    assert timedelta(0) == timedelta() and LTP and train_once
