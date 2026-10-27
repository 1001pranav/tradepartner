"""`replay` (hide the future, then reveal) and `deploy` (to shadow only)."""

from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from sqlalchemy import select
from typer.testing import CliRunner

from tests.fixtures.synth import settings_with_capital, synth_view
from tests.unit.test_cli_m4 import store_view
from tradepartner import cli
from tradepartner.backtest.view import MarketView
from tradepartner.data.candles import CandleStore
from tradepartner.db.models import Model, UniverseMember
from tradepartner.models import registry
from tradepartner.models.registry import ModelError
from tradepartner.models.replay import replay

runner = CliRunner()
S = settings_with_capital()


@pytest.fixture(scope="module")
def view() -> MarketView:
    return synth_view(n_symbols=8, n_days=760, seed=21, momentum=0.25)


def test_replay_trains_only_on_the_past_and_reveals_the_future(view: MarketView) -> None:
    days = view.trading_days()
    as_of = days[600]
    res = replay(view, S, as_of, 5, top_k=3)
    assert (
        res.train_last_date < as_of and (days.index(as_of) - days.index(res.train_last_date)) >= 6
    )  # embargo
    assert len(res.picks) == 3 and [p.score for p in res.picks] == sorted(
        (p.score for p in res.picks), reverse=True
    )
    h = view.history(res.picks[0].symbol, days[-1]).reset_index(drop=True)
    i = int(h.index[h["date"] == as_of][0])
    assert res.picks[0].actual_return == pytest.approx(
        float(h["close"].iloc[i + 5] / h["open"].iloc[i + 1] - 1)
    )
    assert (
        res.index_return is not None and res.universe_mean is not None and res.hit_rate is not None
    )


def test_predictions_cannot_depend_on_anything_after_as_of(view: MarketView) -> None:
    days = view.trading_days()
    as_of = days[600]
    scrambled = {}
    for s in view.symbols:
        df = view.history(s, days[-1]).copy()
        df.loc[df["date"] > as_of, ["open", "high", "low", "close"]] = 1e6
        scrambled[s] = df
    idx = view.index.copy()  # type: ignore[union-attr]
    idx.loc[idx["date"] > as_of, ["open", "high", "low", "close"]] = 1e6
    other = MarketView(scrambled, idx, view.sectors)
    a, b = replay(view, S, as_of, 5, top_k=3), replay(other, S, as_of, 5, top_k=3)
    assert [(p.symbol, p.score) for p in a.picks] == [
        (p.symbol, p.score) for p in b.picks
    ]  # same picks
    assert a.picks[0].actual_return != b.picks[0].actual_return  # only the revealed part differs


def test_replay_near_the_end_has_no_reveal_yet_and_bad_dates_are_refused(view: MarketView) -> None:
    days = view.trading_days()
    res = replay(view, S, days[-3], 5, top_k=2)
    assert all(p.actual_return is None for p in res.picks) and res.hit_rate is None
    with pytest.raises(ModelError, match="not a trading day"):
        replay(view, S, date(2030, 1, 1), 5)
    with pytest.raises(ModelError, match="not enough history"):
        replay(view, S, days[40], 5)
    with pytest.raises(ModelError, match="index"):
        replay(MarketView({s: view.history(s, days[-1]) for s in view.symbols}), S, days[600], 5)


def test_replay_command() -> None:
    v = synth_view(n_symbols=8, n_days=760, seed=21, momentum=0.25)
    with cli._session() as s:
        for sym in v.symbols:
            s.add(UniverseMember(universe_name="nifty100", symbol=sym, valid_from=date(2000, 1, 1)))
        s.commit()
    store_view(CandleStore(), v)
    day = v.trading_days()[600].isoformat()
    r = runner.invoke(cli.app, ["replay", "--as-of", day, "--days", "5", "--top", "3"])
    assert r.exit_code == 0, r.output
    assert (
        "never saw anything after that" in r.output
        and "vs Nifty" in r.output
        and "Picks that beat Nifty" in r.output
    )
    assert runner.invoke(cli.app, ["replay", "--as-of", "2030-01-01"]).exit_code == 1


def _model(session, ready: bool) -> Model:  # type: ignore[no-untyped-def]
    m = Model(strategy="swing_v1", version=f"v{ready}", algo="lightgbm", feature_set_version="fs2", train_start=date(2020, 1, 1),
              train_end=date(2025, 1, 1), params_json={}, metrics_json={"gate": {"ready_for_review": ready, "failed": [] if ready else ["min_trades"]}},
              artifact_path="x", status="candidate")  # fmt: skip
    session.add(m)
    session.commit()
    return m


def test_deploy_goes_to_shadow_only_and_refuses_failed_models() -> None:
    with cli._session() as s:
        good, bad = _model(s, True).id, _model(s, False).id
    refused = runner.invoke(cli.app, ["deploy", str(bad)])
    assert refused.exit_code == 1 and "promotion gate failed (min_trades)" in refused.output
    ok = runner.invoke(cli.app, ["deploy", str(good)])
    assert (
        ok.exit_code == 0
        and "SHADOW" in ok.output
        and "Nothing was approved or ordered" in ok.output
    )
    with cli._session() as s:
        m = registry.get(s, good)
        assert (
            m.status == "shadow"
            and m.approved_by is None
            and registry.get(s, bad).status == "candidate"
        )
        assert s.scalars(select(Model).where(Model.status == "approved")).all() == []
    assert runner.invoke(cli.app, ["deploy", str(good)]).exit_code == 1  # already in shadow
    assert runner.invoke(cli.app, ["deploy", "9999"]).exit_code == 1
    assert datetime.now(UTC) and Path(".")


def test_replay_command_saves_a_report_the_ui_can_show() -> None:
    from tradepartner.paths import reports_dir

    v = synth_view(n_symbols=8, n_days=760, seed=21, momentum=0.25)
    with cli._session() as s:
        for sym in v.symbols:
            s.add(UniverseMember(universe_name="nifty100", symbol=sym, valid_from=date(2000, 1, 1)))
        s.commit()
    store_view(CandleStore(), v)
    r = runner.invoke(
        cli.app, ["replay", "--as-of", v.trading_days()[600].isoformat(), "--days", "5"]
    )
    assert r.exit_code == 0 and "Saved report:" in r.output
    saved = list(reports_dir().glob("replay-swing_v1-*.html"))
    assert len(saved) == 1 and "never saw anything after that" in saved[0].read_text()
