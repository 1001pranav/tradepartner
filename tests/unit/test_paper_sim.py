"""Historical paper trading: a retraining model that can only use what it could have known."""

from datetime import date

import pytest
from typer.testing import CliRunner

from tests.fixtures.synth import settings_with_capital, synth_view
from tests.unit.test_cli_m4 import store_view
from tradepartner import cli
from tradepartner.backtest.engine import run_backtest
from tradepartner.backtest.view import MarketView
from tradepartner.data.candles import CandleStore
from tradepartner.db.models import UniverseMember
from tradepartner.features import labels
from tradepartner.models import pipeline
from tradepartner.strategies.walkforward import WalkForwardModel

runner = CliRunner()
S = settings_with_capital()


@pytest.fixture(scope="module")
def view() -> MarketView:
    return synth_view(n_symbols=8, n_days=760, seed=21, momentum=0.25)


def make(view: MarketView, every: int = 21) -> WalkForwardModel:
    p = {**pipeline.strategy_params(S, "swing_v1"), "min_prob": 0.0, "max_stop_pct": 7.0}
    lab = pipeline._labels(view, p, S)
    return WalkForwardModel("swing_v1", view, lab, p, "binary", every)


def test_retraining_never_uses_labels_from_the_future(view: MarketView) -> None:
    wf = make(view)
    days = view.trading_days()
    res = run_backtest(wf, view, S, days[560], days[700])
    assert len(wf.retrains) >= 5 and res.trades
    pos = {d: i for i, d in enumerate(days)}
    for as_of, last_train in wf.retrains:
        # the last training row's label window (5 days) ended before as_of, with a safety gap
        assert pos[as_of] - pos[last_train] >= 6


def test_picks_do_not_change_if_the_future_is_scrambled(view: MarketView) -> None:
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
    a, b = make(view), make(other)
    for d in days[560:601:10]:
        assert a.candidates(d, view) == b.candidates(d, other), d


def test_no_candidates_before_enough_history(view: MarketView) -> None:
    days = view.trading_days()
    assert make(view).candidates(days[100], view) == []


def test_positional_variant_only_acts_on_the_first_day_of_the_month() -> None:
    v = synth_view(n_symbols=8, n_days=1100, seed=3, momentum=0.2)
    p = {**pipeline.strategy_params(S, "positional_v1"), "max_stop_pct": 8.0}
    wf = WalkForwardModel("positional_v1", v, pipeline._labels(v, p, S), p, "regression", 21)
    days = v.trading_days()[800:880]
    active = [d for d in days if wf.candidates(d, v)]
    firsts = {d for d in days if d.month != v.trading_days()[v.trading_days().index(d) - 1].month}
    assert active and set(active) <= firsts


def test_paper_sim_command(monkeypatch: pytest.MonkeyPatch) -> None:
    v = synth_view(n_symbols=8, n_days=760, seed=21, momentum=0.25)
    with cli._session() as s:
        for sym in v.symbols:
            s.add(UniverseMember(universe_name="nifty100", symbol=sym, valid_from=date(2000, 1, 1)))
        s.commit()
    store_view(CandleStore(), v)
    start = v.trading_days()[600].isoformat()
    r = runner.invoke(
        cli.app,
        ["paper-sim", "--strategy", "swing_v1", "--start", start, "--capital-inr", "1000000"],
    )
    assert r.exit_code == 0, r.output
    assert (
        "paper-traded" in r.output
        and "nifty50_buy_and_hold" in r.output
        and "Simulated only" in r.output
    )
    assert (
        runner.invoke(
            cli.app, ["paper-sim", "--strategy", "intraday_v1", "--start", start]
        ).exit_code
        == 2
    )
    assert labels.EXTRA_MARGIN > 0
