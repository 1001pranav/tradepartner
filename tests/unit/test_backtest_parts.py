import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.synth import bars_from_closes, bdays, synth_view
from tradepartner.backtest import benchmarks, fills, metrics
from tradepartner.backtest.view import MarketView
from tradepartner.strategies.base import round_tick

# ---------- fills ----------


def test_slippage_is_adverse_and_integer() -> None:
    assert fills.slip(100_000, 10, up=True) == 100_100
    assert fills.slip(100_000, 10, up=False) == 99_900
    assert fills.slip(100_000, 0, up=True) == 100_000


def test_buy_fill_rules() -> None:
    # opens below the limit: fills at open plus slippage, never above the limit
    assert fills.buy_fill(10_000, 10_100, 9_900, 10_050, 10, None) == 10_010
    assert fills.buy_fill(10_045, 10_100, 9_900, 10_050, 10, None) == 10_050  # capped at limit
    # opens above the limit but the range touches it
    assert fills.buy_fill(10_100, 10_200, 10_000, 10_050, 10, None) == 10_050
    # never touches the limit
    assert fills.buy_fill(10_100, 10_200, 10_060, 10_050, 10, None) is None


def test_locked_circuit_bars_do_not_fill() -> None:
    assert fills.circuit_blocked("BUY", 11_000, 11_000, 11_000, 10_000)
    assert not fills.circuit_blocked(
        "BUY", 11_000, 11_000, 11_000, 12_000
    )  # locked down, buying is fine
    assert fills.circuit_blocked("SELL", 9_000, 9_000, 9_000, 10_000)
    assert not fills.circuit_blocked("SELL", 11_000, 11_000, 11_000, 10_000)
    assert not fills.circuit_blocked("BUY", 11_000, 11_050, 10_900, 10_000)  # an ordinary gap-up
    assert not fills.circuit_blocked("BUY", 11_000, 11_000, 11_000, None)
    assert fills.buy_fill(11_000, 11_000, 11_000, 11_500, 10, 10_000) is None


def test_exit_fill_rules() -> None:
    stop, target = 9_000, 11_000
    ex = fills.exit_fill
    assert ex(False, 8_800, 9_000, 8_700, stop, target, 10, None) == (
        8_792,
        "stop",
    )  # gap through stop
    assert ex(False, 11_200, 11_300, 11_100, stop, target, 10, None) == (
        11_200,
        "target",
    )  # gap above target
    assert ex(False, 10_000, 10_100, 8_900, stop, target, 10, None) == (8_991, "stop")
    assert ex(False, 10_000, 11_100, 9_900, stop, target, 10, None) == (11_000, "target")
    # both inside one bar: the stop is assumed to be hit first
    assert ex(False, 10_000, 11_500, 8_500, stop, target, 10, None) == (8_991, "stop")
    assert ex(False, 10_000, 10_100, 9_900, stop, target, 10, None) is None
    # on the entry day the open gap rules do not apply
    assert ex(True, 11_200, 11_300, 11_100, stop, target, 10, None) == (
        11_000,
        "target",
    )  # no gap fill
    assert ex(True, 10_000, 10_100, 9_900, stop, target, 10, None) is None
    assert ex(False, 9_000, 9_000, 9_000, stop, target, 10, 10_000) is None  # locked lower circuit


# ---------- view ----------


def test_view_never_returns_the_future() -> None:
    days = bdays(date(2024, 1, 1), 30)
    view = MarketView({"A": bars_from_closes(days, [100.0 + i for i in range(30)])})
    cut = days[9]
    h = view.history("A", cut)
    assert h["date"].max() == cut and len(h) == 10
    assert len(view.history("A", cut, 3)) == 3 and view.history("A", cut, 3)["date"].iloc[-1] == cut
    assert view.history("A", date(2020, 1, 1)).empty
    assert view.bar("A", days[3]) is not None and view.bar("A", date(2024, 1, 6)) is None
    assert view.bar("ZZZ", cut) is None and view.last_close("A", cut) == 109.0
    assert view.last_close("A", date(2020, 1, 1)) is None
    assert view.index_history(cut).empty and view.symbols == ["A"]
    assert view.sector("A").startswith("UNMAPPED")
    assert (
        view.avg_daily_value("A", cut, 5) > 0 and view.avg_daily_value("A", date(2020, 1, 1)) == 0.0
    )


def test_view_universe_membership_and_index() -> None:
    days = bdays(date(2024, 1, 1), 10)
    bars = {"A": bars_from_closes(days, [10.0] * 10), "B": bars_from_closes(days[5:], [10.0] * 5)}
    view = MarketView(
        bars, bars_from_closes(days, [100.0] * 10), members=lambda d: ["A", "B", "GONE"]
    )
    assert view.symbols_as_of(days[2]) == ["A"]  # B has no bars yet; GONE has none at all
    assert view.symbols_as_of(days[7]) == ["A", "B"]
    assert len(view.index_history(days[4])) == 5
    assert view.trading_days() == days


def test_round_tick() -> None:
    assert round_tick(100_003) == 100_005 and round_tick(100_002) == 100_000


# ---------- metrics ----------


def _equity(values: list[float]) -> pd.Series:
    return pd.Series(values, index=bdays(date(2024, 1, 1), len(values)))


def test_metrics_on_known_curves() -> None:
    flat = metrics.compute(_equity([100.0] * 30), [])
    assert flat.cagr == pytest.approx(0.0) and flat.max_drawdown == 0.0 and math.isnan(flat.sharpe)
    up = metrics.compute(_equity([100.0 * 1.001**i for i in range(300)]), [10, 20, -5])
    assert up.cagr > 0.29 and up.max_drawdown == 0.0 and up.hit_rate == pytest.approx(2 / 3)
    assert up.avg_win == 15 and up.avg_loss == -5 and up.n_trades == 3
    down = metrics.compute(_equity([100, 90, 80, 85, 70]), [-1])
    assert down.max_drawdown == pytest.approx(70 / 100 - 1) and down.calmar < 0 and down.sortino < 0


def test_metrics_edge_cases() -> None:
    short = metrics.compute(_equity([100.0]), [1, 2])
    assert math.isnan(short.cagr) and short.n_trades == 2 and short.start is None
    ruin = metrics.compute(_equity([100, 50, -10]), [])
    assert ruin.cagr == -1.0
    one = metrics.compute(_equity([100, 101]), [], trade_notional=0, cost_paise=5, gross_profit=0)
    assert math.isnan(one.cost_drag) and math.isnan(one.hit_rate) and math.isnan(one.exposure)
    with_exposure = metrics.compute(_equity([100, 101, 102]), [], cost_paise=1, gross_profit=10,
                                    trade_notional=100, exposure=pd.Series([0.5, 0.5, 0.5]))  # fmt: skip
    assert (
        with_exposure.cost_drag == 0.1
        and with_exposure.exposure == 0.5
        and with_exposure.turnover > 0
    )
    assert metrics.Metrics.as_dict(with_exposure)["n_trades"] == 0


def test_by_year_and_regime() -> None:
    days = bdays(date(2023, 1, 2), 520)
    eq = pd.Series([100.0 * 1.0005**i for i in range(520)], index=days)
    years = metrics.by_year(eq)
    assert list(years["year"]) == [2023, 2024] and (years["return"] > 0).all()
    idx = pd.Series([100.0 + (i if i < 300 else 600 - i) for i in range(520)], index=days)
    regime = metrics.by_regime(eq, idx)
    assert set(regime["regime"]) == {"above_200dma", "below_200dma"}
    assert (regime["days"] > 0).all()


# ---------- benchmarks ----------


def test_buy_and_hold_and_equal_weight() -> None:
    view = synth_view(n_symbols=5, n_days=300)
    days = view.trading_days()
    bh = benchmarks.buy_and_hold(view, days[0], days[-1], 1_000_000_00)
    assert bh.iloc[0] == 1_000_000_00 and len(bh) == 300 and bh.dtype == "int64"
    ew = benchmarks.equal_weight_universe(view, days[0], days[-1], 1_000_000_00, 10)
    assert len(ew) == 300 and ew.iloc[0] < 1_000_000_00  # the first rebalance pays costs
    assert benchmarks.buy_and_hold(MarketView({}), days[0], days[-1], 1).empty
    assert benchmarks.buy_and_hold(view, date(2030, 1, 1), date(2030, 2, 1), 1).empty
    assert benchmarks.equal_weight_universe(view, date(2030, 1, 1), date(2030, 2, 1), 1).empty
    zero_cost = benchmarks.equal_weight_universe(view, days[0], days[-1], 1_000_000_00, 0)
    assert zero_cost.iloc[0] == 1_000_000_00
    assert np.isfinite(zero_cost.to_numpy()).all()
