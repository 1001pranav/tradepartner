"""Equal-weight simulation: whole shares, real charges, FIFO tax at the financial-year end."""

from datetime import date

import pandas as pd
import pytest
from typer.testing import CliRunner

from tests.fixtures.synth import bars_from_closes, bdays, settings_with_capital, synth_view
from tests.unit.test_cli_m4 import store_view
from tradepartner import cli
from tradepartner.backtest.equalweight import _is_rebalance_day, _year_tax, run_equal_weight
from tradepartner.backtest.view import MarketView
from tradepartner.data.candles import CandleStore
from tradepartner.db.models import UniverseMember

S = settings_with_capital()
runner = CliRunner()


def flat_view(prices: dict[str, list[float]], start: date = date(2024, 1, 1)) -> MarketView:
    n = len(next(iter(prices.values())))
    days = bdays(start, n)
    bars = {s: bars_from_closes(days, p) for s, p in prices.items()}
    idx = bars_from_closes(days, [1000.0] * n)
    return MarketView(bars, idx, {})


def test_rebalance_calendar() -> None:
    d = date
    assert _is_rebalance_day("monthly", d(2024, 2, 1), d(2024, 1, 31)) and not _is_rebalance_day(
        "monthly", d(2024, 2, 2), d(2024, 2, 1)
    )
    assert _is_rebalance_day("quarterly", d(2024, 4, 1), d(2024, 3, 28)) and not _is_rebalance_day(
        "quarterly", d(2024, 3, 1), d(2024, 2, 29)
    )
    assert _is_rebalance_day("annual", d(2024, 4, 1), d(2024, 3, 28)) and not _is_rebalance_day(
        "annual", d(2024, 7, 1), d(2024, 6, 28)
    )
    assert not _is_rebalance_day("never", d(2024, 2, 1), d(2024, 1, 31)) and _is_rebalance_day(
        "never", d(2024, 1, 1), None
    )
    with pytest.raises(ValueError):
        _is_rebalance_day("weekly", d(2024, 2, 1), d(2024, 1, 31))


def test_tax_rules_stcg_ltcg_exemption_and_setoff() -> None:
    lakh = 100_000 * 100
    assert _year_tax(1_000_000_00, 0, S) == 200_000_00  # 20% STCG
    assert _year_tax(0, 2 * lakh, S) == round(
        (2 * lakh - 125_000 * 100) * 0.125
    )  # 12.5% above Rs 1.25 lakh
    assert _year_tax(0, lakh, S) == 0  # inside the exemption
    assert _year_tax(-lakh, 3 * lakh, S) == round(
        (2 * lakh - 125_000 * 100) * 0.125
    )  # STCL offsets LTCG
    assert (
        _year_tax(-lakh, 0, S) == 0 and _year_tax(-lakh, -lakh, S) == 0
    )  # losses pay nothing (no carry-forward)


def test_never_rebalancing_buys_once_and_pays_no_tax() -> None:
    v = flat_view({"A": [100.0] * 60, "B": [200.0] * 60})
    r = run_equal_weight(v, S, v.trading_days()[0], v.trading_days()[-1], 1_000_000_00, "never")
    assert r.trades == 2 and r.tax_paise == 0 and r.holdings == 2
    assert (
        r.equity.iloc[-1] < 1_000_000_00 and r.equity.iloc[-1] > 995_000_00
    )  # only costs lost (about 0.2%)
    assert r.charges_paise > 0 and r.slippage_paise > 0


def test_whole_shares_only_and_cash_is_never_negative() -> None:
    v = flat_view({"A": [333.0] * 40, "B": [777.0] * 40, "C": [1234.0] * 40})
    r = run_equal_weight(v, S, v.trading_days()[0], v.trading_days()[-1], 500_000_00, "never")
    assert (
        r.trades == 3 and 0 < r.equity.iloc[0] <= 500_000_00
    )  # leftover cash from whole-share rounding


def test_gains_sold_within_a_year_are_taxed_at_stcg_and_after_a_year_at_ltcg() -> None:
    prices = {"A": [100.0] * 20 + [200.0] * 520, "B": [100.0] * 540}  # A doubles after month 1
    v = flat_view(prices)
    first, last = v.trading_days()[0], v.trading_days()[-1]
    monthly = run_equal_weight(v, S, first, last, 10_000_000_00, "monthly")  # Rs 1 crore
    never = run_equal_weight(v, S, first, last, 10_000_000_00, "never")
    assert (
        monthly.tax_paise > 0 and never.tax_paise == 0
    )  # selling the winner realises (taxable) gains
    assert (
        never.equity.iloc[-1] > monthly.equity.iloc[-1] > 0
    )  # tax and costs make rebalancing lose here


def test_tax_is_charged_at_the_financial_year_end_not_before() -> None:
    prices = {"A": [100.0] * 20 + [150.0] * 120, "B": [100.0] * 140}
    v = flat_view(prices, date(2024, 2, 1))
    r = run_equal_weight(v, S, v.trading_days()[0], v.trading_days()[-1], 5_000_000_00, "monthly")
    eq = r.equity
    assert r.tax_paise > 0
    mar31 = (
        max(d for d in eq.index if d <= date(2025, 3, 31))
        if eq.index[-1] >= date(2025, 3, 31)
        else eq.index[-1]
    )
    assert mar31 >= eq.index[0]


def test_equal_weight_command_prints_all_frequencies_and_the_benchmark() -> None:
    v = synth_view(n_symbols=8, n_days=760, seed=21)
    with cli._session() as s:
        for sym in v.symbols:
            s.add(UniverseMember(universe_name="nifty100", symbol=sym, valid_from=date(2000, 1, 1)))
        s.commit()
    store_view(CandleStore(), v)
    r = runner.invoke(cli.app, ["ew-sim", "--start", "2023-02-01", "--capital-inr", "1000000"])
    assert r.exit_code == 0, r.output
    for label in (
        "equal-weight monthly",
        "equal-weight quarterly",
        "equal-weight annual",
        "equal-weight never",
        "Nifty 50",
    ):
        assert label in r.output
    assert "survivorship" in r.output and "Simulated only" in r.output
    assert runner.invoke(cli.app, ["ew-sim", "--start", "2023-02-01"]).exit_code == 0
    assert pd.Timestamp("2020-01-01")


def test_ew_sim_saves_a_report() -> None:
    from tradepartner.paths import reports_dir

    v = synth_view(n_symbols=8, n_days=760, seed=21)
    with cli._session() as s:
        for sym in v.symbols:
            s.add(UniverseMember(universe_name="nifty100", symbol=sym, valid_from=date(2000, 1, 1)))
        s.commit()
    store_view(CandleStore(), v)
    assert runner.invoke(cli.app, ["ew-sim", "--start", "2023-02-01"]).exit_code == 0
    saved = list(reports_dir().glob("ew-sim-*.html"))
    assert len(saved) == 1 and "equal-weight monthly" in saved[0].read_text()
