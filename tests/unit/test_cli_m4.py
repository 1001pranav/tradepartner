from datetime import date
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from tests.fixtures.synth import synth_view
from tradepartner import cli
from tradepartner.calendar import session_close_utc
from tradepartner.data.candles import CandleStore
from tradepartner.db.models import UniverseMember

runner = CliRunner()


def store_view(store: CandleStore, view) -> None:  # type: ignore[no-untyped-def]
    """Write a synthetic MarketView into the Parquet store as raw paise candles."""
    names = {s: view._bars[s] for s in view.symbols}
    if view.index is not None:
        names["NIFTY 50"] = view.index
    for sym, df in names.items():
        raw = pd.DataFrame(
            {
                "ts": [pd.Timestamp(session_close_utc(d)) for d in df["date"]],
                "date": list(df["date"]),
                **{
                    c: (df[c] * 100).round().astype("int64")
                    for c in ("open", "high", "low", "close")
                },
                "volume": df["volume"].astype("int64"),
            }
        )
        store.write("day", sym, raw)


def seed(view) -> None:  # type: ignore[no-untyped-def]
    with cli._session() as s:
        for sym in view.symbols:
            s.add(UniverseMember(universe_name="nifty100", symbol=sym, valid_from=date(2000, 1, 1)))
        s.commit()
    store_view(CandleStore(), view)


def test_backtest_command_end_to_end() -> None:
    seed(synth_view(n_symbols=8, n_days=450))
    r = runner.invoke(cli.app, ["backtest", "--capital-inr", "1000000"])
    assert r.exit_code == 0, r.output
    assert "baseline_momentum20: CAGR" in r.output and "Report:" in r.output
    assert "survivorship" in r.output.lower()  # static seeded membership -> warning
    path = r.output.split("Report: ")[1].strip().splitlines()[0]
    html = Path(path).read_text()
    assert "nifty50_buy_and_hold" in html and "equal_weight_universe" in html


def test_backtest_positional_with_dates_and_slippage() -> None:
    seed(synth_view(n_symbols=8, n_days=600))
    r = runner.invoke(cli.app, ["backtest", "--strategy", "baseline_momentum12_1", "--capital-inr", "1000000",
                                "--start", "2023-06-01", "--slippage-bps", "20"])  # fmt: skip
    assert r.exit_code == 0, r.output


def test_backtest_errors() -> None:
    assert runner.invoke(cli.app, ["backtest"]).exit_code == 2  # no capital configured
    assert (
        runner.invoke(
            cli.app, ["backtest", "--capital-inr", "1000000", "--strategy", "nope"]
        ).exit_code
        == 2
    )
    r = runner.invoke(cli.app, ["backtest", "--capital-inr", "1000000"])
    assert r.exit_code == 1 and "backfill" in r.output
