from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tests.fixtures.market import FakeKite, make_cal
from tradepartner import cli
from tradepartner.data import kite_auth, kite_client

runner = CliRunner()


def _kite_with_instruments(symbols: list[str]) -> FakeKite:
    kite = FakeKite(make_cal())
    kite.instrument_rows = [
        {
            "instrument_token": 1000 + i,
            "exchange": "NSE",
            "segment": "NSE",
            "instrument_type": "EQ",
            "tradingsymbol": s,
            "tick_size": 0.05,
        }
        for i, s in enumerate(symbols)
    ]
    return kite


def test_seed_universe_and_instruments(monkeypatch: pytest.MonkeyPatch) -> None:
    r = runner.invoke(cli.app, ["data", "seed-universe"])
    assert r.exit_code == 0 and "survivorship" in r.output
    monkeypatch.setattr(cli, "_kite", lambda: _kite_with_instruments(["INFY"]))
    r = runner.invoke(cli.app, ["data", "instruments"])
    assert r.exit_code == 0 and "1 NSE equity" in r.output


def test_backfill_then_quality_then_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    kite = _kite_with_instruments(["INFY", "TCS"])
    monkeypatch.setattr(cli, "_kite", lambda: kite)
    monkeypatch.setattr(
        cli.candles,
        "RateLimiter",
        lambda rate: cli.candles.RateLimiter.__new__(cli.candles.RateLimiter),
    )

    # a limiter that never waits
    class Free:
        def acquire(self) -> None: ...

    monkeypatch.setattr(cli.candles, "RateLimiter", lambda rate: Free())
    r = runner.invoke(cli.app, ["data", "backfill", "--years", "1"])
    assert r.exit_code == 0, r.output
    assert "failed=0" in r.output and "done=" in r.output
    r2 = runner.invoke(cli.app, ["data", "backfill", "--years", "1"])
    assert "done=0" in r2.output
    q = runner.invoke(cli.app, ["data", "quality"])
    assert "issues across" in q.output


def test_backfill_failure_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    kite = _kite_with_instruments(["INFY"])
    kite.fail_tokens = {1000}
    monkeypatch.setattr(cli, "_kite", lambda: kite)
    monkeypatch.setattr(
        cli.candles, "RateLimiter", lambda rate: type("F", (), {"acquire": lambda s: None})()
    )
    import time

    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setattr(
        cli.candles,
        "_time",
        type("T", (), {"sleep": staticmethod(lambda s: None), "monotonic": time.monotonic}),
    )
    r = runner.invoke(cli.app, ["data", "backfill", "--years", "1"])
    assert r.exit_code == 1 and "FAILED INFY" in r.output


def test_quality_errors_exit_nonzero(tmp_path: Path) -> None:
    from tradepartner.data import candles

    store = candles.CandleStore()
    df = candles.normalize_rows(
        [
            {
                "date": __import__("datetime").date(2026, 3, 2),
                "open": 10,
                "high": 1,
                "low": 5,
                "close": 7,
                "volume": 1,
            }
        ],
        "day",
    )
    store.write("day", "BAD", df)
    r = runner.invoke(cli.app, ["data", "quality"])
    assert r.exit_code == 1 and "ohlc_inconsistent" in r.output


def test_actions_import(tmp_path: Path) -> None:
    f = tmp_path / "ca.csv"
    f.write_text("symbol,action_type,ex_date,ratio_num,ratio_den\nX,split,2026-03-05,5,1\n")
    r = runner.invoke(cli.app, ["data", "actions-import", str(f)])
    assert r.exit_code == 0 and "1 corporate actions" in r.output


def test_kite_helper_requires_login(monkeypatch: pytest.MonkeyPatch) -> None:
    r = runner.invoke(cli.app, ["data", "instruments"])
    assert r.exit_code == 2 and "run `tradepartner login`" in r.output
    kite_auth.save_credentials("k", "s")
    r = runner.invoke(cli.app, ["data", "instruments"])
    assert r.exit_code == 2 and "expired" in r.output
    kite_auth.save_access_token("t")
    assert cli._kite() is not None


class _Login:
    def login_url(self) -> str:
        return "http://example.invalid/login"

    def generate_session(self, request_token: str, api_secret: str) -> dict[str, Any]:
        return {"access_token": "AT"}


def test_login_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    import webbrowser

    monkeypatch.setattr(webbrowser, "open", lambda url: True)
    monkeypatch.setattr(kite_client, "login_client", lambda: _Login())
    monkeypatch.setattr(kite_auth, "wait_for_request_token", lambda port: "rt")
    r = runner.invoke(cli.app, ["login"], input="mykey\nmysecret\n")
    assert r.exit_code == 0, r.output
    assert kite_auth.load_credentials() == ("mykey", "mysecret")
    assert kite_auth.load_access_token() == "AT"
    r = runner.invoke(cli.app, ["login"])  # credentials already stored: no prompts
    assert r.exit_code == 0

    def boom(port: int) -> str:
        raise kite_auth.LoginError("timeout")

    monkeypatch.setattr(kite_auth, "wait_for_request_token", boom)
    assert runner.invoke(cli.app, ["login"]).exit_code == 1


def test_login_bad_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TRADEPARTNER_CONFIG_DIR", str(tmp_path))
    assert runner.invoke(cli.app, ["login"]).exit_code == 2


def test_login_client_requires_credentials() -> None:
    with pytest.raises(kite_client.NotLoggedInError):
        kite_client.login_client()


def test_universe_import_command_warns_about_members_without_prices(tmp_path: Path) -> None:
    f = tmp_path / "m.csv"
    f.write_text("symbol,valid_from,valid_to\nINFY,2010-01-01,\nGONECO,2010-01-01,2019-01-01\n")
    r = runner.invoke(cli.app, ["data", "universe-import", str(f)])
    assert r.exit_code == 0 and "2 membership rows added" in r.output
    assert "NO price history" in r.output and "GONECO" in r.output
    f.write_text("symbol\nX\n")
    assert runner.invoke(cli.app, ["data", "universe-import", str(f)]).exit_code == 1
