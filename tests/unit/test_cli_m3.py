from datetime import date

import pytest
from typer.testing import CliRunner

from tests.fixtures.market import dt
from tests.fixtures.world import STRATEGY, World
from tradepartner import cli
from tradepartner.data.quotes import QuoteBook
from tradepartner.db.models import Model
from tradepartner.modes.controller import ModeController, UnlockEvidence
from tradepartner.modes.evidence import collect_unlock_evidence
from tradepartner.modes.state import GLOBAL
from tradepartner.risk import kill
from tradepartner.runtime import build_runtime
from tradepartner.settings import load_settings

runner = CliRunner()


def _global_line() -> str:
    out = runner.invoke(cli.app, ["mode"]).output
    return next(ln for ln in out.splitlines() if ln.startswith("global"))


def test_mode_show_and_set() -> None:
    out = runner.invoke(cli.app, ["mode"]).output
    assert "global" in out and "strategy:swing_v1" in out and "OFF" in out and "AUTO locked" in out
    ok = runner.invoke(cli.app, ["mode", "global", "manual"])
    assert ok.exit_code == 0 and "global -> MANUAL" in ok.output
    assert "MANUAL" in _global_line()


def test_mode_refuses_locked_auto_and_bad_input() -> None:
    r = runner.invoke(cli.app, ["mode", "swing_v1", "AUTO"])
    assert r.exit_code == 1 and "not unlocked" in r.output
    assert runner.invoke(cli.app, ["mode", "global", "TURBO"]).exit_code == 1
    runner.invoke(cli.app, ["mode", "global", "AUTO"])  # global AUTO is a one-click user action
    assert "AUTO" in _global_line()


def test_unlock_auto_refused_without_evidence() -> None:
    r = runner.invoke(cli.app, ["unlock-auto", "swing_v1"], input="swing_v1\n")
    assert r.exit_code == 1 and "no approved model" in r.output
    wrong = runner.invoke(cli.app, ["unlock-auto", "swing_v1"], input="nope\n")
    assert wrong.exit_code == 1 and "type the strategy name" in wrong.output


def test_kill_command_arms_file_and_turns_global_off() -> None:
    with cli._session() as s:
        ModeController(s).set_mode(GLOBAL, "MANUAL", "user", "x", dt(2026, 3, 10, 9, 0))
    r = runner.invoke(cli.app, ["kill"])
    assert r.exit_code == 0 and "Global mode OFF" in r.output and kill.kill_active()
    with cli._session() as s:
        assert ModeController(s).current(GLOBAL).mode == "OFF"


def test_kill_reports_errors_but_keeps_file_armed(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> None:
        raise RuntimeError("db down")

    monkeypatch.setattr(cli, "_session", boom)
    r = runner.invoke(cli.app, ["kill"])
    assert r.exit_code == 1 and "still armed" in r.output and kill.kill_active()


def test_collect_unlock_evidence_counts_paper_trades(db) -> None:  # type: ignore[no-untyped-def]
    w = World(db)
    assert collect_unlock_evidence(db, STRATEGY) == UnlockEvidence(False, 0, None, False)
    w.go_auto()
    from tests.fixtures.risk import LTP
    from tests.fixtures.risk import exit_ as make_exit

    assert w.submit_entry().status == "placed"
    w.advance(seconds=5)
    held = w.context_for(STRATEGY).positions[0]
    w.set_price("INFY", LTP + 500)
    w.router.submit(
        make_exit(
            idempotency_key="x1", exit_qty=held.qty, signal_time=w.now, limit_price_paise=LTP + 500
        ),
        w.now,
    )
    w.context_for(STRATEGY)
    ev = collect_unlock_evidence(db, STRATEGY)
    assert (
        ev.closed_trades == 1 and ev.first_trade_date == date(2026, 3, 10) and not ev.model_approved
    )
    db.add(
        Model(
            strategy=STRATEGY,
            version="v1",
            algo="lgbm",
            feature_set_version="f1",
            train_start=date(2020, 1, 1),
            train_end=date(2025, 1, 1),
            params_json={},
            metrics_json={"paper_gate_passed": True},
            artifact_path="x",
            status="approved",
        )
    )
    db.commit()
    ev = collect_unlock_evidence(db, STRATEGY)
    assert ev.model_approved and ev.gate_passed


def test_runtime_wiring(db) -> None:  # type: ignore[no-untyped-def]
    settings = load_settings()
    rt = build_runtime(db, settings, quotes=QuoteBook(), meta_provider=lambda: {})
    assert rt.context().mode == "OFF" and rt.context(STRATEGY).mode == "OFF"
    assert rt.quotes.get("X") is None
    assert rt.router.broker is rt.broker and rt.kill.router is rt.router
