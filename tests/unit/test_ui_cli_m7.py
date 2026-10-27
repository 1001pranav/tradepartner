"""Today / Positions / System pages, daily report, `run` and `job` commands."""

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from sqlalchemy import select
from streamlit.testing.v1 import AppTest
from typer.testing import CliRunner

from tests.fixtures.market import dt
from tradepartner import cli, jobs
from tradepartner.db.models import (
    AuditLog,
    Fill,
    JobRun,
    Lot,
    OrderProposal,
    RiskEvent,
    Signal,
    SignalEvent,
)
from tradepartner.reports import daily
from tradepartner.signals import proposals
from tradepartner.ui import data

PAGES = Path(__file__).parents[2] / "src" / "tradepartner" / "ui" / "pages"
runner = CliRunner()


def page(name: str) -> AppTest:
    return AppTest.from_file(str(PAGES / name), default_timeout=30).run()


def seed_signal(session, *, approved: bool = True) -> int:  # type: ignore[no-untyped-def]
    now = datetime.now(UTC)
    sig = Signal(strategy="swing_v1", model_id=None, symbol="INFY", side="BUY", as_of=now, score=0.7, prob=0.7,
                 horizon_days=5, explanation="Momentum is strong; wrong if the stop at 1440 is hit.", status="new",
                 created_at=now)  # fmt: skip
    session.add(sig)
    session.flush()
    session.add(SignalEvent(signal_id=sig.id, status="new", at=now, created_at=now,
                            detail=json.dumps({"limit": 150000, "stop": 144000, "target": 162000, "horizon": 5, "prob": 0.7, "contrib": [["ret_20", 0.3]], "product": "CNC"})))  # fmt: skip
    session.add(OrderProposal(signal_id=sig.id, symbol="INFY", side="BUY", product="CNC", order_type="LIMIT", qty=66,
                              limit_price_paise=150000, risk_amount_paise=396000, mode_at_creation="MANUAL",
                              risk_decision="approved" if approved else "rejected",
                              risk_reasons_json=[] if approved else ["quantity: too small"], idempotency_key=f"k{sig.id}"))  # fmt: skip
    session.add(
        SignalEvent(
            signal_id=sig.id,
            status="shown" if approved else "rejected_by_risk",
            at=now,
            created_at=now,
            detail="{}",
        )
    )
    session.commit()
    return sig.id


def test_today_page_shows_signals_and_the_exact_kite_order() -> None:
    with cli._session() as s:
        sid = seed_signal(s)
        seed_signal(s, approved=False)
    at = page("today.py")
    assert not at.exception and at.metric[0].value == "OFF"
    assert any("INFY" in e.label for e in at.expander)
    copy = next(b for b in at.button if b.key == f"copy-{sid}")
    copy.click().run()
    code = at.code[0].value
    assert (
        "BUY 66 x INFY" in code
        and "1500.00" in code
        and "stop-loss trigger 1440.00" in code
        and "target trigger 1620.00" in code
    )
    assert any("quantity: too small" in e.value for e in at.error)
    next(b for b in at.button if b.key == f"ign-{sid}").click().run()
    with cli._session() as s:
        assert proposals.current_status(s, sid) == "ignored"


def test_today_page_empty_and_kill_confirmation() -> None:
    at = page("today.py")
    assert at.info and "08:50" in at.info[0].value
    at.text_input[0].set_value("nope")
    at.button[0].click().run()
    assert at.error and "Type KILL" in at.error[0].value


def test_kite_order_text_and_protective_status_helpers() -> None:
    row = {
        "symbol": "TCS",
        "qty": 10,
        "entry": 300000,
        "stop": 288000,
        "target": 330000,
        "horizon": 7,
    }
    t = data.kite_order_text(row)
    assert (
        "BUY 10 x TCS" in t
        and "3000.00" in t
        and "2880.00" in t
        and "3300.00" in t
        and "day 7" in t
    )
    with cli._session() as s:
        assert data.protective_status(s) == {}
        s.add(
            AuditLog(
                at=datetime.now(UTC),
                actor="system",
                action="protective_check",
                payload_json={"symbols": {"INFY": "OK", "TCS": "MISSING"}},
                payload_hash="h",
            )
        )
        s.commit()
        assert data.protective_status(s) == {"INFY": "OK", "TCS": "MISSING"}


def test_positions_page_flags_missing_protection(db) -> None:  # type: ignore[no-untyped-def]
    with cli._session() as s:
        f = Fill(broker_trade_id="t1", origin="paper", account="p", symbol="INFY", exchange="NSE", side="BUY", product="CNC", qty=10,
                 price_paise=150000, filled_at=datetime.now(UTC), trade_date=date.today(), source="paper")  # fmt: skip
        s.add(f)
        s.flush()
        s.add(
            Lot(
                symbol="INFY",
                open_fill_id=f.id,
                open_date=date.today(),
                qty_open=10,
                qty_remaining=10,
                cost_paise_per_share=150000,
            )
        )
        s.commit()
    at = page("positions.py")
    assert not at.exception and at.dataframe[0].value.shape[0] == 1 and at.warning
    assert "INFY" in at.warning[0].value


def test_system_page_lists_job_runs_and_errors() -> None:
    with cli._session() as s:
        s.add(
            JobRun(
                job="sync_ledger",
                started_at=datetime.now(UTC),
                ended_at=datetime.now(UTC),
                status="error",
                error="boom",
            )
        )
        s.commit()
    at = page("system.py")
    assert not at.exception and at.error and "failed job runs" in at.error[0].value
    assert at.metric[0].label == "Ollama"


def test_daily_report_aggregates_the_day(db) -> None:  # type: ignore[no-untyped-def]
    from tests.unit.test_ledger_db import BOOK, tb, write
    from tradepartner.ledger import importers, rebuild
    from tradepartner.settings import load_settings

    s = load_settings()
    path = write(Path(__import__("tempfile").mkdtemp()), "tb.csv", tb(BOOK))
    importers.import_tradebook(db, path)
    rebuild.rebuild(db, s.costs, s.tax)
    db.add(
        RiskEvent(
            at=dt(2026, 5, 20, 11, 0), rule="x", severity="block", scope="global", details_json={}
        )
    )
    db.add(JobRun(job="eod", started_at=dt(2026, 5, 20, 16, 30), status="error", error="x"))
    db.add(OrderProposal(symbol="TCS", side="BUY", product="CNC", order_type="LIMIT", qty=1, mode_at_creation="MANUAL", risk_decision="approved",
                         idempotency_key="z", created_at=dt(2026, 5, 20, 9, 0)))  # fmt: skip
    db.commit()
    rep = daily.build(db, date(2026, 5, 20))
    assert (
        rep.fills == 2
        and set(rep.realized_by_category) == {"intraday_spec"}
        and rep.realized_total != 0
    )
    assert (
        rep.risk_events == {"block": 1}
        and rep.proposals == {"approved": 1}
        and rep.failed_jobs == ["eod"]
    )
    assert rep.fy_realized["ltcg"] > 0 and rep.open_lots == 0
    text, html = daily.render_text(rep), daily.render_html(rep)
    assert (
        "FAILED JOBS: eod" in text and "Daily report 2026-05-20" in html and "Failed jobs" in html
    )
    empty = daily.build(db, date(2030, 1, 1))
    assert "none" in daily.render_html(empty) and daily.render_text(empty).startswith("2030-01-01")


def test_job_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    out = runner.invoke(cli.app, ["job", "list"]).output
    assert "generate_signals" in out and "Sat 10:00" in out and "heavy" in out
    assert runner.invoke(cli.app, ["job", "run", "nope"]).exit_code == 2
    assert (
        runner.invoke(cli.app, ["job", "run", "daily_report"]).exit_code == 2
    )  # no capital -> refuses to start


def test_job_run_and_run_command_with_capital(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import shutil

    import yaml

    from tradepartner import scheduler

    cfg = tmp_path / "config"
    shutil.copytree(Path(__file__).parents[2] / "config", cfg)
    d = yaml.safe_load((cfg / "risk.yaml").read_text())
    d["capital_base_inr"] = 500_000
    (cfg / "risk.yaml").write_text(yaml.safe_dump(d))
    monkeypatch.setenv("TRADEPARTNER_CONFIG_DIR", str(cfg))
    r = runner.invoke(cli.app, ["job", "run", "daily_report"])
    assert r.exit_code == 0 and "daily_report: ok" in r.output
    boom = jobs.JobSpec("boom", lambda c: 1 / 0, "never")
    monkeypatch.setitem(jobs.BY_NAME, "boom", boom)
    assert runner.invoke(cli.app, ["job", "run", "boom"]).exit_code == 1
    from tradepartner.modes.controller import ModeController
    from tradepartner.modes.state import GLOBAL

    with cli._session() as s:
        ModeController(s).set_mode(GLOBAL, "AUTO", "user", "x", datetime.now(UTC))
    assert runner.invoke(cli.app, ["job", "run", "daily_report"]).exit_code == 0
    with cli._session() as s:  # an ad-hoc job process never inherits AUTO either
        assert ModeController(s).current(GLOBAL).mode == "MANUAL"
    started: list[object] = []
    monkeypatch.setattr(scheduler, "start", lambda ctx: started.append(ctx))
    from tradepartner.modes.controller import ModeController
    from tradepartner.modes.state import GLOBAL

    with cli._session() as s:
        ModeController(s).set_mode(GLOBAL, "AUTO", "user", "x", datetime.now(UTC))
    res = runner.invoke(cli.app, ["run"])
    assert res.exit_code == 0 and "Startup clamped AUTO -> MANUAL" in res.output and started
    with cli._session() as s:
        assert ModeController(s).current(GLOBAL).mode == "MANUAL"
        assert s.scalar(select(JobRun.id).where(JobRun.job == "daily_report")) is not None


def test_today_page_api_approve_buttons_never_report_false_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import shutil

    import yaml

    c = tmp_path / "cfg"
    shutil.copytree(Path(__file__).parents[2] / "config", c)
    app = yaml.safe_load((c / "app.yaml").read_text())
    app["manual_execution"] = "api_approve"
    (c / "app.yaml").write_text(yaml.safe_dump(app))
    monkeypatch.setenv("TRADEPARTNER_CONFIG_DIR", str(c))
    with cli._session() as s:
        sid = seed_signal(s)
    at = page("today.py")
    assert not at.exception
    approve = next(b for b in at.button if b.key == f"approve-{sid}")
    approve.click().run()
    # paper broker, no quotes and no capital in this fixture: the re-check refuses, and the page says so
    assert at.error and "Not placed" in at.error[0].value and not at.success
    with cli._session() as s:
        assert (
            proposals.current_status(s, sid) == "rejected_by_risk"
        )  # the failed re-check is recorded
        from tradepartner.db.models import Order

        assert s.scalar(select(Order.id)) is None  # and nothing was placed
        second = seed_signal(s)
    at2 = page("today.py")
    next(b for b in at2.button if b.key == f"reject-{second}").click().run()
    assert not at2.exception
    with cli._session() as s:
        assert proposals.current_status(s, second) == "ignored"
