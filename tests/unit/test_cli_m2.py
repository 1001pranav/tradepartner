from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.unit.test_ledger_db import BOOK, tb, write
from tests.unit.test_sync import FakeLedgerKite, order, trade
from tradepartner import cli

runner = CliRunner()


def test_import_rebuild_report_flow(tmp_path: Path) -> None:
    p = write(tmp_path, "tb.csv", tb(BOOK))
    r = runner.invoke(cli.app, ["ledger", "import-tradebook", str(p)])
    assert r.exit_code == 0, r.output
    assert "5 fills imported" in r.output
    assert (
        "0 fills imported" in runner.invoke(cli.app, ["ledger", "import-tradebook", str(p)]).output
    )
    assert "Rebuilt" in runner.invoke(cli.app, ["ledger", "rebuild"]).output
    rep = runner.invoke(cli.app, ["report", "tax", "--fy", "2025-26"])
    assert rep.exit_code == 0 and "for CA review" in rep.output


def test_import_failure_exit_codes(tmp_path: Path) -> None:
    bad = write(tmp_path, "bad.csv", "nope\n1\n")
    assert runner.invoke(cli.app, ["ledger", "import-tradebook", str(bad)]).exit_code == 1
    assert runner.invoke(cli.app, ["ledger", "import-charges", str(bad)]).exit_code == 1


def test_import_charges_ok(tmp_path: Path) -> None:
    runner.invoke(cli.app, ["ledger", "import-tradebook", str(write(tmp_path, "tb.csv", tb(BOOK)))])
    note = write(
        tmp_path,
        "cn.csv",
        "trade_date,symbol,trade_id,component,amount_inr\n2025-06-10,INFY,2,stt,16.00\n",
    )
    r = runner.invoke(cli.app, ["ledger", "import-charges", str(note)])
    assert r.exit_code == 0 and "1 charge lines" in r.output


def test_unmatched_alert_printed(tmp_path: Path) -> None:
    only_sell = tb(["X,INE1,2026-01-05,NSE,EQ,EQ,sell,false,3,10.00,t1,o1,2026-01-05T10:00:00"])
    r = runner.invoke(
        cli.app, ["ledger", "import-tradebook", str(write(tmp_path, "s.csv", only_sell))]
    )
    assert "ALERT" in r.output
    assert "ALERT" in runner.invoke(cli.app, ["ledger", "rebuild"]).output


def test_reconcile_command(tmp_path: Path) -> None:
    runner.invoke(cli.app, ["ledger", "import-tradebook", str(write(tmp_path, "tb.csv", tb(BOOK)))])
    assert runner.invoke(cli.app, ["ledger", "reconcile"]).exit_code == 2
    off = write(tmp_path, "c.csv", "fy,category,pnl_inr\n2025-26,stcg,1.00\n")
    r = runner.invoke(cli.app, ["ledger", "reconcile", "--console-summary", str(off)])
    assert r.exit_code == 1 and "AUTO entries blocked" in r.output
    ack = runner.invoke(cli.app, ["ledger", "reconcile", "--acknowledge", "known"])
    assert ack.exit_code == 0 and "Acknowledged" in ack.output


def test_reconcile_clean(tmp_path: Path) -> None:
    from sqlalchemy import func, select

    from tradepartner.db.models import Realized

    runner.invoke(cli.app, ["ledger", "import-tradebook", str(write(tmp_path, "tb.csv", tb(BOOK)))])
    with cli._session() as s:
        rows = list(
            s.execute(
                select(Realized.fy, Realized.category, func.sum(Realized.pnl_paise)).group_by(
                    Realized.fy, Realized.category
                )
            )
        )
    csv = "fy,category,pnl_inr\n" + "".join(f"{fy},{c},{t / 100:.2f}\n" for fy, c, t in rows)
    r = runner.invoke(
        cli.app, ["ledger", "reconcile", "--console-summary", str(write(tmp_path, "ok.csv", csv))]
    )
    assert r.exit_code == 0 and "Reconciled." in r.output


class KiteWithProfile(FakeLedgerKite):
    def profile(self) -> dict[str, str]:
        return {"user_id": "AB1234"}


def test_sync_command(monkeypatch: pytest.MonkeyPatch) -> None:
    k = KiteWithProfile()
    k._orders = [order("A1")]
    k._trades = [trade("T1", "A1")]
    k._holdings = [
        {"exchange": "NSE", "tradingsymbol": "INFY", "quantity": 7, "average_price": 1500.0}
    ]
    monkeypatch.setattr(cli, "_kite", lambda: k)
    r = runner.invoke(cli.app, ["sync"])
    assert r.exit_code == 0, r.output
    assert "fills+1" in r.output and "MISMATCH INFY" in r.output  # broker 7 vs ledger 10


def test_sync_needs_login() -> None:
    assert runner.invoke(cli.app, ["sync"]).exit_code == 2


def test_ui_command_binds_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    seen: list[list[str]] = []
    monkeypatch.setattr(subprocess, "call", lambda cmd: seen.append(cmd) or 0)
    assert runner.invoke(cli.app, ["ui"]).exit_code == 0
    cmd = seen[0]
    assert cmd[cmd.index("--server.address") + 1] == "127.0.0.1" and "streamlit" in cmd
