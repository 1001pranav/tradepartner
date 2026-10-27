"""Rebuild, importers, corrections, reconciliation, tax export against a real (temp) SQLite DB."""

from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import DatabaseError
from sqlalchemy.orm import Session

from tradepartner.db.models import Charge, Fill, Lot, Realized, RiskEvent
from tradepartner.ledger import corrections, importers, rebuild, reconcile
from tradepartner.reports import tax_export
from tradepartner.settings import load_settings

S = load_settings()
HEADER = ("symbol,isin,trade_date,exchange,segment,series,trade_type,auction,quantity,price,"
          "trade_id,order_id,order_execution_time\n")  # fmt: skip


def tb(rows: list[str]) -> str:
    return HEADER + "\n".join(rows) + "\n"


def write(tmp: Path, name: str, content: str) -> Path:
    p = tmp / name
    p.write_text(content)
    return p


def row(sym: str, day: str, side: str, qty: int, px: str, tid: str, t: str = "10:00:00") -> str:
    return f"{sym},INE001,{day},NSE,EQ,EQ,{side},false,{qty},{px},{tid},o{tid},{day}T{t}"


BOOK = [
    row("INFY", "2025-04-10", "buy", 10, "1500.00", "1"),
    row("INFY", "2025-06-10", "sell", 4, "1600.00", "2"),  # STCG
    row("INFY", "2026-05-12", "sell", 6, "1700.00", "3"),  # LTCG (held > 12 months)
    row("TCS", "2026-05-20", "buy", 5, "3000.50", "4", "09:30:00"),
    row("TCS", "2026-05-20", "sell", 5, "3010.00", "5", "14:00:00"),  # intraday
]


@pytest.fixture
def loaded(db: Session, tmp_path: Path) -> Session:
    importers.import_tradebook(db, write(tmp_path, "tb.csv", tb(BOOK)))
    rebuild.rebuild(db, S.costs, S.tax, now=datetime(2026, 6, 1, tzinfo=UTC))
    return db


def test_import_idempotent_and_paise(db: Session, tmp_path: Path) -> None:
    p = write(tmp_path, "tb.csv", tb(BOOK))
    assert importers.import_tradebook(db, p) == 5
    assert importers.import_tradebook(db, p) == 0
    fill = db.scalars(select(Fill).where(Fill.broker_trade_id == "4")).one()
    assert fill.price_paise == 300050 and fill.source == "console_csv"
    assert fill.filled_at == datetime(2026, 5, 20, 4, 0, tzinfo=UTC)  # 09:30 IST


@pytest.mark.parametrize(
    ("content", "msg"),
    [
        (HEADER.replace("auction", "mystery") + "\n", "unknown columns"),
        ("symbol,trade_date\nX,2026-01-01\n", "missing columns"),
        (tb([row("X", "2026-01-01", "buy", 1, "1.00", "1").replace(",NSE,", ",BSE,")]), "only NSE"),
        (
            tb([row("X", "2026-01-01", "buy", 1, "1.00", "1").replace(",EQ,EQ,", ",FO,EQ,")]),
            "only NSE",
        ),
        (tb([row("X", "2026-01-01", "hold", 1, "1.00", "1")]), "trade_type"),
        (tb([row("X", "2026-01-01", "buy", 0, "1.00", "1")]), "positive"),
        (tb([row("X", "2026-01-01", "buy", 1, "1.005", "1")]), "sub-paise"),
    ],
)
def test_import_fails_loudly(db: Session, tmp_path: Path, content: str, msg: str) -> None:
    with pytest.raises(importers.LedgerImportError, match=msg):
        importers.import_tradebook(db, write(tmp_path, "bad.csv", content))
    assert db.scalar(select(func.count()).select_from(Fill)) == 0


def test_import_default_time_and_naive_vs_aware(db: Session, tmp_path: Path) -> None:
    r1 = "X,INE9,2026-01-05,NSE,EQ,EQ,buy,false,1,10.00,t1,o1,"
    r2 = "X,INE9,2026-01-05,NSE,EQ,EQ,buy,false,1,10.00,t2,o2,2026-01-05T10:00:00+05:30"
    importers.import_tradebook(db, write(tmp_path, "x.csv", tb([r1, r2])))
    got = {f.broker_trade_id: f.filled_at for f in db.scalars(select(Fill))}
    assert got["t1"] == datetime(2026, 1, 4, 18, 30, tzinfo=UTC)  # midnight IST
    assert got["t2"] == datetime(2026, 1, 5, 4, 30, tzinfo=UTC)


def test_rebuild_categories_and_pnl(loaded: Session) -> None:
    rows = {(r.symbol, r.category): r for r in loaded.scalars(select(Realized))}
    assert set(rows) == {("INFY", "stcg"), ("INFY", "ltcg"), ("TCS", "intraday_spec")}
    stcg = rows[("INFY", "stcg")]
    assert stcg.buy_value_paise == 4 * 150000 and stcg.sell_value_paise == 4 * 160000
    assert stcg.fy == "2025-26" and rows[("INFY", "ltcg")].fy == "2026-27"
    intr = rows[("TCS", "intraday_spec")]
    assert (
        intr.pnl_paise
        == intr.sell_value_paise - intr.buy_value_paise - intr.charges_paise - intr.stt_paise
    )
    # capital gains: STT not deducted
    assert stcg.pnl_paise == stcg.sell_value_paise - stcg.buy_value_paise - stcg.charges_paise
    assert stcg.stt_paise > 0
    assert loaded.scalar(select(func.sum(Lot.qty_remaining))) is None  # nothing left open


def test_rebuild_is_deterministic_and_idempotent(loaded: Session) -> None:
    def snap() -> list[tuple[object, ...]]:
        r = [
            tuple(
                x.__dict__[k]
                for k in (
                    "symbol",
                    "qty",
                    "pnl_paise",
                    "category",
                    "fy",
                    "charges_paise",
                    "stt_paise",
                )
            )
            for x in loaded.scalars(select(Realized).order_by(Realized.id))
        ]
        lots_ = [
            (x.symbol, x.qty_remaining, x.cost_paise_per_share)
            for x in loaded.scalars(select(Lot).order_by(Lot.id))
        ]
        return r + lots_  # type: ignore[return-value]

    first = snap()
    n_charges = loaded.scalar(select(func.count()).select_from(Charge))
    rebuild.rebuild(loaded, S.costs, S.tax)
    rebuild.rebuild(loaded, S.costs, S.tax)
    assert snap() == first
    assert (
        loaded.scalar(select(func.count()).select_from(Charge)) == n_charges
    )  # no duplicate charges


def test_open_lots_persisted(db: Session, tmp_path: Path) -> None:
    importers.import_tradebook(
        db, write(tmp_path, "t.csv", tb([row("INFY", "2026-01-05", "buy", 10, "100.00", "1")]))
    )
    res = rebuild.rebuild(db, S.costs, S.tax)
    assert (res.lots, res.realized) == (1, 0)
    lot = db.scalars(select(Lot)).one()
    assert lot.qty_remaining == 10 and lot.cost_paise_per_share == 10000 and not lot.adjusted_for_ca


def test_contract_note_charges_supersede_computed(loaded: Session, tmp_path: Path) -> None:
    before = {
        f.broker_trade_id: rebuild.effective_charges(loaded)[f.id]
        for f in loaded.scalars(select(Fill))
    }
    note = write(
        tmp_path,
        "cn.csv",
        "trade_date,symbol,trade_id,component,amount_inr\n2025-06-10,INFY,2,brokerage,0.00\n2025-06-10,INFY,2,stt,16.00\n",
    )
    assert importers.import_contract_charges(loaded, note) == 2
    assert importers.import_contract_charges(loaded, note) == 0
    fid = loaded.scalars(select(Fill.id).where(Fill.broker_trade_id == "2")).one()
    assert rebuild.effective_charges(loaded)[fid] == {"brokerage": 0, "stt": 1600}
    assert before["2"] != {"brokerage": 0, "stt": 1600}
    rebuild.rebuild(loaded, S.costs, S.tax)
    row_ = loaded.scalars(select(Realized).where(Realized.category == "stcg")).one()
    # sell-side STT from the note (1600) + 4/10 of the computed buy-side STT (600)
    assert row_.stt_paise == 2200


@pytest.mark.parametrize(
    ("content", "msg"),
    [
        ("a,b\n1,2\n", "exactly columns"),
        (
            "trade_date,symbol,trade_id,component,amount_inr\n2025-06-10,INFY,nope,stt,1\n",
            "unknown trade_id",
        ),
        (
            "trade_date,symbol,trade_id,component,amount_inr\n2025-06-10,INFY,2,tip,1\n",
            "unknown component",
        ),
    ],
)
def test_contract_charges_errors(loaded: Session, tmp_path: Path, content: str, msg: str) -> None:
    with pytest.raises(importers.LedgerImportError, match=msg):
        importers.import_contract_charges(loaded, write(tmp_path, "cn.csv", content))


def test_append_only_charges_and_fills_still_enforced(loaded: Session) -> None:
    with pytest.raises(DatabaseError, match="append-only"):
        loaded.execute(text("DELETE FROM charges"))
    loaded.rollback()


def test_unmatched_reported_and_excluded_from_pnl(db: Session, tmp_path: Path) -> None:
    importers.import_tradebook(
        db, write(tmp_path, "t.csv", tb([row("INFY", "2026-01-05", "sell", 3, "100.00", "1")]))
    )
    res = rebuild.rebuild(db, S.costs, S.tax)
    assert res.alerts and db.scalars(select(Realized)).one().pnl_paise == 0


def test_corrections_reverse_and_replace(loaded: Session) -> None:
    fid = loaded.scalars(select(Fill.id).where(Fill.broker_trade_id == "1")).one()
    rev, new = corrections.add_correction(loaded, fid, 12, 149000, "typo in qty")
    assert (rev.side, new.side, new.qty, rev.qty) == ("SELL", "BUY", 12, 10)
    assert rev.source == new.source == "correction"
    rev2, new2 = corrections.add_correction(loaded, fid, 11, 149000, "again")
    assert rev2.broker_trade_id.endswith("-2-rev") and new2.broker_trade_id.endswith("-2-new")
    with pytest.raises(ValueError, match="positive"):
        corrections.add_correction(loaded, fid, 0, 1, "x")
    with pytest.raises(ValueError, match="reason"):
        corrections.add_correction(loaded, fid, 1, 1, "  ")
    with pytest.raises(ValueError, match="not found"):
        corrections.add_correction(loaded, 9999, 1, 1, "x")
    with pytest.raises(ValueError, match="reversing"):
        corrections.add_correction(loaded, rev.id, 1, 1, "x")
    from tradepartner.db.models import AuditLog

    assert loaded.scalar(select(func.count()).select_from(AuditLog)) == 2


# ---------- reconciliation ----------


def test_position_reconciliation_and_ack(db: Session, tmp_path: Path) -> None:
    from tradepartner.db.models import PositionSnapshot

    importers.import_tradebook(
        db, write(tmp_path, "t.csv", tb([row("INFY", "2026-01-05", "buy", 10, "100.00", "1")]))
    )
    rebuild.rebuild(db, S.costs, S.tax)
    day = date(2026, 1, 6)
    assert not reconcile.has_reconciliation_block(db)

    def snap(kind: str, sym: str, qty: int, avg: int, product: str | None = "CNC") -> None:
        db.add(
            PositionSnapshot(
                snapshot_date=day,
                kind=kind,
                symbol=sym,
                isin=None,
                product=product,
                qty=qty,
                avg_price_paise=avg,
            )
        )
        db.commit()

    snap("holding", "INFY", 10, 10040)  # within +/-0.5%
    assert reconcile.reconcile_positions(db, day) == []
    assert not reconcile.has_reconciliation_block(db)
    snap("position", "ZZZ", 3, 500)  # broker has something the ledger does not
    out = reconcile.reconcile_positions(db, day)
    assert [m.symbol for m in out] == ["ZZZ"] and out[0].reason == "quantity differs"
    assert reconcile.has_reconciliation_block(db)
    reconcile.acknowledge(db, "known manual transfer")
    assert not reconcile.has_reconciliation_block(db)
    db.add(
        PositionSnapshot(
            snapshot_date=date(2026, 1, 7),
            kind="holding",
            symbol="INFY",
            isin=None,
            product="CNC",
            qty=10,
            avg_price_paise=11000,
        )
    )
    db.commit()
    bad = reconcile.reconcile_positions(db, date(2026, 1, 7))
    assert bad[0].reason == "average price differs" and reconcile.has_reconciliation_block(db)
    assert (
        db.scalar(select(func.count()).select_from(RiskEvent).where(RiskEvent.severity == "block"))
        == 2
    )


def test_realized_reconciliation_tolerance(loaded: Session, tmp_path: Path) -> None:
    ledger = {
        (fy, cat): total
        for fy, cat, total in loaded.execute(
            select(Realized.fy, Realized.category, func.sum(Realized.pnl_paise)).group_by(
                Realized.fy, Realized.category
            )
        )
    }
    console = {k: v + 900 for k, v in ledger.items()}  # Rs 9 off: within +/-Rs 10
    assert reconcile.check_realized(loaded, console) == []
    console[("2025-26", "stcg")] += 5000  # Rs 50 more
    bad = reconcile.check_realized(loaded, console)
    assert len(bad) == 1 and bad[0].category == "stcg" and bad[0].diff_paise == -5900
    assert reconcile.has_reconciliation_block(loaded)
    csvp = write(tmp_path, "c.csv", "fy,category,pnl_inr\n2025-26,stcg,123.45\n")
    assert reconcile.load_console_summary(csvp) == {("2025-26", "stcg"): 12345}
    with pytest.raises(ValueError, match="exactly columns"):
        reconcile.load_console_summary(write(tmp_path, "d.csv", "a,b\n1,2\n"))


# ---------- tax export ----------


def test_tax_report_and_export(loaded: Session, tmp_path: Path) -> None:
    rep = tax_export.build_report(loaded, "2025-26", S.tax)
    assert rep.totals["stcg"] > 0 and rep.totals["ltcg"] == 0
    q = rep.quarterly.set_index("quarter")
    assert q.loc["Q1 (Apr-Jun)", "stcg"] > 0 and q.loc["Q2 (Jul-Sep)", "stcg"] == 0
    assert set(rep.charges["component"]) >= {"stt", "gst", "exchange_txn"}
    csv_path, xlsx_path = tax_export.export(loaded, "2026-27", S.tax, tmp_path / "out")
    sheets = pd.read_excel(xlsx_path, sheet_name=None)
    assert {
        "Summary",
        "Intraday speculative",
        "STCG",
        "LTCG",
        "Charges",
        "Quarterly (advance tax)",
    } <= set(sheets)
    summary = sheets["Summary"]
    assert summary["remark"].astype(str).str.contains("For CA review").any()
    assert len(sheets["LTCG"]) == 1 and len(sheets["Intraday speculative"]) == 1
    assert "pnl_paise" not in pd.read_csv(csv_path).columns
    empty = tax_export.build_report(loaded, "2030-31", S.tax)
    assert empty.realized.empty and all(v == 0 for v in empty.totals.values())
    assert tax_export.today_fy(date(2026, 10, 26)) == "2026-27"
    assert len(tax_export.quarterly(empty.realized)) == 4
