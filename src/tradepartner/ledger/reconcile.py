"""Reconciliation (spec 9.5): ledger vs broker positions, and ledger realized P&L vs Console."""

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from tradepartner.db.models import Lot, PositionSnapshot, Realized, RiskEvent

POSITION_TOLERANCE_PCT = 0.5  # qty exact; average price within +/-0.5%
REALIZED_TOLERANCE_PAISE = 10 * 100  # +/- Rs 10 per FY per category (M2 acceptance)
RULE_POSITIONS = "reconcile_positions"
RULE_REALIZED = "reconcile_realized"
RULE_ACK = "reconcile_ack"


@dataclass(frozen=True)
class PositionMismatch:
    symbol: str
    ledger_qty: int
    broker_qty: int
    ledger_avg_paise: int
    broker_avg_paise: int
    reason: str


@dataclass(frozen=True)
class RealizedDiff:
    fy: str
    category: str
    ledger_paise: int
    console_paise: int

    @property
    def diff_paise(self) -> int:
        return self.ledger_paise - self.console_paise


def ledger_positions(session: Session) -> dict[str, tuple[int, int]]:
    """symbol -> (qty, average cost in paise) from open lots."""
    qty: dict[str, int] = defaultdict(int)
    cost: dict[str, int] = defaultdict(int)
    for lot in session.scalars(select(Lot)):
        qty[lot.symbol] += lot.qty_remaining
        cost[lot.symbol] += lot.qty_remaining * lot.cost_paise_per_share
    return {s: (q, cost[s] // q) for s, q in qty.items() if q > 0}


def broker_positions(session: Session, day: date) -> dict[str, tuple[int, int]]:
    """symbol -> (qty, avg paise) from the day's snapshot.

    SPEC-QUESTION: holdings (incl. T1) are preferred; CNC day-positions are used only for symbols
    that are not yet holdings, to avoid double counting today's delivery buys.
    """
    rows = list(
        session.scalars(select(PositionSnapshot).where(PositionSnapshot.snapshot_date == day))
    )
    out: dict[str, tuple[int, int]] = {
        r.symbol: (r.qty, r.avg_price_paise) for r in rows if r.kind == "holding"
    }
    for r in rows:
        if r.kind == "position" and r.product == "CNC" and r.symbol not in out:
            out[r.symbol] = (r.qty, r.avg_price_paise)
    return out


def reconcile_positions(
    session: Session, day: date, now: datetime | None = None
) -> list[PositionMismatch]:
    """Compare and record. Any mismatch writes a ``block`` risk event, which blocks AUTO entries."""
    ledger = ledger_positions(session)
    broker = broker_positions(session, day)
    out: list[PositionMismatch] = []
    for sym in sorted(set(ledger) | set(broker)):
        lq, la = ledger.get(sym, (0, 0))
        bq, ba = broker.get(sym, (0, 0))
        if lq != bq:
            out.append(PositionMismatch(sym, lq, bq, la, ba, "quantity differs"))
        elif lq and abs(la - ba) * 1000 > ba * int(POSITION_TOLERANCE_PCT * 10):
            out.append(PositionMismatch(sym, lq, bq, la, ba, "average price differs"))
    at = now or datetime.now(UTC)
    session.add(
        RiskEvent(
            at=at,
            rule=RULE_POSITIONS,
            severity="block" if out else "info",
            scope="global",
            details_json={
                "date": day.isoformat(),
                "mismatches": [m.__dict__ for m in out],
            },
        )
    )
    session.commit()
    return out


def load_console_summary(path: Path) -> dict[tuple[str, str], int]:
    """Normalised Console Tax P&L summary CSV: columns fy,category,pnl_inr -> paise.

    The raw Console XLSX layout is not parsed here; export its category totals to this shape.
    """
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if set(reader.fieldnames or []) != {"fy", "category", "pnl_inr"}:
            raise ValueError("console summary needs exactly columns fy,category,pnl_inr")
        return {
            (r["fy"].strip(), r["category"].strip()): int(Decimal(r["pnl_inr"]) * 100)
            for r in reader
        }


def compare_realized(session: Session, console: dict[tuple[str, str], int]) -> list[RealizedDiff]:
    ledger = {
        (fy, cat): int(total)
        for fy, cat, total in session.execute(
            select(Realized.fy, Realized.category, func.sum(Realized.pnl_paise)).group_by(
                Realized.fy, Realized.category
            )
        )
    }
    diffs = [
        RealizedDiff(fy, cat, ledger.get((fy, cat), 0), console.get((fy, cat), 0))
        for fy, cat in sorted(set(ledger) | set(console))
    ]
    return diffs


def check_realized(
    session: Session, console: dict[tuple[str, str], int], now: datetime | None = None
) -> list[RealizedDiff]:
    """Diffs beyond tolerance are recorded as a ``block`` event; returns just those."""
    bad = [
        d
        for d in compare_realized(session, console)
        if abs(d.diff_paise) > REALIZED_TOLERANCE_PAISE
    ]
    session.add(
        RiskEvent(
            at=now or datetime.now(UTC),
            rule=RULE_REALIZED,
            severity="block" if bad else "info",
            scope="global",
            details_json={"diffs": [{**d.__dict__, "diff": d.diff_paise} for d in bad]},
        )
    )
    session.commit()
    return bad


def acknowledge(session: Session, reason: str, now: datetime | None = None) -> None:
    """Pranav acknowledges known differences; clears the AUTO block."""
    session.add(
        RiskEvent(
            at=now or datetime.now(UTC),
            rule=RULE_ACK,
            severity="info",
            scope="global",
            details_json={"reason": reason},
        )
    )
    session.commit()


def has_reconciliation_block(session: Session) -> bool:
    """True if the latest reconcile result of either kind is a block newer than the last ack."""
    last_ack = session.scalar(select(func.max(RiskEvent.id)).where(RiskEvent.rule == RULE_ACK)) or 0
    for rule in (RULE_POSITIONS, RULE_REALIZED):
        latest = session.scalar(
            select(RiskEvent).where(RiskEvent.rule == rule).order_by(RiskEvent.id.desc()).limit(1)
        )
        if latest is not None and latest.severity == "block" and latest.id > last_ack:
            return True
    return False
