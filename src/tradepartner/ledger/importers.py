"""Importers for history and authoritative data (spec 9.3). All idempotent; unknown columns fail."""

import csv
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from tradepartner.calendar import IST
from tradepartner.db.models import Charge, Fill

TRADEBOOK_COLUMNS = {
    "symbol", "isin", "trade_date", "exchange", "segment", "series", "trade_type",
    "auction", "quantity", "price", "trade_id", "order_id", "order_execution_time",
}  # fmt: skip
_REQUIRED = {"symbol", "trade_date", "exchange", "segment", "trade_type", "quantity", "price",
             "trade_id"}  # fmt: skip
CHARGE_COMPONENTS = {"brokerage", "stt", "exchange_txn", "sebi", "stamp", "gst", "dp", "other"}


class LedgerImportError(Exception):
    """Raised for unreadable or unsupported import files."""


def _paise(text: str) -> int:
    value = Decimal(text.strip().replace(",", ""))
    paise = value * 100
    if paise != paise.to_integral_value():
        raise LedgerImportError(f"price {text!r} has sub-paise precision")
    return int(paise)


def _parse_time(row: dict[str, str], trade_date: date) -> datetime:
    raw = (row.get("order_execution_time") or "").strip()
    if not raw:
        return datetime(trade_date.year, trade_date.month, trade_date.day, tzinfo=IST).astimezone(
            UTC
        )
    ts = datetime.fromisoformat(raw)
    return (ts.replace(tzinfo=IST) if ts.tzinfo is None else ts).astimezone(UTC)


def import_tradebook(session: Session, path: Path, account: str = "console") -> int:
    """Zerodha Console tradebook CSV -> ``fills``. Returns rows added.

    The tradebook has no product column, so rows are recorded as CNC; same-day buy+sell netting
    is applied by the lot matcher, not by this importer. # SPEC-QUESTION: product is unknown here.
    """
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        cols = set(reader.fieldnames or [])
        if cols - TRADEBOOK_COLUMNS:
            raise LedgerImportError(
                f"unknown columns in tradebook: {sorted(cols - TRADEBOOK_COLUMNS)}"
            )
        if _REQUIRED - cols:
            raise LedgerImportError(f"missing columns in tradebook: {sorted(_REQUIRED - cols)}")
        rows = list(reader)
    existing = set(session.scalars(select(Fill.broker_trade_id)))
    added = 0
    for n, row in enumerate(rows, start=2):
        if row["exchange"].strip().upper() != "NSE" or row["segment"].strip().upper() != "EQ":
            raise LedgerImportError(
                f"row {n}: only NSE equity is supported, got "
                f"{row['exchange']}/{row['segment']} for {row['symbol']}"
            )
        side = {"buy": "BUY", "sell": "SELL"}.get(row["trade_type"].strip().lower())
        if side is None:
            raise LedgerImportError(f"row {n}: unknown trade_type {row['trade_type']!r}")
        qty = int(row["quantity"])
        if qty <= 0:
            raise LedgerImportError(f"row {n}: quantity must be positive")
        tid = row["trade_id"].strip()
        if tid in existing:
            continue
        existing.add(tid)
        tdate = date.fromisoformat(row["trade_date"].strip())
        session.add(
            Fill(
                broker_trade_id=tid,
                broker_order_id=(row.get("order_id") or "").strip() or None,
                origin="manual_external",
                account=account,
                symbol=row["symbol"].strip(),
                isin=(row.get("isin") or "").strip() or None,
                exchange="NSE",
                side=side,
                product="CNC",
                qty=qty,
                price_paise=_paise(row["price"]),
                filled_at=_parse_time(row, tdate),
                trade_date=tdate,
                source="console_csv",
            )
        )
        added += 1
    session.commit()
    return added


def import_contract_charges(session: Session, path: Path) -> int:
    """Contract-note charges CSV (trade_date,symbol,trade_id,component,amount_inr).

    These supersede computed charges for the referenced fill (spec 8). Idempotent per
    (fill, component, source).
    """
    required = {"trade_date", "symbol", "trade_id", "component", "amount_inr"}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        cols = set(reader.fieldnames or [])
        if cols != required:
            raise LedgerImportError(f"contract-note CSV needs exactly columns {sorted(required)}")
        rows = list(reader)
    fills = {f.broker_trade_id: f for f in session.scalars(select(Fill))}
    seen = {
        (c.fill_id, c.component)
        for c in session.scalars(select(Charge).where(Charge.source == "contract_note"))
    }
    added = 0
    for n, row in enumerate(rows, start=2):
        fill = fills.get(row["trade_id"].strip())
        if fill is None:
            raise LedgerImportError(f"row {n}: unknown trade_id {row['trade_id']!r}")
        comp = row["component"].strip().lower()
        if comp not in CHARGE_COMPONENTS:
            raise LedgerImportError(f"row {n}: unknown component {comp!r}")
        if (fill.id, comp) in seen:
            continue
        seen.add((fill.id, comp))
        session.add(
            Charge(
                fill_id=fill.id,
                trade_date=fill.trade_date,
                symbol=fill.symbol,
                component=comp,
                amount_paise=_paise(row["amount_inr"]),
                computed=False,
                source="contract_note",
            )
        )
        added += 1
    session.commit()
    return added
