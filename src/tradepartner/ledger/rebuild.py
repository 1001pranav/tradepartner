"""Deterministic rebuild of the derived tables ``lots`` and ``realized`` from ``fills``
(spec 9.1). ``fills`` and ``charges`` are append-only; derived tables are disposable."""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from fractions import Fraction

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from tradepartner.costs import zerodha
from tradepartner.data.corporate_actions import load_actions
from tradepartner.db.models import Charge, Fill, Lot, Realized
from tradepartner.ledger import lots as lots_mod
from tradepartner.ledger import tax
from tradepartner.ledger.lots import FillRec, RealizedRow, round_half_up
from tradepartner.settings import CostsConfig, TaxConfig


@dataclass
class RebuildResult:
    lots: int = 0
    realized: int = 0
    alerts: list[str] = field(default_factory=list)


def _order_ref(f: Fill) -> str:
    return f.broker_order_id or f"fill-{f.id}"


def ensure_charges(session: Session, cfg: CostsConfig) -> int:
    """Compute and append charges for fills that have none. Returns lines inserted."""
    fills = list(session.scalars(select(Fill).order_by(Fill.id)))
    have = set(session.scalars(select(Charge.fill_id).where(Charge.fill_id.is_not(None))))
    by_id = {f.id: f for f in fills}
    inputs = [
        zerodha.FillIn(
            f.id, _order_ref(f), f.symbol, f.side, f.product, f.qty, f.price_paise, f.trade_date
        )
        for f in fills
    ]
    added = 0
    for line in zerodha.compute(inputs, cfg):
        if line.fill_id in have:
            continue
        f = by_id[line.fill_id]
        session.add(
            Charge(
                fill_id=f.id,
                trade_date=f.trade_date,
                symbol=f.symbol,
                component=line.component,
                amount_paise=line.amount_paise,
                computed=True,
                source="computed",
            )
        )
        added += 1
    session.commit()
    return added


def effective_charges(session: Session) -> dict[int, dict[str, int]]:
    """Per fill: contract-note charges if any exist, otherwise the computed ones."""
    rows = list(session.scalars(select(Charge).where(Charge.fill_id.is_not(None))))
    note: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    comp: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for c in rows:
        assert c.fill_id is not None
        target = note if c.source == "contract_note" else comp
        target[c.fill_id][c.component] += c.amount_paise
    out: dict[int, dict[str, int]] = {fid: dict(v) for fid, v in comp.items()}
    out.update({fid: dict(v) for fid, v in note.items()})
    return out


def _share(charges: dict[str, int] | None, qty: int, fill_qty: int) -> tuple[Fraction, Fraction]:
    ex_stt = Fraction(0)
    stt = Fraction(0)
    for comp, amount in (charges or {}).items():
        part = Fraction(amount * qty, fill_qty)
        if comp == "stt":
            stt += part
        else:
            ex_stt += part
    return ex_stt, stt


def _persist_row(
    row: RealizedRow,
    fills: dict[int, Fill],
    charges: dict[int, dict[str, int]],
    cfg: TaxConfig,
) -> Realized:
    ex_stt = Fraction(0)
    stt = Fraction(0)
    for fid in (row.buy_fill_id, row.sell_fill_id):
        if fid is None:
            continue
        a, b = _share(charges.get(fid), row.qty, fills[fid].qty)
        ex_stt += a
        stt += b
    category = tax.classify(row, cfg)
    ex_stt_p, stt_p = round_half_up(ex_stt), round_half_up(stt)
    return Realized(
        symbol=row.symbol,
        open_fill_id=row.open_fill_id,
        close_fill_id=row.close_fill_id,
        qty=row.qty,
        buy_value_paise=row.buy_value_paise,
        sell_value_paise=row.sell_value_paise,
        charges_paise=ex_stt_p,
        stt_paise=stt_p,
        holding_days=row.holding_days,
        category=category,
        fy=tax.fy_of(row.close_date),
        pnl_paise=tax.pnl_paise(
            category, row.sell_value_paise, row.buy_value_paise, ex_stt_p, stt_p
        ),
    )


def rebuild(
    session: Session, costs_cfg: CostsConfig, tax_cfg: TaxConfig, now: datetime | None = None
) -> RebuildResult:
    ensure_charges(session, costs_cfg)
    fills = {f.id: f for f in session.scalars(select(Fill).order_by(Fill.id))}
    recs = [
        FillRec(f.id, f.symbol, f.isin, f.side, f.qty, f.price_paise, f.filled_at, f.trade_date)
        for f in fills.values()
    ]
    matched = lots_mod.match(recs, load_actions(session))
    charges = effective_charges(session)
    created = now or datetime.now(UTC)

    session.execute(delete(Lot))
    session.execute(delete(Realized))
    rows = sorted(
        matched.realized,
        key=lambda r: (r.close_date, r.close_fill_id, r.open_fill_id or 0, r.kind, r.qty),
    )
    for r in rows:
        obj = _persist_row(r, fills, charges, tax_cfg)
        obj.created_at = created
        session.add(obj)
    for lot in matched.lots:
        session.add(
            Lot(
                symbol=lot.symbol,
                isin=lot.isin,
                open_fill_id=lot.open_fill_id,
                open_date=lot.open_date,
                qty_open=lot.qty_open,
                qty_remaining=lot.qty_rem,
                cost_paise_per_share=lot.cost_per_share_paise,
                adjusted_for_ca=lot.adjusted,
                created_at=created,
            )
        )
    session.commit()
    return RebuildResult(len(matched.lots), len(rows), matched.alerts)
