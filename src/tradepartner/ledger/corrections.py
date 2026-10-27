"""Ledger corrections (spec 11, page 4): no edits, only a reversing row plus a replacement row."""

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from tradepartner.db import audit
from tradepartner.db.models import Fill


def add_correction(
    session: Session,
    fill_id: int,
    new_qty: int,
    new_price_paise: int,
    reason: str,
    now: datetime | None = None,
) -> tuple[Fill, Fill]:
    """Append a reversing fill (opposite side, same qty/price) and a corrected replacement."""
    if new_qty <= 0 or new_price_paise <= 0:
        raise ValueError("corrected qty and price must be positive")
    if not reason.strip():
        raise ValueError("a reason is required")
    original = session.get(Fill, fill_id)
    if original is None:
        raise ValueError(f"fill {fill_id} not found")
    if original.source == "correction" and original.broker_trade_id.endswith("-rev"):
        raise ValueError("cannot correct a reversing row")
    existing = session.scalars(
        select(Fill.id).where(Fill.broker_trade_id.like(f"corr-{original.id}-%"))
    ).all()
    seq = len(existing) // 2 + 1
    ts = now or datetime.now(UTC)
    common = dict(
        broker_order_id=None, order_id=None, origin="manual_external", account=original.account,
        symbol=original.symbol, isin=original.isin, exchange="NSE", product=original.product,
        filled_at=original.filled_at, trade_date=original.trade_date, source="correction",
        created_at=ts,
    )  # fmt: skip
    reverse = Fill(
        broker_trade_id=f"corr-{original.id}-{seq}-rev",
        side="SELL" if original.side == "BUY" else "BUY",
        qty=original.qty,
        price_paise=original.price_paise,
        **common,
    )
    replacement = Fill(
        broker_trade_id=f"corr-{original.id}-{seq}-new",
        side=original.side,
        qty=new_qty,
        price_paise=new_price_paise,
        **common,
    )
    session.add_all([reverse, replacement])
    audit.log(
        session,
        "user",
        "ledger_correction",
        {"fill_id": original.id, "new_qty": new_qty, "new_price_paise": new_price_paise,
         "reason": reason},
        ts,
    )  # fmt: skip
    session.commit()
    return reverse, replacement
