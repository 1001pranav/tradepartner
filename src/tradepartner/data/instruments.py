"""Kite instrument master -> ``instruments`` table. Only NSE cash equity is kept (spec 6.1)."""

from collections.abc import Iterable
from datetime import datetime
from decimal import Decimal
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from tradepartner.db.models import Instrument


class InstrumentSource(Protocol):
    def instruments(self, exchange: str | None = None) -> list[dict[str, Any]]: ...


def is_nse_equity(row: dict[str, Any]) -> bool:
    return (
        row.get("exchange") == "NSE"
        and row.get("segment") == "NSE"
        and row.get("instrument_type") == "EQ"
    )


def _paise(value: object) -> int:
    return int((Decimal(str(value)) * 100).to_integral_value())


def upsert_instruments(session: Session, rows: Iterable[dict[str, Any]], now: datetime) -> int:
    """Insert or refresh NSE equity rows. Returns how many rows were kept."""
    kept = 0
    for row in rows:
        if not is_nse_equity(row):
            continue
        token = int(row["instrument_token"])
        inst = session.get(Instrument, token) or Instrument(instrument_token=token)
        inst.exchange = "NSE"
        inst.tradingsymbol = str(row["tradingsymbol"])
        inst.isin = row.get("isin") or None
        inst.name = row.get("name") or None
        inst.segment = "NSE"
        inst.lot_size = int(row.get("lot_size") or 1)
        inst.tick_size_paise = _paise(row.get("tick_size", "0.05"))
        inst.last_updated = now
        session.add(inst)
        kept += 1
    session.commit()
    return kept


def refresh_instruments(session: Session, source: InstrumentSource, now: datetime) -> int:
    return upsert_instruments(session, source.instruments("NSE"), now)


def token_for(session: Session, symbol: str) -> int | None:
    return session.scalar(
        select(Instrument.instrument_token).where(Instrument.tradingsymbol == symbol)
    )


def symbol_tokens(session: Session) -> dict[str, int]:
    return {
        sym: tok
        for tok, sym in session.execute(
            select(Instrument.instrument_token, Instrument.tradingsymbol)
        )
    }
