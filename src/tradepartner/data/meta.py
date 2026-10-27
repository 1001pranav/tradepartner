"""Per-symbol metadata the risk engine needs: sector, 20-day average traded value, surveillance
flag, results/ex-date (spec 7.2). Built from stored candles, config and the database."""

import csv
from datetime import date
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from tradepartner.data.candles import CandleStore
from tradepartner.data.universe import read_sectors, sector_of
from tradepartner.db.models import CorporateAction, ResultDate
from tradepartner.paths import config_dir
from tradepartner.risk.types import InstrumentMeta

ADV_DAYS = 20


def read_asm_gsm(path: Path | None = None) -> set[str]:
    p = path or config_dir() / "asm_gsm.csv"
    if not p.exists():
        return set()
    lines = [ln for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    return {r["symbol"].strip() for r in csv.DictReader(lines) if r.get("symbol", "").strip()}


def avg_daily_value_paise(store: CandleStore, symbol: str, as_of: date) -> int:
    df = store.read("day", symbol, end=as_of).tail(ADV_DAYS)
    if df.empty:
        return 0
    return int((df["close"].astype("float64") * df["volume"].astype("float64")).mean())


def build_meta(
    session: Session,
    store: CandleStore,
    symbols: list[str],
    today: date,
    sectors: dict[str, str] | None = None,
    asm_gsm: set[str] | None = None,
) -> dict[str, InstrumentMeta]:
    sec = sectors if sectors is not None else read_sectors()
    asm = asm_gsm if asm_gsm is not None else read_asm_gsm()
    results = {
        s: d
        for s, d in session.execute(
            select(ResultDate.symbol, ResultDate.result_date).where(ResultDate.result_date == today)
        )
    }
    ex_dates = {
        s: d
        for s, d in session.execute(
            select(CorporateAction.symbol, CorporateAction.ex_date).where(
                CorporateAction.ex_date == today
            )
        )
    }
    return {
        s: InstrumentMeta(
            s,
            sector_of(s, sec),
            avg_daily_value_paise(store, s, today),
            s in asm,
            results.get(s),
            ex_dates.get(s),
        )
        for s in symbols
    }
