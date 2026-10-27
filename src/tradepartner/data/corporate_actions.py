"""Corporate actions: import, and price adjustment (spec 10.3.5).

Convention for ``ratio_num:ratio_den``:
  split: ``num`` new shares for every ``den`` old (5:1 means one share becomes five)
  bonus: ``num`` bonus shares for every ``den`` held (1:1 means one extra per share)
Raw prices are stored; adjusted prices are derived for features and labels only.
"""

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from tradepartner.db.models import CorporateAction

_TYPES = {"split", "bonus", "dividend", "rights", "merger"}
_COLUMNS = {
    "symbol",
    "isin",
    "action_type",
    "ex_date",
    "ratio_num",
    "ratio_den",
    "amount",
    "source",
}


@dataclass(frozen=True)
class Action:
    symbol: str
    action_type: str
    ex_date: date
    ratio_num: int | None = None
    ratio_den: int | None = None


def share_multiplier(action: Action) -> tuple[int, int]:
    """(numerator, denominator) by which share count is multiplied on the ex-date."""
    if action.ratio_num is None or action.ratio_den is None:
        raise ValueError(f"{action.action_type} needs a ratio")
    if action.action_type == "split":
        return action.ratio_num, action.ratio_den
    if action.action_type == "bonus":
        return action.ratio_den + action.ratio_num, action.ratio_den
    raise ValueError(f"{action.action_type} does not change share count")


def import_actions_csv(session: Session, path: Path, source: str = "csv") -> int:
    """Idempotent import. Unknown columns fail loudly."""
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        unknown = set(reader.fieldnames or []) - _COLUMNS
        if unknown:
            raise ValueError(f"unknown columns in {path.name}: {sorted(unknown)}")
        added = 0
        for row in reader:
            kind = row["action_type"].strip().lower()
            if kind not in _TYPES:
                raise ValueError(f"unknown action_type {kind!r}")
            ex = date.fromisoformat(row["ex_date"].strip())
            sym = row["symbol"].strip()
            exists = session.scalar(
                select(CorporateAction.id).where(
                    CorporateAction.symbol == sym,
                    CorporateAction.ex_date == ex,
                    CorporateAction.action_type == kind,
                )
            )
            if exists:
                continue
            amount = row.get("amount", "").strip()
            session.add(
                CorporateAction(
                    symbol=sym,
                    isin=(row.get("isin") or "").strip() or None,
                    action_type=kind,
                    ex_date=ex,
                    ratio_num=int(row["ratio_num"]) if row.get("ratio_num", "").strip() else None,
                    ratio_den=int(row["ratio_den"]) if row.get("ratio_den", "").strip() else None,
                    amount_paise=int(Decimal(amount) * 100) if amount else None,
                    source=(row.get("source") or source).strip(),
                )
            )
            added += 1
    session.commit()
    return added


def load_actions(session: Session, symbol: str | None = None) -> list[Action]:
    stmt = select(CorporateAction).order_by(CorporateAction.ex_date)
    if symbol:
        stmt = stmt.where(CorporateAction.symbol == symbol)
    return [
        Action(a.symbol, a.action_type, a.ex_date, a.ratio_num, a.ratio_den)
        for a in session.scalars(stmt)
    ]


def adjustment_factors(dates: Sequence[date], actions: Sequence[Action]) -> pd.Series:
    """Price multiplier per date: product of 1/shares-multiplier of every later split/bonus."""
    factors = pd.Series(1.0, index=pd.Index(list(dates)), dtype="float64")
    for act in actions:
        if act.action_type not in ("split", "bonus"):
            continue
        num, den = share_multiplier(act)
        before = factors.index < act.ex_date
        factors[before] = factors[before] * den / num
    return factors


def adjust_candles(df: pd.DataFrame, actions: Sequence[Action]) -> pd.DataFrame:
    """Raw paise candles -> adjusted float candles in rupees. Volume scaled inversely.

    ``df`` needs columns date, open, high, low, close (int paise), volume. Output prices are
    floats because they feed features and labels only, never order prices or the ledger.
    """
    out = df.copy()
    factors = adjustment_factors(list(out["date"]), actions).to_numpy()
    for col in ("open", "high", "low", "close"):
        out[col] = out[col].astype("float64") / 100.0 * factors
    out["volume"] = out["volume"].astype("float64") / factors
    return out
