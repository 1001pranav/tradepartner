"""Universe construction from ``universe_members`` (point-in-time, spec 6.1, 10.3.4)."""

import csv
from collections.abc import Callable
from datetime import date
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from tradepartner.db.models import Instrument, UniverseMember
from tradepartner.paths import config_dir

# Rows seeded from a static list carry this valid_from, so we can tell membership is not historical.
STATIC_SEED_DATE = date(2000, 1, 1)

SURVIVORSHIP_WARNING = (
    "WARNING: universe membership is the CURRENT list applied to all history, so results are "
    "subject to survivorship bias. Treat backtest numbers as optimistic."
)


def read_seed(name: str, directory: Path | None = None) -> list[str]:
    path = (directory or config_dir() / "universe") / f"{name}.csv"
    lines = [ln for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    reader = csv.DictReader(lines)
    return [row["symbol"].strip() for row in reader if row["symbol"].strip()]


def seed_universe(
    session: Session, name: str, directory: Path | None = None, replace: bool = False
) -> int:
    """Idempotent: adds only symbols not already members under this universe. With ``replace``,
    static-seed rows whose symbol is no longer in the seed file are removed (a refreshed list)."""
    if replace:
        keep = set(read_seed(name, directory))
        for m in session.scalars(
            select(UniverseMember).where(UniverseMember.universe_name == name)
        ):
            if m.valid_from == STATIC_SEED_DATE and m.valid_to is None and m.symbol not in keep:
                session.delete(m)
        session.flush()
    existing = set(
        session.scalars(select(UniverseMember.symbol).where(UniverseMember.universe_name == name))
    )
    added = 0
    for symbol in read_seed(name, directory):
        if symbol not in existing:
            session.add(
                UniverseMember(
                    universe_name=name, symbol=symbol, valid_from=STATIC_SEED_DATE, valid_to=None
                )
            )
            added += 1
    session.commit()
    return added


def members_as_of(session: Session, name: str, as_of: date) -> list[str]:
    rows = session.scalars(
        select(UniverseMember).where(
            UniverseMember.universe_name == name,
            UniverseMember.valid_from <= as_of,
        )
    )
    return sorted(m.symbol for m in rows if m.valid_to is None or m.valid_to > as_of)


def has_survivorship_bias(session: Session, name: str) -> bool:
    rows = list(session.scalars(select(UniverseMember).where(UniverseMember.universe_name == name)))
    return bool(rows) and all(m.valid_from == STATIC_SEED_DATE and m.valid_to is None for m in rows)


def tradable_members(session: Session, name: str, as_of: date) -> dict[str, int]:
    """Members that map to a known NSE equity instrument -> instrument token."""
    symbols = members_as_of(session, name, as_of)
    rows = session.execute(
        select(Instrument.tradingsymbol, Instrument.instrument_token).where(
            Instrument.tradingsymbol.in_(symbols), Instrument.segment == "NSE"
        )
    )
    return {sym: tok for sym, tok in rows}


def read_sectors(directory: Path | None = None) -> dict[str, str]:
    """symbol -> sector. Unknown symbols get their own sector (never one shared 'unknown' bucket
    that would wrongly aggregate unrelated names under the 30% sector limit)."""
    path = (directory or config_dir() / "universe") / "sectors.csv"
    if not path.exists():
        return {}
    lines = [ln for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    return {r["symbol"].strip(): r["sector"].strip() for r in csv.DictReader(lines)}


def sector_of(symbol: str, sectors: dict[str, str]) -> str:
    return sectors.get(symbol, f"UNMAPPED:{symbol}")


MEMBERSHIP_COLUMNS = {"symbol", "valid_from", "valid_to"}


def import_membership_csv(session: Session, name: str, path: Path) -> tuple[int, int]:
    """Real historical membership: ``symbol,valid_from,valid_to`` (blank valid_to = still a member).

    Replaces the static seed rows for this universe (they say "member since 2000, forever", which
    is exactly the survivorship bias). Idempotent. Returns (rows added, static rows removed)."""
    with path.open(newline="") as fh:
        reader = csv.DictReader(ln for ln in fh if ln.strip() and not ln.startswith("#"))
        cols = set(reader.fieldnames or [])
        if cols != MEMBERSHIP_COLUMNS:
            raise ValueError(f"membership CSV needs exactly columns {sorted(MEMBERSHIP_COLUMNS)}")
        rows = list(reader)
    existing = {
        (m.symbol, m.valid_from)
        for m in session.scalars(select(UniverseMember).where(UniverseMember.universe_name == name))
    }
    added = 0
    for r in rows:
        start = date.fromisoformat(r["valid_from"].strip())
        end = date.fromisoformat(r["valid_to"].strip()) if r["valid_to"].strip() else None
        if end is not None and end <= start:
            raise ValueError(f"{r['symbol']}: valid_to must be after valid_from")
        if (r["symbol"].strip(), start) in existing:
            continue
        session.add(
            UniverseMember(
                universe_name=name, symbol=r["symbol"].strip(), valid_from=start, valid_to=end
            )
        )
        existing.add((r["symbol"].strip(), start))
        added += 1
    removed = 0
    if added:
        for m in session.scalars(
            select(UniverseMember).where(UniverseMember.universe_name == name)
        ):
            if m.valid_from == STATIC_SEED_DATE and m.valid_to is None:
                session.delete(m)
                removed += 1
    session.commit()
    return added, removed


def read_excluded(directory: Path | None = None) -> dict[str, str]:
    """symbol -> reason, for symbols excluded for data-quality reasons (see excluded.csv)."""
    path = (directory or config_dir() / "universe") / "excluded.csv"
    if not path.exists():
        return {}
    lines = [ln for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    return {r["symbol"].strip(): r["reason"].strip() for r in csv.DictReader(lines)}


def all_symbols(session: Session, name: str) -> list[str]:
    """Every symbol that was ever a member (needed to load history for past members), minus the
    data-quality exclusions."""
    rows = session.scalars(
        select(UniverseMember.symbol).where(UniverseMember.universe_name == name)
    )
    return sorted(set(rows) - set(read_excluded()))


def membership_fn(session: Session, name: str) -> "Callable[[date], list[str]]":
    """Point-in-time membership from preloaded intervals (no DB query per call)."""
    spans = [
        (m.symbol, m.valid_from, m.valid_to)
        for m in session.scalars(select(UniverseMember).where(UniverseMember.universe_name == name))
    ]

    excluded = set(read_excluded())

    def members(as_of: date) -> list[str]:
        return sorted(
            {s for s, a, b in spans if a <= as_of and (b is None or b > as_of)} - excluded
        )

    return members
