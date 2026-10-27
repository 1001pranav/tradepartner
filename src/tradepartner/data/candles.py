"""Historical candles: Kite download, Parquet store, gap detection (spec 4, 5, 16-M1).

Layout: ``<root>/<interval>/<symbol>.parquet`` (interval/symbol partitioning), queried with DuckDB.
Prices are stored as integer paise. For daily bars ``ts`` is the session close (15:30 IST, in UTC):
the moment the bar's information becomes available, which makes as-of filtering leak-proof.
"""

import time as _time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

import duckdb
import pandas as pd

from tradepartner.calendar import IST, TradingCalendar, session_close_utc
from tradepartner.paths import home_dir

COLUMNS = ["ts", "date", "open", "high", "low", "close", "volume"]
CHUNK_DAYS = {"day": 2000, "60minute": 400, "15minute": 200, "5minute": 100, "minute": 60}


def parquet_root() -> Path:
    return home_dir() / "parquet"


class HistoricalSource(Protocol):
    def historical_data(
        self, instrument_token: int, from_date: Any, to_date: Any, interval: str
    ) -> list[dict[str, Any]]: ...


class RateLimiter:
    """Token bucket: at most ``rate`` calls per second, with injectable clock and sleep."""

    def __init__(
        self,
        rate: float,
        clock: Callable[[], float] = _time.monotonic,
        sleep: Callable[[float], None] = _time.sleep,
    ) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self._interval = 1.0 / rate
        self._clock = clock
        self._sleep = sleep
        self._next = clock()

    def acquire(self) -> None:
        now = self._clock()
        if now < self._next:
            self._sleep(self._next - now)
            now = self._next
        self._next = now + self._interval


def _paise(value: object) -> int:
    return int((Decimal(str(value)) * 100).to_integral_value())


def normalize_rows(rows: list[dict[str, Any]], interval: str) -> pd.DataFrame:
    """Kite rows (rupee floats) -> store schema (paise ints)."""
    recs: list[dict[str, Any]] = []
    for r in rows:
        d = r["date"]
        if interval == "day":
            day = d.date() if isinstance(d, datetime) else d
            ts = session_close_utc(day)
        else:
            if d.tzinfo is None:
                d = d.replace(tzinfo=IST)
            ts = d.astimezone(IST).astimezone(session_close_utc(d.date()).tzinfo)
            day = d.astimezone(IST).date()
        recs.append(
            {
                "ts": pd.Timestamp(ts),
                "date": day,
                "open": _paise(r["open"]),
                "high": _paise(r["high"]),
                "low": _paise(r["low"]),
                "close": _paise(r["close"]),
                "volume": int(r["volume"]),
            }
        )
    if not recs:
        return empty_frame()
    return _typed(pd.DataFrame(recs))


def empty_frame() -> pd.DataFrame:
    return _typed(pd.DataFrame({c: [] for c in COLUMNS}))


def _typed(df: pd.DataFrame) -> pd.DataFrame:
    df = df[COLUMNS].copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype("int64")
    return df


class CandleStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or parquet_root()

    def path(self, interval: str, symbol: str) -> Path:
        safe = symbol.replace("&", "_and_").replace("/", "_")
        return self.root / interval / f"{safe}.parquet"

    def read(
        self,
        interval: str,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
        as_of: datetime | None = None,
    ) -> pd.DataFrame:
        p = self.path(interval, symbol)
        if not p.exists():
            return empty_frame()
        df = _typed(pd.read_parquet(p))
        if start is not None:
            df = df[df["date"] >= start]
        if end is not None:
            df = df[df["date"] <= end]
        if as_of is not None:  # point-in-time: only bars whose info was available by as_of
            df = df[df["ts"] <= pd.Timestamp(as_of)]
        return df.sort_values("ts").reset_index(drop=True)

    def write(self, interval: str, symbol: str, new: pd.DataFrame) -> int:
        """Merge, dedupe on ts (new rows win) and persist. Returns the stored row count."""
        existing = self.read(interval, symbol)
        merged = pd.concat([existing, new], ignore_index=True) if len(existing) else new
        merged = merged.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)
        p = self.path(interval, symbol)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        merged.to_parquet(tmp, index=False)
        tmp.replace(p)  # atomic: an interrupted backfill never leaves a torn file
        return len(merged)

    def last_date(self, interval: str, symbol: str) -> date | None:
        df = self.read(interval, symbol)
        return None if df.empty else df["date"].max()

    def symbols(self, interval: str) -> list[str]:
        d = self.root / interval
        return sorted(p.stem for p in d.glob("*.parquet")) if d.exists() else []

    def query(self, sql: str) -> pd.DataFrame:
        """Run DuckDB SQL; use ``{root}`` in the text for the store location."""
        con = duckdb.connect()
        try:
            return con.execute(sql.replace("{root}", str(self.root))).df()
        finally:
            con.close()


@dataclass
class BackfillResult:
    done: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    rows: int = 0


def last_completed_session(now: datetime, calendar: TradingCalendar) -> date:
    local = now.astimezone(IST)
    today = local.date()
    if calendar.is_trading_day(today) and now >= session_close_utc(today):
        return today
    return calendar.prev_trading_day(today)


def backfill(
    source: HistoricalSource,
    store: CandleStore,
    targets: dict[str, int],
    *,
    years: int,
    now: datetime,
    calendar: TradingCalendar,
    limiter: RateLimiter,
    interval: str = "day",
    retries: int = 3,
    sleep: Callable[[float], None] = _time.sleep,
) -> BackfillResult:
    """Download history for ``targets`` (symbol -> token). Resumable: restarts from the last
    stored date, so a rerun after an interruption only fetches what is missing."""
    result = BackfillResult()
    end = last_completed_session(now, calendar)
    default_start = end - timedelta(days=365 * years)
    chunk = CHUNK_DAYS[interval]
    for symbol, token in sorted(targets.items()):
        last = store.last_date(interval, symbol)
        if last is not None and last >= end:
            result.skipped.append(symbol)
            continue
        start = last + timedelta(days=1) if last else default_start
        try:
            cursor = start
            while cursor <= end:
                stop = min(cursor + timedelta(days=chunk - 1), end)
                rows = _fetch(source, limiter, token, cursor, stop, interval, retries, sleep)
                frame = normalize_rows(rows, interval)
                if not frame.empty:
                    store.write(interval, symbol, frame)  # persisted per chunk -> resumable
                    result.rows += len(frame)
                cursor = stop + timedelta(days=1)
            result.done.append(symbol)
        except Exception as exc:  # one bad symbol must not abort the whole run
            result.failed[symbol] = f"{type(exc).__name__}: {exc}"
    return result


def real_gaps(df: pd.DataFrame, calendar: TradingCalendar) -> list[date]:
    """Missing trading days counted from the first real (positive-price) bar: junk bars before a
    stock listed are not history."""
    px = df[["open", "high", "low", "close"]] > 0
    return detect_gaps(df[px.all(axis=1).cummax()], calendar)


def refill_gaps(
    source: HistoricalSource,
    store: CandleStore,
    targets: dict[str, int],
    *,
    calendar: TradingCalendar,
    limiter: RateLimiter,
    interval: str = "day",
    only: dict[str, tuple[date, date]] | None = None,
    retries: int = 3,
    sleep: Callable[[float], None] = _time.sleep,
) -> tuple[dict[str, list[date]], dict[str, list[date]], dict[str, str]]:
    """Re-download the missing days INSIDE stored history (``backfill`` only extends the end).
    Days missing for most symbols are market holidays, not holes, and are left alone.
    Returns (filled, still_missing, failed) per symbol. Without ``only`` it finds gaps itself."""
    gaps = {sym: g for sym in targets if (g := real_gaps(store.read(interval, sym), calendar))}
    if only:
        gaps = {s: [d for d in gaps.get(s, []) if lo <= d <= hi] for s, (lo, hi) in only.items()}
    holes = {d for g in gaps.values() for d in g}
    common = {
        d for d in holes if sum(d in g for g in gaps.values()) > len(gaps) / 2 and len(gaps) > 3
    }
    filled: dict[str, list[date]] = {}
    still: dict[str, list[date]] = {}
    failed: dict[str, str] = {}
    for sym, days in gaps.items():
        days = [d for d in days if d not in common]
        if not days:
            continue
        try:
            rows = _fetch(
                source, limiter, targets[sym], min(days), max(days), interval, retries, sleep
            )
            frame = normalize_rows(rows, interval)
            if not frame.empty:
                store.write(interval, sym, frame)
            got = set(frame["date"]) if not frame.empty else set()
        except Exception as exc:
            failed[sym] = f"{type(exc).__name__}: {exc}"
            continue
        filled[sym] = [d for d in days if d in got]
        still[sym] = [d for d in days if d not in got]
    return filled, still, failed


def _fetch(
    source: HistoricalSource,
    limiter: RateLimiter,
    token: int,
    start: date,
    stop: date,
    interval: str,
    retries: int,
    sleep: Callable[[float], None],
) -> list[dict[str, Any]]:
    for attempt in range(retries):
        limiter.acquire()
        try:
            return source.historical_data(token, start, stop, interval)
        except Exception:
            if attempt == retries - 1:
                raise
            sleep(2.0**attempt)
    raise AssertionError("unreachable")


def detect_gaps(df: pd.DataFrame, calendar: TradingCalendar) -> list[date]:
    """Trading days missing between the first and last stored daily bar."""
    if df.empty:
        return []
    have = set(df["date"])
    expected = calendar.trading_days(min(have), max(have))
    return [d for d in expected if d not in have]
