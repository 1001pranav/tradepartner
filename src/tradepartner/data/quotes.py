"""Live quotes from Kite (REST, polled) into the in-memory ``QuoteBook`` (spec 12 at 09:15)."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from tradepartner.calendar import IST
from tradepartner.data.live import Tick
from tradepartner.risk.types import Quote


@dataclass
class QuoteBook:
    """Latest quotes, filled by the live feed (or by tests)."""

    quotes: dict[str, Quote] = field(default_factory=dict)

    def get(self, symbol: str) -> Quote | None:
        return self.quotes.get(symbol)


BATCH = 500  # Kite allows up to 500 instruments per quote call


class QuoteSource(Protocol):
    def quote(self, instruments: list[str]) -> dict[str, dict[str, Any]]: ...


def _paise(x: float) -> int:
    return round(x * 100)


def to_quote(raw: dict[str, Any], fallback_now: datetime) -> Quote:
    ts = raw.get("timestamp") or raw.get("last_trade_time") or fallback_now
    if isinstance(ts, datetime) and ts.tzinfo is None:
        ts = ts.replace(tzinfo=IST)
    ltp = _paise(float(raw["last_price"]))
    upper = float(raw.get("upper_circuit_limit") or 0)
    lower = float(raw.get("lower_circuit_limit") or 0)
    prev = float(raw.get("ohlc", {}).get("close") or 0)
    return Quote(
        ltp_paise=ltp,
        as_of=ts.astimezone(UTC),
        prev_close_paise=_paise(prev) if prev else None,
        upper_circuit_hit=upper > 0 and ltp >= _paise(upper),
        lower_circuit_hit=lower > 0 and ltp <= _paise(lower),
    )


def poll(
    source: QuoteSource,
    symbols: Sequence[str],
    book: QuoteBook,
    now: datetime,
    acquire: Callable[[], None] = lambda: None,
    on_tick: Callable[[Tick], None] | None = None,
) -> int:
    """Refresh quotes for ``symbols``. Returns how many were updated."""
    updated = 0
    names = sorted(set(symbols))
    for i in range(0, len(names), BATCH):
        chunk = names[i : i + BATCH]
        acquire()
        raw = source.quote([f"NSE:{s}" for s in chunk])
        for key, data in raw.items():
            sym = key.split(":", 1)[1]
            if data.get("last_price"):
                book.quotes[sym] = to_quote(data, now)
                updated += 1
                if on_tick is not None:  # feed the 5-minute bar builder
                    q = book.quotes[sym]
                    vol = int(data.get("volume") or data.get("volume_traded") or 0)
                    on_tick(Tick(sym, q.ltp_paise, vol, q.as_of))
    return updated
