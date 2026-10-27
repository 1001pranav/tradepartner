"""Live ticks -> 5-minute bars (spec 5, 16-M9). No LLM and no network in this hot path.

``BarBuilder`` is pure and deterministic: feed it ticks (from Kite's WebSocket, or from polled
quotes) and it emits completed bars. A bar's ``ts`` is its END time in UTC, which is the moment
its information becomes available (same convention as daily bars, so as-of filtering is
leak-proof).
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd

from tradepartner.calendar import IST, SESSION_CLOSE, SESSION_OPEN

INTERVAL_MIN = 5
COLUMNS = ["ts", "date", "open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class Tick:
    symbol: str
    ltp_paise: int
    day_volume: int  # cumulative traded volume for the day (as Kite reports it)
    ts: datetime  # timezone-aware


@dataclass(frozen=True)
class Bar:
    symbol: str
    end: datetime  # UTC
    day: date
    open: int
    high: int
    low: int
    close: int
    volume: int


def bucket_end(ts: datetime, interval_min: int = INTERVAL_MIN) -> datetime | None:
    """End of the bar a tick belongs to (IST session 09:15-15:30), or None outside the session."""
    local = ts.astimezone(IST)
    open_dt = datetime.combine(local.date(), SESSION_OPEN, tzinfo=IST)
    close_dt = datetime.combine(local.date(), SESSION_CLOSE, tzinfo=IST)
    if not (open_dt <= local < close_dt):
        return None
    n = int((local - open_dt).total_seconds() // (interval_min * 60)) + 1
    return (open_dt + timedelta(minutes=n * interval_min)).astimezone(UTC)


class BarBuilder:
    def __init__(self, interval_min: int = INTERVAL_MIN) -> None:
        self.interval_min = interval_min
        self._open: dict[str, dict[str, Any]] = {}
        self._last_cum: dict[str, int] = {}

    def on_tick(self, t: Tick) -> list[Bar]:
        end = bucket_end(t.ts, self.interval_min)
        if end is None:
            return []
        done: list[Bar] = []
        cur = self._open.get(t.symbol)
        if cur is not None and cur["end"] != end:
            done.append(self._close(t.symbol))
            cur = None
        if cur is None:
            base = self._last_cum.get(t.symbol, t.day_volume)
            self._open[t.symbol] = cur = {
                "end": end,
                "open": t.ltp_paise,
                "high": t.ltp_paise,
                "low": t.ltp_paise,
                "close": t.ltp_paise,
                "cum_start": base,
                "cum": t.day_volume,
            }
        else:
            cur["high"] = max(cur["high"], t.ltp_paise)
            cur["low"] = min(cur["low"], t.ltp_paise)
            cur["close"] = t.ltp_paise
            cur["cum"] = max(cur["cum"], t.day_volume)
        return done

    def _close(self, symbol: str) -> Bar:
        c = self._open.pop(symbol)
        self._last_cum[symbol] = c["cum"]
        end: datetime = c["end"]
        return Bar(
            symbol,
            end,
            end.astimezone(IST).date(),
            c["open"],
            c["high"],
            c["low"],
            c["close"],
            max(0, c["cum"] - c["cum_start"]),
        )

    def flush(self, now: datetime) -> list[Bar]:
        """Close every bar whose period has ended by ``now`` (call every few seconds)."""
        return [self._close(s) for s, c in sorted(self._open.items()) if c["end"] <= now]


def bars_to_frame(bars: list[Bar]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts": pd.Timestamp(b.end),
                "date": b.day,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
            }
            for b in bars
        ],
        columns=COLUMNS,
    )


class TickerService:  # pragma: no cover - needs the live WebSocket
    """Thin wrapper over ``kiteconnect.KiteTicker`` (injected, so nothing here needs the network
    in tests). Reconnects are handled by KiteTicker itself."""

    def __init__(
        self,
        ticker_factory: Callable[[], Any],
        tokens: dict[int, str],
        on_bars: Callable[[list[Bar]], None],
        builder: BarBuilder | None = None,
    ) -> None:
        self.ticker = ticker_factory()
        self.tokens = tokens
        self.builder = builder or BarBuilder()
        self.on_bars = on_bars
        self.ticker.on_ticks = self._on_ticks
        self.ticker.on_connect = lambda ws, resp: (
            ws.subscribe(list(tokens)) or ws.set_mode("full", list(tokens))
        )

    def _on_ticks(self, ws: Any, ticks: list[dict[str, Any]]) -> None:
        for raw in ticks:
            sym = self.tokens.get(raw["instrument_token"])
            if sym is None or "last_price" not in raw:
                continue
            ts = raw.get("exchange_timestamp") or datetime.now(UTC)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=IST)
            done = self.builder.on_tick(
                Tick(sym, round(raw["last_price"] * 100), int(raw.get("volume_traded", 0)), ts)
            )
            if done:
                self.on_bars(done)

    def start(self) -> None:
        self.ticker.connect(threaded=True)

    def stop(self) -> None:
        self.ticker.close()
