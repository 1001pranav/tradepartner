"""Point-in-time market view: strategies can only ever see bars dated at or before ``as_of``."""

from collections.abc import Callable
from datetime import date

import numpy as np
import pandas as pd


class MarketView:
    """Adjusted daily bars (float rupees; volume adjusted inversely) for a universe.

    Each frame needs columns ``date`` (python date), ``open``, ``high``, ``low``, ``close``,
    ``volume`` and be sorted by date.
    """

    def __init__(
        self,
        bars: dict[str, pd.DataFrame],
        index: pd.DataFrame | None = None,
        sectors: dict[str, str] | None = None,
        members: Callable[[date], list[str]] | None = None,
    ) -> None:
        self._bars = {s: df.reset_index(drop=True) for s, df in bars.items() if len(df)}
        self._dates = {
            s: np.array([d.toordinal() for d in df["date"]]) for s, df in self._bars.items()
        }
        self.index = index.reset_index(drop=True) if index is not None else None
        self._index_dates = (
            np.array([d.toordinal() for d in self.index["date"]])
            if self.index is not None
            else None
        )
        self.sectors = sectors or {}
        self._members = members

    @property
    def symbols(self) -> list[str]:
        return sorted(self._bars)

    def symbols_as_of(self, as_of: date) -> list[str]:
        """Universe members with at least one bar by ``as_of``."""
        names = self._members(as_of) if self._members else self.symbols
        return [
            s for s in sorted(names) if s in self._bars and self._dates[s][0] <= as_of.toordinal()
        ]

    def trading_days(self) -> list[date]:
        days: set[date] = set()
        for df in self._bars.values():
            days.update(df["date"])
        return sorted(days)

    def history(self, symbol: str, as_of: date, n: int | None = None) -> pd.DataFrame:
        """Bars with ``date <= as_of`` (the last ``n`` if given). Never returns the future."""
        df = self._bars[symbol]
        hi = int(np.searchsorted(self._dates[symbol], as_of.toordinal(), side="right"))
        lo = max(0, hi - n) if n else 0
        return df.iloc[lo:hi]

    def bar(self, symbol: str, day: date) -> pd.Series | None:
        if symbol not in self._bars:
            return None
        i = int(np.searchsorted(self._dates[symbol], day.toordinal(), side="left"))
        if i < len(self._dates[symbol]) and self._dates[symbol][i] == day.toordinal():
            return self._bars[symbol].iloc[i]
        return None

    def last_close(self, symbol: str, as_of: date) -> float | None:
        h = self.history(symbol, as_of, 1)
        return None if h.empty else float(h["close"].iloc[0])

    def index_history(self, as_of: date) -> pd.DataFrame:
        if self.index is None or self._index_dates is None:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        hi = int(np.searchsorted(self._index_dates, as_of.toordinal(), side="right"))
        return self.index.iloc[:hi]

    def avg_daily_value(self, symbol: str, as_of: date, n: int = 20) -> float:
        """Mean traded value (rupees) over the last ``n`` bars up to ``as_of``."""
        h = self.history(symbol, as_of, n)
        return 0.0 if h.empty else float((h["close"] * h["volume"]).mean())

    def sector(self, symbol: str) -> str:
        return self.sectors.get(symbol, f"UNMAPPED:{symbol}")

    def close_matrix(self, days: list[date]) -> pd.DataFrame:
        """Close prices (rows = ``days``, columns = symbols); NaN where a symbol has no bar."""
        return pd.DataFrame(
            {s: pd.Series(df.set_index("date")["close"]) for s, df in self._bars.items()}
        ).reindex(days)
