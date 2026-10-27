"""Backtest performance metrics (spec 10.5). Derived statistics use floats; the ledger-style
amounts they are computed from (P&L, charges) stay integer paise upstream."""

import math
from dataclasses import asdict, dataclass
from datetime import date

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class Metrics:
    cagr: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float  # negative number
    calmar: float
    hit_rate: float
    avg_win: float  # paise
    avg_loss: float  # paise (negative)
    turnover: float  # annualised traded notional / average equity
    cost_drag: float  # (charges + slippage) / gross profit; nan if no gross profit
    exposure: float  # average invested fraction of equity
    n_trades: int
    start: date | None
    end: date | None

    def as_dict(self) -> dict[str, float | int | date | None]:
        return asdict(self)


def _nan() -> float:
    return float("nan")


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def compute(
    equity: pd.Series,
    trade_pnl: list[int],
    trade_notional: int = 0,
    cost_paise: int = 0,
    gross_profit: int = 0,
    exposure: pd.Series | None = None,
) -> Metrics:
    """``equity`` is indexed by date (paise). ``trade_pnl`` is net P&L per closed trade."""
    if len(equity) < 2:
        n = _nan()
        return Metrics(n, n, n, n, n, n, n, n, n, n, n, n, len(trade_pnl), None, None)
    eq = equity.astype("float64")
    rets = eq.pct_change().dropna()
    days = (equity.index[-1] - equity.index[0]).days or 1
    years = days / 365.25
    total = float(eq.iloc[-1] / eq.iloc[0])
    cagr = total ** (1 / years) - 1 if total > 0 else -1.0
    vol = float(rets.std(ddof=1) * math.sqrt(TRADING_DAYS)) if len(rets) > 1 else _nan()
    sharpe = (
        float(rets.mean() / rets.std(ddof=1) * math.sqrt(TRADING_DAYS))
        if vol and vol > 0
        else _nan()
    )
    downside = rets[rets < 0]
    dd_dev = float(np.sqrt((downside**2).mean())) if len(downside) else 0.0
    sortino = float(rets.mean() / dd_dev * math.sqrt(TRADING_DAYS)) if dd_dev > 0 else _nan()
    mdd = float(drawdown_series(eq).min())
    calmar = cagr / abs(mdd) if mdd < 0 else _nan()
    wins = [p for p in trade_pnl if p > 0]
    losses = [p for p in trade_pnl if p < 0]
    avg_equity = float(eq.mean())
    return Metrics(
        cagr=cagr,
        ann_vol=vol,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=mdd,
        calmar=calmar,
        hit_rate=len(wins) / len(trade_pnl) if trade_pnl else _nan(),
        avg_win=float(np.mean(wins)) if wins else 0.0,
        avg_loss=float(np.mean(losses)) if losses else 0.0,
        turnover=trade_notional / avg_equity / years if avg_equity > 0 else _nan(),
        cost_drag=cost_paise / gross_profit if gross_profit > 0 else _nan(),
        exposure=float(exposure.mean()) if exposure is not None and len(exposure) else _nan(),
        n_trades=len(trade_pnl),
        start=equity.index[0],
        end=equity.index[-1],
    )


def by_year(equity: pd.Series) -> pd.DataFrame:
    rows = []
    for year, part in equity.groupby([d.year for d in equity.index]):
        m = compute(part, [])
        rows.append({"year": year, "return": float(part.iloc[-1] / part.iloc[0] - 1),
                     "max_drawdown": m.max_drawdown, "sharpe": m.sharpe})  # fmt: skip
    return pd.DataFrame(rows, columns=["year", "return", "max_drawdown", "sharpe"])


def by_regime(equity: pd.Series, index_close: pd.Series, window: int = 200) -> pd.DataFrame:
    """Daily-return stats when Nifty was above/below its 200-day average, using the *previous*
    day's regime (no lookahead)."""
    sma = index_close.rolling(window).mean()
    above = (index_close > sma).shift(1)
    rets = equity.astype("float64").pct_change()
    rows = []
    for label, flag in (("above_200dma", True), ("below_200dma", False)):
        mask = above.reindex(rets.index).fillna(False).astype(bool) == flag
        valid = sma.shift(1).reindex(rets.index).notna()
        r = rets[mask & valid].dropna()
        rows.append(
            {
                "regime": label,
                "days": len(r),
                "mean_daily": float(r.mean()) if len(r) else _nan(),
                "ann_return": float((1 + r.mean()) ** TRADING_DAYS - 1) if len(r) else _nan(),
                "sharpe": float(r.mean() / r.std(ddof=1) * math.sqrt(TRADING_DAYS))
                if len(r) > 1 and r.std(ddof=1) > 0
                else _nan(),
            }
        )
    return pd.DataFrame(rows)
