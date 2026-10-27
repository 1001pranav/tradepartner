"""Equal-weight portfolio simulation with REAL costs and tax (simulation only: no broker).

Rebalances at the open of the first trading day of each period, in whole shares, paying the
Zerodha delivery charges from ``config/costs.yaml`` plus slippage. Sales are matched FIFO and each
financial year's capital-gains tax (rates from ``config/tax.yaml``) is deducted at the year end:
STCG at 20%, LTCG at 12.5% above the yearly exemption, short-term losses set off against
long-term gains. Loss carry-forward and dividends are ignored (conservative / simplified).
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from tradepartner.backtest.metrics import Metrics, compute
from tradepartner.backtest.view import MarketView
from tradepartner.calendar import indian_fy, months_between_exceeds
from tradepartner.costs import zerodha
from tradepartner.settings import Settings

Frequency = str  # monthly | quarterly | annual | never


@dataclass
class EwResult:
    frequency: Frequency
    equity: pd.Series  # date -> paise, AFTER costs and tax
    metrics: Metrics
    charges_paise: int
    slippage_paise: int
    tax_paise: int
    turnover_paise: int
    trades: int
    holdings: int
    notes: list[str] = field(default_factory=list)


@dataclass
class _Lot:
    qty: int
    price: int
    day: date


def _is_rebalance_day(freq: Frequency, d: date, prev: date | None) -> bool:
    if prev is None:
        return True  # the initial purchase
    if freq == "never":
        return False
    if freq == "monthly":
        return d.month != prev.month
    if freq == "quarterly":
        return d.month != prev.month and d.month in (1, 4, 7, 10)
    if freq == "annual":
        return d.month != prev.month and d.month == 4  # start of the financial year
    raise ValueError(f"unknown frequency {freq!r}")


def _year_tax(stcg: int, ltcg: int, settings: Settings) -> int:
    t = settings.tax
    if stcg < 0 < ltcg:  # short-term loss may be set off against long-term gains
        used = min(-stcg, ltcg)
        stcg, ltcg = stcg + used, ltcg - used
    taxable_lt = max(0, ltcg - t.ltcg_exemption_inr * 100)
    return round(max(0, stcg) * t.stcg_pct / 100 + taxable_lt * t.ltcg_pct / 100)


def run_equal_weight(
    view: MarketView,
    settings: Settings,
    start: date,
    end: date,
    capital_paise: int,
    frequency: Frequency,
    slippage_bps: float | None = None,
) -> EwResult:
    bps = settings.costs.slippage.default_bps if slippage_bps is None else slippage_bps
    days = [d for d in view.trading_days() if start <= d <= end]
    px = {
        s: {
            d: (round(float(r["open"]) * 100), round(float(r["close"]) * 100))
            for d, r in zip(df["date"], df.to_dict("records"), strict=True)
        }
        for s in view.symbols
        for df in [view.history(s, days[-1])]
    }
    cash = capital_paise
    lots: dict[str, deque[_Lot]] = {s: deque() for s in view.symbols}
    equity: dict[date, int] = {}
    charges = slip = tax_total = turnover = trades = 0
    fy_stcg = fy_ltcg = 0
    counter = 0
    prev: date | None = None
    last_close: dict[str, int] = {}
    notes: list[str] = []

    def shares(sym: str) -> int:
        return sum(lot.qty for lot in lots[sym])

    def fill(sym: str, side: str, qty: int, price: int, d: date) -> int:
        nonlocal counter, charges, slip, turnover, trades
        counter += 1
        c = zerodha.total_paise(
            zerodha.compute(
                [zerodha.FillIn(counter, f"ew{counter}", sym, side, "CNC", qty, price, d)],
                settings.costs,
            )
        )
        charges += c
        slip += qty * price * round(bps * 100) // 1_000_000
        turnover += qty * price
        trades += 1
        return c

    for d in days:
        if _is_rebalance_day(frequency, d, prev):
            avail = [s for s in view.symbols_as_of(d) if d in px[s]]
            if avail:
                value = cash + sum(
                    shares(s) * px[s][d][0] for s in view.symbols if shares(s) and d in px[s]
                )
                target_each = value // len(avail)
                want = {s: target_each // px[s][d][0] for s in avail}
                # sell everything above target (or no longer a member), then buy up to target
                for s in view.symbols:
                    if d not in px[s]:
                        continue
                    excess = shares(s) - want.get(s, 0)
                    if excess <= 0:
                        continue
                    price = px[s][d][0] - px[s][d][0] * round(bps * 100) // 1_000_000
                    left = excess
                    while left > 0:
                        lot = lots[s][0]
                        take = min(left, lot.qty)
                        gain = take * (price - lot.price)
                        long_term = months_between_exceeds(
                            lot.day, d, settings.tax.long_term_months
                        )
                        if long_term:
                            fy_ltcg += gain
                        else:
                            fy_stcg += gain
                        lot.qty -= take
                        if lot.qty == 0:
                            lots[s].popleft()
                        left -= take
                    cash += excess * price - fill(s, "SELL", excess, price, d)
                for s in avail:
                    add = want[s] - shares(s)
                    if add <= 0:
                        continue
                    price = px[s][d][0] + px[s][d][0] * round(bps * 100) // 1_000_000
                    add = min(add, max(0, cash // price))
                    if add <= 0:
                        continue
                    cash -= add * price + fill(s, "BUY", add, price, d)
                    lots[s].append(_Lot(add, price, d))
        for s in view.symbols:
            if d in px[s]:
                last_close[s] = px[s][d][1]
        nxt = days[days.index(d) + 1] if d != days[-1] else None
        if nxt is None or indian_fy(nxt) != indian_fy(d):  # financial year end: pay the tax
            t = _year_tax(fy_stcg, fy_ltcg, settings)
            cash -= t
            tax_total += t
            fy_stcg = fy_ltcg = 0
        equity[d] = cash + sum(shares(s) * last_close.get(s, 0) for s in view.symbols if shares(s))
        prev = d
    series = pd.Series(equity, dtype="int64")
    held = sum(1 for s in view.symbols if shares(s))
    if min(capital_paise // max(held, 1), 10**12) < 0:
        notes.append("negative cash")
    return EwResult(
        frequency,
        series,
        compute(series, []),
        charges,
        slip,
        tax_total,
        turnover,
        trades,
        held,
        notes,
    )
