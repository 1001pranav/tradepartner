"""FIFO lot matching (spec 9.4). Pure functions over fills: no database, fully deterministic.

Rules:
* Per scrip per trade date, buys and sells net against each other as **intraday** first
  (the first ``min(buy_qty, sell_qty)`` units on each side, in time order).
* Remaining buys open delivery lots; remaining sells consume the oldest lots (FIFO).
* A sell with no lot to match is an ``unmatched`` row. A cost basis is never invented.
* Splits and bonuses adjust open lots on the ex-date (applied before that day's fills).
"""

from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from fractions import Fraction
from typing import Literal

from tradepartner.data.corporate_actions import Action, share_multiplier


@dataclass(frozen=True)
class FillRec:
    id: int
    symbol: str
    isin: str | None
    side: str  # BUY | SELL
    qty: int
    price_paise: int
    filled_at: datetime
    trade_date: date


@dataclass
class LotState:
    open_fill_id: int
    symbol: str
    isin: str | None
    open_date: date
    qty_open: int
    qty_rem: int
    cost_rem: Fraction  # total cost of the remaining quantity, in paise
    adjusted: bool = False

    @property
    def cost_per_share_paise(self) -> int:
        return round_half_up(self.cost_rem / self.qty_rem) if self.qty_rem else 0


Kind = Literal["intraday", "delivery", "unmatched"]


@dataclass(frozen=True)
class RealizedRow:
    symbol: str
    kind: Kind
    open_fill_id: int | None
    close_fill_id: int
    buy_fill_id: int | None
    sell_fill_id: int
    qty: int
    buy_value_paise: int
    sell_value_paise: int
    holding_days: int
    open_date: date | None
    close_date: date


@dataclass
class MatchResult:
    lots: list[LotState] = field(default_factory=list)
    realized: list[RealizedRow] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)


def round_half_up(x: Fraction) -> int:
    return int((x * 2 + 1) // 2) if x >= 0 else -round_half_up(-x)


@dataclass
class _Chunk:
    fill: FillRec
    rem: int


def match(fills: Sequence[FillRec], actions: Sequence[Action] = ()) -> MatchResult:
    result = MatchResult()
    by_symbol: dict[str, list[FillRec]] = defaultdict(list)
    for f in fills:
        by_symbol[f.symbol].append(f)
    acts_by_symbol: dict[str, list[Action]] = defaultdict(list)
    for a in actions:
        if a.action_type in ("split", "bonus"):
            acts_by_symbol[a.symbol].append(a)
    for symbol in sorted(by_symbol):
        _match_symbol(symbol, by_symbol[symbol], acts_by_symbol.get(symbol, []), result)
    return result


def _match_symbol(
    symbol: str, fills: list[FillRec], actions: list[Action], out: MatchResult
) -> None:
    lots: deque[LotState] = deque()
    days = sorted({f.trade_date for f in fills} | {a.ex_date for a in actions})
    for day in days:
        for act in sorted((a for a in actions if a.ex_date == day), key=lambda a: a.action_type):
            lots = _apply_action(lots, act, day, out)
        day_fills = sorted(
            (f for f in fills if f.trade_date == day), key=lambda f: (f.filled_at, f.id)
        )
        buys = [_Chunk(f, f.qty) for f in day_fills if f.side == "BUY"]
        sells = [_Chunk(f, f.qty) for f in day_fills if f.side == "SELL"]
        _net_intraday(symbol, day, buys, sells, out)
        for b in buys:
            if b.rem:
                lots.append(
                    LotState(
                        b.fill.id,
                        symbol,
                        b.fill.isin,
                        day,
                        b.rem,
                        b.rem,
                        Fraction(b.rem * b.fill.price_paise),
                    )
                )
        for s in sells:
            if s.rem:
                _consume(symbol, day, s, lots, out)
    out.lots.extend(lot for lot in lots if lot.qty_rem > 0)


def _net_intraday(
    symbol: str, day: date, buys: list[_Chunk], sells: list[_Chunk], out: MatchResult
) -> None:
    to_pair = min(sum(b.rem for b in buys), sum(s.rem for s in sells))
    i = j = 0
    while to_pair > 0:
        while buys[i].rem == 0:
            i += 1
        while sells[j].rem == 0:
            j += 1
        b, s = buys[i], sells[j]
        take = min(b.rem, s.rem, to_pair)
        first, second = sorted((b.fill, s.fill), key=lambda f: (f.filled_at, f.id))
        out.realized.append(
            RealizedRow(
                symbol,
                "intraday",
                first.id,
                second.id,
                b.fill.id,
                s.fill.id,
                take,
                take * b.fill.price_paise,
                take * s.fill.price_paise,
                0,
                day,
                day,
            )
        )
        b.rem -= take
        s.rem -= take
        to_pair -= take


def _consume(symbol: str, day: date, sell: _Chunk, lots: deque[LotState], out: MatchResult) -> None:
    while sell.rem > 0:
        if not lots:
            out.realized.append(
                RealizedRow(
                    symbol,
                    "unmatched",
                    None,
                    sell.fill.id,
                    None,
                    sell.fill.id,
                    sell.rem,
                    0,
                    sell.rem * sell.fill.price_paise,
                    0,
                    None,
                    day,
                )
            )
            out.alerts.append(
                f"{symbol}: sell of {sell.rem} on {day} has no matching buy (missing history?)"
            )
            sell.rem = 0
            return
        lot = lots[0]
        take = min(sell.rem, lot.qty_rem)
        cost = lot.cost_rem * take / lot.qty_rem
        lot.cost_rem -= cost
        lot.qty_rem -= take
        out.realized.append(
            RealizedRow(
                symbol,
                "delivery",
                lot.open_fill_id,
                sell.fill.id,
                lot.open_fill_id,
                sell.fill.id,
                take,
                round_half_up(cost),
                take * sell.fill.price_paise,
                (day - lot.open_date).days,
                lot.open_date,
                day,
            )
        )
        sell.rem -= take
        if lot.qty_rem == 0:
            lots.popleft()


def _apply_action(
    lots: deque[LotState], act: Action, day: date, out: MatchResult
) -> deque[LotState]:
    num, den = share_multiplier(act)
    kept: deque[LotState] = deque()
    bonus: list[LotState] = []
    for lot in lots:
        if act.action_type == "split":
            new_rem, frac = divmod(lot.qty_rem * num, den)
            if frac:
                out.alerts.append(
                    f"{lot.symbol}: fractional shares dropped on {day} split (cash in lieu)"
                )
            lot.qty_open = lot.qty_open * num // den
            lot.qty_rem = new_rem
            lot.adjusted = True
            if new_rem > 0:
                kept.append(lot)
        else:  # bonus: existing lots untouched, a zero-cost lot is added at allotment
            kept.append(lot)
            extra = lot.qty_rem * (num - den) // den
            if extra > 0:
                bonus.append(
                    LotState(
                        lot.open_fill_id, lot.symbol, lot.isin, day, extra, extra, Fraction(0), True
                    )
                )
    kept.extend(bonus)
    return kept
