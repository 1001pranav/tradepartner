"""Zerodha equity charge computation (spec section 8). All rates come from ``config/costs.yaml``.

Charges are computed per *order* (the broker's unit for the brokerage cap and the contract note),
then allocated back to the individual fills so every fill carries its share. Component amounts are
integer paise; the allocation preserves the order total exactly (largest remainder).
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

from tradepartner.settings import CostsConfig

_PAISA = Decimal(1)
_RUPEE_PAISE = Decimal(100)
_CRORE_PAISE = Decimal(10**9)  # 1 crore rupees expressed in paise


@dataclass(frozen=True)
class FillIn:
    id: int
    order_ref: str  # broker order id (fills sharing it belong to one order)
    symbol: str
    side: str  # BUY | SELL
    product: str  # CNC | MIS
    qty: int
    price_paise: int
    trade_date: date

    @property
    def turnover_paise(self) -> int:
        return self.qty * self.price_paise


@dataclass(frozen=True)
class ChargeLine:
    fill_id: int
    component: str  # brokerage|stt|exchange_txn|sebi|stamp|gst|dp
    amount_paise: int


def _round(x: Decimal, unit_paise: Decimal = _PAISA) -> int:
    """Round half up to a multiple of ``unit_paise`` and return paise."""
    return int((x / unit_paise).to_integral_value(ROUND_HALF_UP) * unit_paise)


def _pct(amount: Decimal, pct: float) -> Decimal:
    return amount * Decimal(str(pct)) / Decimal(100)


def compute_order(fills: Sequence[FillIn], cfg: CostsConfig) -> dict[str, int]:
    """Total charge components (paise) for the fills of one order, excluding the DP charge."""
    side = {f.side for f in fills}
    product = {f.product for f in fills}
    if len(side) != 1 or len(product) != 1:
        raise ValueError("fills of one order must share side and product")
    is_buy = side == {"BUY"}
    turnover = Decimal(sum(f.turnover_paise for f in fills))
    stt_unit = _RUPEE_PAISE if cfg.rounding.stt_round_to_rupee else _PAISA

    if product == {"CNC"}:
        d = cfg.equity_delivery
        brokerage = Decimal(str(d.brokerage)) * _RUPEE_PAISE
        stt = _pct(turnover, d.stt_buy_pct if is_buy else d.stt_sell_pct)
        exch = _pct(turnover, d.exchange_txn_pct_nse)
        sebi = turnover * Decimal(str(d.sebi_per_crore_inr)) * _RUPEE_PAISE / _CRORE_PAISE
        stamp = _pct(turnover, d.stamp_buy_pct) if is_buy else Decimal(0)
        gst_pct = d.gst_pct
    elif product == {"MIS"}:
        i = cfg.equity_intraday
        brokerage = min(_pct(turnover, i.brokerage_pct), Decimal(str(i.brokerage_cap_inr)) * 100)
        stt = Decimal(0) if is_buy else _pct(turnover, i.stt_sell_pct)
        exch = _pct(turnover, i.exchange_txn_pct_nse)
        sebi = turnover * Decimal(str(i.sebi_per_crore_inr)) * _RUPEE_PAISE / _CRORE_PAISE
        stamp = _pct(turnover, i.stamp_buy_pct) if is_buy else Decimal(0)
        gst_pct = i.gst_pct
    else:
        raise ValueError(f"unsupported product {product}")

    brokerage_r, exch_r, sebi_r = _round(brokerage), _round(exch), _round(sebi)
    gst = _round(_pct(Decimal(brokerage_r + exch_r + sebi_r), gst_pct))
    return {
        "brokerage": brokerage_r,
        "stt": _round(stt, stt_unit),
        "exchange_txn": exch_r,
        "sebi": sebi_r,
        "stamp": _round(stamp),
        "gst": gst,
    }


def allocate(total: int, weights: Sequence[int]) -> list[int]:
    """Split ``total`` paise across ``weights`` so the parts sum exactly (largest remainder)."""
    wsum = sum(weights)
    if wsum <= 0:
        raise ValueError("weights must sum to a positive number")
    raw = [Decimal(total) * w / wsum for w in weights]
    parts = [int(r.to_integral_value(rounding="ROUND_FLOOR")) for r in raw]
    leftover = total - sum(parts)
    order = sorted(range(len(weights)), key=lambda k: (parts[k] - raw[k], k))
    for k in order[:leftover]:
        parts[k] += 1
    return parts


def compute(fills: Sequence[FillIn], cfg: CostsConfig) -> list[ChargeLine]:
    """Charges for any set of fills, allocated to individual fills.

    The DP charge applies once per scrip per day on delivery sells; it is attached to the first
    sell order of that scrip-day.
    """
    by_order: dict[str, list[FillIn]] = defaultdict(list)
    for f in fills:
        by_order[f.order_ref].append(f)
    lines: list[ChargeLine] = []
    dp_done: set[tuple[str, date]] = set()
    dp_paise = _round(Decimal(str(cfg.equity_delivery.dp_charge_per_scrip_sell_day_inr)) * 100)
    for ref in sorted(by_order):
        group = sorted(by_order[ref], key=lambda f: f.id)
        comps = compute_order(group, cfg)
        first = group[0]
        if first.side == "SELL" and first.product == "CNC":
            key = (first.symbol, first.trade_date)
            if key not in dp_done:
                dp_done.add(key)
                comps["dp"] = dp_paise
        weights = [f.turnover_paise for f in group]
        for comp, total in comps.items():
            if total == 0:
                continue
            for f, part in zip(group, allocate(total, weights), strict=True):
                if part:
                    lines.append(ChargeLine(f.id, comp, part))
    return lines


def total_paise(lines: Sequence[ChargeLine]) -> int:
    return sum(line.amount_paise for line in lines)
