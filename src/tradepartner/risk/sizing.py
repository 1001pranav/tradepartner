"""Quantity is computed by the risk engine, never by the strategy (spec 7.2).

``qty = floor(min(risk_budget / (entry - stop), max_position_notional / entry, adv_cap / entry))``
All arithmetic is exact (``Fraction``); no floats touch money.
"""

from fractions import Fraction
from math import floor

from tradepartner.risk.types import Context, Proposal, Sizing
from tradepartner.settings import RiskConfig


def pct(x: float) -> Fraction:
    """A percentage from config as an exact fraction of 1 (0.75 -> 3/400)."""
    return Fraction(str(x)) / 100


def effective_position_pct(cfg: RiskConfig, mode: str) -> float:
    if mode == "AUTO":
        return min(cfg.per_trade.max_position_pct, cfg.auto_only.max_position_pct)
    return cfg.per_trade.max_position_pct


def effective_daily_loss_pct(cfg: RiskConfig, mode: str) -> float:
    if mode == "AUTO":
        return min(cfg.loss_limits.daily_loss_pct, cfg.auto_only.daily_loss_pct)
    return cfg.loss_limits.daily_loss_pct


def effective_orders_per_day(cfg: RiskConfig, mode: str) -> int:
    if mode == "AUTO":
        return min(cfg.rate_limits.max_orders_per_day, cfg.auto_only.max_orders_per_day)
    return cfg.rate_limits.max_orders_per_day


def entry_price(p: Proposal) -> int | None:
    return p.limit_price_paise if p.order_type == "LIMIT" else None


def compute_sizing(p: Proposal, ctx: Context, cfg: RiskConfig) -> Sizing:
    capital = cfg.capital_base_paise
    if p.kind == "EXIT":
        qty = p.exit_qty or 0
        px = p.limit_price_paise or (
            ctx.quotes[p.symbol].ltp_paise if p.symbol in ctx.quotes else 0
        )
        return Sizing(qty, qty * px, 0)
    entry, stop = entry_price(p), p.stop_loss_paise
    if capital <= 0 or entry is None or entry <= 0 or stop is None or stop <= 0:
        return Sizing()
    per_share = entry - stop if p.side == "BUY" else stop - entry
    if per_share <= 0:
        return Sizing()
    risk_budget = capital * pct(cfg.per_trade.max_risk_pct)
    max_notional = capital * pct(effective_position_pct(cfg, ctx.mode))
    meta = ctx.instruments.get(p.symbol)
    adv = meta.avg_daily_value_paise if meta else 0
    adv_cap = adv * pct(cfg.per_trade.max_pct_of_adv)
    qty = floor(min(risk_budget / per_share, max_notional / entry, adv_cap / entry))
    if p.max_qty is not None:
        qty = min(qty, p.max_qty)  # a cap can only reduce the size
    if qty < 1:
        return Sizing()
    return Sizing(qty, qty * entry, qty * per_share)
