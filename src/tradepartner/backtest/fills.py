"""Fill and slippage model for daily bars (spec 13). All prices are integer paise.

* Signals at bar T fill at bar T+1's open +/- slippage; limit orders fill only if the bar's range
  touches the limit.
* Stop and target inside one bar: the stop is assumed to be hit first (conservative).
* Locked-circuit bars (open == high == low, moving away from the prior close) do not fill.
"""

from typing import Literal

ExitReason = Literal["stop", "target"]


def slip(price: int, bps: float, up: bool) -> int:
    """``price`` moved adversely by ``bps`` (up for buys, down for sells), in whole paise."""
    delta = price * round(bps * 100) // 1_000_000
    return price + delta if up else price - delta


def circuit_blocked(side: str, open_: int, high: int, low: int, prev_close: int | None) -> bool:
    # SPEC-QUESTION: spec says "open equals high" for buys; taken literally that blocks ordinary
    # gap-up opens. We require a fully locked bar (open == high == low) moving away from the
    # prior close, which is what an upper/lower circuit looks like in daily data.
    if prev_close is None or not (open_ == high == low):
        return False
    return open_ > prev_close if side == "BUY" else open_ < prev_close


def buy_fill(
    open_: int, high: int, low: int, limit: int, bps: float, prev_close: int | None
) -> int | None:
    if circuit_blocked("BUY", open_, high, low, prev_close):
        return None
    if open_ <= limit:
        return min(limit, slip(open_, bps, up=True))
    if low <= limit:
        return limit
    return None


def exit_fill(
    entry_day: bool, open_: int, high: int, low: int, stop: int, target: int, bps: float,
    prev_close: int | None,
) -> tuple[int, ExitReason] | None:  # fmt: skip
    """Stop/target exit for a long position on this bar, or None.

    Gap rules apply only after the entry day. A stop is a market-on-trigger (slipped); a target is
    a resting limit (no slippage).
    """
    if circuit_blocked("SELL", open_, high, low, prev_close):
        return None
    if not entry_day:
        if open_ <= stop:
            return slip(open_, bps, up=False), "stop"
        if open_ >= target:
            return open_, "target"
    if low <= stop:
        return slip(stop, bps, up=False), "stop"
    if high >= target:
        return target, "target"
    return None
