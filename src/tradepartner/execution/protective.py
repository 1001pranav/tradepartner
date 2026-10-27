"""Protective orders live at the broker, not on the laptop (spec 7.3).

For every open bot CNC position there must be a GTT OCO (stop-loss + target). This module
verifies that on a schedule (every 5 minutes and on startup), recreates or resizes a missing GTT
where ``allow_place`` permits, and flags positions that must be sold. Statuses are written to
the audit log so the Positions page can show a green tick or a red alert.
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from tradepartner.db import audit
from tradepartner.db.models import Fill, Lot, Order, OrderProposal
from tradepartner.execution.broker import Broker, GttInfo

ACTIVE = "active"
TRIGGERED = "triggered"
# Statuses that mean the position must be sold now, not fixed with a GTT.
EXIT_STATUSES = ("BREACHED", "TRIGGERED")


@dataclass(frozen=True)
class Intent:
    symbol: str
    qty: int
    stop_paise: int
    target_paise: int
    strategy: str | None
    origin: str


def intents(session: Session) -> list[Intent]:
    """What protection each open CNC position should have, from the proposal that opened it."""
    qty: dict[str, int] = defaultdict(int)
    levels: dict[str, tuple[int, int, str | None, str]] = {}
    for lot in session.scalars(select(Lot).order_by(Lot.id)):
        fill = session.get(Fill, lot.open_fill_id)
        if fill is None or fill.product != "CNC":
            continue
        order = session.get(Order, fill.order_id) if fill.order_id else None
        prop = (
            session.get(OrderProposal, order.proposal_id) if order and order.proposal_id else None
        )
        if prop is None or prop.stop_loss_paise is None or prop.target_paise is None:
            continue  # not bot-managed: nothing to protect on its behalf
        qty[lot.symbol] += lot.qty_remaining
        levels[lot.symbol] = (prop.stop_loss_paise, prop.target_paise, prop.strategy, fill.origin)
    return [
        Intent(sym, q, levels[sym][0], levels[sym][1], levels[sym][2], levels[sym][3])
        for sym, q in sorted(qty.items())
        if q > 0
    ]


class ProtectiveManager:
    def __init__(
        self,
        session: Session,
        broker: Broker,
        last_price: Callable[[str], int | None],
        notify: Callable[[str, str], None],
        clock: Callable[[], datetime],
    ) -> None:
        self.session = session
        self.broker = broker
        self.last_price = last_price
        self.notify = notify
        self.clock = clock

    def _active(self) -> dict[str, GttInfo]:
        return {g.symbol: g for g in self.broker.list_gtts() if g.status == ACTIVE}

    def _triggered(self) -> set[str]:
        return {g.symbol for g in self.broker.list_gtts() if g.status == TRIGGERED}

    def _fix(self, it: Intent, existing: GttInfo | None) -> str:
        """Create or resize one GTT. Returns the resulting status."""
        last = self.last_price(it.symbol)
        if last is None:
            return "NO_QUOTE"
        if last <= it.stop_paise:
            return "BREACHED"  # price is already through the stop: a GTT cannot be placed
        if last >= it.target_paise:
            return "TARGET_REACHED"
        if existing is None:
            self.broker.place_gtt_oco(it.symbol, it.qty, last, it.stop_paise, it.target_paise)
            return "CREATED"
        self.broker.modify_gtt_oco(existing.gtt_id, it.qty, last, it.stop_paise, it.target_paise)
        return "RESIZED"

    def check(self, allow_place: dict[str, bool] | bool = False) -> dict[str, str]:
        """Verify every bot position. ``allow_place`` is per strategy (AUTO) or a global flag."""
        active = self._active()
        triggered = self._triggered()
        report: dict[str, str] = {}
        wanted = intents(self.session)
        for it in wanted:
            allowed = (
                allow_place
                if isinstance(allow_place, bool)
                else allow_place.get(it.strategy or "", False)
            )
            g = active.get(it.symbol)
            correct = (
                g is not None
                and g.qty == it.qty
                and g.stop_paise == it.stop_paise
                and g.target_paise == it.target_paise
            )
            if correct:
                report[it.symbol] = "OK"
                continue
            problem = "MISSING" if g is None else "MISMATCH"
            if g is None and it.symbol in triggered:
                # The GTT fired but the position is still held: its limit leg did not fill
                # (e.g. a gap below the stop). Only a protective exit helps now.
                report[it.symbol] = "TRIGGERED"
                self.notify(
                    "TradePartner PROTECTION", f"{it.symbol}: GTT fired, still held: exiting"
                )
            elif allowed:
                status = self._fix(it, g)
                report[it.symbol] = status
                if status not in ("CREATED", "RESIZED"):
                    self.notify("TradePartner PROTECTION", f"{it.symbol}: {status}: act now")
            else:
                report[it.symbol] = f"{problem} (alert only)"
                self.notify(
                    "TradePartner PROTECTION",
                    f"{it.symbol} has no correct GTT ({problem}): set it in Kite",
                )
        held = {it.symbol for it in wanted}
        for sym in sorted(set(active) - held):
            report[sym] = "ORPHAN"  # a GTT with no position behind it: alert, never auto-delete
            self.notify("TradePartner PROTECTION", f"GTT for {sym} has no matching position")
        audit.log(self.session, "system", "protective_check", {"symbols": report}, self.clock())
        self.session.commit()
        return report

    def place_for(self, symbol: str) -> str:
        """User-initiated (Positions page button): create or fix one symbol's GTT now."""
        it = next((i for i in intents(self.session) if i.symbol == symbol), None)
        if it is None:
            return "NOT_MANAGED"
        status = self._fix(it, self._active().get(symbol))
        audit.log(
            self.session,
            "user",
            "protective_place",
            {"symbol": symbol, "status": status},
            self.clock(),
        )
        self.session.commit()
        return status
