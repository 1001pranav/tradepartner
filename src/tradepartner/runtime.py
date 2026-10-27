"""Composition root: wires the controller, risk engine, broker, router and kill switch.

This is the one place that owns a ``ModeController`` and hands the router a halt-enforcer
callable, so the execution package never imports the controller itself.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from tradepartner.calendar import TradingCalendar, utc_now
from tradepartner.data.quotes import QuoteBook
from tradepartner.execution.broker import Broker
from tradepartner.execution.killswitch import KillSwitch, Pending
from tradepartner.execution.paper import PaperBroker
from tradepartner.execution.router import Router
from tradepartner.ledger import rebuild
from tradepartner.modes.controller import ModeController
from tradepartner.modes.state import GLOBAL
from tradepartner.notify import notify
from tradepartner.risk import kill
from tradepartner.risk.context import build_context
from tradepartner.risk.engine import RiskEngine, enforce_halts
from tradepartner.risk.types import Context, InstrumentMeta
from tradepartner.settings import Settings

MetaProvider = Callable[[], dict[str, InstrumentMeta]]


@dataclass
class Runtime:
    session: Session
    settings: Settings
    calendar: TradingCalendar
    quotes: QuoteBook
    controller: ModeController
    engine: RiskEngine
    broker: Broker
    router: Router
    kill: KillSwitch
    meta_provider: MetaProvider
    clock: Callable[[], datetime]

    def enforce_limits(self, strategies: list[str]) -> list[str]:
        """The 30-second tracker: evaluate loss limits and force halts (spec 7.2). Returns the
        scopes switched OFF."""
        ctx = self.context(None)
        results = self.engine.check_halts(ctx, strategies)
        return enforce_halts(self.session, self.controller, results, self.clock())

    def downgrade_on_drift(self, strategy: str, reasons: list[str]) -> bool:
        """Drift monitors: AUTO -> MANUAL for the strategy (spec 10.6)."""
        return self.controller.downgrade_to_manual(
            strategy, "drift: " + "; ".join(reasons), self.clock()
        )

    def no_token_safeguard(self) -> list[str]:
        """No valid Kite token at 09:10: never stay in AUTO (spec 14). Returns scopes lowered."""
        lowered: list[str] = []
        for scope in self.controller.scopes():
            if self.controller.current(scope).mode == "AUTO":
                self.controller.set_mode(scope, "MANUAL", "system", "no Kite token", self.clock())
                lowered.append(scope)
        return lowered

    def context(self, strategy: str | None = None) -> Context:
        """A fresh risk context (rebuilds the derived ledger tables first)."""
        rebuild.rebuild(self.session, self.settings.costs, self.settings.tax)
        mode = (
            self.controller.effective(strategy)
            if strategy
            else self.controller.current(GLOBAL).mode
        )
        return build_context(
            self.session, self.settings.risk, self.clock(), self.calendar, self.quotes.quotes,
            self.meta_provider(), mode, kill.kill_active(),
            live_broker=self.settings.app.broker == "kite",
        )  # fmt: skip


def build_runtime(
    session: Session,
    settings: Settings,
    broker: Broker | None = None,
    quotes: QuoteBook | None = None,
    meta_provider: MetaProvider | None = None,
    kill_pending: list[Pending] | None = None,
    clock: Callable[[], datetime] = utc_now,
    notifier: Callable[[str, str], None] = notify,
    calendar: TradingCalendar | None = None,
) -> Runtime:
    book = quotes or QuoteBook()
    controller = ModeController(session)
    engine = RiskEngine(settings.risk)
    chosen: Broker = broker or PaperBroker(
        session, book.get, settings.costs.slippage.default_bps, clock
    )
    runtime: Runtime

    def ctx_for(strategy: str) -> Context:
        return runtime.context(strategy)

    router = Router(
        session, engine, controller, chosen, ctx_for,
        lambda results, now: enforce_halts(session, controller, results, now),
        settings.app.manual_execution,
    )  # fmt: skip
    ks = KillSwitch(
        session, router, chosen, engine, lambda: runtime.context(None),
        lambda reason, now: controller.halt(GLOBAL, reason, now), notifier,
        pending=kill_pending,
    )  # fmt: skip
    runtime = Runtime(
        session, settings, calendar or TradingCalendar.from_config(), book, controller, engine,
        chosen, router, ks, meta_provider or (lambda: {}), clock,
    )  # fmt: skip
    return runtime
