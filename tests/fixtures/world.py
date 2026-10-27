"""A wired-up paper-trading world: DB, quotes, paper broker, router, modes, kill switch."""

from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy.orm import Session

from tests.fixtures.market import dt, make_cal
from tests.fixtures.risk import CAPITAL_INR, LTP, cfg, entry, meta, quote
from tradepartner.execution.killswitch import KillSwitch
from tradepartner.execution.paper import PaperBroker
from tradepartner.execution.router import Router
from tradepartner.ledger import rebuild
from tradepartner.modes.controller import ModeController
from tradepartner.modes.state import GLOBAL, strategy_scope
from tradepartner.risk.context import build_context
from tradepartner.risk.engine import RiskEngine, enforce_halts
from tradepartner.risk.types import Context, InstrumentMeta, Quote
from tradepartner.settings import load_settings

STRATEGY = "swing_v1"


@dataclass
class World:
    db: Session
    now: datetime = field(default_factory=lambda: dt(2026, 3, 10, 10, 0))
    quotes: dict[str, Quote] = field(default_factory=dict)
    instruments: dict[str, InstrumentMeta] = field(default_factory=dict)
    notes: list[tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.settings = load_settings()
        self.risk_cfg = cfg()
        self.calendar = make_cal()
        self.set_price("INFY", LTP)
        self.instruments["INFY"] = meta("INFY")
        self.controller = ModeController(self.db)
        self.engine = RiskEngine(self.risk_cfg)
        self.broker = PaperBroker(
            self.db,
            lambda s: self.quotes.get(s),
            10.0,
            lambda: self.now.astimezone(dt(2026, 1, 1).tzinfo),
        )
        self.router = Router(
            self.db, self.engine, self.controller, self.broker, self.context_for,
            lambda results, now: enforce_halts(self.db, self.controller, results, now),
        )  # fmt: skip
        self.kill_path = None
        self.kill = KillSwitch(
            self.db, self.router, self.broker, self.engine, self.plain_context,
            lambda reason, now: self.controller.halt(GLOBAL, reason, now),
            lambda t, m: self.notes.append((t, m)), None,
        )  # fmt: skip

    def set_price(self, symbol: str, ltp: int, prev_close: int | None = None) -> None:
        self.quotes[symbol] = quote(ltp, 0, self.now, prev_close_paise=prev_close)

    def advance(self, **delta: int) -> None:
        from datetime import timedelta

        self.now = self.now + timedelta(**delta)
        self.quotes = {
            s: Quote(q.ltp_paise, self.now, q.prev_close_paise) for s, q in self.quotes.items()
        }

    def go_auto(self) -> None:
        from datetime import date

        from tradepartner.modes.controller import UnlockEvidence

        good = UnlockEvidence(True, 40, date(2025, 12, 1), True)
        self.controller.set_mode(GLOBAL, "AUTO", "user", "test", self.now)
        self.controller.unlock_auto(STRATEGY, good, STRATEGY, self.now)
        self.controller.set_mode(strategy_scope(STRATEGY), "AUTO", "user", "test", self.now)

    def go_manual(self) -> None:
        self.controller.set_mode(GLOBAL, "MANUAL", "user", "test", self.now)
        self.controller.set_mode(strategy_scope(STRATEGY), "MANUAL", "user", "test", self.now)

    def plain_context(self) -> Context:
        rebuild.rebuild(self.db, self.settings.costs, self.settings.tax)
        return build_context(
            self.db, self.risk_cfg, self.now, self.calendar, self.quotes, self.instruments,
            self.controller.effective(STRATEGY),
        )  # fmt: skip

    def context_for(self, strategy: str) -> Context:
        rebuild.rebuild(self.db, self.settings.costs, self.settings.tax)
        from tradepartner.risk import kill

        return build_context(
            self.db, self.risk_cfg, self.now, self.calendar, self.quotes, self.instruments,
            self.controller.effective(strategy), kill.kill_active(self.kill_path),
        )  # fmt: skip

    def submit_entry(self, symbol: str = "INFY", key: str = "k1", **kw: object):  # type: ignore[no-untyped-def]
        px = int(kw.pop("limit_price_paise", self.quotes[symbol].ltp_paise))  # type: ignore[call-overload]
        fields: dict[str, object] = dict(
            symbol=symbol, idempotency_key=key, signal_time=self.now, limit_price_paise=px,
            stop_loss_paise=int(px * 0.96), target_paise=int(px * 1.08),
        )  # fmt: skip
        fields.update(kw)
        return self.router.submit(entry(**fields), self.now)


__all__ = ["CAPITAL_INR", "STRATEGY", "World"]
