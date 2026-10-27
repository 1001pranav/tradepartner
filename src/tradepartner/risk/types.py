"""Plain data types for the risk engine. ``RiskEngine.check`` is a pure function of these."""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

from tradepartner.calendar import TradingCalendar
from tradepartner.risk.approved import ApprovedOrder

Mode = Literal["OFF", "MANUAL", "AUTO"]
Severity = Literal["block", "halt"]
Kind = Literal["ENTRY", "EXIT"]


@dataclass(frozen=True)
class Proposal:
    """What a strategy (or exit logic) asks for. Strategies never set quantity (spec 7.2)."""

    strategy: str
    symbol: str
    side: str  # BUY | SELL
    product: str  # CNC | MIS
    order_type: str  # LIMIT | SL | SL-M | MARKET
    kind: Kind
    signal_time: datetime
    idempotency_key: str
    limit_price_paise: int | None = None
    trigger_price_paise: int | None = None
    stop_loss_paise: int | None = None
    target_paise: int | None = None
    time_stop_date: date | None = None
    exit_qty: int | None = None  # EXIT only
    exit_reason: str | None = None  # EXIT only: stop | target | time_stop | manual | kill
    max_qty: int | None = None  # a hard cap on size (e.g. 1-share live tests); only ever lowers
    protective: bool = False  # protective exits still run when mode is OFF
    emergency: bool = False  # kill-switch exits may use protected market orders
    segment: str = "NSE_EQ"
    exchange: str = "NSE"
    signal_id: int | None = None
    proposal_id: int | None = None


@dataclass(frozen=True)
class Quote:
    ltp_paise: int
    as_of: datetime
    prev_close_paise: int | None = None
    upper_circuit_hit: bool = False
    lower_circuit_hit: bool = False


@dataclass(frozen=True)
class InstrumentMeta:
    symbol: str
    sector: str = "UNKNOWN"
    avg_daily_value_paise: int = 0  # 20-day average traded value
    asm_gsm: bool = False
    result_date: date | None = None
    ex_date: date | None = None


@dataclass(frozen=True)
class Position:
    symbol: str
    qty: int
    avg_price_paise: int
    product: str
    strategy: str | None = None
    bot_created: bool = False


@dataclass(frozen=True)
class LimitState:
    """Loss trackers and counters, recomputed from the ledger every 30 s (spec 7.2)."""

    daily_pnl_paise: int = 0  # realized + unrealized, today
    weekly_pnl_paise: int = 0
    equity_paise: int = 0
    peak_equity_paise: int = 0
    consecutive_losses: dict[str, int] = field(default_factory=dict)
    orders_today: int = 0
    last_order_at: datetime | None = None
    last_stop_dates: dict[str, date] = field(default_factory=dict)
    seen_idempotency_keys: frozenset[str] = frozenset()
    reconciliation_blocked: bool = False
    kill_active: bool = False
    intraday_live_ok: bool = False  # 3 months of passing paper trading + approved model (M9)


@dataclass(frozen=True)
class Context:
    now: datetime
    mode: Mode  # effective mode for the proposal's strategy
    calendar: TradingCalendar
    quotes: dict[str, Quote]
    instruments: dict[str, InstrumentMeta]
    positions: tuple[Position, ...] = ()
    pending_entries: tuple[Position, ...] = ()  # unfilled remainder of open bot BUY orders
    limits: LimitState = field(default_factory=LimitState)
    live_broker: bool = False  # True when orders would go to the real broker


def exposure_positions(ctx: Context) -> tuple[Position, ...]:
    """Held positions plus resting entry orders: what portfolio limits must count."""
    return ctx.positions + ctx.pending_entries


@dataclass(frozen=True)
class Sizing:
    qty: int = 0
    notional_paise: int = 0
    risk_amount_paise: int = 0


@dataclass(frozen=True)
class RuleResult:
    rule: str
    passed: bool
    severity: Severity = "block"
    message: str = ""
    scope: str = "global"  # 'global' or 'strategy:<name>' for halt scoping


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    results: tuple[RuleResult, ...]
    sizing: Sizing
    order: ApprovedOrder | None

    @property
    def failures(self) -> tuple[RuleResult, ...]:
        return tuple(r for r in self.results if not r.passed)

    @property
    def halts(self) -> tuple[RuleResult, ...]:
        return tuple(r for r in self.failures if r.severity == "halt")

    @property
    def reasons(self) -> list[str]:
        return [f"{r.rule}: {r.message}" for r in self.failures]
