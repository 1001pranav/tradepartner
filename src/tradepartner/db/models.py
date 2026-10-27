"""SQLAlchemy tables (spec section 6).

Conventions: timestamps are UTC (stored as ISO-8601 text), money is integer paise,
every table has ``created_at``. Append-only tables are protected by SQLite triggers
created in the migration (see ``APPEND_ONLY_TABLES``).
"""

from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Date,
    ForeignKey,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class UTCDateTime(TypeDecorator[datetime]):
    """Stores timezone-aware datetimes as UTC ISO-8601 text; rejects naive values."""

    impl = String
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> str | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetime: timestamps must be timezone-aware")
        return value.astimezone(UTC).isoformat()

    def process_result_value(self, value: str | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        return datetime.fromisoformat(value).astimezone(UTC)


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class _Created:
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=_now)


def _pk() -> Mapped[int]:
    return mapped_column(Integer, primary_key=True, autoincrement=True)


# SPEC-QUESTION: spec 6.3 lists `signals` as append-only but also gives it a mutable `status`.
# Safer option taken: signals is fully append-only; a status change is a new row (M7 decides how).
APPEND_ONLY_TABLES = (
    "news_raw",
    "news_scores",
    "signals",
    "signal_outcomes",
    "signal_events",
    "order_events",
    "fills",
    "charges",
    "cash_ledger",
    "risk_events",
    "audit_log",
)

# ---------------- 6.1 Reference ----------------


class Instrument(_Created, Base):
    __tablename__ = "instruments"
    instrument_token: Mapped[int] = mapped_column(Integer, primary_key=True)
    exchange: Mapped[str] = mapped_column(String)
    tradingsymbol: Mapped[str] = mapped_column(String, index=True)
    isin: Mapped[str | None] = mapped_column(String, index=True)
    name: Mapped[str | None] = mapped_column(String)
    segment: Mapped[str] = mapped_column(String)
    lot_size: Mapped[int] = mapped_column(Integer, default=1)
    tick_size_paise: Mapped[int] = mapped_column(Integer, default=5)
    last_updated: Mapped[datetime] = mapped_column(UTCDateTime, default=_now)


class UniverseMember(_Created, Base):
    __tablename__ = "universe_members"
    id: Mapped[int] = _pk()
    universe_name: Mapped[str] = mapped_column(String, index=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    valid_from: Mapped[date] = mapped_column(Date)
    valid_to: Mapped[date | None] = mapped_column(Date)


class CorporateAction(_Created, Base):
    __tablename__ = "corporate_actions"
    id: Mapped[int] = _pk()
    symbol: Mapped[str] = mapped_column(String, index=True)
    isin: Mapped[str | None] = mapped_column(String)
    action_type: Mapped[str] = mapped_column(String)
    ex_date: Mapped[date] = mapped_column(Date)
    ratio_num: Mapped[int | None] = mapped_column(Integer)
    ratio_den: Mapped[int | None] = mapped_column(Integer)
    amount_paise: Mapped[int | None] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String)
    __table_args__ = (
        CheckConstraint("action_type IN ('split','bonus','dividend','rights','merger')"),
    )


class TradingCalendar(_Created, Base):
    __tablename__ = "trading_calendar"
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    is_trading_day: Mapped[bool] = mapped_column(Boolean)
    session_open: Mapped[str | None] = mapped_column(String)
    session_close: Mapped[str | None] = mapped_column(String)
    note: Mapped[str | None] = mapped_column(String)


# ---------------- 6.2 News and LLM ----------------


class NewsRaw(_Created, Base):
    __tablename__ = "news_raw"
    id: Mapped[int] = _pk()
    source: Mapped[str] = mapped_column(String)
    url: Mapped[str] = mapped_column(String, unique=True)
    title: Mapped[str] = mapped_column(Text)
    body: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    published_at_estimated: Mapped[bool] = mapped_column(Boolean, default=False)
    fetched_at: Mapped[datetime] = mapped_column(UTCDateTime)
    content_hash: Mapped[str] = mapped_column(String, unique=True)
    lang: Mapped[str | None] = mapped_column(String)


class ResultDate(_Created, Base):
    """Board-meeting / results calendar (for the 'results in next N days' feature and skip rule)."""

    __tablename__ = "result_dates"
    id: Mapped[int] = _pk()
    symbol: Mapped[str] = mapped_column(String, index=True)
    result_date: Mapped[date] = mapped_column(Date, index=True)
    purpose: Mapped[str | None] = mapped_column(String)
    source: Mapped[str] = mapped_column(String)
    __table_args__ = (UniqueConstraint("symbol", "result_date"),)


class NewsEntity(_Created, Base):
    __tablename__ = "news_entities"
    news_id: Mapped[int] = mapped_column(ForeignKey("news_raw.id"), primary_key=True)
    symbol: Mapped[str] = mapped_column(String, primary_key=True)
    match_method: Mapped[str] = mapped_column(String)
    confidence: Mapped[float] = mapped_column()
    __table_args__ = (CheckConstraint("match_method IN ('alias','isin','llm')"),)


class NewsScore(_Created, Base):
    __tablename__ = "news_scores"
    id: Mapped[int] = _pk()
    news_id: Mapped[int] = mapped_column(ForeignKey("news_raw.id"), index=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    llm_model: Mapped[str] = mapped_column(String)
    llm_model_digest: Mapped[str | None] = mapped_column(String)
    prompt_version: Mapped[str] = mapped_column(String)
    relevance: Mapped[float] = mapped_column()
    sentiment: Mapped[float] = mapped_column()
    event_type: Mapped[str] = mapped_column(String)
    materiality: Mapped[float] = mapped_column()
    horizon: Mapped[str] = mapped_column(String)
    confidence: Mapped[float] = mapped_column()
    summary: Mapped[str] = mapped_column(Text)
    raw_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    scored_at: Mapped[datetime] = mapped_column(UTCDateTime)
    __table_args__ = (
        CheckConstraint("relevance BETWEEN 0 AND 1"),
        CheckConstraint("sentiment BETWEEN -1 AND 1"),
        CheckConstraint("materiality BETWEEN 0 AND 1"),
        CheckConstraint("confidence BETWEEN 0 AND 1"),
        CheckConstraint("horizon IN ('intraday','days','weeks','months')"),
    )


# ---------------- 6.3 Features, models, signals ----------------


class FeatureSnapshot(_Created, Base):
    __tablename__ = "feature_snapshots"
    id: Mapped[int] = _pk()
    as_of: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    feature_set_version: Mapped[str] = mapped_column(String)
    parquet_path: Mapped[str] = mapped_column(String)


class Model(_Created, Base):
    __tablename__ = "models"
    id: Mapped[int] = _pk()
    strategy: Mapped[str] = mapped_column(String, index=True)
    version: Mapped[str] = mapped_column(String)
    algo: Mapped[str] = mapped_column(String)
    feature_set_version: Mapped[str] = mapped_column(String)
    train_start: Mapped[date] = mapped_column(Date)
    train_end: Mapped[date] = mapped_column(Date)
    params_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    artifact_path: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="candidate")
    approved_by: Mapped[str | None] = mapped_column(String)
    approved_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    notes: Mapped[str | None] = mapped_column(Text)
    __table_args__ = (
        CheckConstraint("status IN ('candidate','shadow','approved','retired')"),
        UniqueConstraint("strategy", "version"),
    )


class Signal(_Created, Base):
    __tablename__ = "signals"
    id: Mapped[int] = _pk()
    strategy: Mapped[str] = mapped_column(String, index=True)
    model_id: Mapped[int | None] = mapped_column(ForeignKey("models.id"))
    symbol: Mapped[str] = mapped_column(String, index=True)
    side: Mapped[str] = mapped_column(String)
    as_of: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    score: Mapped[float] = mapped_column()
    prob: Mapped[float | None] = mapped_column()
    expected_return: Mapped[float | None] = mapped_column()
    horizon_days: Mapped[int] = mapped_column(Integer)
    explanation: Mapped[str | None] = mapped_column(Text)
    features_ref: Mapped[str | None] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="new")
    __table_args__ = (
        CheckConstraint("side IN ('BUY','SELL')"),
        CheckConstraint(
            "status IN ('new','proposed','rejected_by_risk','shown','approved',"
            "'executed','expired','ignored')"
        ),
    )


class SignalEvent(_Created, Base):
    """Lifecycle of a signal (shown, ignored, proposed, executed, expired...). The ``signals`` row
    itself is immutable, so its 'current status' is the latest event (append-only)."""

    __tablename__ = "signal_events"
    id: Mapped[int] = _pk()
    signal_id: Mapped[int] = mapped_column(ForeignKey("signals.id"), index=True)
    status: Mapped[str] = mapped_column(String)
    detail: Mapped[str | None] = mapped_column(Text)
    at: Mapped[datetime] = mapped_column(UTCDateTime)
    __table_args__ = (
        CheckConstraint(
            "status IN ('new','proposed','rejected_by_risk','shown','approved','executed',"
            "'expired','ignored','shadow')"
        ),
    )


class SignalOutcome(_Created, Base):
    """Revealed outcome of a signal once its horizon has passed (spec 10.6, append-only)."""

    __tablename__ = "signal_outcomes"
    id: Mapped[int] = _pk()
    signal_id: Mapped[int] = mapped_column(ForeignKey("signals.id"), index=True)
    realized_return: Mapped[float] = mapped_column()
    excess_return: Mapped[float | None] = mapped_column()
    label: Mapped[int | None] = mapped_column(Integer)  # 1 if it beat the target definition
    revealed_at: Mapped[datetime] = mapped_column(UTCDateTime)


class OrderProposal(_Created, Base):
    __tablename__ = "order_proposals"
    id: Mapped[int] = _pk()
    signal_id: Mapped[int | None] = mapped_column(ForeignKey("signals.id"))
    symbol: Mapped[str] = mapped_column(String, index=True)
    side: Mapped[str] = mapped_column(String)
    product: Mapped[str] = mapped_column(String)
    order_type: Mapped[str] = mapped_column(String)
    qty: Mapped[int] = mapped_column(Integer)
    limit_price_paise: Mapped[int | None] = mapped_column(Integer)
    trigger_price_paise: Mapped[int | None] = mapped_column(Integer)
    stop_loss_paise: Mapped[int | None] = mapped_column(Integer)
    target_paise: Mapped[int | None] = mapped_column(Integer)
    time_stop_date: Mapped[date | None] = mapped_column(Date)
    risk_amount_paise: Mapped[int | None] = mapped_column(Integer)
    mode_at_creation: Mapped[str] = mapped_column(String)
    risk_decision: Mapped[str | None] = mapped_column(String)
    risk_reasons_json: Mapped[list[object] | None] = mapped_column(JSON)
    idempotency_key: Mapped[str] = mapped_column(String, unique=True)
    # Added beyond spec 6.3: needed for exits, cooldowns and per-strategy loss tracking.
    strategy: Mapped[str | None] = mapped_column(String, index=True)
    kind: Mapped[str | None] = mapped_column(String)  # ENTRY | EXIT
    exit_reason: Mapped[str | None] = mapped_column(String)  # stop|target|time_stop|manual|kill
    max_qty: Mapped[int | None] = mapped_column(Integer)  # hard size cap (e.g. 1-share live test)
    __table_args__ = (
        CheckConstraint("side IN ('BUY','SELL')"),
        CheckConstraint("product IN ('CNC','MIS')"),
        CheckConstraint("order_type IN ('LIMIT','SL','SL-M','MARKET')"),
        CheckConstraint("mode_at_creation IN ('OFF','MANUAL','AUTO')"),
        CheckConstraint("risk_decision IS NULL OR risk_decision IN ('approved','rejected')"),
    )


# ---------------- 6.4 Execution and ledger ----------------

_ORIGINS = "('bot_auto','bot_manual_api','manual_external','paper')"


class Order(_Created, Base):
    __tablename__ = "orders"
    id: Mapped[int] = _pk()
    proposal_id: Mapped[int | None] = mapped_column(ForeignKey("order_proposals.id"))
    broker_order_id: Mapped[str | None] = mapped_column(String, unique=True)
    origin: Mapped[str] = mapped_column(String)
    symbol: Mapped[str] = mapped_column(String, index=True)
    side: Mapped[str] = mapped_column(String)
    product: Mapped[str] = mapped_column(String)
    order_type: Mapped[str] = mapped_column(String)
    qty: Mapped[int] = mapped_column(Integer)
    price_paise: Mapped[int | None] = mapped_column(Integer)
    trigger_paise: Mapped[int | None] = mapped_column(Integer)
    tag: Mapped[str | None] = mapped_column(String(20))
    placed_at: Mapped[datetime] = mapped_column(UTCDateTime)
    status: Mapped[str] = mapped_column(String)
    __table_args__ = (
        CheckConstraint(f"origin IN {_ORIGINS}"),
        CheckConstraint("side IN ('BUY','SELL')"),
    )


class OrderEvent(_Created, Base):
    __tablename__ = "order_events"
    id: Mapped[int] = _pk()
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), index=True)
    status: Mapped[str] = mapped_column(String)
    filled_qty: Mapped[int] = mapped_column(Integer, default=0)
    avg_price_paise: Mapped[int | None] = mapped_column(Integer)
    message: Mapped[str | None] = mapped_column(Text)
    at: Mapped[datetime] = mapped_column(UTCDateTime)


class Fill(_Created, Base):
    __tablename__ = "fills"
    id: Mapped[int] = _pk()
    broker_trade_id: Mapped[str] = mapped_column(String, unique=True)
    broker_order_id: Mapped[str | None] = mapped_column(String)
    order_id: Mapped[int | None] = mapped_column(ForeignKey("orders.id"))
    origin: Mapped[str] = mapped_column(String)
    account: Mapped[str] = mapped_column(String)
    symbol: Mapped[str] = mapped_column(String, index=True)
    isin: Mapped[str | None] = mapped_column(String, index=True)
    exchange: Mapped[str] = mapped_column(String)
    side: Mapped[str] = mapped_column(String)
    product: Mapped[str] = mapped_column(String)
    qty: Mapped[int] = mapped_column(Integer)
    price_paise: Mapped[int] = mapped_column(Integer)
    filled_at: Mapped[datetime] = mapped_column(UTCDateTime)
    trade_date: Mapped[date] = mapped_column(Date, index=True)
    source: Mapped[str] = mapped_column(String)
    __table_args__ = (
        CheckConstraint(f"origin IN {_ORIGINS}"),
        CheckConstraint("side IN ('BUY','SELL')"),
        CheckConstraint("qty > 0"),
        CheckConstraint("price_paise > 0"),
        CheckConstraint("source IN ('kite_api','console_csv','paper','correction')"),
        CheckConstraint("exchange = 'NSE'"),
    )


class Charge(_Created, Base):
    __tablename__ = "charges"
    id: Mapped[int] = _pk()
    fill_id: Mapped[int | None] = mapped_column(ForeignKey("fills.id"))
    trade_date: Mapped[date] = mapped_column(Date, index=True)
    symbol: Mapped[str] = mapped_column(String)
    component: Mapped[str] = mapped_column(String)
    amount_paise: Mapped[int] = mapped_column(Integer)
    computed: Mapped[bool] = mapped_column(Boolean)
    source: Mapped[str] = mapped_column(String)
    __table_args__ = (
        CheckConstraint(
            "component IN ('brokerage','stt','exchange_txn','sebi','stamp','gst','dp','other')"
        ),
        CheckConstraint("source IN ('computed','contract_note')"),
    )


class Lot(_Created, Base):
    """Derived and rebuildable from ``fills``."""

    __tablename__ = "lots"
    id: Mapped[int] = _pk()
    symbol: Mapped[str] = mapped_column(String, index=True)
    isin: Mapped[str | None] = mapped_column(String)
    open_fill_id: Mapped[int] = mapped_column(ForeignKey("fills.id"))
    open_date: Mapped[date] = mapped_column(Date)
    qty_open: Mapped[int] = mapped_column(Integer)
    qty_remaining: Mapped[int] = mapped_column(Integer)
    cost_paise_per_share: Mapped[int] = mapped_column(Integer)
    adjusted_for_ca: Mapped[bool] = mapped_column(Boolean, default=False)


class Realized(_Created, Base):
    """Derived and rebuildable from ``fills``."""

    __tablename__ = "realized"
    id: Mapped[int] = _pk()
    symbol: Mapped[str] = mapped_column(String, index=True)
    open_fill_id: Mapped[int | None] = mapped_column(ForeignKey("fills.id"))
    close_fill_id: Mapped[int] = mapped_column(ForeignKey("fills.id"))
    qty: Mapped[int] = mapped_column(Integer)
    buy_value_paise: Mapped[int] = mapped_column(Integer)
    sell_value_paise: Mapped[int] = mapped_column(Integer)
    charges_paise: Mapped[int] = mapped_column(Integer)  # excludes STT
    stt_paise: Mapped[int] = mapped_column(Integer)
    holding_days: Mapped[int] = mapped_column(Integer)
    category: Mapped[str] = mapped_column(String)
    fy: Mapped[str] = mapped_column(String, index=True)
    pnl_paise: Mapped[int] = mapped_column(Integer)
    __table_args__ = (
        CheckConstraint("category IN ('intraday_spec','stcg','ltcg','business','unmatched')"),
    )


class PositionSnapshot(_Created, Base):
    __tablename__ = "positions_snapshot"
    id: Mapped[int] = _pk()
    snapshot_date: Mapped[date] = mapped_column(Date, index=True)
    kind: Mapped[str] = mapped_column(String)
    symbol: Mapped[str] = mapped_column(String)
    isin: Mapped[str | None] = mapped_column(String)
    product: Mapped[str | None] = mapped_column(String)
    qty: Mapped[int] = mapped_column(Integer)
    avg_price_paise: Mapped[int] = mapped_column(Integer)
    __table_args__ = (CheckConstraint("kind IN ('position','holding')"),)


class CashLedger(_Created, Base):
    __tablename__ = "cash_ledger"
    id: Mapped[int] = _pk()
    date: Mapped[date] = mapped_column(Date, index=True)
    type: Mapped[str] = mapped_column(String)
    amount_paise: Mapped[int] = mapped_column(Integer)
    ref: Mapped[str | None] = mapped_column(String)
    __table_args__ = (
        CheckConstraint(
            "type IN ('funds_in','funds_out','charge','dividend','pnl_settlement','adjustment')"
        ),
    )


# ---------------- 6.5 Control and audit ----------------


class ModeState(_Created, Base):
    """Latest row per scope is current."""

    __tablename__ = "mode_state"
    id: Mapped[int] = _pk()
    scope: Mapped[str] = mapped_column(String, index=True)
    mode: Mapped[str] = mapped_column(String)
    auto_unlocked: Mapped[bool] = mapped_column(Boolean, default=False)
    changed_by: Mapped[str] = mapped_column(String)
    reason: Mapped[str | None] = mapped_column(Text)
    changed_at: Mapped[datetime] = mapped_column(UTCDateTime)
    __table_args__ = (
        CheckConstraint("mode IN ('OFF','MANUAL','AUTO')"),
        CheckConstraint("changed_by IN ('user','system')"),
    )


class RiskEvent(_Created, Base):
    __tablename__ = "risk_events"
    id: Mapped[int] = _pk()
    at: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    rule: Mapped[str] = mapped_column(String)
    severity: Mapped[str] = mapped_column(String)
    scope: Mapped[str] = mapped_column(String)
    details_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    __table_args__ = (CheckConstraint("severity IN ('info','block','halt')"),)


class AuditLog(_Created, Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = _pk()
    at: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    actor: Mapped[str] = mapped_column(String)
    action: Mapped[str] = mapped_column(String)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    payload_hash: Mapped[str] = mapped_column(String)
    __table_args__ = (CheckConstraint("actor IN ('user','system','scheduler')"),)


class JobRun(_Created, Base):
    __tablename__ = "job_runs"
    id: Mapped[int] = _pk()
    job: Mapped[str] = mapped_column(String, index=True)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime)
    ended_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    status: Mapped[str] = mapped_column(String)
    error: Mapped[str | None] = mapped_column(Text)
