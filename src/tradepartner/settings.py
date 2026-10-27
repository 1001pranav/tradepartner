"""Typed models for every config file (spec section 5: config/*.yaml)."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tradepartner.paths import config_dir


class ConfigError(Exception):
    """Raised when configuration is missing or unsafe. The app must not start."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---- risk.yaml (spec 7.2) ----


class PerTradeRisk(_Strict):
    max_risk_pct: float = Field(gt=0, le=100)
    max_position_pct: float = Field(gt=0, le=100)
    require_stop_loss: bool
    max_stop_distance_pct: float = Field(gt=0, le=100)
    min_reward_risk: float = Field(gt=0)
    limit_price_band_pct: float = Field(gt=0, le=100)
    min_price_inr: float = Field(ge=0)
    min_avg_daily_value_inr: int = Field(ge=0)
    max_pct_of_adv: float = Field(gt=0, le=100)


class PortfolioRisk(_Strict):
    max_open_positions: int = Field(ge=1)
    max_sector_exposure_pct: float = Field(gt=0, le=100)
    max_gross_exposure_pct: float = Field(gt=0, le=100)  # no leverage on CNC
    max_mis_exposure_pct: float = Field(ge=0, le=100)
    no_averaging_down: bool
    reentry_cooldown_days_after_stop: int = Field(ge=0)


class LossLimits(_Strict):
    daily_loss_pct: float = Field(gt=0, le=100)
    weekly_loss_pct: float = Field(gt=0, le=100)
    max_drawdown_pct: float = Field(gt=0, le=100)
    consecutive_losses_halt: int = Field(ge=1)


class RateLimits(_Strict):
    # Spec 2.1: hard cap of 1 order/s and 30/day. Configurable lower, never higher.
    max_orders_per_second: int = Field(ge=1, le=1)
    max_orders_per_day: int = Field(ge=1, le=30)
    max_order_modifications_per_order: int = Field(ge=0)


class TimeRules(_Strict):
    no_new_entries_before: str = Field(pattern=r"^\d{2}:\d{2}$")
    no_new_entries_after: str = Field(pattern=r"^\d{2}:\d{2}$")
    mis_square_off_at: str = Field(pattern=r"^\d{2}:\d{2}$")
    skip_days: list[str]


class DataRules(_Strict):
    max_quote_staleness_sec: int = Field(ge=1)
    max_signal_age_minutes: int = Field(ge=1)
    block_if_circuit_hit: bool
    block_asm_gsm: bool


class AutoOnly(_Strict):
    max_orders_per_day: int = Field(ge=1, le=30)
    max_position_pct: float = Field(gt=0, le=100)
    daily_loss_pct: float = Field(gt=0, le=100)


class RiskConfig(_Strict):
    capital_base_inr: int = Field(ge=0)
    allowed_segments: list[Literal["NSE_EQ"]]
    allowed_products: list[Literal["CNC", "MIS"]]
    allow_intraday_short: bool
    allow_market_orders: bool
    market_protection_pct: float = Field(gt=0)
    per_trade: PerTradeRisk
    portfolio: PortfolioRisk
    loss_limits: LossLimits
    rate_limits: RateLimits
    time_rules: TimeRules
    data_rules: DataRules
    auto_only: AutoOnly

    @property
    def capital_base_paise(self) -> int:
        return self.capital_base_inr * 100

    @model_validator(mode="after")
    def _auto_not_looser_than_manual(self) -> "RiskConfig":
        # AUTO-only limits may only tighten the general ones.
        if self.auto_only.max_orders_per_day > self.rate_limits.max_orders_per_day:
            raise ValueError("auto_only.max_orders_per_day exceeds rate_limits")
        if self.auto_only.max_position_pct > self.per_trade.max_position_pct:
            raise ValueError("auto_only.max_position_pct exceeds per_trade")
        if self.auto_only.daily_loss_pct > self.loss_limits.daily_loss_pct:
            raise ValueError("auto_only.daily_loss_pct exceeds loss_limits")
        return self


# ---- costs.yaml (spec 8) ----


class DeliveryCosts(_Strict):
    brokerage: float
    stt_buy_pct: float
    stt_sell_pct: float
    exchange_txn_pct_nse: float
    sebi_per_crore_inr: float
    stamp_buy_pct: float
    gst_pct: float
    dp_charge_per_scrip_sell_day_inr: float


class IntradayCosts(_Strict):
    brokerage_pct: float
    brokerage_cap_inr: float
    stt_sell_pct: float
    exchange_txn_pct_nse: float
    sebi_per_crore_inr: float
    stamp_buy_pct: float
    gst_pct: float


class Slippage(_Strict):
    default_bps: float = Field(ge=0)
    intraday_bps: float = Field(ge=0)


class Rounding(_Strict):
    stt_round_to_rupee: bool


class CostsConfig(_Strict):
    equity_delivery: DeliveryCosts
    equity_intraday: IntradayCosts
    slippage: Slippage
    rounding: Rounding


# ---- tax.yaml (spec 2.4) ----


class TaxConfig(_Strict):
    swing_treatment: Literal["capital_gains", "business_income"] = "capital_gains"
    stcg_pct: float
    ltcg_pct: float
    ltcg_exemption_inr: int
    long_term_months: int
    capital_loss_carry_forward_years: int
    speculative_loss_carry_forward_years: int


# ---- app.yaml ----


class LLMConfig(_Strict):
    model: str
    knowledge_cutoff: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    cutoff_buffer_days: int = Field(ge=0)
    num_ctx: int = Field(ge=512)
    keep_alive: str


class UIConfig(_Strict):
    host: Literal["127.0.0.1"]  # UI binds to localhost only (spec 14)
    port: int = Field(ge=1024, le=65535)


class KiteConfig(_Strict):
    redirect_port: int = Field(ge=1024, le=65535)


class AppConfig(_Strict):
    timezone: Literal["Asia/Kolkata"]
    universe: str
    llm: LLMConfig
    rss_feeds: list[str]
    broker: Literal["paper", "kite"] = "paper"
    manual_execution: Literal["kite_app", "api_approve"]
    ui: UIConfig
    kite: KiteConfig


class StrategyConfig(BaseModel):
    """Strategy files carry extra, strategy-specific parameters added in later milestones."""

    model_config = ConfigDict(extra="allow", frozen=True)

    name: str
    enabled: bool = False


class Settings(_Strict):
    app: AppConfig
    risk: RiskConfig
    costs: CostsConfig
    tax: TaxConfig
    strategies: dict[str, StrategyConfig]


def _load_yaml(path: Path) -> dict[str, object]:
    try:
        data = yaml.safe_load(path.read_text())
    except FileNotFoundError as exc:
        raise ConfigError(f"missing config file: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping")
    return data


def load_settings(directory: Path | None = None) -> Settings:
    """Load and validate every config file. Raises ConfigError on any problem."""
    base = directory or config_dir()
    strategies_dir = base / "strategies"
    strategies: dict[str, object] = {}
    if strategies_dir.is_dir():
        for path in sorted(strategies_dir.glob("*.yaml")):
            strategies[path.stem] = _load_yaml(path)
    try:
        return Settings.model_validate(
            {
                "app": _load_yaml(base / "app.yaml"),
                "risk": _load_yaml(base / "risk.yaml"),
                "costs": _load_yaml(base / "costs.yaml"),
                "tax": _load_yaml(base / "tax.yaml"),
                "strategies": strategies,
            }
        )
    except ValueError as exc:  # pydantic.ValidationError subclasses ValueError
        raise ConfigError(f"invalid configuration: {exc}") from exc


def require_ready(settings: Settings) -> None:
    """Startup gate (spec 7.2): refuse to run without a capital base."""
    if settings.risk.capital_base_inr <= 0:
        raise ConfigError(
            "risk.capital_base_inr is 0. Set it in config/risk.yaml before starting; "
            "the system refuses to run without a capital base."
        )
