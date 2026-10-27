"""A full daily-pipeline environment on synthetic data (no network, no real broker, no Ollama)."""

import threading
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from tests.fixtures.llm import FakeOllama
from tests.fixtures.market import dt
from tests.fixtures.synth import bdays, settings_with_capital, synth_view
from tests.unit.test_cli_m4 import store_view
from tradepartner.calendar import TradingCalendar
from tradepartner.data.candles import CandleStore
from tradepartner.data.quotes import QuoteBook
from tradepartner.db.models import Model, UniverseMember
from tradepartner.jobs import JobContext
from tradepartner.models import pipeline, registry
from tradepartner.settings import StrategyConfig

_NEXT = bdays(date(2023, 1, 2), 761)[
    -1
]  # the first weekday after the last synthetic bar (n_days=760)
NOW0 = datetime(_NEXT.year, _NEXT.month, _NEXT.day, 8, 50, tzinfo=dt(2026, 1, 1).tzinfo)


def fast_settings():  # type: ignore[no-untyped-def]
    s = settings_with_capital()
    cfg = {
        **s.strategies["swing_v1"].model_dump(),
        "n_trials": 1,
        "min_train_days": 300,
        "min_prob": 0.0,
    }
    return s.model_copy(update={"strategies": {**s.strategies, "swing_v1": StrategyConfig(**cfg)}})


EXPLAIN_FALLBACK = {"explanation": "The model likes this setup because of momentum."}


class FakeKite:
    """The slice of KiteConnect the jobs use, served from memory."""

    def __init__(self, view: Any, price_moves: dict[str, float] | None = None) -> None:
        self.view = view
        self.moves = price_moves or {}
        self.quote_calls = 0
        self._orders: list[dict[str, Any]] = []
        self._trades: list[dict[str, Any]] = []

    def instruments(self, exchange: str | None = None) -> list[dict[str, Any]]:
        return [
            {"instrument_token": 100 + i, "exchange": "NSE", "segment": "NSE", "instrument_type": "EQ",
             "tradingsymbol": s, "name": f"{s} LTD", "tick_size": 0.05}
            for i, s in enumerate(self.view.symbols)
        ]  # fmt: skip

    def quote(self, instruments: list[str]) -> dict[str, dict[str, Any]]:
        self.quote_calls += 1
        out: dict[str, dict[str, Any]] = {}
        for key in instruments:
            sym = key.split(":")[1]
            last = float(self.view.history(sym, self.view.trading_days()[-1])["close"].iloc[-1])
            px = round(last * (1 + self.moves.get(sym, 0.0)), 2)
            out[key] = {"last_price": px, "ohlc": {"close": last}, "timestamp": NOW0.replace(hour=9, minute=20),
                        "upper_circuit_limit": last * 1.2, "lower_circuit_limit": last * 0.8}  # fmt: skip
        return out

    def historical_data(self, token: int, a: date, b: date, interval: str) -> list[dict[str, Any]]:
        return []

    def profile(self) -> dict[str, str]:
        return {"user_id": "AB1234"}

    def orders(self) -> list[dict[str, Any]]:
        return self._orders

    def trades(self) -> list[dict[str, Any]]:
        return self._trades

    def positions(self) -> dict[str, list[dict[str, Any]]]:
        return {"net": []}

    def holdings(self) -> list[dict[str, Any]]:
        return []


@dataclass
class Env:
    db: Session
    ctx: JobContext
    view: Any
    kite: FakeKite
    ollama: FakeOllama
    notes: list[tuple[str, str]] = field(default_factory=list)
    clock_now: datetime = NOW0

    def set_clock(self, hh: int, mm: int, day: date | None = None) -> None:
        d = day or NOW0.date()
        self.clock_now = datetime(d.year, d.month, d.day, hh, mm, tzinfo=NOW0.tzinfo)


@dataclass
class TrainedArtifact:
    path: str
    metrics: dict[str, Any]
    params: dict[str, Any]
    train_start: date
    train_end: date


def train_once(root: Path) -> TrainedArtifact:
    """Train a real (tiny) swing model once per test module."""
    from tests.unit.test_cli_m4 import synth_view as _sv  # noqa: F401
    from tradepartner.db.migrate import upgrade
    from tradepartner.db.session import make_engine, make_session_factory

    view = synth_view(n_symbols=8, n_days=760, seed=21, momentum=0.25)
    url = f"sqlite:///{root / 'train.sqlite'}"
    upgrade(url)
    with make_session_factory(make_engine(url))() as s:
        out = pipeline.train_strategy(
            s, "swing_v1", view, fast_settings(), NOW0.astimezone(UTC), models_root=root / "m"
        )
        m = registry.get(s, out.model_id)  # type: ignore[arg-type]
        return TrainedArtifact(
            m.artifact_path, dict(m.metrics_json), dict(m.params_json), m.train_start, m.train_end
        )


def add_model(
    db: Session, art: TrainedArtifact, status: str = "approved", strategy: str = "swing_v1"
) -> Model:
    metrics = {**art.metrics, "gate": {**art.metrics.get("gate", {}), "ready_for_review": True}}
    m = Model(strategy=strategy, version=f"t{status}{len(list(db.query(Model)))}", algo="lightgbm", feature_set_version="fs1",
              train_start=art.train_start, train_end=art.train_end, params_json=art.params, metrics_json=metrics,
              artifact_path=art.path, status=status)  # fmt: skip
    db.add(m)
    db.commit()
    return m


def make_env(db: Session, tmp_path: Path, settings: Any = None, broker_factory: Any = None) -> Env:
    from tradepartner import cli  # noqa: F401  (ensures migrations exist)

    settings = settings or fast_settings()
    view = synth_view(n_symbols=8, n_days=760, seed=21, momentum=0.25)
    store = CandleStore()
    store_view(store, view)
    for sym in view.symbols:
        db.add(UniverseMember(universe_name="nifty100", symbol=sym, valid_from=date(2000, 1, 1)))
    db.commit()
    kite = FakeKite(view)
    ollama = FakeOllama(default=EXPLAIN_FALLBACK)
    env = Env(db, None, view, kite, ollama)  # type: ignore[arg-type]
    http_llm = ollama.client()
    from tradepartner.db.session import make_session_factory

    factory = make_session_factory(db.get_bind())
    env.ctx = JobContext(
        settings=settings, session_factory=factory, store=store, quotes=QuoteBook(), kite=lambda: kite,
        llm=lambda: http_llm, clock=lambda: env.clock_now.astimezone(UTC),
        notifier=lambda t, m: env.notes.append((t, m)),
        calendar=TradingCalendar(extra_trading_days=frozenset(view.trading_days())),
        broker_factory=broker_factory, internet_check=lambda: True, heavy=threading.Lock(),
    )  # fmt: skip
    return env


__all__ = ["NOW0", "Env", "FakeKite", "TrainedArtifact", "add_model", "make_env", "train_once"]
_ = (pd, timedelta)
