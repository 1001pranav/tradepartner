"""Findings 2 and 5: stop upkeep and protective exits must not depend on the mode."""

from types import SimpleNamespace

from sqlalchemy.orm import Session

from tests.fixtures.risk import LTP
from tests.fixtures.world import STRATEGY, World
from tradepartner import jobs
from tradepartner.db.models import Fill
from tradepartner.execution.broker import GttInfo
from tradepartner.execution.protective import EXIT_STATUSES, ProtectiveManager
from tradepartner.modes.state import GLOBAL, strategy_scope


def _off(w: World) -> None:
    w.controller.set_mode(strategy_scope(STRATEGY), "OFF", "user", "halt", w.now)
    w.controller.set_mode(GLOBAL, "OFF", "user", "halt", w.now)


def _bought(w: World) -> int:
    w.go_auto()
    r = w.submit_entry(key="k1")
    assert r.status == "placed"
    w.context_for(STRATEGY)  # rebuild lots
    return r.decision.sizing.qty  # type: ignore[union-attr]


def _pm(w: World, last: int) -> ProtectiveManager:
    return ProtectiveManager(w.db, w.broker, lambda s: last, lambda t, m: None, lambda: w.now)


def test_exit_limit_is_integer_paise_on_the_tick() -> None:
    assert jobs._exit_limit_paise(150_000, 20) == 149_700
    assert jobs._exit_limit_paise(150_123, 50) % 5 == 0
    assert isinstance(jobs._exit_limit_paise(99_999), int)


def test_missing_gtt_is_recreated_when_allowed_regardless_of_mode(db: Session) -> None:
    w = World(db)
    qty = _bought(w)
    _off(w)
    report = _pm(w, LTP).check(True)  # what job_protective_check now passes
    assert report == {"INFY": "CREATED"} and w.broker.list_gtts()[0].qty == qty


def test_triggered_gtt_with_position_still_held_is_flagged_for_exit(db: Session) -> None:
    w = World(db)
    _bought(w)
    w.broker.list_gtts = lambda: [GttInfo("g1", "INFY", 1, 1, 2, "triggered")]  # type: ignore[method-assign]
    assert _pm(w, LTP).check(True) == {"INFY": "TRIGGERED"}
    assert "TRIGGERED" in EXIT_STATUSES and "BREACHED" in EXIT_STATUSES


def test_breached_position_is_sold_by_a_protective_exit_in_off_mode(db: Session) -> None:
    w = World(db)
    qty = _bought(w)
    _off(w)
    w.advance(minutes=5)
    w.set_price("INFY", int(LTP * 0.9))
    ctx = SimpleNamespace(clock=lambda: w.now)
    rt = SimpleNamespace(router=w.router, broker=w.broker)
    assert jobs._protective_exit(
        ctx, rt, "INFY", "CNC", qty, STRATEGY, "stop", w.quotes["INFY"].ltp_paise
    )  # type: ignore[arg-type]
    sells = [f for f in db.query(Fill).all() if f.side == "SELL"]
    assert sum(f.qty for f in sells) == qty
