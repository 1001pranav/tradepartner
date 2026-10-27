"""Mode state machine (spec 7.1): every transition rule is tested."""

from datetime import date

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from tests.fixtures.market import dt
from tradepartner.db.models import AuditLog, ModeState
from tradepartner.modes.controller import (
    ModeController,
    ModeError,
    UnlockEvidence,
    unlock_blockers,
)
from tradepartner.modes.state import GLOBAL, ModeReader, restrictive, strategy_scope

NOW = dt(2026, 6, 1, 9, 0)
S = strategy_scope("swing_v1")
GOOD = UnlockEvidence(True, 40, date(2026, 3, 1), True)


@pytest.fixture
def ctl(db: Session) -> ModeController:
    return ModeController(db)


def test_unknown_scope_defaults_to_off_and_locked(ctl: ModeController) -> None:
    row = ctl.current(S)
    assert (row.mode, row.auto_unlocked) == ("OFF", False)
    assert ctl.effective("swing_v1") == "OFF" and not ctl.allows("swing_v1")


def test_effective_is_the_more_restrictive_of_global_and_strategy(ctl: ModeController) -> None:
    assert restrictive("OFF", "AUTO") == "OFF" and restrictive("AUTO", "MANUAL") == "MANUAL"
    assert restrictive("AUTO", "AUTO") == "AUTO"
    ctl.set_mode(GLOBAL, "MANUAL", "user", "x", NOW)
    ctl.set_mode(S, "MANUAL", "user", "x", NOW)
    assert ctl.effective("swing_v1") == "MANUAL"
    ctl.set_mode(GLOBAL, "OFF", "user", "x", NOW)
    assert ctl.effective("swing_v1") == "OFF"


def test_allows_semantics(ctl: ModeController) -> None:
    assert not ctl.allows("swing_v1", "ENTRY") and not ctl.allows("swing_v1", "EXIT")
    assert ctl.allows("swing_v1", "EXIT", protective=True)  # protective exits run in OFF
    ctl.set_mode(GLOBAL, "MANUAL", "user", "x", NOW)
    ctl.set_mode(S, "MANUAL", "user", "x", NOW)
    assert ctl.allows("swing_v1", "ENTRY")


def test_auto_requires_unlock_for_strategies(ctl: ModeController) -> None:
    with pytest.raises(ModeError, match="not unlocked"):
        ctl.set_mode(S, "AUTO", "user", "x", NOW)
    ctl.unlock_auto("swing_v1", GOOD, "swing_v1", NOW)
    ctl.set_mode(S, "AUTO", "user", "go", NOW)
    assert ctl.current(S).mode == "AUTO" and ctl.current(S).auto_unlocked


def test_global_auto_is_one_click_but_user_only(ctl: ModeController) -> None:
    ctl.set_mode(GLOBAL, "AUTO", "user", "session start", NOW)
    assert ctl.current(GLOBAL).mode == "AUTO"
    ctl.set_mode(GLOBAL, "OFF", "system", "halt", NOW)
    with pytest.raises(ModeError, match="only lower"):
        ctl.set_mode(GLOBAL, "AUTO", "system", "sneaky", NOW)


def test_system_can_only_lower_never_raise(ctl: ModeController) -> None:
    ctl.set_mode(GLOBAL, "MANUAL", "user", "x", NOW)
    with pytest.raises(ModeError, match="only lower"):
        ctl.set_mode(GLOBAL, "MANUAL", "system", "same", NOW)
    ctl.set_mode(GLOBAL, "OFF", "system", "lower", NOW)
    with pytest.raises(ModeError, match="only lower"):
        ctl.set_mode(GLOBAL, "MANUAL", "system", "raise", NOW)


def test_input_validation(ctl: ModeController) -> None:
    with pytest.raises(ModeError, match="scope"):
        ctl.set_mode("strategy:", "MANUAL", "user", "x", NOW)
    with pytest.raises(ModeError, match="scope"):
        ctl.set_mode("everything", "MANUAL", "user", "x", NOW)
    with pytest.raises(ModeError, match="actor"):
        ctl.set_mode(GLOBAL, "MANUAL", "robot", "x", NOW)
    with pytest.raises(ModeError, match="unknown mode"):
        ctl.set_mode(GLOBAL, "TURBO", "user", "x", NOW)  # type: ignore[arg-type]
    with pytest.raises(ModeError, match="scope"):
        ctl.halt("nope", "x", NOW)


def test_system_never_starts_in_auto(db: Session, ctl: ModeController) -> None:
    ctl.set_mode(GLOBAL, "AUTO", "user", "x", NOW)
    ctl.unlock_auto("swing_v1", GOOD, "swing_v1", NOW)
    ctl.set_mode(S, "AUTO", "user", "x", NOW)
    ctl.set_mode(strategy_scope("positional_v1"), "MANUAL", "user", "x", NOW)
    # a brand-new controller == a process restart
    restarted = ModeController(db)
    clamped = restarted.startup(NOW)
    assert sorted(clamped) == sorted([GLOBAL, S])
    assert restarted.current(GLOBAL).mode == "MANUAL" and restarted.current(S).mode == "MANUAL"
    assert restarted.current(
        S
    ).auto_unlocked  # unlock survives; AUTO must be re-enabled by the user
    assert restarted.current(strategy_scope("positional_v1")).mode == "MANUAL"
    assert restarted.startup(NOW) == []  # idempotent
    assert all(r.mode != "AUTO" for r in (restarted.current(s) for s in restarted.scopes()))


def test_startup_keeps_off_as_off(ctl: ModeController) -> None:
    ctl.set_mode(GLOBAL, "MANUAL", "user", "x", NOW)
    ctl.set_mode(GLOBAL, "OFF", "user", "x", NOW)
    ctl.startup(NOW)
    assert ctl.current(GLOBAL).mode == "OFF"


def test_halt_forces_off_and_only_user_can_reset(ctl: ModeController) -> None:
    ctl.set_mode(GLOBAL, "AUTO", "user", "x", NOW)
    assert ctl.halt(GLOBAL, "daily loss", NOW) is True
    row = ctl.current(GLOBAL)
    assert row.mode == "OFF" and row.changed_by == "system"
    assert ctl.halt(GLOBAL, "again", NOW) is False  # already OFF
    with pytest.raises(ModeError, match="only lower"):
        ctl.set_mode(GLOBAL, "MANUAL", "system", "auto-recover", NOW)
    with pytest.raises(ModeError, match="RESET"):
        ctl.reset_halt("reset", NOW)
    ctl.reset_halt("RESET", NOW)
    assert ctl.current(GLOBAL).mode == "MANUAL"  # never AUTO after a reset
    with pytest.raises(ModeError, match="not halted"):
        ctl.reset_halt("RESET", NOW)


def test_unlock_requirements() -> None:
    assert unlock_blockers(GOOD, date(2026, 6, 1)) == []
    assert unlock_blockers(GOOD, date(2026, 4, 30)) != []  # < 2 months from Mar 1
    assert unlock_blockers(GOOD, date(2026, 5, 1)) == []  # exactly 2 months
    bad = UnlockEvidence(False, 39, None, False)
    assert len(unlock_blockers(bad, date(2026, 6, 1))) == 4
    assert any("39" in b for b in unlock_blockers(bad, date(2026, 6, 1)))


def test_unlock_needs_typed_name_and_full_evidence(ctl: ModeController) -> None:
    with pytest.raises(ModeError, match="type the strategy name"):
        ctl.unlock_auto("swing_v1", GOOD, "swing", NOW)
    for field_value in (
        UnlockEvidence(False, 40, date(2026, 3, 1), True),
        UnlockEvidence(True, 39, date(2026, 3, 1), True),
        UnlockEvidence(True, 40, date(2026, 5, 2), True),
        UnlockEvidence(True, 40, date(2026, 3, 1), False),
    ):
        with pytest.raises(ModeError, match="cannot unlock AUTO"):
            ctl.unlock_auto("swing_v1", field_value, "swing_v1", NOW)
    assert not ctl.current(S).auto_unlocked


def test_lock_auto_and_downgrade(ctl: ModeController) -> None:
    ctl.unlock_auto("swing_v1", GOOD, "swing_v1", NOW)
    ctl.set_mode(S, "AUTO", "user", "x", NOW)
    assert ctl.downgrade_to_manual("swing_v1", "drift", NOW) is True
    assert ctl.current(S).mode == "MANUAL" and ctl.current(S).auto_unlocked
    assert ctl.downgrade_to_manual("swing_v1", "drift", NOW) is False
    ctl.set_mode(S, "AUTO", "user", "x", NOW)
    ctl.lock_auto("swing_v1", "user", "locking", NOW)
    assert ctl.current(S).mode == "MANUAL" and not ctl.current(S).auto_unlocked
    with pytest.raises(ModeError, match="not unlocked"):
        ctl.set_mode(S, "AUTO", "user", "x", NOW)
    ctl.set_mode(S, "OFF", "user", "x", NOW)
    ctl.lock_auto("swing_v1", "user", "again", NOW)
    assert ctl.current(S).mode == "OFF"


def test_every_transition_is_audited(db: Session, ctl: ModeController) -> None:
    ctl.set_mode(GLOBAL, "MANUAL", "user", "a", NOW)
    ctl.set_mode(GLOBAL, "AUTO", "user", "b", NOW)
    ctl.halt(GLOBAL, "c", NOW)
    rows = db.scalar(select(func.count()).select_from(ModeState))
    audits = db.scalar(select(func.count()).select_from(AuditLog))
    assert rows == audits == 3
    assert len(ctl.history()) == 3 and ctl.history(1)[0].mode == "OFF"
    assert ModeReader(db).scopes() == [GLOBAL]
