"""Mode controller: the OFF / MANUAL / AUTO state machine (spec 7.1).

Callers of mutating methods are restricted by import-linter to ``cli``, ``ui`` and
``risk.engine`` (for halts). Strategy, model and LLM code can never reach this module.
Every transition is written to ``mode_state`` and ``audit_log``.
"""

from dataclasses import dataclass
from datetime import date, datetime

from sqlalchemy.orm import Session

from tradepartner.calendar import add_months
from tradepartner.db import audit
from tradepartner.db.models import ModeState
from tradepartner.modes.state import (
    GLOBAL,
    RANK,
    Mode,
    ModeReader,
    restrictive,
    strategy_scope,
)

MIN_CLOSED_TRADES = 40
MIN_TRACKING_MONTHS = 2
# Intraday is the riskiest strategy: its AUTO unlock is separate and needs longer tracking (M9).
MIN_MONTHS_BY_STRATEGY = {"intraday_v1": 3}
RESET_PHRASE = "RESET"


class ModeError(Exception):
    """A transition the state machine does not allow."""


@dataclass(frozen=True)
class UnlockEvidence:
    """Everything spec 7.1 rule 2 requires before AUTO can be unlocked for a strategy."""

    model_approved: bool
    closed_trades: int
    first_trade_date: date | None
    gate_passed: bool


def unlock_blockers(ev: UnlockEvidence, today: date, strategy: str = "") -> list[str]:
    out: list[str] = []
    if not ev.model_approved:
        out.append("no approved model")
    if ev.closed_trades < MIN_CLOSED_TRADES:
        out.append(f"{ev.closed_trades} closed trades (need {MIN_CLOSED_TRADES})")
    months = MIN_MONTHS_BY_STRATEGY.get(strategy, MIN_TRACKING_MONTHS)
    if ev.first_trade_date is None or today < add_months(ev.first_trade_date, months):
        out.append(f"fewer than {months} calendar months of tracking")
    if not ev.gate_passed:
        out.append("paper/manual metrics do not pass the promotion gate")
    return out


class ModeController(ModeReader):
    def __init__(self, session: Session) -> None:
        super().__init__(session)

    # ---- internals ----

    def _write(
        self,
        scope: str,
        mode: Mode,
        unlocked: bool,
        actor: str,
        reason: str,
        now: datetime,
        action: str,
    ) -> None:
        prev = self.current(scope)
        self.session.add(
            ModeState(
                scope=scope,
                mode=mode,
                auto_unlocked=unlocked,
                changed_by=actor,
                reason=reason,
                changed_at=now,
            )
        )
        audit.log(
            self.session,
            actor,
            action,
            {"scope": scope, "from": prev.mode, "to": mode, "unlocked": unlocked, "reason": reason},
            now,
        )
        self.session.commit()

    @staticmethod
    def _check_scope(scope: str) -> None:
        if scope != GLOBAL and not (scope.startswith("strategy:") and len(scope) > 9):
            raise ModeError(f"invalid scope {scope!r}")

    # ---- transitions ----

    def startup(self, now: datetime) -> list[str]:
        """Rule 1: never start in AUTO. Every scope becomes ``min(persisted, MANUAL)``."""
        clamped: list[str] = []
        for scope in self.scopes():
            cur = self.current(scope)
            if cur.mode == "AUTO":
                self._write(
                    scope, "MANUAL", cur.auto_unlocked, "system",
                    "startup: AUTO is never resumed", now, "mode_startup_clamp",
                )  # fmt: skip
                clamped.append(scope)
        return clamped

    def set_mode(self, scope: str, mode: Mode, actor: str, reason: str, now: datetime) -> None:
        self._check_scope(scope)
        if actor not in ("user", "system"):
            raise ModeError(f"unknown actor {actor!r}")
        if mode not in RANK:
            raise ModeError(f"unknown mode {mode!r}")
        cur = self.current(scope)
        if actor == "system" and RANK[mode] >= RANK[cur.mode]:
            raise ModeError("the system may only lower a mode; Pranav raises it")
        # AUTO is always a raise, so the rule above already restricts it to the user.
        if mode == "AUTO" and scope != GLOBAL and not cur.auto_unlocked:
            raise ModeError(f"AUTO is not unlocked for {scope}")
        self._write(scope, mode, cur.auto_unlocked, actor, reason, now, "mode_change")

    def halt(self, scope: str, reason: str, now: datetime) -> bool:
        """A halt risk event forces OFF immediately (rule 3). Returns False if already OFF."""
        self._check_scope(scope)
        cur = self.current(scope)
        if cur.mode == "OFF":
            return False
        self._write(scope, "OFF", cur.auto_unlocked, "system", f"HALT: {reason}", now, "mode_halt")
        return True

    def reset_halt(self, confirmation: str, now: datetime) -> None:
        """Pranav lifts a global halt: OFF -> MANUAL (never AUTO). Needs the typed phrase."""
        if confirmation != RESET_PHRASE:
            raise ModeError(f"type {RESET_PHRASE} to confirm")
        cur = self.current(GLOBAL)
        if cur.mode != "OFF":
            raise ModeError("global mode is not halted")
        self._write(GLOBAL, "MANUAL", cur.auto_unlocked, "user", "halt reset", now, "mode_reset")

    def unlock_auto(
        self, strategy: str, evidence: UnlockEvidence, confirmation: str, now: datetime
    ) -> None:
        """Rule 2: all evidence present, and the strategy name typed to confirm."""
        if confirmation != strategy:
            raise ModeError("type the strategy name exactly to confirm")
        blockers = unlock_blockers(evidence, now.date(), strategy)
        if blockers:
            raise ModeError("cannot unlock AUTO: " + "; ".join(blockers))
        scope = strategy_scope(strategy)
        cur = self.current(scope)
        self._write(scope, cur.mode, True, "user", "AUTO unlocked", now, "auto_unlock")

    def lock_auto(self, strategy: str, actor: str, reason: str, now: datetime) -> None:
        """Re-lock AUTO (drift, kill, manual). If the strategy is in AUTO it drops to MANUAL."""
        scope = strategy_scope(strategy)
        cur = self.current(scope)
        mode: Mode = restrictive(cur.mode, "MANUAL")
        self._write(scope, mode, False, actor, reason, now, "auto_lock")

    def downgrade_to_manual(self, strategy: str, reason: str, now: datetime) -> bool:
        """Drift monitor action (spec 10.6): AUTO -> MANUAL for one strategy."""
        scope = strategy_scope(strategy)
        cur = self.current(scope)
        if cur.mode != "AUTO":
            return False
        self._write(scope, "MANUAL", cur.auto_unlocked, "system", reason, now, "mode_downgrade")
        return True
