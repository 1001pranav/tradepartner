"""Read-only view of mode state. Importable by anything (the router, scheduler, UI).

Mutation lives in ``modes.controller``, which only ``cli``, ``ui`` and ``risk.engine`` may import.
"""

from dataclasses import dataclass
from typing import Literal, cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from tradepartner.db.models import ModeState

Mode = Literal["OFF", "MANUAL", "AUTO"]
RANK: dict[str, int] = {"OFF": 0, "MANUAL": 1, "AUTO": 2}
GLOBAL = "global"


def restrictive(a: str, b: str) -> Mode:
    """The more restrictive of two modes (OFF < MANUAL < AUTO)."""
    return cast(Mode, a if RANK[a] <= RANK[b] else b)


def strategy_scope(name: str) -> str:
    return f"strategy:{name}"


@dataclass(frozen=True)
class ModeRow:
    scope: str
    mode: Mode
    auto_unlocked: bool
    changed_by: str


DEFAULT = ("OFF", False)  # a scope with no history is OFF and locked


class ModeReader:
    def __init__(self, session: Session) -> None:
        self.session = session

    def current(self, scope: str) -> ModeRow:
        row = self.session.scalar(
            select(ModeState).where(ModeState.scope == scope).order_by(ModeState.id.desc()).limit(1)
        )
        if row is None:
            return ModeRow(scope, "OFF", False, "system")
        return ModeRow(scope, cast(Mode, row.mode), row.auto_unlocked, row.changed_by)

    def scopes(self) -> list[str]:
        return sorted(set(self.session.scalars(select(ModeState.scope))))

    def effective(self, strategy: str) -> Mode:
        """More restrictive of global and strategy mode."""
        return restrictive(self.current(GLOBAL).mode, self.current(strategy_scope(strategy)).mode)

    def allows(self, strategy: str, kind: str = "ENTRY", protective: bool = False) -> bool:
        """May this strategy produce an order of this kind right now?"""
        mode = self.effective(strategy)
        if mode == "OFF":
            return kind == "EXIT" and protective
        return True

    def history(self, limit: int = 50) -> list[ModeState]:
        return list(
            self.session.scalars(select(ModeState).order_by(ModeState.id.desc()).limit(limit))
        )
