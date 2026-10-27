"""Modes page (spec 11.2): global and per-strategy modes, AUTO unlock flow, history."""

import pandas as pd
import streamlit as st

from tradepartner.calendar import fmt_ist, utc_now
from tradepartner.modes.controller import ModeController, ModeError
from tradepartner.modes.evidence import collect_unlock_evidence
from tradepartner.modes.state import GLOBAL, strategy_scope
from tradepartner.ui.common import get_session, get_settings

st.title("Modes")
session = get_session()
ctl = ModeController(session)
settings = get_settings()
st.caption(
    "Effective mode is the more restrictive of global and strategy. The system never starts in "
    "AUTO; a halt forces OFF and only you can raise it."
)


def mode_row(scope: str, label: str) -> None:
    row = ctl.current(scope)
    options = ["OFF", "MANUAL"] + (["AUTO"] if scope == GLOBAL or row.auto_unlocked else [])
    cols = st.columns([2, 2, 2])
    cols[0].markdown(
        f"**{label}**  \nCurrent: `{row.mode}`"
        + (
            ""
            if scope == GLOBAL
            else ("  \nAUTO unlocked" if row.auto_unlocked else "  \nAUTO locked")
        )
    )
    choice = cols[1].selectbox("Set to", options, index=options.index(row.mode), key=f"sel-{scope}")
    if cols[2].button("Apply", key=f"apply-{scope}") and choice != row.mode:
        try:
            ctl.set_mode(scope, choice, "user", "UI", utc_now())  # type: ignore[arg-type]
            st.rerun()
        except ModeError as exc:
            st.error(str(exc))


mode_row(GLOBAL, "Global")
for name in sorted(settings.strategies):
    mode_row(strategy_scope(name), name)

st.subheader("Unlock AUTO")
target = st.selectbox("Strategy", sorted(settings.strategies), key="unlock-target")
ev = collect_unlock_evidence(session, target)
st.write(
    f"Approved model: **{ev.model_approved}** | closed trades: **{ev.closed_trades}** (need 40) | "
    f"first trade: **{ev.first_trade_date}** (need 2 months) | gate passed: **{ev.gate_passed}**"
)
typed = st.text_input("Type the strategy name to confirm")
if st.button("Unlock AUTO"):
    try:
        ctl.unlock_auto(target, ev, typed, utc_now())
        st.success(f"AUTO unlocked for {target}.")
    except ModeError as exc:
        st.error(str(exc))

st.subheader("History")
hist = pd.DataFrame(
    [
        {
            "at (IST)": fmt_ist(h.changed_at),
            "scope": h.scope,
            "mode": h.mode,
            "by": h.changed_by,
            "reason": h.reason,
        }
        for h in ctl.history(100)
    ]
)
st.dataframe(hist, hide_index=True, use_container_width=True)
