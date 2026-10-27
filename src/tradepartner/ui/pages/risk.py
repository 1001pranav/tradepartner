"""Risk page (spec 11.7): limit usage, risk events, halt reset. Config values are read-only."""

import pandas as pd
import streamlit as st
from sqlalchemy import select

from tradepartner.calendar import fmt_ist, utc_now
from tradepartner.db.models import RiskEvent
from tradepartner.modes.controller import ModeController, ModeError
from tradepartner.modes.state import GLOBAL
from tradepartner.risk import kill
from tradepartner.risk.limits import compute_limit_state
from tradepartner.runtime import build_runtime
from tradepartner.ui.common import get_session, get_settings

st.title("Risk")
session = get_session()
settings = get_settings()
risk = settings.risk
ctl = ModeController(session)

if risk.capital_base_inr <= 0:
    st.error("capital_base_inr is 0 in config/risk.yaml: the system will not run.")
else:
    state = compute_limit_state(session, risk, {}, utc_now(), kill.kill_active())
    cap = risk.capital_base_paise

    def gauge(label: str, used: int, limit_pct: float) -> None:
        limit = cap * limit_pct / 100
        frac = min(max(used / limit, 0.0), 1.0) if limit else 0.0
        st.write(
            f"**{label}**: {used / 100:,.0f} of {limit / 100:,.0f} INR ({limit_pct}% of capital)"
        )
        st.progress(frac)

    gauge("Daily loss", max(0, -state.daily_pnl_paise), risk.loss_limits.daily_loss_pct)
    gauge("Weekly loss", max(0, -state.weekly_pnl_paise), risk.loss_limits.weekly_loss_pct)
    dd_limit_pct = risk.loss_limits.max_drawdown_pct * state.peak_equity_paise / cap
    gauge("Drawdown from peak", max(0, state.peak_equity_paise - state.equity_paise), dd_limit_pct)
    st.write(f"**Orders today**: {state.orders_today} of {risk.rate_limits.max_orders_per_day}")
    st.progress(min(state.orders_today / risk.rate_limits.max_orders_per_day, 1.0))

st.subheader("Limits (read-only; edit config/risk.yaml by hand)")
st.json(risk.model_dump())

st.subheader("Risk events")
events = pd.DataFrame(
    [
        {
            "at (IST)": fmt_ist(e.at),
            "rule": e.rule,
            "severity": e.severity,
            "scope": e.scope,
            "details": str(e.details_json),
        }
        for e in session.scalars(select(RiskEvent).order_by(RiskEvent.id.desc()).limit(200))
    ]
)
st.dataframe(events, hide_index=True, use_container_width=True)

st.subheader("Halt reset")
st.write(f"Global mode: `{ctl.current(GLOBAL).mode}`")
phrase = st.text_input("Type RESET to lift a halt (goes to MANUAL, never AUTO)")
if st.button("Reset halt"):
    try:
        ctl.reset_halt(phrase, utc_now())
        st.success("Global mode is now MANUAL.")
    except ModeError as exc:
        st.error(str(exc))

st.subheader("Kill switch")
kill_phrase = st.text_input("Type KILL to confirm")
if st.button("KILL SWITCH", type="primary"):
    if kill_phrase != "KILL":
        st.error("Type KILL to confirm.")
    else:
        kill.arm()
        report = build_runtime(session, settings).kill.trigger(utc_now(), "UI kill")
        st.warning(
            f"Global OFF. Cancelled {len(report.cancelled)}; "
            f"squared off {report.squared_off or 'nothing'}."
        )
if kill.kill_active():
    st.error("Kill file is present. Delete ~/.tradepartner/KILL to allow trading again.")
