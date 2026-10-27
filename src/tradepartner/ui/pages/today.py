"""Today page (spec 11.1): status, kill switch and the proposals table with explanations."""

import streamlit as st

from tradepartner.appctx import build_job_context
from tradepartner.calendar import IST, utc_now
from tradepartner.data import kite_auth
from tradepartner.db.models import NewsScore
from tradepartner.modes.state import GLOBAL, ModeReader
from tradepartner.risk import kill
from tradepartner.runtime import build_runtime
from tradepartner.signals import proposals
from tradepartner.ui import data
from tradepartner.ui.common import get_session, get_settings

st.title("Today")
session = get_session()
settings = get_settings()
now = utc_now()
today = now.astimezone(IST).date()
modes = ModeReader(session)

c1, c2, c3, c4 = st.columns(4)
mode = modes.current(GLOBAL).mode
c1.metric("Global mode", mode)
c2.metric("Kite login", "yes" if kite_auth.load_access_token(today) else "NO")
from sqlalchemy import func, select  # noqa: E402

scored_today = session.scalar(select(func.count()).select_from(NewsScore)) or 0
c3.metric("News scored (all time)", scored_today)
c4.metric("Kill file", "ARMED" if kill.kill_active() else "off")

with st.expander("Kill switch"):
    phrase = st.text_input("Type KILL to confirm", key="kill-phrase")
    if st.button("KILL SWITCH", type="primary"):
        if phrase != "KILL":
            st.error("Type KILL to confirm.")
        else:
            kill.arm()
            rep = build_runtime(session, settings).kill.trigger(utc_now(), "UI kill")
            st.warning(
                f"Global OFF. Cancelled {len(rep.cancelled)}; squared off {rep.squared_off or 'nothing'}."
            )

rows = data.today_rows(session, today)
st.subheader(f"Signals and proposals ({len(rows)})")
manual_kite = settings.app.manual_execution == "kite_app"
if not rows:
    st.info("No signals yet today. They are generated at 08:50 and become actionable at 09:20.")
for row in rows:
    head = f"{row['symbol']} | {row['strategy']} | {row['status']} | decision: {row['decision'] or 'pending'}"
    with st.expander(head):
        st.write(
            f"Entry {int(row['entry']) / 100:.2f}  Stop {int(row['stop']) / 100:.2f}  "  # type: ignore[call-overload]
            f"Target {int(row['target']) / 100:.2f}  Qty {row['qty'] or '-'}  "  # type: ignore[call-overload]
            f"Risk {(int(row['risk'] or 0)) / 100:.2f}  Model score {row['prob']}"  # type: ignore[call-overload]
        )
        st.markdown(f"**Why:** {row['explanation'] or 'shadow signal (not shown)'}")
        if row["reasons"]:
            st.error("Risk engine: " + "; ".join(str(r) for r in row["reasons"]))  # type: ignore[attr-defined]
        if row["decision"] == "approved" and row["status"] in ("shown", "approved"):
            if manual_kite:
                if st.button("Copy order for Kite", key=f"copy-{row['signal_id']}"):
                    st.code(data.kite_order_text(row), language="text")
            else:
                a, r = st.columns(2)
                sid = int(row["signal_id"])  # type: ignore[call-overload]
                if a.button("Approve (send via API)", key=f"approve-{sid}"):
                    try:
                        rt = build_job_context(settings).runtime(session)
                        res = rt.router.approve_and_place(int(row["proposal_id"]), utc_now())  # type: ignore[call-overload]
                        if res.order is None:
                            why = res.decision.reasons if res.decision else []
                            st.error(f"Not placed ({res.status}): {why}")
                            if res.status == "rejected":  # the fresh re-check refused it
                                proposals.record_event(
                                    session, sid, "rejected_by_risk", str(why), utc_now()
                                )
                        else:
                            proposals.record_event(
                                session, sid, "approved", "approved by user", utc_now()
                            )
                            st.success(f"Order placed: {res.order.broker_order_id}")
                    except Exception as exc:  # shown to the user, never swallowed silently
                        st.error(f"Not placed: {exc}")
                if r.button("Reject", key=f"reject-{sid}"):
                    proposals.ignore(session, sid, utc_now())
                    st.rerun()
        if row["status"] in ("new", "shown") and st.button("Ignore", key=f"ign-{row['signal_id']}"):
            proposals.ignore(session, int(row["signal_id"]), utc_now())  # type: ignore[call-overload]
            st.rerun()
