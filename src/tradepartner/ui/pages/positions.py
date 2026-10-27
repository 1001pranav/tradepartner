"""Positions page (spec 11.3): open lots with origin and protective-order status."""

import pandas as pd
import streamlit as st
from sqlalchemy import select

from tradepartner.db.models import Fill, Lot
from tradepartner.ui import data
from tradepartner.ui.common import get_session
from tradepartner.ui.data import rupees

st.title("Positions")
session = get_session()
status = data.protective_status(session)
rows = []
for lot in session.scalars(select(Lot).order_by(Lot.symbol, Lot.open_date)):
    fill = session.get(Fill, lot.open_fill_id)
    ps = status.get(lot.symbol, "unknown (checked every 5 minutes by the scheduler)")
    rows.append(
        {
            "symbol": lot.symbol,
            "qty": lot.qty_remaining,
            "avg cost": rupees(lot.cost_paise_per_share),
            "opened": str(lot.open_date),
            "origin": fill.origin if fill else "?",
            "product": fill.product if fill else "?",
            "protective order": ps,
            "adjusted for CA": lot.adjusted_for_ca,
        }
    )
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
missing = [
    r["symbol"] for r in rows if str(r["protective order"]).startswith(("MISSING", "unknown"))
]
if rows and missing:
    st.warning(f"No confirmed protective order for: {', '.join(sorted(set(map(str, missing))))}")

if rows:
    st.subheader("Set a missing GTT yourself")
    sym = st.selectbox("Symbol", sorted({str(r["symbol"]) for r in rows}))
    st.caption(
        "Creates or fixes the GTT OCO for this position (needs the Kite broker and a registered static IP)."
    )
    if st.button("Place GTT"):
        from tradepartner.appctx import build_job_context
        from tradepartner.calendar import utc_now
        from tradepartner.execution.protective import ProtectiveManager
        from tradepartner.ui.common import get_settings

        try:
            rt = build_job_context(get_settings()).runtime(session)
            pm = ProtectiveManager(session, rt.broker, lambda s: None, lambda t, m: None, utc_now)
            st.info(f"{sym}: {pm.place_for(sym)}")
        except Exception as exc:
            st.error(f"Could not place the GTT: {exc}")
