"""Ledger page: searchable fills; no edits, only the 'add correction' flow (spec 11.4)."""

import streamlit as st

from tradepartner.ledger import corrections
from tradepartner.ui import data
from tradepartner.ui.common import get_session

st.title("Ledger")
session = get_session()

c1, c2, c3 = st.columns(3)
origins = c1.multiselect("Origin", data.ORIGINS)
products = c2.multiselect("Product", ["CNC", "MIS"])
symbol = c3.text_input("Symbol contains")
df = data.ledger_frame(session, origins, products, symbol=symbol)
st.caption(f"{len(df)} fills. Fills are append-only: corrections add a reversing and a new row.")
st.dataframe(df, use_container_width=True, hide_index=True)

with st.expander("Add correction"):
    fill_id = st.number_input("Fill id", min_value=1, step=1)
    qty = st.number_input("Corrected quantity", min_value=1, step=1)
    price = st.number_input("Corrected price (INR)", min_value=0.01, step=0.05, format="%.2f")
    reason = st.text_input("Reason (required)")
    if st.button("Add correction"):
        try:
            corrections.add_correction(session, int(fill_id), int(qty), round(price * 100), reason)
            st.success("Correction rows appended. Run `tradepartner ledger rebuild`.")
        except ValueError as exc:
            st.error(str(exc))
