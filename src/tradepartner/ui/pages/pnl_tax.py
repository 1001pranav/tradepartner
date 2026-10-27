"""P&L and Tax page (spec 11.5): FY selector, categories, charges, advance-tax helper, export."""

from datetime import date

import streamlit as st

from tradepartner.calendar import utc_now
from tradepartner.ledger import reconcile
from tradepartner.paths import reports_dir
from tradepartner.reports import tax_export
from tradepartner.ui.common import get_session, get_settings
from tradepartner.ui.data import rupees

st.title("P&L and Tax")
session = get_session()
cfg = get_settings().tax
this_fy = tax_export.today_fy(utc_now().date())
start_year = int(this_fy[:4])
fys = [f"{y}-{str(y + 1)[-2:]}" for y in range(start_year, start_year - 6, -1)]
fy = st.selectbox("Financial year", fys)
rep = tax_export.build_report(session, fy, cfg)

labels = {
    "intraday_spec": "Intraday (speculative)",
    "stcg": "STCG",
    "ltcg": "LTCG",
    "business": "Business income",
    "unmatched": "Unmatched (excluded)",
}
cols = st.columns(len(labels))
for col, (key, label) in zip(cols, labels.items(), strict=True):
    col.metric(label, rupees(rep.totals[key]))

st.subheader("Quarterly realized (advance tax helper)")
st.dataframe(rep.quarterly, hide_index=True, use_container_width=True)
st.subheader("Charges by component")
st.dataframe(rep.charges, hide_index=True, use_container_width=True)
st.subheader("Loss set-off and LTCG exemption (for CA review)")
for note in rep.setoff.notes or ("No set-off needed.",):
    st.write(f"- {note}")
st.write(
    f"LTCG exemption used: {rupees(rep.setoff.ltcg_exemption_used)}; "
    f"taxable LTCG: {rupees(rep.setoff.ltcg_taxable)}."
)

st.subheader("Reconciliation")
st.write(
    "BLOCKED: unresolved mismatch (blocks AUTO entries)."
    if reconcile.has_reconciliation_block(session)
    else "No unresolved reconciliation differences."
)
if st.button("Export CSV + XLSX"):
    csv_path, xlsx_path = tax_export.export(session, fy, cfg, reports_dir())
    st.success(f"Wrote {csv_path} and {xlsx_path}")
st.caption(f"Treatment for swing trades: {cfg.swing_treatment}. Today: {date.today()}.")
