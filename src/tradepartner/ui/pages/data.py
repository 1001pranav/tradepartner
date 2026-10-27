"""Data page: stored prices, quality warnings and the universe. Read-only except the buttons that
start background data jobs (market data download only; nothing here can place an order)."""

import pandas as pd
import streamlit as st

from tradepartner.calendar import TradingCalendar, utc_now
from tradepartner.data.candles import CandleStore
from tradepartner.data.corporate_actions import load_actions
from tradepartner.ui import configedit, data, jobs
from tradepartner.ui.common import get_session

st.title("Data")
session = get_session()
store = CandleStore()
if not store.symbols("day"):
    st.info("No candles yet. Run the backfill below (or `tradepartner data backfill`).")

cov = data.coverage(store)
c1, c2, c3 = st.columns(3)
c1.metric("Symbols with prices", len(cov))
c2.metric("First day", str(cov["first"].min()) if len(cov) else "-")
c3.metric("Last day", str(cov["last"].max()) if len(cov) else "-")


@st.cache_data(ttl=300, show_spinner="Checking data quality...")
def _quality(_stamp: str) -> pd.DataFrame:
    acts: dict[str, list] = {}  # type: ignore[type-arg]
    for a in load_actions(session):
        acts.setdefault(a.symbol, []).append(a)
    return data.quality_frame(store, TradingCalendar.from_config(), acts)


stamp = str(cov["last"].max()) + str(len(cov))
issues = _quality(stamp)

tab_issues, tab_symbol, tab_universe, tab_jobs, tab_fix = st.tabs(
    ["Quality issues", "Symbol explorer", "Universe", "Download", "Fix data"]
)

with tab_issues:
    st.caption(
        "Jumps of 20% or more in one day are flagged. Most are real events (crash, results, Budget "
        "day); only a few are missing splits or demergers. Gaps should be near zero once the "
        "holiday calendar is complete."
    )
    errors = int((issues["severity"] == "error").sum()) if len(issues) else 0
    st.write(f"{len(issues)} issues, **{errors} errors**.")
    kinds = sorted(issues["kind"].unique()) if len(issues) else []
    pick = st.multiselect("Kind", kinds, default=[k for k in kinds if k != "zero_volume"])
    sev = st.multiselect("Severity", ["error", "warn"], default=["error", "warn"])
    query = st.text_input("Symbol contains").upper()
    view = issues[issues["kind"].isin(pick) & issues["severity"].isin(sev)]
    if query:
        view = view[view["symbol"].str.contains(query)]
    st.dataframe(view, hide_index=True, use_container_width=True)
    if len(issues):
        st.bar_chart(issues.groupby("kind").size())
    st.subheader("Why did these jump?")
    st.caption(
        "The local LLM reads web headlines and NSE announcements around each date and names the "
        "reason. Annotation only: it changes no data. 'unknown' means no evidence explained it."
    )
    reasons = data.jump_reasons_frame()
    jump_n = int((issues["kind"] == "unexplained_jump").sum()) if len(issues) else 0
    st.write(f"{len(reasons)} of {jump_n} unexplained jumps have a reason.")
    batch = st.number_input("Explain this many now", 1, 500, 20)
    if st.button("Search the web and explain"):
        st.success(f"Started {jobs.launch('explain-jumps', {'limit': int(batch)}).id}")
    if len(reasons):
        st.dataframe(reasons, hide_index=True, use_container_width=True)

with tab_symbol:
    sym = st.selectbox("Symbol", sorted(store.symbols("day")))
    frame = data.symbol_frame(store, sym) if sym else pd.DataFrame()
    if len(frame):
        st.line_chart(frame["close"], height=260)
        st.bar_chart(frame["volume"], height=120)
        st.write(
            f"{len(frame)} bars, {frame.index.min()} to {frame.index.max()} (prices in rupees)."
        )
        mine = issues[issues["symbol"] == sym]
        if len(mine):
            st.warning("Quality issues for this symbol:")
            st.dataframe(mine, hide_index=True, use_container_width=True)
        rets = frame["close"].pct_change().abs()
        big = frame.loc[rets > 0.15, ["close"]].assign(move=rets[rets > 0.15].map("{:.1%}".format))
        if len(big):
            st.caption("Largest single-day moves (over 15%)")
            st.dataframe(big, use_container_width=True)
        st.dataframe(frame.tail(15), use_container_width=True)

with tab_universe:
    u = data.universe_summary(session, store)
    st.write(
        f"**{u.members}** symbols in the universe. {len(u.without_data)} have no stored prices."
    )
    if u.static_bias:
        st.warning(
            "Static membership (the current list applied to all history): backtests are survivorship-biased. See `data universe-import`."
        )
    if u.without_data:
        st.write("No prices: " + ", ".join(u.without_data))
    st.subheader("Excluded for data reasons")
    st.dataframe(
        pd.DataFrame(list(u.excluded.items()), columns=["symbol", "reason"]), hide_index=True
    )
    st.caption("Edit this on the Config page.")
    st.dataframe(cov, hide_index=True, use_container_width=True)

with tab_jobs:
    st.write("Downloads market data (read-only). It resumes where it stopped.")
    years = st.number_input("Years of history", 1, 20, 10)
    if st.button("Backfill candles"):
        job = jobs.launch("data-backfill", {"years": int(years)})
        st.success(f"Started {job.id}. Watch it on the Train and Test page.")

with tab_fix:
    st.caption(
        "Corrections are saved to the config files (validated, backed up, audited) and never "
        "touch stored prices. Re-run the quality check afterwards."
    )
    st.subheader("Days with no data")
    gaps = data.missing_days(store, TradingCalendar.from_config())
    holidays_like = gaps[gaps["likely"] == "market holiday"]
    if gaps.empty:
        st.success("No missing trading days.")
    else:
        st.dataframe(gaps, hide_index=True, use_container_width=True)
    if len(holidays_like):
        st.write(
            f"{len(holidays_like)} days are missing for nearly every symbol: the market was "
            "closed (e.g. Diwali Laxmi Pujan). Verify, then mark them as holidays."
        )
        pick_day = st.selectbox("Day", list(holidays_like["date"]))
        if st.button("Mark as market holiday"):
            try:
                configedit.add_holiday(pick_day, session, utc_now())
                st.cache_data.clear()
                st.success(f"{pick_day} added to holidays.yaml.")
                st.rerun()
            except configedit.ConfigEditError as exc:
                st.error(str(exc))
    manual = st.date_input("Or add any holiday by date", value=None)
    if manual and st.button("Add this holiday"):
        try:
            configedit.add_holiday(manual, session, utc_now())
            st.cache_data.clear()
            st.success(f"{manual} added.")
            st.rerun()
        except configedit.ConfigEditError as exc:
            st.error(str(exc))

    st.subheader("Exclude a symbol (data problems only)")
    ex_sym = st.selectbox("Symbol", sorted(store.symbols("day")), key="ex-sym")
    ex_why = st.text_input("Reason (e.g. demerger distorts the price)")
    if st.button("Exclude from training and backtests"):
        try:
            configedit.exclude_symbol(ex_sym, ex_why, session, utc_now())
            st.success(f"{ex_sym} excluded.")
        except configedit.ConfigEditError as exc:
            st.error(str(exc))
