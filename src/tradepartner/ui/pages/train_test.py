"""Train and Test page: start research jobs in the background and read their results.

Everything here is simulation or model training. There is no button that trades, approves a model,
or touches the broker: approving stays on the Strategies and Models page.
"""

import json
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from tradepartner.paths import reports_dir
from tradepartner.ui import jobs

st.title("Train and Test")
st.caption(
    "Simulation only: training, replay, paper trading and backtests. No orders, no approvals."
)

action = st.selectbox("What to run", list(jobs.ACTIONS), format_func=lambda a: jobs.ACTIONS[a])
params: dict[str, object] = {}
if action in ("train", "paper-sim"):
    params["strategy"] = st.selectbox("Strategy", jobs.ML_STRATEGIES)
if action == "backtest":
    params["strategy"] = st.selectbox("Baseline rule", jobs.BASELINES)
if action == "train":
    params["trials"] = st.number_input("Optuna trials per fold (0 to 50)", 0, 50, 10)
if action == "train":
    params["news"] = st.checkbox(
        "Also train the with-news variant (needs scored news; news features only exist from 2025-03-31)",
        False,
    )
if action == "explain-jumps":
    params["limit"] = st.number_input("Explain at most (this run)", 1, 500, 20)
if action == "news-fetch":
    params["hours"] = st.number_input("Look back (hours)", 1, 720, 48)
if action == "news-score":
    params["limit"] = st.number_input("Max (article, symbol) pairs", 1, 500, 20)
    params["max_minutes"] = st.number_input("Stop after (minutes)", 1, 120, 10)
if action == "replay":
    params["as_of"] = st.date_input(
        "Predict from this day", date.today() - timedelta(days=30)
    ).isoformat()
    params["days"] = st.number_input("Days to reveal", 1, 60, 5)
    params["top"] = st.number_input("Top picks", 1, 20, 5)
if action == "paper-sim":
    params["start"] = st.date_input("Start trading on", date(2024, 1, 1)).isoformat()
    params["capital_inr"] = st.number_input(
        "Simulated capital (INR)", 10_000, 100_000_000, 1_000_000, step=100_000
    )
if action == "backtest":
    params["capital_inr"] = st.number_input(
        "Simulated capital (INR)", 10_000, 100_000_000, 1_000_000, step=100_000
    )
if action == "data-backfill":
    params["years"] = st.number_input("Years", 1, 20, 10)

if st.button("Start in the background", type="primary"):
    try:
        job = jobs.launch(action, params)
        st.success(f"Started {job.id}. Scroll down for progress.")
    except jobs.JobError as exc:
        st.error(str(exc))


@st.fragment(run_every=3)
def progress() -> None:
    st.subheader("Recent jobs")
    for j in jobs.list_jobs():
        with st.expander(f"{j.id}: {j.status}", expanded=j.status == "running"):
            st.code(" ".join(j.args), language="text")
            st.code(jobs.tail(j) or "(no output yet)", language="text")


progress()

st.subheader("Reports")
files = (
    sorted(reports_dir().glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)[:40]
    if reports_dir().exists()
    else []
)
if not files:
    st.info("No reports yet.")
else:
    chosen = st.selectbox("Open a report", files, format_func=lambda p: p.name)
    st.caption(f"{chosen}  ({chosen.stat().st_size // 1024} KB)")
    st.download_button("Download this report", chosen.read_bytes(), file_name=chosen.name)
    if chosen.suffix == ".html":
        st.iframe(chosen, height=800)  # our own generated, escaped report files
    elif chosen.name.startswith("train-") and chosen.suffix == ".json":
        rep = json.loads(chosen.read_text())
        st.write(
            f"**{rep['strategy']}**, variant {rep.get('chosen_variant')}: {rep.get('chosen_reason')}"
        )
        checks = pd.DataFrame(rep["gate"]["checks"])[["name", "passed", "value", "threshold"]]
        st.dataframe(checks, hide_index=True, use_container_width=True)
        st.write("Prediction quality:", rep.get("quality"))
        imp = pd.Series(rep.get("importance", {})).sort_values(ascending=False).head(12)
        st.bar_chart(imp)
        for w in rep.get("warnings", []):
            st.warning(w)
    else:
        st.write(chosen.read_text()[:5000])
