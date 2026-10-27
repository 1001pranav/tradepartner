"""News page: see what was fetched, linked to symbols and scored by the local LLM, and start the
fetch/score jobs. The LLM only scores text; nothing here can trade (spec 10)."""

import streamlit as st

from tradepartner.ui import data, jobs
from tradepartner.ui.common import get_session

st.title("News")
st.caption(
    "Articles are fetched, linked to symbols and scored by the local LLM (sentiment, event type). "
    "The model never decides trades. News features only exist from 2025-03-31 because the LLM "
    "knows nothing after its training cutoff, so older history stays price-only."
)
session = get_session()
s = data.news_summary(session)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Articles", s.articles)
c2.metric("Scored", s.scored)
c3.metric("Schema-valid rate", f"{s.valid_rate:.0%}" if s.valid_rate is not None else "-")
c4.metric("Throughput / hour", f"{s.per_hour:.0f}" if s.per_hour is not None else "-")
if s.by_source:
    st.bar_chart(s.by_source)
if s.valid_rate is not None and s.valid_rate < 0.95:
    st.warning("Valid rate is under the 95% target.")
if s.per_hour is not None and s.per_hour < 300:
    st.info("Throughput is under 300/hour; scoring a backlog will take a while.")

st.subheader("Run")
f1, f2 = st.columns(2)
with f1:
    hours = st.number_input("Fetch: look back (hours)", 1, 720, 48)
    if st.button("1. Fetch news"):
        st.success(f"Started {jobs.launch('news-fetch', {'hours': int(hours)}).id}")
with f2:
    limit = st.number_input("Score: max article/symbol pairs", 1, 500, 20)
    minutes = st.number_input("Score: stop after (minutes)", 1, 120, 10)
    if st.button("2. Score with the LLM"):
        job = jobs.launch("news-score", {"limit": int(limit), "max_minutes": int(minutes)})
        st.success(f"Started {job.id}")
st.caption(
    "3. Then train with news on the Train and Test page (tick the with-news box). Progress of "
    "these jobs is listed there too. Do not score while training: both are heavy."
)

st.subheader("Recent articles")
frame = data.recent_news(session)
if frame.empty:
    st.info("No articles yet. Press Fetch news.")
else:
    only_scored = st.checkbox("Only scored", False)
    sym = st.text_input("Symbol contains").upper()
    view = frame[frame["sentiment"].notna()] if only_scored else frame
    if sym:
        view = view[view["symbols"].str.contains(sym)]
    st.dataframe(view, hide_index=True, use_container_width=True)
