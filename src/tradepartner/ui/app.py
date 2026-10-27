"""Streamlit entry point. Run with ``tradepartner ui`` (binds to 127.0.0.1 only).

There is deliberately no 'share' or 'publish' feature (spec 2.3).
"""

from pathlib import Path

import streamlit as st

PAGES_DIR = Path(__file__).parent / "pages"
# (file, title) in display order; pages are added by the milestone that builds them.
PAGE_ORDER = [
    ("today.py", "Today"),
    ("data.py", "Data"),
    ("news.py", "News"),
    ("train_test.py", "Train and Test"),
    ("modes.py", "Modes"),
    ("positions.py", "Positions"),
    ("ledger.py", "Ledger"),
    ("pnl_tax.py", "P&L and Tax"),
    ("strategies.py", "Strategies and Models"),
    ("risk.py", "Risk"),
    ("config.py", "Config"),
    ("system.py", "System"),
]

st.set_page_config(page_title="TradePartner", layout="wide")
pages = [
    st.Page(str(PAGES_DIR / f), title=title, url_path=f.removesuffix(".py"))
    for f, title in PAGE_ORDER
    if (PAGES_DIR / f).exists()
]
st.navigation(pages).run()
