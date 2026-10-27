"""Config page: edit the safe settings with validation, a diff and an automatic backup.

risk.yaml (limits, capital) and app.yaml (broker, execution mode) are shown read-only: loosening
risk or going live is done by hand in the file.
"""

import streamlit as st

from tradepartner.calendar import utc_now
from tradepartner.paths import config_dir
from tradepartner.ui import configedit
from tradepartner.ui.common import get_session

st.title("Config")
files = list(configedit.editable())
name = st.selectbox("File", files)
current = configedit.read(name)
text = st.text_area("Contents", current, height=420, key=f"edit-{name}")

errors = configedit.validate(name, text)
if errors:
    st.error("Not valid: " + "; ".join(errors))
changes = configedit.diff(current, text)
if changes:
    st.subheader("Changes")
    st.code(changes, language="diff")
if st.button("Save", disabled=bool(errors) or not changes):
    try:
        configedit.save(name, text, get_session(), utc_now())
        st.success(
            "Saved. The previous version was backed up under ~/.tradepartner/backups/config-edits/."
        )
        st.rerun()
    except configedit.ConfigEditError as exc:
        st.error(str(exc))

st.subheader("Read-only (edit by hand)")
for ro, why in configedit.READ_ONLY.items():
    with st.expander(f"{ro}: {why}"):
        st.code((config_dir() / ro).read_text(), language="yaml")
