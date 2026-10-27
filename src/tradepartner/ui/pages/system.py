"""System page (spec 11.8): job runs, errors, Ollama, disk, last backup."""

import shutil

import pandas as pd
import streamlit as st
from sqlalchemy import select

from tradepartner.calendar import fmt_ist
from tradepartner.db.models import JobRun
from tradepartner.llm.client import OllamaClient
from tradepartner.paths import backups_dir, db_path, home_dir, kill_file
from tradepartner.ui.common import get_session, get_settings

st.title("System")
session = get_session()
settings = get_settings()

llm = OllamaClient(settings.app.llm.model, base_url="http://127.0.0.1:11434", timeout=3, retries=0)
up = llm.health()
c1, c2, c3 = st.columns(3)
c1.metric("Ollama", "running" if up else "NOT running")
usage = shutil.disk_usage(home_dir() if home_dir().exists() else ".")
c2.metric("Disk free", f"{usage.free / 1e9:.1f} GB")
c3.metric("DB size", f"{(db_path().stat().st_size / 1e6) if db_path().exists() else 0:.1f} MB")
backups = sorted(backups_dir().glob("*")) if backups_dir().exists() else []
st.write(
    f"Last backup: **{backups[-1].name if backups else 'none yet'}**. Kill file present: {kill_file().exists()}"
)

runs = list(session.scalars(select(JobRun).order_by(JobRun.id.desc()).limit(200)))
st.subheader("Job runs")
st.dataframe(
    pd.DataFrame(
        [
            {
                "job": r.job,
                "started (IST)": fmt_ist(r.started_at),
                "status": r.status,
                "detail": r.error,
            }
            for r in runs
        ]
    ),
    hide_index=True,
    use_container_width=True,
)
errors = [r for r in runs if r.status == "error"]
if errors:
    st.error(f"{len(errors)} failed job runs in the last {len(runs)}.")
