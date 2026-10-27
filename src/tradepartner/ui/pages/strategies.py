"""Strategies and Models page (spec 11.6): status, gate checklist, shadow, approve, rollback."""

import pandas as pd
import streamlit as st
from sqlalchemy import select

from tradepartner.calendar import utc_now
from tradepartner.db.models import Model
from tradepartner.models import approval, registry
from tradepartner.ui.common import get_session, get_settings

st.title("Strategies and Models")
session = get_session()
settings = get_settings()
st.caption(
    "The system never approves a model. A candidate must pass the gate, run in shadow for "
    f"{approval.MIN_SHADOW_DAYS} days and pass the shadow comparison; then you approve it here."
)

models = list(session.scalars(select(Model).order_by(Model.strategy, Model.id.desc())))
st.dataframe(
    pd.DataFrame(
        [
            {
                "id": m.id,
                "strategy": m.strategy,
                "version": m.version,
                "status": m.status,
                "gate ready": m.metrics_json.get("gate", {}).get("ready_for_review"),
                "variant": m.metrics_json.get("chosen_variant"),
                "trained to": str(m.train_end),
            }
            for m in models
        ]
    ),
    hide_index=True,
    use_container_width=True,
)

if models:
    chosen_id = st.selectbox("Model", [m.id for m in models])
    model = registry.get(session, int(chosen_id))
    gate = model.metrics_json.get("gate", {})
    st.subheader("Promotion gate")
    for c in gate.get("checks", []):
        icon = "✅" if c["passed"] else ("⏳" if c["name"] == "human_approval" else "❌")
        st.write(
            f"{icon} **{c['name']}**: {c['value']} (needs {c['threshold']}) {c.get('detail', '')}"
        )
    if model.metrics_json.get("llm_period"):
        st.info(f"LLM features tested for {model.metrics_json['llm_period']}")
    for w in model.metrics_json.get("warnings", []):
        st.warning(w)
    st.subheader("Shadow comparison")
    st.json(model.metrics_json.get("shadow_comparison", {"status": "not run yet"}))
    for blocker in approval.approval_blockers(model, utc_now()):
        st.write(f"- blocker: {blocker}")

    c1, c2, c3 = st.columns(3)
    if c1.button("Start shadow") and model.status == "candidate":
        registry.start_shadow(session, model.id, utc_now())
        st.rerun()
    phrase = c2.text_input(f"Type APPROVE {model.strategy}")
    if c2.button("Approve"):
        if phrase != f"APPROVE {model.strategy}":
            st.error("Type the confirmation phrase exactly.")
        else:
            try:
                approval.approve(session, model.id, "user", utc_now())
                st.success("Approved. The previous model is kept for rollback.")
            except registry.ModelError as exc:
                st.error(str(exc))
    if c3.button("Rollback strategy"):
        try:
            prev = approval.rollback(session, model.strategy, "user", utc_now())
            st.success(f"Rolled back to model {prev.id}.")
        except registry.ModelError as exc:
            st.error(str(exc))
    st.subheader("Feature importance (gain share)")
    imp = model.metrics_json.get("importance", {})
    st.bar_chart(pd.Series(imp).sort_values(ascending=False).head(15))
st.caption(f"Strategies configured: {', '.join(sorted(settings.strategies))}")
