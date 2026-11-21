"""PAGE 1 — Overview Dashboard."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import app_utils
from config import settings
from services.review_service import load_reviews

NAVY = settings.COLORS["navy"]


def _kpi(label: str, value, color: str = NAVY) -> None:
    st.markdown(
        f'<div class="metric-card"><div style="font-size:0.8rem;color:#6C757D">'
        f'{label}</div><div style="font-size:1.6rem;font-weight:700;color:{color}">'
        f'{value}</div></div>', unsafe_allow_html=True)


def render() -> None:
    app_utils.header("Overview Dashboard",
                     "Real-time snapshot of triage activity. Predicted values "
                     "are kept distinct from human-verified values.")

    preds = app_utils.get_predictions()
    if preds is None or preds.empty:
        st.info("No tickets processed yet. Go to **Ticket Processing** to "
                "upload and run the triage pipeline on a CSV.")
        if st.button("Load bundled placeholder dataset (300 tickets)"):
            df = app_utils.load_default_tickets()
            st.session_state.input_df = df
            st.switch_page("ticket-processing")
        return

    reviews = load_reviews()
    verified_ids = set(reviews["ticket_id"].astype(str)) if not reviews.empty else set()

    total = len(preds)
    classified = int(((preds["needs_review"] == "no")).sum())
    pending = int((preds["needs_review"] == "yes").sum())
    urgent = int((preds["priority"] == "Urgent").sum())
    high = int((preds["priority"] == "High").sum())
    avg_time = float(preds["total_processing_seconds"].mean()) if "total_processing_seconds" in preds else 0.0
    review_pct = round(100 * pending / total, 1) if total else 0.0

    c1, c2, c3 = st.columns(3)
    with c1:
        _kpi("Total tickets received", total)
        _kpi("Successfully classified", classified, settings.COLORS["green"])
    with c2:
        _kpi("Pending human review", pending, settings.COLORS["yellow"])
        _kpi("Review percentage", f"{review_pct}%", settings.COLORS["yellow"])
    with c3:
        _kpi("Urgent tickets", urgent, settings.COLORS["red"])
        _kpi("High-priority tickets", high, settings.COLORS["yellow"])
    st.markdown("")
    _kpi("Average processing time (s)", f"{avg_time:.3f}")

    st.markdown("#### Distribution")
    col_a, col_b = st.columns(2)

    cat_df = preds[preds["category"].astype(str) != ""].copy()
    with col_a:
        fig = px.bar(
            cat_df["category"].value_counts().reindex(settings.CATEGORIES).fillna(0).reset_index(),
            x="index", y="category", labels={"index": "Category", "category": "Count"},
            color_discrete_sequence=[NAVY],
        )
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Category distribution (predicted).")

    pri_df = preds[preds["priority"].astype(str) != ""].copy()
    order = settings.PRIORITIES
    colors = [settings.COLORS["green"], settings.COLORS["navy"],
              settings.COLORS["yellow"], settings.COLORS["red"]]
    with col_b:
        counts = pri_df["priority"].value_counts().reindex(order).fillna(0)
        fig = go.Figure([go.Bar(x=list(counts.index), y=list(counts.values),
                                marker_color=colors)])
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300,
                          xaxis_title="Priority", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Priority distribution (predicted).")

    st.markdown("#### Predicted vs human-verified")
    v1, v2, v3 = st.columns(3)
    with v1:
        _kpi("Predicted (auto)", classified, settings.COLORS["blue"])
    with v2:
        _kpi("Human-verified decisions", len(reviews), settings.COLORS["green"])
    with v3:
        _kpi("Still awaiting review", pending - len(verified_ids), settings.COLORS["yellow"])

    st.caption("Zero-shot scores are relative ranking scores, not calibrated "
               "probabilities. Statistics shown here describe predictions, not "
               "model accuracy.")
