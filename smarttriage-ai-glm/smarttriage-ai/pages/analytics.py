"""PAGE 4 — Analytics.

Prediction distributions, urgent-ticket analysis, human-review statistics,
ticket arrival trends and processing-time stats. Sentiment distribution is
shown only if the optional sentiment module is enabled.

Important: these are descriptive statistics of the system's predictions, NOT
model accuracy. Ground-truth accuracy can only be measured on the human-verified
holdout after supervised training.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import app_utils
from config import settings
from services.review_service import load_reviews

NAVY = settings.COLORS["navy"]
PRI_COLORS = [settings.COLORS["green"], NAVY,
              settings.COLORS["yellow"], settings.COLORS["red"]]


def render() -> None:
    app_utils.header("Analytics",
                     "Descriptive statistics of triage predictions and review "
                     "activity. Not model accuracy.")

    preds = app_utils.get_predictions()
    if preds is None or preds.empty:
        st.info("No predictions to analyse yet. Process tickets first.")
        return

    st.subheader("Category distribution")
    cat_counts = preds["category"].replace("", pd.NA).dropna().value_counts()
    cat_counts = cat_counts.reindex(settings.CATEGORIES).fillna(0)
    fig = px.bar(cat_counts.reset_index(), x="index", y="category",
                 labels={"index": "Category", "category": "Count"},
                 color_discrete_sequence=[NAVY])
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Priority distribution")
    pri_counts = preds["priority"].replace("", pd.NA).dropna().value_counts()
    pri_counts = pri_counts.reindex(settings.PRIORITIES).fillna(0)
    fig = go.Figure([go.Bar(x=list(pri_counts.index), y=list(pri_counts.values),
                            marker_color=PRI_COLORS)])
    fig.update_layout(height=320, xaxis_title="Priority", yaxis_title="Count",
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Urgent-ticket analysis")
        urgent = preds[preds["priority"] == "Urgent"]
        st.metric("Urgent tickets", len(urgent))
        if not urgent.empty:
            u_by_cat = urgent["category"].replace("", pd.NA).dropna().value_counts()
            fig = px.bar(u_by_cat.reset_index(), x="index", y="category",
                         labels={"index": "Category", "category": "Urgent count"},
                         color_discrete_sequence=[settings.COLORS["red"]])
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No urgent tickets in this run.")

    with c2:
        st.subheader("Human-review statistics")
        reviews = load_reviews()
        pending = int((preds["needs_review"] == "yes").sum())
        approved = int((reviews["decision"] == "approved").sum()) if not reviews.empty else 0
        corrected = int((reviews["decision"] == "corrected").sum()) if not reviews.empty else 0
        rejected = int((reviews["decision"] == "rejected").sum()) if not reviews.empty else 0
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Pending", pending)
        r2.metric("Approved", approved)
        r3.metric("Corrected", corrected)
        r4.metric("Rejected", rejected)
        if not reviews.empty:
            fig = px.pie(reviews, names="decision",
                         color_discrete_sequence=[settings.COLORS["green"],
                                                  settings.COLORS["yellow"],
                                                  settings.COLORS["red"]])
            fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Ticket arrival trends")
    if "received_at" in preds.columns:
        ts = pd.to_datetime(preds["received_at"], errors="coerce").dropna()
        if not ts.empty:
            per_day = ts.dt.date.value_counts().sort_index()
            fig = px.line(per_day.reset_index(), x="index", y="received_at",
                          labels={"index": "Date", "received_at": "Tickets"},
                          color_discrete_sequence=[NAVY])
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No valid timestamps found.")
    else:
        st.caption("received_at column not available.")

    st.subheader("Processing-time statistics")
    if "total_processing_seconds" in preds.columns:
        times = preds["total_processing_seconds"]
        tc1, tc2, tc3 = st.columns(3)
        tc1.metric("Mean (s)", f"{times.mean():.3f}")
        tc2.metric("Median (s)", f"{times.median():.3f}")
        tc3.metric("Max (s)", f"{times.max():.3f}")
        fig = px.histogram(times, nbins=30,
                           color_discrete_sequence=[NAVY])
        fig.update_layout(height=280, xaxis_title="Seconds per ticket",
                          yaxis_title="Count", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    if settings.ENABLE_SENTIMENT:
        st.subheader("Sentiment distribution")
        st.caption("Sentiment module enabled — connect the analyzer to populate "
                   "this chart.")
