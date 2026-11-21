"""PAGE 3 — Human Review Center.

Lists tickets routed to review, lets a reviewer read the original message, see
the model prediction + confidence + reason, then correct category/priority and
approve or reject. Corrections are stored SEPARATELY from model predictions
(see services/review_service) and can be exported into a verified training
dataset for the future supervised pipeline.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import app_utils
from config import settings
from services import review_service


def render() -> None:
    app_utils.header("Human Review Center",
                     "Ambiguous or critical tickets routed here. Reviewers "
                     "correct and approve; decisions are stored separately from "
                     "model predictions.")

    preds = app_utils.get_predictions()
    if preds is None or preds.empty:
        st.info("No predictions yet. Process tickets first on the "
                "**Ticket Processing** page.")
        return

    pending = review_service.pending_review_tickets(preds)
    if pending.empty:
        st.success("Nothing pending review — every ticket was classified "
                   "above threshold.")
        return

    reviewed_ids = review_service.already_reviewed_ids()

    st.markdown(f"**{len(pending)}** ticket(s) routed to review · "
                f"**{len(reviewed_ids & set(pending['id'].astype(str)))}** "
                "already reviewed.")

    # Filter controls
    cats = ["All"] + settings.CATEGORIES
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        show_only = st.radio("Show", ["Pending", "Already reviewed"],
                             horizontal=True)
    with fcol2:
        cat_filter = st.selectbox("Filter by predicted category", cats)

    queue = pending.copy()
    if show_only == "Already reviewed":
        queue = queue[queue["id"].astype(str).isin(reviewed_ids)]
    else:
        queue = queue[~queue["id"].astype(str).isin(reviewed_ids)]
    if cat_filter != "All":
        queue = queue[queue["category"].astype(str) == cat_filter]
    queue = queue.reset_index(drop=True)

    if queue.empty:
        st.info("No tickets match this filter.")
        return

    st.markdown("---")
    for _, row in queue.iterrows():
        tid = str(row["id"])
        with st.expander(f"Ticket {tid} · {row.get('subject_raw','')[:60]}",
                         expanded=False):
            cc1, cc2 = st.columns([2, 1])
            with cc1:
                st.markdown(f"**Subject:** {row.get('subject_raw','')}")
                st.markdown(f"**Customer:** {row.get('customer','')} · "
                            f"**Received:** {row.get('received_at','')}")
                st.markdown("**Original message:**")
                st.text(row.get("body_raw", "")[:1500])
            with cc2:
                st.markdown("**Model prediction**")
                st.metric("Category", row.get("category") or "—")
                st.metric("Priority", row.get("priority") or "—")
                st.metric("Cat conf.", row.get("category_confidence"))
                st.metric("Pri conf.", row.get("priority_confidence"))
                st.caption(f"Reason: {row.get('review_reasons','')}")

            rc1, rc2, rc3, rc4 = st.columns([1, 1, 2, 1])
            with rc1:
                new_cat = st.selectbox("Corrected category", settings.CATEGORIES,
                                       index=settings.CATEGORIES.index(row["category"])
                                       if row.get("category") in settings.CATEGORIES else 0,
                                       key=f"cat_{tid}")
            with rc2:
                new_pri = st.selectbox("Corrected priority", settings.PRIORITIES,
                                       index=settings.PRIORITIES.index(row["priority"])
                                       if row.get("priority") in settings.PRIORITIES else 1,
                                       key=f"pri_{tid}")
            with rc3:
                notes = st.text_input("Reviewer notes", key=f"notes_{tid}")
            with rc4:
                st.write("")
                st.write("")

            bcol1, bcol2, bcol3 = st.columns(3)
            with bcol1:
                if st.button("✅ Approve", key=f"appr_{tid}"):
                    review_service.store_review(
                        row, new_cat, new_pri, decision="approved",
                        notes=notes)
                    st.success(f"Approved ticket {tid}. Stored in human_reviews.csv.")
            with bcol2:
                if st.button("✏️ Correct & save", key=f"corr_{tid}"):
                    review_service.store_review(
                        row, new_cat, new_pri, decision="corrected",
                        notes=notes)
                    st.success(f"Corrected ticket {tid}. Stored in human_reviews.csv.")
            with bcol3:
                if st.button("⛔ Reject", key=f"rej_{tid}"):
                    review_service.store_review(
                        row, "", "", decision="rejected", notes=notes)
                    st.warning(f"Rejected ticket {tid} (not exported as a label).")

    st.markdown("---")
    st.subheader("Verified training data export")
    reviews = review_service.load_reviews()
    if not reviews.empty:
        st.dataframe(reviews[["ticket_id", "decision", "model_category",
                              "corrected_category", "model_priority",
                              "corrected_priority", "reviewed_at"]],
                     use_container_width=True)
    if st.button("Export verified_training_data.csv", type="primary"):
        out = review_service.export_verified_training_data()
        st.success(f"Exported {len(out)} verified rows → "
                   f"`{settings.VERIFIED_TRAINING_CSV}`")
        st.caption("This file is the input for `python -m ml.training`. "
                   "It contains only human-approved or corrected labels, never "
                   "raw model predictions.")
