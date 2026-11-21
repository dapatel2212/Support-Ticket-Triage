"""PAGE 2 — Ticket Processing (single + bulk)."""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

import app_utils
from config import settings
from ml.preprocessing import validate_dataframe
from services import export_service
from services.ticket_service import process_single, process_tickets


def render() -> None:
    app_utils.header("Ticket Processing",
                     "Upload a CSV, validate it, and run the triage pipeline. "
                     "Single-ticket mode lets you test the model on ad-hoc text.")

    tab_bulk, tab_single = st.tabs(["Bulk processing", "Single ticket"])

    # --------------------------------------------------------------- bulk
    with tab_bulk:
        uploaded = st.file_uploader("Upload tickets.csv", type=["csv"],
                                    key="tickets_upload")
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("Use bundled dataset (data/tickets.csv)"):
                st.session_state.input_df = app_utils.load_default_tickets()
        with col2:
            if uploaded is not None:
                try:
                    df = pd.read_csv(uploaded)
                    st.session_state.input_df = df
                except Exception as exc:
                    st.error(f"Could not read CSV: {exc}")

        df = st.session_state.get("input_df")
        if df is not None:
            st.markdown("##### Preview & validation")
            report = validate_dataframe(df)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rows", report.n_rows)
            m2.metric("Missing subject", report.missing_subject)
            m3.metric("Missing body", report.missing_body)
            m4.metric("Duplicate IDs", report.duplicate_ids)
            if not report.ok:
                st.error("Validation errors: " + "; ".join(report.errors))
            else:
                st.success("Required columns present and valid.")
            st.dataframe(df.head(10), use_container_width=True, height=240)

            backend = app_utils.cached_backend()
            if hasattr(backend, "is_loaded") and not backend.is_loaded and \
               getattr(backend, "load_error", None):
                st.error(f"Model could not be loaded: {backend.load_error}. "
                         "Tickets will still be processed and routed to review.")

            if st.button("Process all tickets", type="primary"):
                progress = st.progress(0.0, text="Processing...")
                status = st.empty()

                def cb(done, total, tid):
                    progress.progress(done / total,
                                       text=f"Processing {done}/{total} (id={tid})")

                start = time.time()
                preds = process_tickets(df, backend, progress_cb=cb)
                elapsed = time.time() - start
                progress.empty()
                status.success(f"Done — {len(preds)} tickets in {elapsed:.1f}s")
                app_utils.set_predictions(preds)

                # Persist to disk for the other pages / files.
                export_service.write_predictions(preds)
                export_service.write_results(preds)
                _show_results(preds)
        else:
            st.info("Upload a CSV or load the bundled dataset to begin.")

    # --------------------------------------------------------------- single
    with tab_single:
        subject = st.text_input("Subject", value="Cannot log in to my account")
        body = st.text_area("Body", value="My password reset email never arrives "
                            "and I am completely locked out. This is urgent.")
        if st.button("Classify ticket", type="primary"):
            backend = app_utils.cached_backend()
            text = f"{subject}. {body}" if subject and body else (subject or body)
            from ml.preprocessing import combine_subject_body
            text = combine_subject_body(subject, body)
            row = pd.Series({"id": "single", "subject_raw": subject,
                             "body_raw": body, "text": text,
                             "category": "", "priority": "",
                             "category_confidence": "", "priority_confidence": "",
                             "review_reasons": ""})
            pred = process_single(text, backend)
            r1, r2, r3 = st.columns(3)
            r1.metric("Category", pred["category"] or "—")
            r2.metric("Priority", pred["priority"] or "—")
            r3.metric("Needs review", pred["needs_review"])
            st.markdown(f"**Category confidence:** {pred['category_confidence']}")
            st.markdown(f"**Priority confidence:** {pred['priority_confidence']}")
            if pred["needs_review"] == "yes":
                st.warning(f"Routed to review: {pred['review_reasons']}")
            else:
                st.success("Auto-classified with sufficient confidence.")


def _show_results(preds: pd.DataFrame) -> None:
    st.markdown("##### Predictions")
    display_cols = ["id", "category", "priority", "needs_review",
                    "category_confidence", "priority_confidence",
                    "review_reasons"]
    st.dataframe(preds[display_cols], use_container_width=True, height=320)

    st.markdown("##### Download")
    b1, b2, b3 = st.columns(3)
    with b1:
        st.download_button("Download results.csv",
                           export_service.results_to_csv_bytes(preds),
                           file_name="results.csv", mime="text/csv",
                           type="primary")
    with b2:
        st.download_button("Download full predictions.csv",
                           export_service.predictions_to_csv_bytes(preds),
                           file_name="predictions.csv", mime="text/csv")
    with b3:
        review_count = int((preds["needs_review"] == "yes").sum())
        st.metric("Routed to review", review_count)
