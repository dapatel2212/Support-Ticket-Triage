"""Ticket processing service.

Orchestrates preprocessing -> category prediction -> priority prediction ->
confidence assessment for a single ticket or a full DataFrame. Produces a
prediction DataFrame that the export service turns into results.csv.

Design notes:
  * One model instance is reused for every ticket (no per-ticket reload).
  * Every input row produces exactly one output row — nothing is dropped.
  * On model failure, the row is marked ``needs_review=yes`` with empty
    category/priority rather than being lost.
  * Sentiment is added only when the optional module is enabled and available.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd

from config import settings
from ml import category_classifier, priority_classifier
from ml.confidence import assess
from ml.preprocessing import prepare_tickets, validate_dataframe
from models.zero_shot import ZeroShotClassifier

logger = logging.getLogger(__name__)


def get_backend() -> ZeroShotClassifier:
    """Return the active backend. The Streamlit layer wraps this in
    ``st.cache_resource`` so the model loads once per process."""
    if settings.ACTIVE_BACKEND == "zero_shot":
        clf = ZeroShotClassifier()
        clf.load()
        return clf
    # Traditional / transformer backends are loaded lazily by the trained-model
    # loader once a supervised model has been trained and activated.
    try:
        from models.trained_loader import load_active_trained_backend
        backend = load_active_trained_backend()
        if backend is not None:
            return backend
    except Exception as exc:  # pragma: no cover
        logger.warning("Trained backend unavailable, falling back to zero-shot: %s", exc)
    clf = ZeroShotClassifier()
    clf.load()
    return clf


def process_single(text: str, backend) -> dict:
    """Process one ticket text. Returns a dict of prediction fields."""
    cat_res = category_classifier.classify(text, backend)
    pri_res = priority_classifier.classify(text, backend)
    assessment = assess(cat_res, pri_res, text=text)
    return {
        "category": assessment.category if not cat_res.failed else "",
        "priority": assessment.priority if not pri_res.failed else "",
        "needs_review": "yes" if assessment.needs_review else "no",
        "category_confidence": assessment.category_confidence,
        "priority_confidence": assessment.priority_confidence,
        "top_alt_category": assessment.top_alt_category or "",
        "top_alt_priority": assessment.top_alt_priority or "",
        "category_margin": assessment.category_margin,
        "priority_margin": assessment.priority_margin,
        "review_reasons": "; ".join(assessment.review_reasons),
        "category_inference_seconds": round(cat_res.inference_seconds, 4),
        "priority_inference_seconds": round(pri_res.inference_seconds, 4),
    }


def process_tickets(df: pd.DataFrame, backend,
                    progress_cb=None) -> pd.DataFrame:
    """Process a raw tickets DataFrame end-to-end.

    ``progress_cb(done, total, current_id)`` is an optional callback for the UI.
    Returns a DataFrame with one row per input ticket, including prediction
    fields and audit columns (timestamps, original text).
    """
    report = validate_dataframe(df)
    if not report.ok:
        raise ValueError("; ".join(report.errors))

    prepared = prepare_tickets(df)
    rows = []
    total = len(prepared)
    for i, (_, row) in enumerate(prepared.iterrows()):
        started = time.time()
        text = row["text"]
        if row["is_empty"]:
            # Empty ticket -> cannot predict; route straight to review.
            pred = {
                "category": "", "priority": "", "needs_review": "yes",
                "category_confidence": 0.0, "priority_confidence": 0.0,
                "top_alt_category": "", "top_alt_priority": "",
                "category_margin": 0.0, "priority_margin": 0.0,
                "review_reasons": "empty subject and body",
                "category_inference_seconds": 0.0,
                "priority_inference_seconds": 0.0,
            }
        else:
            pred = process_single(text, backend)
        pred["id"] = row["id"]
        pred["subject_raw"] = row["subject_raw"]
        pred["body_raw"] = row["body_raw"]
        pred["text"] = text
        pred["received_at"] = row["received_at"]
        pred["customer"] = row["customer"]
        pred["processed_at"] = pd.Timestamp.utcnow().isoformat()
        pred["total_processing_seconds"] = round(time.time() - started, 4)
        rows.append(pred)
        if progress_cb:
            progress_cb(i + 1, total, row["id"])

    out = pd.DataFrame(rows)
    # Stable column order.
    cols = ["id", "received_at", "customer", "subject_raw", "body_raw", "text",
            "category", "priority", "needs_review",
            "category_confidence", "priority_confidence",
            "top_alt_category", "top_alt_priority",
            "category_margin", "priority_margin",
            "review_reasons", "category_inference_seconds",
            "priority_inference_seconds", "total_processing_seconds",
            "processed_at"]
    return out[cols]
