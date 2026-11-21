"""Human-review service.

Stores human corrections SEPARATELY from model predictions. A review record
contains the reviewer's decision plus the original model prediction, so the
two can never be confused. Verified corrections can be exported into a
training dataset (``verified_training_data.csv``) for the future supervised
pipeline.

The model prediction column is never overwritten — see ``store_review``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

REVIEW_COLUMNS = [
    "review_id", "ticket_id", "reviewed_at", "reviewer",
    "subject_raw", "body_raw", "text",
    "model_category", "model_priority",
    "model_category_confidence", "model_priority_confidence",
    "review_reasons",
    "corrected_category", "corrected_priority",
    "decision",  # approved | corrected | rejected
    "notes",
]


def _empty_reviews_df() -> pd.DataFrame:
    return pd.DataFrame(columns=REVIEW_COLUMNS)


def load_reviews(path: Path = settings.HUMAN_REVIEWS_CSV) -> pd.DataFrame:
    if not path.exists():
        return _empty_reviews_df()
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.warning("Could not read human_reviews.csv: %s", exc)
        return _empty_reviews_df()
    for col in REVIEW_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[REVIEW_COLUMNS]


def store_review(predictions_row: pd.Series,
                 corrected_category: str, corrected_priority: str,
                 decision: str, reviewer: str = "anonymous",
                 notes: str = "",
                 path: Path = settings.HUMAN_REVIEWS_CSV) -> pd.DataFrame:
    """Append one human review. Updates an existing review for the same ticket
    instead of duplicating it."""
    existing = load_reviews(path)
    review_id = f"R-{len(existing) + 1:05d}"
    record = {
        "review_id": review_id,
        "ticket_id": str(predictions_row.get("id", "")),
        "reviewed_at": pd.Timestamp.utcnow().isoformat(),
        "reviewer": reviewer,
        "subject_raw": predictions_row.get("subject_raw", ""),
        "body_raw": predictions_row.get("body_raw", ""),
        "text": predictions_row.get("text", ""),
        "model_category": predictions_row.get("category", ""),
        "model_priority": predictions_row.get("priority", ""),
        "model_category_confidence": predictions_row.get("category_confidence", ""),
        "model_priority_confidence": predictions_row.get("priority_confidence", ""),
        "review_reasons": predictions_row.get("review_reasons", ""),
        "corrected_category": corrected_category,
        "corrected_priority": corrected_priority,
        "decision": decision,
        "notes": notes,
    }
    # Replace any prior review for the same ticket id (no duplicates).
    existing = existing[existing["ticket_id"].astype(str) != record["ticket_id"]]
    existing = pd.concat([existing, pd.DataFrame([record])], ignore_index=True)
    existing.to_csv(path, index=False)
    return existing


def pending_review_tickets(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return the subset of predictions that need human review."""
    if predictions is None or predictions.empty:
        return predictions
    mask = predictions["needs_review"].astype(str).str.lower() == "yes"
    return predictions[mask].reset_index(drop=True)


def already_reviewed_ids(path: Path = settings.HUMAN_REVIEWS_CSV) -> set[str]:
    df = load_reviews(path)
    return set(df["ticket_id"].astype(str).tolist())


def export_verified_training_data(
        path: Path = settings.VERIFIED_TRAINING_CSV) -> pd.DataFrame:
    """Export approved/corrected reviews into the supervised training format.

    Output columns: ticket_id, subject, body, category, priority.
    Only reviews with decision in {approved, corrected} are exported; rejected
    predictions are dropped (they are not usable as labels). The category and
    priority used are the reviewer's corrected values (which, for an
    "approved" decision, equal the model prediction the reviewer endorsed).
    """
    reviews = load_reviews(path)
    if reviews.empty:
        logger.info("No human reviews to export yet.")
        empty = pd.DataFrame(columns=["ticket_id", "subject", "body",
                                      "category", "priority"])
        empty.to_csv(path, index=False)
        return empty

    keep = reviews[reviews["decision"].astype(str).isin(["approved", "corrected"])]
    out = pd.DataFrame({
        "ticket_id": keep["ticket_id"],
        "subject": keep["subject_raw"],
        "body": keep["body_raw"],
        "category": keep["corrected_category"],
        "priority": keep["corrected_priority"],
    })
    # Drop rows with empty labels (reviewer left them blank).
    out = out[(out["category"].astype(str).str.len() > 0) &
              (out["priority"].astype(str).str.len() > 0)]
    out = out.drop_duplicates(subset=["ticket_id"]).reset_index(drop=True)
    out.to_csv(path, index=False)
    logger.info("Exported %d verified rows -> %s", len(out), path)
    return out
