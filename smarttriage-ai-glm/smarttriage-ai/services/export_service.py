"""Export service.

Writes the deliverable ``results.csv`` with EXACTLY the required columns:
    id, category, priority, needs_review

Plus the richer ``predictions.csv`` for analytics and auditing. Every input
ticket id appears exactly once; rows sent for review may have empty
category/priority.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from config import settings

RESULTS_COLUMNS = ["id", "category", "priority", "needs_review"]


def build_results_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    """Reduce a predictions DataFrame to the exact results.csv schema."""
    if predictions is None or predictions.empty:
        return pd.DataFrame(columns=RESULTS_COLUMNS)
    res = predictions[["id", "category", "priority", "needs_review"]].copy()
    res["id"] = res["id"].astype(str)
    # Normalise needs_review to lowercase yes/no.
    res["needs_review"] = res["needs_review"].astype(str).str.lower().map(
        lambda v: "yes" if v.startswith("y") else "no")
    # Reviews may legitimately have empty category/priority.
    res["category"] = res["category"].fillna("").astype(str)
    res["priority"] = res["priority"].fillna("").astype(str)
    # Guarantee one row per input id (defensive — never duplicate).
    res = res.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)
    return res[RESULTS_COLUMNS]


def write_results(predictions: pd.DataFrame,
                  path: Optional[settings.RESULTS_CSV.__class__] = None) -> str:
    target = path or settings.RESULTS_CSV
    res = build_results_frame(predictions)
    res.to_csv(target, index=False)
    return str(target)


def write_predictions(predictions: pd.DataFrame,
                      path: Optional[settings.PREDICTIONS_CSV.__class__] = None) -> str:
    target = path or settings.PREDICTIONS_CSV
    predictions.to_csv(target, index=False)
    return str(target)


def results_to_csv_bytes(predictions: pd.DataFrame) -> bytes:
    """Return results.csv as bytes (for Streamlit ``st.download_button``)."""
    return build_results_frame(predictions).to_csv(index=False).encode("utf-8")


def predictions_to_csv_bytes(predictions: pd.DataFrame) -> bytes:
    return predictions.to_csv(index=False).encode("utf-8")


def validate_results(results: pd.DataFrame,
                     input_ids: Optional[list] = None) -> list[str]:
    """Return a list of validation errors (empty list = valid)."""
    errors: list[str] = []
    if list(results.columns) != RESULTS_COLUMNS:
        errors.append(f"columns mismatch: {list(results.columns)} != {RESULTS_COLUMNS}")
    if results["needs_review"].astype(str).str.lower().isin(["yes", "no"]).all() is False:
        errors.append("needs_review must contain only yes/no")
    bad_cat = results[results["category"].astype(str) != ""]
    bad_cat = bad_cat[~bad_cat["category"].isin(settings.CATEGORIES)]
    if not bad_cat.empty:
        errors.append(f"unexpected categories: {bad_cat['category'].unique().tolist()}")
    bad_pri = results[results["priority"].astype(str) != ""]
    bad_pri = bad_pri[~bad_pri["priority"].isin(settings.PRIORITIES)]
    if not bad_pri.empty:
        errors.append(f"unexpected priorities: {bad_pri['priority'].unique().tolist()}")
    if results["id"].duplicated().any():
        errors.append("duplicate ticket ids in results")
    if input_ids is not None:
        missing = set(map(str, input_ids)) - set(results["id"].astype(str))
        if missing:
            errors.append(f"{len(missing)} input ids missing from results")
    return errors
