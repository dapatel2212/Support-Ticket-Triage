"""Data preprocessing for SmartTriage AI.

Responsibilities:
  * validate that the input DataFrame has the required columns;
  * handle missing subjects / bodies / duplicate IDs safely;
  * light text normalisation that NEVER removes meaningful words such as
    "not", "never", "cannot", "failed", "refund", "urgent" (these are exactly
    the signals the classifier needs);
  * combine subject + body into a single prediction text while keeping the
    original customer message untouched for display and auditing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config import settings

REQUIRED_COLUMNS = ["id", "subject", "body"]
OPTIONAL_COLUMNS = ["received_at", "customer"]

# Words we explicitly must NOT strip during normalisation.
PROTECTED_WORDS = {
    "not", "never", "cannot", "cant", "wont", "dont", "didnt", "doesnt",
    "failed", "refund", "urgent", "broken", "error", "lost", "missing",
    "cancel", "charged", "overcharged", "unauthorized", "compromised",
}

_WS_RE = re.compile(r"\s+")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


@dataclass
class ValidationReport:
    ok: bool
    n_rows: int
    missing_subject: int
    missing_body: int
    duplicate_ids: int
    invalid_ids: int
    errors: list[str]


def validate_dataframe(df: pd.DataFrame) -> ValidationReport:
    """Check structure & data quality. Never raises; reports instead."""
    errors: list[str] = []
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    n_rows = len(df)
    missing_subject = missing_body = duplicate_ids = invalid_ids = 0
    if not missing_cols:
        missing_subject = int(df["subject"].isna().sum() +
                              (df["subject"].astype(str).str.strip() == "").sum())
        missing_body = int(df["body"].isna().sum() +
                           (df["body"].astype(str).str.strip() == "").sum())
        duplicate_ids = int(df["id"].duplicated().sum())
        invalid_ids = int(df["id"].isna().sum())

    ok = not errors
    return ValidationReport(
        ok=ok, n_rows=n_rows,
        missing_subject=missing_subject, missing_body=missing_body,
        duplicate_ids=duplicate_ids, invalid_ids=invalid_ids, errors=errors,
    )


def _clean_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = _CTRL_RE.sub(" ", text)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = text.strip()
    text = _WS_RE.sub(" ", text)
    return text


def normalize_text(text: str) -> str:
    """Light normalisation preserving negation and risk keywords.

    We deliberately do NOT lowercase-stopword-strip or lemmatise, because the
    zero-shot cross-encoder benefits from full natural sentences including
    negations like "cannot" / "never".
    """
    cleaned = _clean_text(text)
    # Collapse repeated punctuation such as "!!!" -> "!" but keep the mark.
    cleaned = re.sub(r"([!?.,]){3,}", r"\1", cleaned)
    return cleaned


def combine_subject_body(subject: str, body: str) -> str:
    """Build the prediction text. Missing parts fall back gracefully."""
    subj = normalize_text(subject)
    bod = normalize_text(body)
    if subj and bod:
        return f"{subj}. {bod}"
    return subj or bod or ""


def prepare_tickets(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean copy with derived columns ready for classification.

    Adds:
      * ``text``        : combined subject + body used by the model
      * ``subject_raw`` : original subject (for display)
      * ``body_raw``    : original body (for display)
      * ``is_empty``    : True when both subject and body are empty
    Original rows are preserved 1:1; nothing is dropped.
    """
    report = validate_dataframe(df)
    if not report.ok:
        raise ValueError("; ".join(report.errors))

    out = df.copy()
    # Fill missing optional columns so downstream code never sees NaN.
    for col in OPTIONAL_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out["subject"] = out["subject"].fillna("").astype(str)
    out["body"] = out["body"].fillna("").astype(str)
    out["received_at"] = out["received_at"].fillna("").astype(str)
    out["customer"] = out["customer"].fillna("").astype(str)

    # Preserve originals for the UI and audit trail.
    out["subject_raw"] = df["subject"].fillna("").astype(str)
    out["body_raw"] = df["body"].fillna("").astype(str)

    out["text"] = out.apply(lambda r: combine_subject_body(r["subject"], r["body"]), axis=1)
    out["is_empty"] = out["text"].str.strip() == ""
    # Stable string id (handles numeric/object ids uniformly downstream).
    out["id"] = out["id"].astype(str)
    return out
