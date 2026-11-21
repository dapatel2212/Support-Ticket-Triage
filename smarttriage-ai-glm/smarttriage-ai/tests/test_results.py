"""Tests for results.csv schema and uniqueness guarantees."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from services.export_service import (build_results_frame, validate_results,
                                     results_to_csv_bytes)


def _fake_predictions(n=5, review_ids=None):
    review_ids = review_ids or set()
    rows = []
    for i in range(1, n + 1):
        needs = "yes" if i in review_ids else "no"
        rows.append({
            "id": f"T-{i:04d}",
            "category": "" if i in review_ids else "Billing",
            "priority": "" if i in review_ids else "Low",
            "needs_review": needs,
            "category_confidence": 0.9, "priority_confidence": 0.8,
            "top_alt_category": "", "top_alt_priority": "",
            "category_margin": 0.5, "priority_margin": 0.4,
            "review_reasons": "", "category_inference_seconds": 0.01,
            "priority_inference_seconds": 0.01, "total_processing_seconds": 0.02,
            "subject_raw": "s", "body_raw": "b", "text": "s. b",
            "received_at": "", "customer": "", "processed_at": "",
        })
    return pd.DataFrame(rows)


def test_results_columns_exact():
    res = build_results_frame(_fake_predictions())
    assert list(res.columns) == ["id", "category", "priority", "needs_review"]


def test_results_preserves_review_empties():
    res = build_results_frame(_fake_predictions(5, review_ids={3}))
    assert res.loc[2, "category"] == ""
    assert res.loc[2, "priority"] == ""
    assert res.loc[2, "needs_review"] == "yes"
    assert (res["needs_review"].isin(["yes", "no"])).all()


def test_results_no_duplicates_even_if_input_duplicated():
    preds = _fake_predictions()
    preds = pd.concat([preds, preds.iloc[[0]]], ignore_index=True)
    res = build_results_frame(preds)
    assert res["id"].duplicated().sum() == 0
    assert len(res) == 5


def test_results_to_csv_bytes_roundtrip():
    preds = _fake_predictions(3)
    data = results_to_csv_bytes(preds)
    assert data.startswith(b"id,category,priority,needs_review")
    reread = pd.read_csv(__import__("io").BytesIO(data))
    assert len(reread) == 3


def test_validate_results_clean():
    preds = _fake_predictions(10, review_ids={5})
    res = build_results_frame(preds)
    errors = validate_results(res, input_ids=[r["id"] for r in preds.to_dict("records")])
    assert errors == []


def test_validate_results_flags_bad_category():
    preds = _fake_predictions()
    preds.loc[0, "category"] = "NotACategory"
    res = build_results_frame(preds)
    errors = validate_results(res)
    assert any("unexpected categories" in e for e in errors)


def test_every_input_id_appears_once():
    preds = _fake_predictions(20, review_ids={1, 5, 19})
    res = build_results_frame(preds)
    assert sorted(res["id"].tolist()) == sorted(preds["id"].tolist())
    assert res["id"].is_unique
