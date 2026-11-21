"""Tests for preprocessing: validation, normalisation, subject+body combine."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.preprocessing import (combine_subject_body, normalize_text,
                              prepare_tickets, validate_dataframe)


def test_validate_ok():
    df = pd.DataFrame({"id": [1, 2], "subject": ["a", "b"], "body": ["x", "y"]})
    rep = validate_dataframe(df)
    assert rep.ok and rep.n_rows == 2


def test_validate_missing_columns():
    df = pd.DataFrame({"id": [1], "subject": ["a"]})
    rep = validate_dataframe(df)
    assert not rep.ok
    assert "body" in " ".join(rep.errors)


def test_validate_detects_duplicates_and_empties():
    df = pd.DataFrame({"id": [1, 1, 3], "subject": ["a", "", "c"],
                       "body": ["x", "y", ""]})
    rep = validate_dataframe(df)
    assert rep.duplicate_ids == 1
    assert rep.missing_subject == 1
    assert rep.missing_body == 1


def test_normalize_preserves_protected_words():
    text = "I CANNOT log in!!! Refund URGENT please..."
    out = normalize_text(text)
    assert "cannot" in out.lower()  # case preserved but word present
    assert "refund" in out.lower()
    assert "urgent" in out.lower()
    assert "!!!" not in out  # repeated punctuation collapsed


def test_combine_subject_body_both():
    assert combine_subject_body("Login issue", "I cannot log in") == \
        "Login issue. I cannot log in"


def test_combine_subject_body_missing():
    assert combine_subject_body("", "just body") == "just body"
    assert combine_subject_body("", "") == ""


def test_prepare_tickets_preserves_row_count_and_ids():
    df = pd.DataFrame({
        "id": [10, 11, 12],
        "subject": ["a", "", ""],
        "body": ["body a", "body b", ""],
    })
    out = prepare_tickets(df)
    assert len(out) == 3
    assert list(out["id"]) == ["10", "11", "12"]
    assert out.loc[0, "text"] == "a. body a"
    assert out.loc[2, "is_empty"]  # both subject and body empty
