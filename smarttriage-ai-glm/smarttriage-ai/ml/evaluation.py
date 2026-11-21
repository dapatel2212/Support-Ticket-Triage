"""Evaluation utilities for (future) supervised models.

Computes the metrics required by the assignment, separately for the category
and priority tasks, with stratified splitting and a holdout test set kept
distinct from train/validation. No metric is ever fabricated: every number is
computed from real labelled data via scikit-learn.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class EvaluationResult:
    task: str                       # "category" or "priority"
    n_test: int
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    f1_weighted: float
    labels: list[str] = field(default_factory=list)
    confusion_matrix: list[list[int]] = field(default_factory=list)
    report: str = ""

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "n_test": self.n_test,
            "accuracy": round(self.accuracy, 4),
            "precision_macro": round(self.precision_macro, 4),
            "recall_macro": round(self.recall_macro, 4),
            "f1_macro": round(self.f1_macro, 4),
            "f1_weighted": round(self.f1_weighted, 4),
        }


def evaluate(y_true, y_pred, labels: list[str], task: str) -> EvaluationResult:
    """Compute the full metric set for one task."""
    y_true = list(y_true)
    y_pred = list(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return EvaluationResult(
        task=task,
        n_test=len(y_true),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision_macro=float(precision_score(y_true, y_pred, labels=labels,
                                              average="macro", zero_division=0)),
        recall_macro=float(recall_score(y_true, y_pred, labels=labels,
                                        average="macro", zero_division=0)),
        f1_macro=float(f1_score(y_true, y_pred, labels=labels,
                                average="macro", zero_division=0)),
        f1_weighted=float(f1_score(y_true, y_pred, labels=labels,
                                   average="weighted", zero_division=0)),
        labels=list(labels),
        confusion_matrix=cm.tolist(),
        report=classification_report(y_true, y_pred, labels=labels,
                                     zero_division=0),
    )


def stratified_split_with_holdout(X, y, test_size: float = 0.2,
                                  holdout_size: float = 0.2,
                                  random_state: int = 42):
    """Two-level split: first carve a final holdout, then train/val.

    Returns (X_train, X_val, X_test, y_train, y_val, y_test) where X_test is
    the locked holdout used ONLY for the final evaluation. This prevents data
    leakage between model selection and the reported metrics.
    """
    from sklearn.model_selection import train_test_split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=holdout_size, random_state=random_state,
        stratify=y if _can_stratify(y) else None,
    )
    relative_val = test_size / (1 - holdout_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val, random_state=random_state,
        stratify=y_temp if _can_stratify(y_temp) else None,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _can_stratify(y) -> bool:
    """Stratification needs at least 2 samples per class."""
    import collections
    counts = collections.Counter(y)
    return all(c >= 2 for c in counts.values()) and len(counts) > 1


def macro_f1(y_true, y_pred, labels) -> float:
    return float(f1_score(y_true, y_pred, labels=labels,
                          average="macro", zero_division=0))
