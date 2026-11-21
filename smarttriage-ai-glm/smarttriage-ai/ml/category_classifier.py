"""Category classification facade.

Delegates to the active backend (zero-shot by default; a trained TF-IDF or
fine-tuned transformer model in the future) and guarantees that the returned
label belongs to the allowed category taxonomy.
"""
from __future__ import annotations

from typing import Protocol

from config import settings
from models.zero_shot import CandidateResult


class CategoryBackend(Protocol):
    def predict_category(self, text: str) -> CandidateResult: ...


def classify(text: str, backend: CategoryBackend) -> CandidateResult:
    """Run category prediction and sanitise the output."""
    if not text or not text.strip():
        return CandidateResult(label="Other", confidence=0.0, failed=True,
                               error="empty input text")
    res = backend.predict_category(text)
    if not res.failed and res.label not in settings.CATEGORIES:
        res.label = settings.canonical_category(res.label)
    if not res.failed and res.label not in settings.CATEGORIES:
        # Unexpected label even after canonicalisation -> safe fallback.
        res.label = "Other"
    return res
