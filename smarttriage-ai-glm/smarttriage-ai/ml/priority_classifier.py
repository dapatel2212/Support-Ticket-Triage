"""Priority classification facade.

Independent of category classification: priority is predicted from the same
ticket text through its own zero-shot query with priority-specific candidate
labels. The returned label is guaranteed to belong to the allowed priority set.
"""
from __future__ import annotations

from typing import Protocol

from config import settings
from models.zero_shot import CandidateResult


class PriorityBackend(Protocol):
    def predict_priority(self, text: str) -> CandidateResult: ...


def classify(text: str, backend: PriorityBackend) -> CandidateResult:
    if not text or not text.strip():
        return CandidateResult(label="Medium", confidence=0.0, failed=True,
                               error="empty input text")
    res = backend.predict_priority(text)
    if not res.failed and res.label not in settings.PRIORITIES:
        res.label = settings.canonical_priority(res.label)
    if not res.failed and res.label not in settings.PRIORITIES:
        res.label = "Medium"
    return res
