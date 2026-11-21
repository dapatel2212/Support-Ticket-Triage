"""Zero-shot classification backend.

Wraps the Hugging Face zero-shot pipeline so that:
  * the model is loaded once and reused for both category and priority tasks;
  * failures are handled gracefully and surfaced as a structured result so the
    rest of the application can route the ticket to human review instead of
    crashing;
  * the same instance is shared via a module-level cache that Streamlit can
    wrap in ``st.cache_resource``.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class CandidateResult:
    """Full result for a single zero-shot classification task."""

    label: str                       # canonical label (Billing / Urgent / ...)
    confidence: float                # top-1 softmax score (NOT a calibrated prob)
    all_labels: list[str] = field(default_factory=list)
    all_scores: list[float] = field(default_factory=list)
    top_alternative: Optional[str] = None
    margin: float = 0.0              # score(top1) - score(top2)
    failed: bool = False
    error: Optional[str] = None
    inference_seconds: float = 0.0


class ZeroShotClassifier:
    """Loads the zero-shot model once and serves both prediction tasks."""

    def __init__(self, model_id: str = settings.ZERO_SHOT_MODEL_ID):
        self.model_id = model_id
        self._pipe = None
        self._load_error: Optional[str] = None
        self._load_seconds: float = 0.0
        self._revision: Optional[str] = None

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        """Load the HF pipeline. Safe to call multiple times."""
        if self._pipe is not None or self._load_error:
            return
        start = time.time()
        try:
            logger.info("Loading zero-shot model %s ...", self.model_id)
            from transformers import pipeline
            self._pipe = pipeline(
                "zero-shot-classification",
                model=self.model_id,
            )
            # Record the exact revision used (for the Model Information page).
            try:
                from huggingface_hub import model_info
                info = model_info(self.model_id)
                self._revision = info.sha
            except Exception:  # network/permission issue, non-fatal
                self._revision = "unknown"
            self._load_seconds = time.time() - start
            logger.info("Zero-shot model ready in %.2fs", self._load_seconds)
        except Exception as exc:  # pragma: no cover - depends on environment
            self._load_error = str(exc)
            self._load_seconds = time.time() - start
            logger.exception("Failed to load zero-shot model")

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    @property
    def load_seconds(self) -> float:
        return self._load_seconds

    @property
    def revision(self) -> Optional[str]:
        return self._revision

    # ----------------------------------------------------------------- predict
    def predict(self, text: str, candidate_labels: list[str]) -> CandidateResult:
        """Run the zero-shot pipeline and return a structured result."""
        if not text or not text.strip():
            return CandidateResult(
                label="Other", confidence=0.0, failed=True,
                error="empty input text",
            )
        if not self.is_loaded:
            self.load()
        if not self.is_loaded:
            return CandidateResult(
                label="Other", confidence=0.0, failed=True,
                error=self._load_error or "model unavailable",
            )
        start = time.time()
        try:
            out = self._pipe(text, candidate_labels=candidate_labels,
                             multi_label=False)
            labels = list(out["labels"])
            scores = [float(s) for s in out["scores"]]
            top_label = labels[0]
            top_score = scores[0]
            alt = labels[1] if len(labels) > 1 else None
            margin = top_score - (scores[1] if len(scores) > 1 else 0.0)
            return CandidateResult(
                label=top_label, confidence=top_score,
                all_labels=labels, all_scores=scores,
                top_alternative=alt, margin=margin,
                inference_seconds=time.time() - start,
            )
        except Exception as exc:  # inference error -> route to review
            logger.exception("Zero-shot inference failed")
            return CandidateResult(
                label="Other", confidence=0.0, failed=True,
                error=str(exc), inference_seconds=time.time() - start,
            )

    # ------------------------------------------------------ task-specific APIs
    def predict_category(self, text: str) -> CandidateResult:
        res = self.predict(text, list(settings.CATEGORY_LABELS.values()))
        if not res.failed:
            res.label = settings.canonical_category(res.label)
            if res.top_alternative:
                res.top_alternative = settings.canonical_category(res.top_alternative)
        return res

    def predict_priority(self, text: str) -> CandidateResult:
        res = self.predict(text, list(settings.PRIORITY_LABELS.values()))
        if not res.failed:
            res.label = settings.canonical_priority(res.label)
            if res.top_alternative:
                res.top_alternative = settings.canonical_priority(res.top_alternative)
        return res
