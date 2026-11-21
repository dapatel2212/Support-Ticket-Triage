"""Loader for trained supervised backends.

Activated only when ``settings.ACTIVE_BACKEND`` is ``traditional`` or
``transformer`` AND a trained artifact exists under ``models/trained/``. If no
artifact is found, ``load_active_trained_backend`` returns ``None`` and the
caller falls back to the zero-shot backend. This keeps the app usable at every
stage of the project lifecycle.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import joblib

from config import settings

logger = logging.getLogger(__name__)


class TraditionalBackend:
    """Adapts a joblib TF-IDF pipeline to the backend protocol."""

    def __init__(self, category_pipe, priority_pipe):
        self.category_pipe = category_pipe
        self.priority_pipe = priority_pipe
        from models.zero_shot import CandidateResult

    def _predict(self, pipe, text, canonical_fn, fallback: str):
        from models.zero_shot import CandidateResult
        if not text or not text.strip():
            return CandidateResult(label=fallback, confidence=0.0, failed=True,
                                   error="empty input text")
        try:
            import numpy as np
            proba = pipe.predict_proba([text])[0]
            classes = pipe.classes_
            order = np.argsort(proba)[::-1]
            top = int(order[0])
            alt = int(order[1]) if len(order) > 1 else None
            label = canonical_fn(str(classes[top]))
            return CandidateResult(
                label=label, confidence=float(proba[top]),
                all_labels=[str(c) for c in classes],
                all_scores=[float(p) for p in proba],
                top_alternative=canonical_fn(str(classes[alt])) if alt is not None else None,
                margin=float(proba[top] - (proba[alt] if alt is not None else 0.0)),
            )
        except Exception as exc:
            logger.warning("Traditional backend predict failed: %s", exc)
            from models.zero_shot import CandidateResult
            return CandidateResult(label=fallback, confidence=0.0, failed=True,
                                   error=str(exc))

    def predict_category(self, text):
        return self._predict(self.category_pipe, text, settings.canonical_category, "Other")

    def predict_priority(self, text):
        return self._predict(self.priority_pipe, text, settings.canonical_priority, "Medium")


def _find_latest(prefix: str) -> Optional[Path]:
    if not settings.TRAINED_MODELS_DIR.exists():
        return None
    matches = sorted(settings.TRAINED_MODELS_DIR.glob(f"{prefix}_tfidf_*.joblib"))
    return matches[-1] if matches else None


def load_active_trained_backend():
    """Return a TraditionalBackend if artifacts exist, else None."""
    if settings.ACTIVE_BACKEND not in ("traditional", "transformer"):
        return None
    cat_path = _find_latest("category")
    pri_path = _find_latest("priority")
    if not cat_path or not pri_path:
        logger.warning("ACTIVE_BACKEND=%s but no trained artifacts found in %s; "
                       "fall back to zero-shot.",
                       settings.ACTIVE_BACKEND, settings.TRAINED_MODELS_DIR)
        return None
    logger.info("Loading trained backend: %s, %s", cat_path, pri_path)
    return TraditionalBackend(joblib.load(cat_path), joblib.load(pri_path))
