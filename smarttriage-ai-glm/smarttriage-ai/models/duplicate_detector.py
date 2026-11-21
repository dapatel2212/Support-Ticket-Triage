"""Semantic duplicate detection module (FUTURE / optional).

Uses sentence-transformers/all-MiniLM-L6-v2 to embed ticket text and compute
cosine similarity, so support teams can spot near-duplicate tickets.

This module is intentionally self-contained: it is only wired into the app
when ``settings.ENABLE_DUPLICATE_DETECTION`` is True, and it must never break
the core triage flow if the underlying library is missing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class DuplicateMatch:
    ticket_id: object
    similarity: float


class DuplicateDetector:
    def __init__(self, model_id: str = settings.DUPLICATE_MODEL_ID,
                 threshold: float = 0.85):
        self.model_id = model_id
        self.threshold = threshold
        self._model = None
        self._error: Optional[str] = None

    def load(self) -> None:
        if self._model is not None or self._error:
            return
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading duplicate-detection model %s ...", self.model_id)
            self._model = SentenceTransformer(self.model_id)
        except Exception as exc:  # pragma: no cover
            self._error = str(exc)
            logger.warning("Duplicate detection unavailable: %s", exc)

    @property
    def is_available(self) -> bool:
        if self._model is None and self._error is None:
            self.load()
        return self._model is not None

    @property
    def error(self) -> Optional[str]:
        return self._error

    def embed(self, texts: list[str]) -> Optional[np.ndarray]:
        if not self.is_available or not texts:
            return None
        try:
            return self._model.encode(texts, convert_to_numpy=True,
                                      normalize_embeddings=True)
        except Exception as exc:  # pragma: no cover
            logger.warning("Embedding failed: %s", exc)
            return None

    def find_duplicates(self, ids: list, texts: list[str]) -> dict:
        """Return {id: [DuplicateMatch, ...]} for each ticket above threshold."""
        result: dict = {tid: [] for tid in ids}
        emb = self.embed(texts)
        if emb is None:
            return result
        sim = np.matmul(emb, emb.T)  # embeddings are L2-normalised
        n = len(ids)
        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] >= self.threshold:
                    result[ids[i]].append(DuplicateMatch(ids[j], float(sim[i, j])))
                    result[ids[j]].append(DuplicateMatch(ids[i], float(sim[i, j])))
        return result
