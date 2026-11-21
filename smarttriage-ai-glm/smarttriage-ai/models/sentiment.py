"""Optional sentiment-analysis module.

Uses cardiffnlp/twitter-roberta-base-sentiment-latest to produce a
Positive / Neutral / Negative label plus a score. Sentiment is an *additional*
escalation signal only — it never decides the priority on its own.

If the model cannot be loaded (no network, no disk, library missing) the module
degrades gracefully: every call returns a neutral result and a flag telling the
caller that sentiment is unavailable. The main application keeps working.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)

# Map the model's numeric labels to readable ones.
_LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}


@dataclass
class SentimentResult:
    label: str = "Neutral"
    score: float = 0.0
    available: bool = False
    error: Optional[str] = None
    inference_seconds: float = 0.0


class SentimentAnalyzer:
    def __init__(self, model_id: str = settings.SENTIMENT_MODEL_ID):
        self.model_id = model_id
        self._pipe = None
        self._error: Optional[str] = None

    def load(self) -> None:
        if self._pipe is not None or self._error:
            return
        try:
            from transformers import pipeline
            logger.info("Loading sentiment model %s ...", self.model_id)
            self._pipe = pipeline("text-classification", model=self.model_id)
        except Exception as exc:  # pragma: no cover
            self._error = str(exc)
            logger.warning("Sentiment model unavailable: %s", exc)

    @property
    def is_available(self) -> bool:
        if self._pipe is None and self._error is None:
            self.load()
        return self._pipe is not None

    @property
    def error(self) -> Optional[str]:
        return self._error

    def analyze(self, text: str) -> SentimentResult:
        if not text or not text.strip():
            return SentimentResult()
        if not self.is_available:
            return SentimentResult(available=False, error=self._error)
        start = time.time()
        try:
            out = self._pipe(text[:1500])[0]
            raw = out["label"].lower()
            if "neg" in raw:
                label = "Negative"
            elif "pos" in raw:
                label = "Positive"
            else:
                label = "Neutral"
            return SentimentResult(
                label=label, score=float(out["score"]), available=True,
                inference_seconds=time.time() - start,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Sentiment inference failed: %s", exc)
            return SentimentResult(available=False, error=str(exc))
