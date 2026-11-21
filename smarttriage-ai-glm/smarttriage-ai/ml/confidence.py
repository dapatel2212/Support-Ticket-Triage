"""Confidence assessment and human-review decision logic.

A ticket is routed to human review when ANY of the following hold:
  * the category prediction failed (model unavailable / inference error);
  * the priority prediction failed;
  * category confidence < configured threshold;
  * priority confidence < configured threshold;
  * the margin between top-1 and top-2 is too small (ambiguous ranking);
  * a critical-risk keyword appears in the text (e.g. "security breach").

Zero-shot softmax scores are relative ranking scores, NOT calibrated
probabilities — we never present them as P(correct). The thresholds below are
tuneable heuristics, exposed in settings and on the Model Information page.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from config import settings
from models.zero_shot import CandidateResult


@dataclass
class ConfidenceAssessment:
    category: str
    category_confidence: float
    priority: str
    priority_confidence: float
    top_alt_category: Optional[str]
    top_alt_priority: Optional[str]
    category_margin: float
    priority_margin: float
    needs_review: bool
    review_reasons: list[str] = field(default_factory=list)


def _matches_critical_risk(text: str) -> Optional[str]:
    lowered = text.lower()
    for kw in settings.CRITICAL_RISK_KEYWORDS:
        if kw in lowered:
            return kw
    return None


def assess(category_res: CandidateResult, priority_res: CandidateResult,
           text: str = "",
           cat_threshold: float = settings.CATEGORY_CONFIDENCE_THRESHOLD,
           pri_threshold: float = settings.PRIORITY_CONFIDENCE_THRESHOLD,
           margin_threshold: float = settings.CONFIDENCE_MARGIN_THRESHOLD,
           ) -> ConfidenceAssessment:
    """Combine both predictions into a single review decision."""
    reasons: list[str] = []

    if category_res.failed:
        reasons.append(f"category prediction failed: {category_res.error}")
    if priority_res.failed:
        reasons.append(f"priority prediction failed: {priority_res.error}")

    if not category_res.failed and category_res.confidence < cat_threshold:
        reasons.append(
            f"category confidence {category_res.confidence:.2f} < {cat_threshold}")
    if not priority_res.failed and priority_res.confidence < pri_threshold:
        reasons.append(
            f"priority confidence {priority_res.confidence:.2f} < {pri_threshold}")

    if not category_res.failed and category_res.margin < margin_threshold:
        reasons.append(
            f"category margin {category_res.margin:.2f} < {margin_threshold}")
    if not priority_res.failed and priority_res.margin < margin_threshold:
        reasons.append(
            f"priority margin {priority_res.margin:.2f} < {margin_threshold}")

    critical = _matches_critical_risk(text)
    if critical:
        reasons.append(f"critical-risk keyword detected: '{critical}'")

    needs_review = bool(reasons)
    return ConfidenceAssessment(
        category="" if category_res.failed else category_res.label,
        category_confidence=round(category_res.confidence, 4),
        priority="" if priority_res.failed else priority_res.label,
        priority_confidence=round(priority_res.confidence, 4),
        top_alt_category=category_res.top_alternative,
        top_alt_priority=priority_res.top_alternative,
        category_margin=round(category_res.margin, 4),
        priority_margin=round(priority_res.margin, 4),
        needs_review=needs_review,
        review_reasons=reasons,
    )
