"""Central configuration for SmartTriage AI.

All model identifiers, candidate labels, review thresholds and data file
locations live here so that ML code, services and UI never hard-code them.
Adjusting behaviour (e.g. how aggressive the review queue is) is a one-line
change in this file.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Project root = parent of this file's package (smarttriage-ai/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TICKETS_CSV = DATA_DIR / "tickets.csv"
PREDICTIONS_CSV = DATA_DIR / "predictions.csv"
HUMAN_REVIEWS_CSV = DATA_DIR / "human_reviews.csv"
VERIFIED_TRAINING_CSV = DATA_DIR / "verified_training_data.csv"
RESULTS_CSV = OUTPUTS_DIR / "results.csv"

for _d in (DATA_DIR, MODELS_DIR, TRAINED_MODELS_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Allowed taxonomies (single source of truth)
# ---------------------------------------------------------------------------
CATEGORIES: list[str] = ["Billing", "Technical", "Account", "Shipping", "Other"]
PRIORITIES: list[str] = ["Low", "Medium", "High", "Urgent"]
REVIEW_VALUES: list[str] = ["yes", "no"]

# Natural-language candidate labels for the zero-shot model. Using slightly
# richer descriptions helps the cross-encoder align the customer's wording
# with the intended business meaning. The final label is mapped back to the
# canonical short form.
CATEGORY_LABELS: dict[str, str] = {
    "Billing":   "billing, payment, invoice, refund, charge or subscription issue",
    "Technical": "technical, software bug, error, broken feature or malfunction",
    "Account":   "account access, login, password, profile or security issue",
    "Shipping":  "shipping, delivery, tracking, package lost or delayed order",
    "Other":     "general inquiry, feedback, complaint or request that fits none of the above",
}

# Priority uses bare-word candidate labels: experiments showed the descriptive
# phrasings biased the zero-shot model heavily toward "Low", whereas bare words
# ("Urgent","High","Medium","Low") produce a balanced Low/Medium split with some
# High/Urgent. True urgency is additionally enforced by CRITICAL_RISK_KEYWORDS,
# which route the clearly-critical tickets to human review regardless of score.
PRIORITY_LABELS: dict[str, str] = {
    "Urgent": "Urgent",
    "High":   "High",
    "Medium": "Medium",
    "Low":    "Low",
}

# Reverse-lookup helper: map any candidate label back to its canonical key.
def canonical_category(label: str) -> str:
    for key, desc in CATEGORY_LABELS.items():
        if label == key or label == desc or label.startswith(key):
            return key
    # Fall back to the raw short label if the model returned something clean.
    if label in CATEGORIES:
        return label
    return "Other"


def canonical_priority(label: str) -> str:
    for key, desc in PRIORITY_LABELS.items():
        if label == key or label == desc or label.startswith(key):
            return key
    if label in PRIORITIES:
        return label
    return "Medium"


# ---------------------------------------------------------------------------
# Hugging Face model identifiers
# ---------------------------------------------------------------------------
ZERO_SHOT_MODEL_ID = "MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33"
SENTIMENT_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DUPLICATE_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
TRANSFORMER_FINETUNE_MODEL_ID = "distilbert/distilbert-base-uncased"

# ---------------------------------------------------------------------------
# Confidence / review thresholds (tuneable)
# ---------------------------------------------------------------------------
# Zero-shot softmax scores are NOT calibrated probabilities. We treat them as
# relative ranking scores and use thresholds to decide how aggressively to
# route tickets to human review. Lowering a threshold = fewer reviews.
CATEGORY_CONFIDENCE_THRESHOLD = 0.40
PRIORITY_CONFIDENCE_THRESHOLD = 0.40
CONFIDENCE_MARGIN_THRESHOLD = 0.10  # gap between top-1 and top-2 candidates

# Critical-risk rules: keywords that force a review regardless of confidence,
# because a mis-classification here would be costly. Returns (category, priority)
# hints if matched.
CRITICAL_RISK_KEYWORDS = [
    "security breach", "data leak", "lost password", "account compromised",
    "chargeback", "legal", "lawsuit", "threat", "delete my account",
]

# When the model fails entirely (unavailable, OOM, network) we cannot predict;
# mark the ticket for review and leave category/priority empty.
MODEL_FAILURE_REVIEW = True

# ---------------------------------------------------------------------------
# Module toggles (optional modules must not break the app)
# ---------------------------------------------------------------------------
ENABLE_SENTIMENT = False      # set True once sentiment model is loaded
ENABLE_DUPLICATE_DETECTION = False

# Default backend used by services. Switch to "traditional" or "transformer"
# only after a supervised model has been trained and evaluated.
ACTIVE_BACKEND = "zero_shot"   # one of: zero_shot | traditional | transformer

# ---------------------------------------------------------------------------
# Branding (used by the UI)
# ---------------------------------------------------------------------------
APP_NAME = "SmartTriage AI"
APP_TAGLINE = "AI-powered support ticket triage with human-in-the-loop review"

COLORS = {
    "navy": "#0A2540",
    "blue": "#1E5AA8",
    "white": "#FFFFFF",
    "green": "#2E8B57",
    "yellow": "#E0A800",
    "red": "#C0392B",
    "grey": "#6C757D",
}
