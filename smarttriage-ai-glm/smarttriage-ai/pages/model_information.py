"""PAGE 5 — Model Information.

Shows the active model identity, taxonomy, review thresholds, load status and
inference timing. Includes a section explaining the difference between zero-shot
inference and supervised training — so reviewers never confuse zero-shot scores
with calibrated probabilities.
"""
from __future__ import annotations

import streamlit as st

import app_utils
from config import settings


def render() -> None:
    app_utils.header("Model Information",
                     "Active backend, configuration, and the conceptual "
                     "difference between zero-shot and supervised ML.")

    backend = app_utils.cached_backend()

    st.subheader("Active model")
    rows = [
        ("Backend", settings.ACTIVE_BACKEND),
        ("Model name", "xtremedistil zero-shot (DeBERTa-free distilled encoder)"),
        ("Model identifier", settings.ZERO_SHOT_MODEL_ID),
        ("Hugging Face link",
         f"https://huggingface.co/{settings.ZERO_SHOT_MODEL_ID}"),
        ("Model revision", getattr(backend, "revision", "unknown") or "unknown"),
        ("Classification type",
         "Zero-shot cross-encoder (MNLI-style, multi-class softmax)"),
        ("Load time (s)",
         f"{getattr(backend, 'load_seconds', 0):.2f}" if backend else "—"),
        ("Load status",
         "loaded" if (backend and getattr(backend, "is_loaded", False))
         else f"unavailable: {getattr(backend, 'load_error', '')}"),
    ]
    st.table({"Property": [r[0] for r in rows], "Value": [r[1] for r in rows]})

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Allowed categories")
        for cat, desc in settings.CATEGORY_LABELS.items():
            st.markdown(f"**{cat}** — {desc}")
    with c2:
        st.subheader("Allowed priorities")
        for pri, desc in settings.PRIORITY_LABELS.items():
            st.markdown(f"**{pri}** — {desc}")

    st.subheader("Review thresholds (tuneable in config/settings.py)")
    t1, t2, t3 = st.columns(3)
    t1.metric("Category confidence ≥", settings.CATEGORY_CONFIDENCE_THRESHOLD)
    t2.metric("Priority confidence ≥", settings.PRIORITY_CONFIDENCE_THRESHOLD)
    t3.metric("Top-2 margin ≥", settings.CONFIDENCE_MARGIN_THRESHOLD)
    st.caption("A ticket is routed to review if category OR priority confidence "
               "falls below its threshold, the margin is too small, a critical-"
               f"risk keyword appears ({', '.join(settings.CRITICAL_RISK_KEYWORDS[:4])}…), "
               "or the model fails.")

    st.subheader("Optional modules")
    oc1, oc2 = st.columns(2)
    with oc1:
        st.markdown(f"**Sentiment** — `{settings.SENTIMENT_MODEL_ID}`")
        st.caption(f"Enabled: {settings.ENABLE_SENTIMENT}. "
                   "cardiffnlp/twitter-roberta-base-sentiment-latest. "
                   "Additional escalation signal only; never decides priority.")
    with oc2:
        st.markdown(f"**Duplicate detection** — `{settings.DUPLICATE_MODEL_ID}`")
        st.caption(f"Enabled: {settings.ENABLE_DUPLICATE_DETECTION}. "
                   "sentence-transformers/all-MiniLM-L6-v2. Future module.")

    st.subheader("Zero-shot inference vs supervised training")
    st.markdown(
        """
**Zero-shot inference (current backend)** does not require any labelled
support-ticket data. The same pre-trained model answers two independent
questions — *"which of these five categories best matches this text?"* and
*"which priority?"* — by scoring each candidate label against the ticket text
and applying a softmax. The resulting scores are **relative ranking scores, not
calibrated probabilities**: a score of 0.7 does not mean 70% likely-correct.
That is why a confidence threshold + margin + critical-risk rules route
ambiguous tickets to human review instead of trusting the number blindly.

**Supervised training (future)** takes human-verified labels
(`verified_training_data.csv`, produced in the Human Review Center) and fits a
dedicated model — TF-IDF + MultinomialNB / LogisticRegression / LinearSVC, or a
fine-tuned DistilBERT — specifically for this ticket taxonomy. Trained models
produce better-calibrated, faster, cheaper predictions, but only after enough
verified labels exist. The app keeps using the zero-shot backend until a
supervised model has been trained, evaluated on a locked holdout, and activated
via `settings.ACTIVE_BACKEND`.
        """
    )
    st.info("Switch backends by setting `ACTIVE_BACKEND` in "
            "`config/settings.py` to `zero_shot`, `traditional`, or "
            "`transformer` after a supervised model has been trained.")
