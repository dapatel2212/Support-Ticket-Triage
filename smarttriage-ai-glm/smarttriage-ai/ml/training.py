"""Future supervised training pipeline.

This module prepares the architecture for training and comparing traditional ML
and transformer-based classifiers on HUMAN-VERIFIED labels only. It must NEVER
be fed raw model predictions as ground truth — see the README and the
``verified_training_data.csv`` produced by the Human Review Center.

Pipeline:
  1. Load verified labels from ``data/verified_training_data.csv``
     (columns: ticket_id, subject, body, category, priority).
  2. Stratified train/val/holdout split (leakage-safe).
  3. Train three traditional baselines per task
     (MultinomialNB, LogisticRegression, LinearSVC) on TF-IDF features
     (word + char n-grams, class balancing where useful).
  4. Evaluate each candidate with the SAME methodology (macro F1 on the
     locked holdout) and pick the best.
  5. Persist artifacts under ``models/trained/`` with joblib.
  6. (Optional) Fine-tune distilbert-base-uncased with HF Trainer.

Run manually:
    python -m ml.training --task both --out models/trained

This module is intentionally NOT executed at app startup. The app keeps using
the zero-shot backend until a supervised model has been trained, evaluated and
explicitly activated via ``settings.ACTIVE_BACKEND``.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from config import settings
from ml.evaluation import EvaluationResult, evaluate, stratified_split_with_holdout
from ml.preprocessing import combine_subject_body

logger = logging.getLogger(__name__)

ALLOWED_LABELS = {
    "category": settings.CATEGORIES,
    "priority": settings.PRIORITIES,
}


@dataclass
class CandidateModel:
    name: str
    pipeline: Pipeline
    result: EvaluationResult


def _build_tfidf() -> TfidfVectorizer:
    """Word + char n-gram TF-IDF. Negation/risk words are preserved."""
    return TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
        min_df=2,
    )


def _candidate_pipelines() -> dict:
    """Three traditional baselines, each wrapped in a TF-IDF Pipeline."""
    return {
        "MultinomialNB": Pipeline([
            ("tfidf", _build_tfidf()),
            ("clf", MultinomialNB()),
        ]),
        "LogisticRegression": Pipeline([
            ("tfidf", _build_tfidf()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "LinearSVC": Pipeline([
            ("tfidf", _build_tfidf()),
            ("clf", LinearSVC(class_weight="balanced", max_iter=5000)),
        ]),
    }


def load_verified_data(path: Path = settings.VERIFIED_TRAINING_CSV) -> pd.DataFrame:
    """Load human-verified training data. Raises if file is missing/empty."""
    if not path.exists():
        raise FileNotFoundError(
            f"No verified training data at {path}. Have reviewers approved "
            "tickets in the Human Review Center and exported them first?")
    df = pd.read_csv(path)
    required = ["subject", "body", "category", "priority"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"verified_training_data missing columns: {missing}")
    df = df.dropna(subset=["category", "priority"]).copy()
    df["text"] = df.apply(lambda r: combine_subject_body(r["subject"], r["body"]), axis=1)
    # Reject rows whose labels are outside the allowed taxonomies.
    df = df[df["category"].isin(ALLOWED_LABELS["category"])]
    df = df[df["priority"].isin(ALLOWED_LABELS["priority"])]
    if df.empty:
        raise ValueError("Verified training data is empty after filtering.")
    return df


def train_task(df: pd.DataFrame, task: str) -> dict:
    """Train and evaluate all three baselines for one task; return the best."""
    labels = ALLOWED_LABELS[task]
    y = df[task].astype(str).tolist()
    X = df["text"].astype(str).tolist()
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split_with_holdout(X, y)

    candidates: list[CandidateModel] = []
    for name, pipe in _candidate_pipelines().items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        result = evaluate(y_test, y_pred, labels=labels, task=task)
        logger.info("[%s] %s -> macro F1=%.4f acc=%.4f",
                    task, name, result.f1_macro, result.accuracy)
        candidates.append(CandidateModel(name=name, pipeline=pipe, result=result))

    best = max(candidates, key=lambda c: c.result.f1_macro)
    return {"best": best, "candidates": candidates,
            "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test)}


def save_best(task: str, candidate: CandidateModel, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{task}_tfidf_{candidate.name.lower()}.joblib"
    joblib.dump(candidate.pipeline, path)
    meta = {
        "task": task, "model": candidate.name,
        "metrics": candidate.result.to_dict(),
        "labels": candidate.result.labels,
    }
    (out_dir / f"{task}_meta.json").write_text(json.dumps(meta, indent=2))
    return path


# ---------------------------------------------------------------------------
# Optional DistilBERT fine-tuning (transformer-based)
# ---------------------------------------------------------------------------
def train_distilbert(task: str, df: pd.DataFrame, out_dir: Path,
                     epochs: int = 3, batch_size: int = 16) -> Path:
    """Fine-tune distilbert-base-uncased for one task.

    Only call this once you have enough VERIFIED labelled data (rule of thumb:
    >= a few hundred examples per class). It is intentionally heavy and is not
    part of the default MVP path.
    """
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)
    from datasets import Dataset
    from sklearn.preprocessing import LabelEncoder

    labels = ALLOWED_LABELS[task]
    le = LabelEncoder().fit(labels)
    y = le.transform(df[task].astype(str))
    X = df["text"].astype(str).tolist()
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split_with_holdout(X, y)

    tok = AutoTokenizer.from_pretrained(settings.TRANSFORMER_FINETUNE_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        settings.TRANSFORMER_FINETUNE_MODEL_ID, num_labels=len(labels))

    def enc(texts, ys):
        ds = Dataset.from_dict({"text": texts, "label": ys})
        ds = ds.map(lambda b: tok(b["text"], truncation=True, padding="max_length",
                                  max_length=128), batched=True)
        return ds

    args = TrainingArguments(
        output_dir=str(out_dir / f"distilbert_{task}"),
        num_train_epochs=epochs, per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size, eval_strategy="epoch",
        save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="f1", logging_steps=50,
    )

    def compute_metrics(eval_pred):
        import numpy as np
        from sklearn.metrics import f1_score, accuracy_score
        logits, labels_ = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"f1": f1_score(labels_, preds, average="macro"),
                "accuracy": accuracy_score(labels_, preds)}

    trainer = Trainer(model=model, args=args,
                      train_dataset=enc(X_train, y_train),
                      eval_dataset=enc(X_val, y_val),
                      compute_metrics=compute_metrics)
    trainer.train()
    save_path = out_dir / f"distilbert_{task}_final"
    trainer.save_model(str(save_path))
    tok.save_pretrained(str(save_path))
    joblib.dump(le, out_dir / f"distilbert_{task}_label_encoder.joblib")
    return save_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Train supervised triage models.")
    ap.add_argument("--task", choices=["category", "priority", "both"], default="both")
    ap.add_argument("--data", default=str(settings.VERIFIED_TRAINING_CSV))
    ap.add_argument("--out", default=str(settings.TRAINED_MODELS_DIR))
    ap.add_argument("--distilbert", action="store_true",
                    help="also fine-tune DistilBERT (heavy, needs lots of data)")
    args = ap.parse_args()

    df = load_verified_data(Path(args.data))
    out_dir = Path(args.out)
    tasks = ["category", "priority"] if args.task == "both" else [args.task]
    for task in tasks:
        report = train_task(df, task)
        path = save_best(task, report["best"], out_dir)
        logger.info("Saved best %s model -> %s (macro F1=%.4f)",
                    task, path, report["best"].result.f1_macro)
        if args.distilbert:
            train_distilbert(task, df, out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
