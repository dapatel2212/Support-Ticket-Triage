# SmartTriage AI

> AI-powered support ticket triage with zero-shot classification, confidence-based human-in-the-loop review, and a clean path to future supervised ML.

**SmartTriage AI** is an intelligent customer support ticket classification and prioritization system. It automatically processes customer support messages, predicts their **category** and **priority**, detects uncertain predictions, and routes ambiguous or potentially critical tickets to a human review queue. The goal is to reduce manual ticket sorting and help support teams handle urgent customer issues efficiently.

The MVP runs entirely on a zero-shot transformer model — it requires **no labelled training data** — while keeping a clean, modular architecture ready for future supervised model training (TF-IDF baselines and a fine-tuned DistilBERT).

---

## Table of contents

1. [Project description](#1-project-description)
2. [Problem statement](#2-problem-statement)
3. [Project objectives](#3-project-objectives)
4. [Key features](#4-key-features)
5. [System architecture](#5-system-architecture)
6. [Application workflow](#6-application-workflow)
7. [Technology stack](#7-technology-stack)
8. [Exact ML models](#8-exact-ml-models)
9. [Zero-shot classification explained](#9-zero-shot-classification-explained)
10. [Category classification](#10-category-classification)
11. [Priority classification](#11-priority-classification)
12. [Confidence assessment](#12-confidence-assessment)
13. [Human-in-the-loop review](#13-human-in-the-loop-review)
14. [Dataset description](#14-dataset-description)
15. [Dataset limitations](#15-dataset-limitations)
16. [Input CSV format](#16-input-csv-format)
17. [Output CSV format](#17-output-csv-format)
18. [Project folder structure](#18-project-folder-structure)
19. [Installation](#19-installation)
20. [Local execution](#20-local-execution)
21. [Streamlit deployment](#21-streamlit-deployment)
22. [Future supervised training](#22-future-supervised-training)
23. [Data annotation process](#23-data-annotation-process)
24. [Model evaluation methodology](#24-model-evaluation-methodology)
25. [Scalability plan for 50,000 tickets/day](#25-scalability-plan-for-50000-ticketsday)
26. [Monitoring plan for the first month](#26-monitoring-plan-for-the-first-month)
27. [Limitations](#27-limitations)
28. [Future improvements](#28-future-improvements)
29. [Author and contributions](#29-author-and-contributions)
30. [Original assignment design note](#30-original-assignment-design-note)

---

## 1. Project description

SmartTriage AI ingests raw customer support tickets (subject + body), runs two independent zero-shot classification queries — one for **category**, one for **priority** — assesses the confidence of each prediction, and decides whether the ticket can be auto-classified or must be sent to a human reviewer. Every decision is logged with its scores, the top alternative label, the confidence margin and the review reason, so the team can audit and improve the system over time.

The application is a Streamlit dashboard with five pages (Overview, Ticket Processing, Human Review Center, Analytics, Model Information) backed by a modular Python codebase that separates ML logic, business services and UI.

## 2. Problem statement

Support teams receive large volumes of tickets in no particular order. Manual triage — reading each message, deciding a category, assigning a priority and escalating urgent cases — is slow, inconsistent and error-prone. Critical tickets can sit unread behind low-priority questions. There is no ground-truth label set available for these tickets, so a traditional supervised classifier cannot be trained out of the box.

## 3. Project objectives

- Automatically categorize every ticket into one of five categories (Billing, Technical, Account, Shipping, Other).
- Automatically assign one of four priorities (Low, Medium, High, Urgent), independently of category.
- Route uncertain, ambiguous or critical-risk tickets to a human review queue instead of trusting a weak prediction.
- Preserve every input ticket exactly once in the output — never silently drop or duplicate a ticket.
- Keep model predictions and human corrections in separate files so verified labels are never confused with auto-generated ones.
- Prepare the architecture for future supervised training on human-verified labels, without forcing the zero-shot model to act as the final production model.

## 4. Key features

- Zero-shot category + priority classification (no training data required).
- Confidence-based prediction with configurable review thresholds and margins.
- Human-in-the-loop review center with category/priority correction and approve/correct/reject decisions.
- Bulk ticket processing with a live progress bar.
- Single-ticket ad-hoc triage for quick testing.
- Analytics dashboard with Plotly charts (category, priority, urgent, review, arrival trends, processing time).
- CSV export of `results.csv` (exact required schema) and a richer `predictions.csv`.
- Pluggable backend: zero-shot today, trained TF-IDF / fine-tuned DistilBERT tomorrow.
- Optional sentiment module and future duplicate-detection module that never break the core flow.
- Automated tests for preprocessing, confidence/review logic and results-schema validation.

## 5. System architecture

```
            ┌──────────────────────────────────────────────────────────┐
            │                      Streamlit UI                        │
            │  app.py → st.navigation → 5 pages (pages/*.py)           │
            └───────────────▲──────────────────────────▲───────────────┘
                            │                          │
              app_utils (cached backend, state, theme) │
                            │                          │
            ┌───────────────┴──────────────────────────┴───────────────┐
            │                     services/                            │
            │  ticket_service  review_service  export_service          │
            └───────▲────────────────▲──────────────────▲──────────────┘
                    │                │                  │
            ┌───────┴───────┐ ┌──────┴───────┐ ┌────────┴────────┐
            │   ml/         │ │ models/      │ │  config/         │
            │ preprocessing │ │ zero_shot    │ │  settings.py     │
            │ category_*    │ │ sentiment    │ │  (thresholds,    │
            │ priority_*    │ │ duplicate_*  │ │   labels, paths) │
            │ confidence    │ │ trained_*    │ │                  │
            │ evaluation    │ │              │ │                  │
            │ training      │ │              │ │                  │
            └───────────────┘ └──────────────┘ └──────────────────┘
```

- **config/** — single source of truth for model IDs, candidate labels, thresholds and file paths.
- **models/** — model wrappers (zero-shot, sentiment, duplicate detector, trained-model loader).
- **ml/** — pure ML logic: preprocessing, classifiers, confidence, evaluation, training.
- **services/** — orchestration: process tickets, manage reviews, export results.
- **pages/** — Streamlit page render functions; **app.py** wires them with `st.navigation`.
- **data/** — input dataset and runtime output files (predictions, reviews, verified training data).
- **outputs/** — the deliverable `results.csv`.
- **tests/** — pytest suite.

## 6. Application workflow

1. **Load** — the user uploads a CSV (or uses the bundled `data/tickets.csv`) on the Ticket Processing page.
2. **Validate** — required columns (`id`, `subject`, `body`) are checked; missing subjects/bodies and duplicate IDs are reported but never cause a crash.
3. **Preprocess** — subject + body are normalized and combined into a prediction text. Negations and risk keywords (`not`, `never`, `cannot`, `failed`, `refund`, `urgent`) are deliberately preserved.
4. **Classify** — for each ticket, the zero-shot model answers two independent questions: *which category?* and *which priority?* The same loaded model instance is reused for every ticket.
5. **Assess confidence** — each prediction gets a confidence score, a top alternative, and a margin (top-1 − top-2).
6. **Decide review** — the ticket is routed to review if any confidence/margin threshold is missed, a critical-risk keyword is present, or the model failed.
7. **Persist** — `predictions.csv` and `outputs/results.csv` are written; every input id appears exactly once.
8. **Review (human)** — the Human Review Center shows pending tickets with the original message, the model prediction and the review reason. The reviewer corrects category/priority and approves, corrects or rejects. Decisions are stored separately in `human_reviews.csv`.
9. **Export verified data** — approved/corrected reviews are exported to `verified_training_data.csv` for the future supervised pipeline.

## 7. Technology stack

| Layer | Technology |
| --- | --- |
| Language | Python 3.11 / 3.12 |
| Frontend | Streamlit |
| ML | Hugging Face Transformers, PyTorch (CPU build) |
| Data | Pandas, NumPy |
| Visualization | Plotly |
| Traditional ML (future) | scikit-learn, TF-IDF |
| Model persistence | joblib (traditional), HF save format (transformer) |
| Storage | CSV now; SQLite next; PostgreSQL for production |
| Deployment | Streamlit Community Cloud |
| Version control | Git / GitHub |

## 8. Exact ML models

| Role | Model ID | HF link | Status |
| --- | --- | --- | --- |
| Zero-shot category + priority | `MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33` | https://huggingface.co/MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33 | **Active backend** |
| Sentiment (optional) | `cardiffnlp/twitter-roberta-base-sentiment-latest` | https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest | Optional module |
| Duplicate detection (future) | `sentence-transformers/all-MiniLM-L6-v2` | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | Future module |
| Fine-tuning base (future) | `distilbert/distilbert-base-uncased` | https://huggingface.co/distilbert/distilbert-base-uncased | Future supervised |

The zero-shot model is loaded once via Streamlit's `@st.cache_resource` and reused for both the category and priority tasks — it is never reloaded per ticket.

## 9. Zero-shot classification explained

A zero-shot classifier answers *"which of these candidate labels best matches this text?"* **without ever being trained on examples of those labels.** The model is a distilled transformer fine-tuned on natural-language-inference (NLI): it treats each candidate label as a hypothesis ("this text is about billing") and scores how entailment-likely it is. A softmax over the candidates produces a ranking.

**Crucially, the softmax scores are relative ranking scores, NOT calibrated probabilities.** A score of 0.7 does *not* mean "70% likely correct." That is why SmartTriage AI does not trust a single score: it combines the top score, the margin to the runner-up, and critical-risk rules to decide whether to auto-classify or route to a human. Calibrated, trustworthy probabilities come only from a supervised model trained on verified labels — see [§22](#22-future-supervised-training).

## 10. Category classification

- **Input:** normalized `subject + body`.
- **Output:** one of `Billing, Technical, Account, Shipping, Other` plus a confidence score and a top alternative.
- **Candidate labels** (natural-language, to align the cross-encoder with business meaning) are defined in `config/settings.py::CATEGORY_LABELS`.
- The result is canonicalized back to the short taxonomy via `canonical_category()`.

## 11. Priority classification

- **Input:** the same `subject + body` text.
- **Output:** one of `Low, Medium, High, Urgent` plus a confidence score and a top alternative.
- **Independent of category** — a separate zero-shot query with priority-specific candidate labels.
- **Candidate labels are bare words** (`Urgent`, `High`, `Medium`, `Low`). Experiments showed that descriptive priority labels biased the model heavily toward `Low`; bare words produce a balanced Low/Medium split. True urgency is additionally enforced by the critical-risk keyword rules in [§12](#12-confidence-assessment), which route clearly-critical tickets to human review.

## 12. Confidence assessment

For every ticket the system stores: predicted category, category confidence, predicted priority, priority confidence, top alternative category, top alternative priority, both confidence margins, and the review decision (yes/no) with reasons.

A ticket is routed to human review when **any** of these holds:

| Rule | Default threshold (`config/settings.py`) |
| --- | --- |
| Category confidence below threshold | `CATEGORY_CONFIDENCE_THRESHOLD = 0.40` |
| Priority confidence below threshold | `PRIORITY_CONFIDENCE_THRESHOLD = 0.40` |
| Top-1 / top-2 margin too small | `CONFIDENCE_MARGIN_THRESHOLD = 0.10` |
| Critical-risk keyword present | `CRITICAL_RISK_KEYWORDS` (e.g. *security breach*, *data leak*, *account compromised*) |
| Category or priority prediction failed | always review |
| Model unavailable | always review |

All thresholds are tuneable from configuration and are displayed on the Model Information page.

## 13. Human-in-the-loop review

The Human Review Center lists every ticket with `needs_review=yes`. For each ticket the reviewer can:

1. read the **original** customer message (subject + body, untouched);
2. see the **model prediction** (category, priority, both confidence scores, top alternatives, review reason);
3. **correct** the category and/or priority;
4. **approve** (endorse the model's prediction), **correct & save**, or **reject** the prediction;
5. add **notes**.

Reviews are stored in `data/human_reviews.csv` — **separate** from `predictions.csv` — and the original model prediction column is never overwritten. Rejected predictions are not exported as labels.

## 14. Dataset description

The provided dataset `data/tickets.csv` contains **300 support tickets** with columns:

| Column | Description |
| --- | --- |
| `id` | Unique ticket id (`T-0001` … `T-0300`) |
| `received_at` | Timestamp the ticket was received |
| `customer` | Customer name |
| `subject` | Short subject line |
| `body` | Full message body |

There are **no ground-truth category or priority labels** in the dataset.

## 15. Dataset limitations

- **No labels:** category and priority labels are absent, so supervised training cannot be done directly. The MVP therefore uses zero-shot inference, and supervised training waits for human-verified labels from the review queue.
- **Small size:** 300 tickets is enough for an MVP run but far too small for a robust supervised model (especially a transformer).
- **Class imbalance is unknown:** without labels we cannot know the true category/priority distribution, so stratified splitting for future training will depend on the labels reviewers actually produce.
- **One null body** and possible edge cases (empty subject, duplicate id) are handled defensively but cannot be "fixed" — they reflect real-world noisy input.
- **No temporal holdout:** the dataset spans a short window; drift analysis will need a longer history.

## 16. Input CSV format

```csv
id,received_at,customer,subject,body
T-0001,2026-11-16 07:37,Emma Johansson,Wrong item received,"Hello support, ..."
```

**Required columns:** `id`, `subject`, `body`. Optional but recommended: `received_at`, `customer`. Missing optional columns are tolerated; missing required columns are reported and block processing.

## 17. Output CSV format

`outputs/results.csv` contains **exactly** these columns, one row per input ticket:

```csv
id,category,priority,needs_review
T-0001,Shipping,Low,yes
T-0004,Billing,Medium,no
```

- `category` ∈ {Billing, Technical, Account, Shipping, Other} **or empty** (when routed to review).
- `priority` ∈ {Low, Medium, High, Urgent} **or empty** (when routed to review).
- `needs_review` ∈ {yes, no}.
- Every input `id` appears **exactly once**; no ticket is dropped or duplicated.

### Observed run on the provided dataset

Running the bundled pipeline (`python scripts/run_pipeline.py`) on the 300 provided tickets produced:

| Metric | Value |
| --- | --- |
| Tickets processed | 300 |
| Unique ids in results | 300 |
| Auto-classified (`needs_review=no`) | 156 |
| Routed to review (`needs_review=yes`) | 144 (48%) |
| Urgent tickets | 2 |

Category distribution (predicted): Technical 76, Billing 68, Account 57, Other 50, Shipping 49.
Priority distribution (predicted): Low 146, Medium 133, High 19, Urgent 2.

These are **descriptive statistics of the system's predictions on this dataset**, not model accuracy. The 48% review rate is intentional and conservative: zero-shot scores are not calibrated, so ambiguous cases are routed to humans rather than trusted blindly. Thresholds can be tuned in `config/settings.py` to trade automation rate against review load.

## 18. Project folder structure

```
smarttriage-ai/
├── app.py                      # Streamlit entry; st.navigation wires 5 pages
├── app_utils.py                # cached backend, session state, theme
├── config/
│   └── settings.py             # model IDs, labels, thresholds, paths
├── data/
│   ├── tickets.csv             # input dataset (300 tickets)
│   ├── predictions.csv         # full predictions (runtime output)
│   ├── human_reviews.csv       # human corrections (separate from predictions)
│   └── verified_training_data.csv  # exported labels for supervised training
├── models/
│   ├── zero_shot.py            # zero-shot classifier wrapper
│   ├── sentiment.py            # optional sentiment module
│   ├── duplicate_detector.py   # future duplicate-detection module
│   ├── trained_loader.py       # loads trained supervised backend if active
│   └── trained/                # trained artifacts (.gitkeep'd; weights ignored)
├── ml/
│   ├── preprocessing.py
│   ├── category_classifier.py
│   ├── priority_classifier.py
│   ├── confidence.py
│   ├── evaluation.py
│   └── training.py             # future TF-IDF + DistilBERT training pipeline
├── services/
│   ├── ticket_service.py
│   ├── review_service.py
│   └── export_service.py
├── pages/
│   ├── dashboard.py            # Overview
│   ├── ticket_processing.py
│   ├── human_review.py
│   ├── analytics.py
│   └── model_information.py
├── outputs/
│   └── results.csv             # deliverable
├── scripts/
│   ├── run_pipeline.py         # headless end-to-end run
│   └── generate_dataset.py     # synthetic generator (utility only)
├── tests/
│   ├── test_preprocessing.py
│   ├── test_classification.py
│   └── test_results.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

## 19. Installation

```bash
git clone <your-fork-url>.git
cd smarttriage-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11 or 3.12 is recommended. `requirements.txt` pins the CPU build of PyTorch to stay within Streamlit Community Cloud's resource limits. If you run locally with a GPU and want GPU inference, install the matching CUDA build of `torch` from https://pytorch.org instead.

The first run downloads the zero-shot model (~127 MB) from the Hugging Face Hub into the local HF cache; subsequent runs reuse it.

## 20. Local execution

```bash
# 1. Run the headless pipeline directly on data/tickets.csv
python scripts/run_pipeline.py

# 2. Launch the dashboard
streamlit run app.py
```

Then open the printed URL (default http://localhost:8501). Go to **Ticket Processing → Use bundled dataset → Process all tickets**, then explore the Overview, Human Review Center, Analytics and Model Information pages.

### Run the tests

```bash
pytest tests/ -q
```

## 21. Streamlit deployment

1. Push the project to a GitHub repository (do not commit model weights — `.gitignore` already excludes `models/trained/*` and weight files).
2. Go to https://share.streamlit.io and click **New app**.
3. Select your repository and the branch, set the main file to `app.py`.
4. Add the dependencies from `requirements.txt` (Streamlit installs them automatically).
5. (Optional) If you hit Hugging Face rate limits, set a `HF_TOKEN` secret in the app's advanced settings.
6. Deploy. The first cold start downloads the model; subsequent runs use the cache.

The CPU-only `torch` pin keeps memory usage within Community Cloud's free tier. The model is loaded once per process via `@st.cache_resource`, so reruns do not reload it.

## 22. Future supervised training

The zero-shot model is a starting point, not the final production model. The codebase prepares for proper supervised ML:

- **Traditional baselines** (`ml/training.py`): `MultinomialNB`, `LogisticRegression`, `LinearSVC`, each wrapped in a `TfidfVectorizer` pipeline (word 1-2 grams, char n-grams where useful, `class_weight="balanced"`). Separate models are trained for category and priority.
- **Transformer fine-tuning** (`ml/training.py::train_distilbert`): `distilbert-base-uncased` with the HF `Trainer`. Heavy — only run once enough verified labels exist.
- **Leakage-safe splitting** (`ml/evaluation.py::stratified_split_with_holdout`): a final locked holdout is carved off first, then train/val. The holdout is used ONLY for the final reported metrics.
- **Model saving**: traditional pipelines → `joblib` under `models/trained/`; transformer → HF save format. A `meta.json` records the metrics of every saved model.
- **Backend switching**: set `ACTIVE_BACKEND` in `config/settings.py` to `traditional` or `transformer` once a supervised model has been trained, evaluated and activated. If no artifact exists, the app transparently falls back to the zero-shot backend.

Run training manually:

```bash
# after exporting verified_training_data.csv from the Human Review Center
python -m ml.training --task both --out models/trained
# optionally also fine-tune DistilBERT (needs lots of verified data)
python -m ml.training --task both --distilbert
```

**Never train on unverified model predictions.** Only `verified_training_data.csv` (human-approved or corrected labels) is a valid training input. If pseudo-labeling is ever introduced, pseudo-labels must be clearly flagged and kept separate from verified labels.

## 23. Data annotation process

1. Process tickets on the Ticket Processing page.
2. Open the Human Review Center; tickets with `needs_review=yes` are queued with the model prediction, scores and review reason.
3. For each ticket, the reviewer reads the original message, corrects the category/priority if needed, and chooses **Approve**, **Correct & save**, or **Reject**.
4. Reviews are stored in `data/human_reviews.csv` (model prediction kept alongside the correction).
5. Click **Export verified_training_data.csv** to produce the supervised-training input (`ticket_id, subject, body, category, priority`). Only `approved` and `corrected` decisions are exported; `rejected` rows are dropped.

### Annotation template

`data/verified_training_data.csv`:

```csv
ticket_id,subject,body,category,priority
T-0004,Login issue,I cannot log in...,Account,High
```

Reviewers should aim for label consistency: agree on what counts as "Urgent" vs "High", and what distinguishes "Technical" from "Account". A small shared guideline document is recommended before bulk annotation.

## 24. Model evaluation methodology

For each task (category, priority) separately:

1. Load `verified_training_data.csv`.
2. Carve a **final holdout** (e.g. 20%) that is touched only once for the final report.
3. Split the remainder into train/validation (stratified when class counts permit).
4. Train every candidate (MultinomialNB, LogisticRegression, LinearSVC, and optionally DistilBERT) with the **same** train/val/holdout split.
5. Evaluate on the locked holdout: **accuracy, precision (macro), recall (macro), F1 (macro), F1 (weighted), confusion matrix, full classification report**.
6. Pick the best model by **macro F1** on the holdout (macro F1 is robust to class imbalance).
7. Persist the winner with `joblib` / HF save format and a `meta.json` recording its metrics.

All metrics are computed from real labelled data via scikit-learn — **no metric is ever fabricated.** Candidate models are compared with identical methodology so the comparison is fair.

## 25. Scalability plan for 50,000 tickets/day

The Streamlit MVP is a single-process app and is **not** load-tested for 50,000 tickets/day — do not claim it has achieved that throughput. The production design below describes how to reach it:

- **Asynchronous ingestion:** Kafka or RabbitMQ topic as the entry point; producers publish raw tickets, consumers pull batches.
- **Parallel inference workers:** stateless worker processes (or containers) each holding one loaded zero-shot model instance; a Kubernetes deployment scales horizontally with queue depth. CPU inference of the distilled model is ~80 ms/ticket on this sandbox, so a handful of workers comfortably exceeds 50k/day.
- **Batch processing:** workers call the HF pipeline with batches of N texts per forward pass instead of one-at-a-time, cutting per-ticket overhead.
- **Database persistence:** SQLite for local dev → PostgreSQL (with indexes on `category`, `priority`, `needs_review`, `received_at`) for production; predictions and reviews become tables, not CSV files.
- **Model versioning:** every trained artifact is saved under `models/trained/<task>_<model>_<version>.joblib` with a `meta.json`; a model registry (or MLflow) tracks which version is active and its holdout metrics.
- **Logging & exception handling:** structured JSON logs per ticket (`ticket_id`, `inference_ms`, `backend`, `version`, `status`); failed inferences are caught and routed to review rather than crashing the worker.
- **Human-review queues:** review work items stored in the database with a claim/lease mechanism so multiple reviewers do not double-handle the same ticket.
- **Decoupled UI:** Streamlit becomes a read-only dashboard over the database; triage itself happens in the worker fleet.

## 26. Monitoring plan for the first month

- **Throughput:** tickets processed per minute/hour/day per worker and overall; alert if sustained throughput drops below the incoming rate (queue growing).
- **Latency:** p50 / p95 / p99 inference time per ticket and per batch; alert if p95 exceeds the SLO.
- **Failure rates:** ratio of model-failure reviews and worker exceptions; alert above a threshold (e.g. > 2%).
- **Model prediction distribution:** daily category and priority distributions; a sudden shift may indicate drift or an upstream data change.
- **Human-review rate:** percentage routed to review; a sustained jump signals the model is becoming less decisive or input is shifting.
- **Verified-label evaluation:** once reviewers have approved a labelled set, run the active model against it weekly and report macro F1 / accuracy — this is the only true quality signal (prediction statistics are not accuracy).
- **Model drift:** track the distribution of confidence scores and the review rate over time; trigger a retraining investigation when drift is detected.

## 27. Limitations

- Zero-shot scores are **not calibrated probabilities**; do not interpret a 0.7 score as 70% confidence.
- The dataset has **no labels**, so no accuracy can be reported for the current MVP. Any number that looks like accuracy would be fabricated.
- Priority urgency detection is conservative: bare-word labels balance Low/Medium but under-detect Urgent; truly critical tickets are caught by keyword rules and routed to review.
- The review rate (~48% on this dataset) is intentionally conservative; tune thresholds to trade automation against review load.
- The Streamlit app is single-process; it is not designed for 50,000 tickets/day without the production architecture in [§25](#25-scalability-plan-for-50000-ticketsday).
- Sentiment and duplicate-detection modules are scaffolds and are not wired into the active decision path.

## 28. Future improvements

- Train supervised TF-IDF baselines and a fine-tuned DistilBERT on verified labels; switch `ACTIVE_BACKEND` once evaluated.
- Activate the sentiment module as an additional escalation signal.
- Activate semantic duplicate detection to cluster near-duplicate tickets.
- Add RAG-based response suggestions using the ticket text and a knowledge base.
- Move persistence from CSV to PostgreSQL.
- Add Prometheus metrics and a Grafana dashboard for the monitoring plan.
- Add an A/B framework to compare backend versions on live traffic before promotion.

## 29. Author and contributions

Author and contribution details are intentionally left blank — no real names, GitHub links, deployment URLs, or benchmark numbers are invented. Add your own author/contributor information here when you fork the project.

## 30. Original assignment design note

> *How would the system operate at 50,000 tickets per day, and how would you monitor it a month later?*

- **Throughput & scaling at 50k/day:** ingest via Kafka/RabbitMQ; run stateless inference workers (each holding one loaded zero-shot model) behind an autoscaling Kubernetes deployment; use batched `pipeline()` calls and PostgreSQL for persistence. With ~80 ms/ticket CPU inference, a modest worker fleet clears 50k/day with headroom.
- **Model failures:** every inference is wrapped in exception handling; on failure or model-unavailability the ticket is marked `needs_review=yes` with empty category/priority and pushed to the review queue — no ticket is lost and no worker crashes.
- **Routing uncertain tickets:** a ticket goes to review when category/priority confidence is below threshold, the top-1/top-2 margin is too small, a critical-risk keyword matches, or the model fails — keeping humans in the loop exactly where the model is weakest.
- **Latency & reliability monitoring:** track p50/p95/p99 inference time, throughput vs. incoming rate, failure/exception rate, and queue depth; alert on SLO breaches so problems surface before users do.
- **Quality after one month:** measure the active model against the human-verified labelled set accumulated in the review queue (macro F1, accuracy, per-class precision/recall, confusion matrix) — this is the only true quality signal, since prediction statistics are not accuracy.
- **Drift detection & retraining:** watch the daily prediction distribution, confidence-score distribution and review rate for drift; when drift or a quality drop is detected, export fresh verified labels and retrain/evaluate candidate models on a locked holdout, then promote the winner via the versioned model registry.
