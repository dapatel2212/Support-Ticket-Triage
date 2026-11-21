"""Headless end-to-end pipeline run.

Processes data/tickets.csv through the zero-shot backend, writes predictions.csv
and outputs/results.csv, and validates the results against the assignment rules:
  * exact columns id, category, priority, needs_review
  * category in the allowed set (or empty when routed to review)
  * priority in the allowed set (or empty when routed to review)
  * needs_review is yes/no
  * every input id appears exactly once, no drops

Run:  python scripts/run_pipeline.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import settings
from services import export_service
from services.ticket_service import get_backend, process_tickets
from services.export_service import validate_results


def main() -> int:
    if not settings.TICKETS_CSV.exists():
        print(f"ERROR: {settings.TICKETS_CSV} not found")
        return 1

    df = pd.read_csv(settings.TICKETS_CSV)
    print(f"Loaded {len(df)} tickets ({df['id'].nunique()} unique ids) "
          f"from {settings.TICKETS_CSV}")

    print("Loading backend...")
    backend = get_backend()
    if not backend.is_loaded:
        print(f"Backend not loaded: {backend.load_error}")
        return 2
    print(f"Backend ready (revision={backend.revision}, "
          f"load={backend.load_seconds:.1f}s)")

    t0 = time.time()

    def cb(done, total, tid):
        if done % 25 == 0 or done == total:
            print(f"  processed {done}/{total} (last id={tid})", flush=True)

    preds = process_tickets(df, backend, progress_cb=cb)
    elapsed = time.time() - t0
    print(f"Processed {len(preds)} tickets in {elapsed:.1f}s "
          f"({elapsed/len(preds):.3f}s/ticket)")

    export_service.write_predictions(preds)
    results_path = export_service.write_results(preds)
    print(f"Wrote predictions -> {settings.PREDICTIONS_CSV}")
    print(f"Wrote results     -> {results_path}")

    results = export_service.build_results_frame(preds)
    input_ids = df["id"].astype(str).tolist()
    errors = validate_results(results, input_ids=input_ids)
    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(" -", e)
        return 3

    print("VALIDATION PASSED")
    print(f"  total rows        : {len(results)}")
    print(f"  unique ids        : {results['id'].nunique()}")
    print(f"  routed to review  : {(results['needs_review']=='yes').sum()}")
    print(f"  auto-classified   : {(results['needs_review']=='no').sum()}")
    print(f"  urgent tickets    : {(results['priority']=='Urgent').sum()}")
    print("\nCategory distribution:")
    print(results["category"].replace("", "(review)").value_counts().to_string())
    print("\nPriority distribution:")
    print(results["priority"].replace("", "(review)").value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
