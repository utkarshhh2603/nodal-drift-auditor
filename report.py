"""
CLI entrypoint: replay -> classify -> (optional) LLM explain -> report.

Usage:
    python data/generate_data.py   # once, to produce data/generated/*.csv
    python report.py
"""
import csv
import os
import time
from datetime import datetime
from pathlib import Path

from engine import classify, llm_explain, replay

OUT_PATH = Path(__file__).parent / "report.csv"


def main():
    transactions, settlements, refunds, fee_sweeps, nodal_ledger = replay.load_tables()
    merged = replay.run()

    total_days = len(merged)
    events = classify.find_events(merged)
    classified = classify.classify_events(events, settlements, refunds, fee_sweeps, None)
    classified = llm_explain.explain_unexplained(classified, transactions, settlements, refunds, fee_sweeps)

    drift_days = int((merged["drift"].abs() > classify.TOLERANCE).sum())

    # "fee_sweep_timing_resolved" events are the same incident as the
    # fee_sweep_timing event that flagged them, not a second incident, so
    # they're excluded from the incident count and value totals below.
    ACTION_NEEDED_BUCKETS = {"duplicate_settlement", "stuck_refund", "unexplained"}
    SELF_RESOLVING_BUCKETS = {"fee_sweep_timing"}
    incidents = [ev for ev in classified if ev.bucket not in {"fee_sweep_timing_resolved"}]
    value_at_risk = sum(abs(ev.delta) for ev in classified if ev.bucket in ACTION_NEEDED_BUCKETS)
    self_resolving_value = sum(abs(ev.delta) for ev in classified if ev.bucket in SELF_RESOLVING_BUCKETS)
    rule_classified = sum(1 for ev in incidents if ev.bucket != "unexplained")

    print(f"Transactions audited:   {len(transactions)}")
    print(f"Days audited:            {total_days}  ({drift_days} with a book-vs-bank mismatch)")
    print(f"Incidents detected:      {len(incidents)}")
    print(f"  Value at risk (action needed):   Rs.{value_at_risk:,.2f}")
    print(f"  Self-resolving timing noise:     Rs.{self_resolving_value:,.2f}  (delayed fee sweeps, not a real loss)")
    print(f"Detection: {rule_classified}/{len(incidents)} incidents explained same-day by rules; "
          f"{len(incidents) - rule_classified} handed to LLM review")
    print()

    bucket_counts = {}
    for ev in classified:
        bucket_counts[ev.bucket] = bucket_counts.get(ev.bucket, 0) + 1
    for bucket, count in sorted(bucket_counts.items()):
        print(f"  {bucket:<26} {count}")
    print()

    for ev in classified:
        sign = "+" if ev.delta > 0 else ""
        print(f"[{ev.date.date()}] {sign}Rs.{ev.delta:.2f}  ({ev.bucket}, {ev.confidence})")
        print(f"    {ev.detail}")

    rows = [["date", "delta", "bucket", "confidence", "detail"]]
    rows += [[ev.date.date(), ev.delta, ev.bucket, ev.confidence, ev.detail] for ev in classified]
    written_to = _write_csv_resilient(OUT_PATH, rows)
    print(f"\nWrote {written_to}")


def _write_csv_resilient(path, rows, retries=5, delay=0.3):
    """Writes via a temp file + atomic replace, retrying on PermissionError.

    On Windows, a file under a synced folder (OneDrive) or open in another
    program (Excel, a preview pane) can hold a transient lock right after
    it was last written. Falls back to a timestamped filename rather than
    crashing the whole report if the target path stays locked.
    """
    tmp_path = path.with_suffix(".csv.tmp")
    with open(tmp_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    last_error = None
    for attempt in range(retries):
        try:
            os.replace(tmp_path, path)
            return path
        except PermissionError as exc:
            last_error = exc
            time.sleep(delay)

    fallback = path.with_name(f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
    os.replace(tmp_path, fallback)
    print(f"Note: {path} is locked ({last_error}); wrote to {fallback} instead.")
    return fallback


if __name__ == "__main__":
    main()
