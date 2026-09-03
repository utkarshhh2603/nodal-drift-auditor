"""
CLI entrypoint: replay -> classify -> (optional) LLM explain -> report.

Usage:
    python data/generate_data.py   # once, to produce data/generated/*.csv
    python report.py
"""
import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

from engine import classify, llm_explain, remediate, replay

OUT_PATH = Path(__file__).parent / "report.csv"


# "fee_sweep_timing_resolved" events are the same incident as the
# fee_sweep_timing event that flagged them, not a second incident, so
# they're excluded from the incident count and value totals below.
ACTION_NEEDED_BUCKETS = {"duplicate_settlement", "stuck_refund", "chargeback_duplicate", "unexplained"}
SELF_RESOLVING_BUCKETS = {"fee_sweep_timing"}


def compute_report(data_dir=None):
    """Runs the full pipeline and returns everything report.py's CLI output
    and dashboard/build.py's HTML both need, so neither has to re-derive it."""
    transactions, settlements, refunds, fee_sweeps, chargebacks, nodal_ledger = replay.load_tables(data_dir)
    merged = replay.run(data_dir)

    total_days = len(merged)
    events = classify.find_events(merged)
    classified = classify.classify_events(events, settlements, refunds, fee_sweeps, chargebacks)
    classified = llm_explain.explain_unexplained(classified, transactions, settlements, refunds, fee_sweeps, chargebacks)

    drift_days = int((merged["drift"].abs() > classify.TOLERANCE).sum())
    incidents = [ev for ev in classified if ev.bucket != "fee_sweep_timing_resolved"]
    value_at_risk = sum(abs(ev.delta) for ev in classified if ev.bucket in ACTION_NEEDED_BUCKETS)
    self_resolving_value = sum(abs(ev.delta) for ev in classified if ev.bucket in SELF_RESOLVING_BUCKETS)
    rule_classified = sum(1 for ev in incidents if ev.bucket != "unexplained")

    bucket_counts = {}
    for ev in classified:
        bucket_counts[ev.bucket] = bucket_counts.get(ev.bucket, 0) + 1

    return {
        "transactions_audited": len(transactions),
        "total_days": total_days,
        "drift_days": drift_days,
        "classified": classified,
        "incidents": incidents,
        "value_at_risk": value_at_risk,
        "self_resolving_value": self_resolving_value,
        "rule_classified": rule_classified,
        "llm_classified": len(incidents) - rule_classified,
        "bucket_counts": bucket_counts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None, help="Directory with the 6 recorded CSVs (default: data/generated/)")
    parser.add_argument("--apply-fixes", action="store_true",
                         help="Actually correct the incidents that are safe to fix without moving "
                              "money (duplicate settlement/chargeback rows), write the corrected "
                              "books to <data-dir>/corrected/, and prove each fix worked by "
                              "re-auditing. Everything else gets a numbered checklist instead.")
    args = parser.parse_args()

    r = compute_report(args.data_dir)

    print(f"Transactions audited:   {r['transactions_audited']}")
    print(f"Days audited:            {r['total_days']}  ({r['drift_days']} with a book-vs-bank mismatch)")
    print(f"Incidents detected:      {len(r['incidents'])}")
    print(f"  Value at risk (action needed):   Rs.{r['value_at_risk']:,.2f}")
    print(f"  Self-resolving timing noise:     Rs.{r['self_resolving_value']:,.2f}  (delayed fee sweeps, not a real loss)")
    print(f"Detection: {r['rule_classified']}/{len(r['incidents'])} incidents explained same-day by rules; "
          f"{r['llm_classified']} handed to LLM review")
    print()

    for bucket, count in sorted(r["bucket_counts"].items()):
        print(f"  {bucket:<26} {count}")
    print()

    for ev in r["classified"]:
        sign = "+" if ev.delta > 0 else ""
        print(f"[{ev.date.date()}] {sign}Rs.{ev.delta:.2f}  ({ev.bucket}, {ev.confidence})")
        print(f"    {ev.detail}")
        if ev.suggested_fix:
            print(f"    Fix: {ev.suggested_fix}")

    if args.apply_fixes:
        _run_remediation(r, args.data_dir)

    rows = [["date", "delta", "bucket", "confidence", "txn_id", "detail", "suggested_fix", "resolution_steps"]]
    rows += [
        [ev.date.date(), ev.delta, ev.bucket, ev.confidence, ev.txn_id, ev.detail, ev.suggested_fix,
         " | ".join(ev.resolution_steps)]
        for ev in r["classified"]
    ]
    # Writing against a --data-dir batch goes alongside that data instead of
    # the default report.csv, so a one-off run (e.g. the clean-batch check)
    # never clobbers the main demo's report.
    out_path = Path(args.data_dir) / "report.csv" if args.data_dir else OUT_PATH
    written_to = _write_csv_resilient(out_path, rows)
    print(f"\nWrote {written_to}")


def _run_remediation(r, data_dir_arg):
    print("\n" + "=" * 70)
    data_dir = Path(data_dir_arg) if data_dir_arg else replay.DATA_DIR
    corrected_dir = data_dir / "corrected"

    corrections = remediate.apply_corrections(r["classified"], data_dir, corrected_dir)
    needs_human = [
        ev for ev in r["incidents"]
        if not ev.auto_fix and ev.bucket != "fee_sweep_timing"
    ]

    if corrections:
        verified = remediate.verify_corrections(corrections, corrected_dir)
        total_corrected = sum(c["value_corrected"] for c in verified)
        print(f"AUTO-FIXED BY THE AGENT ({len(verified)} incident(s), "
              f"Rs.{total_corrected:,.2f} corrected)")
        for c in verified:
            resolved = abs(c["remaining_drift_on_date"]) <= classify.TOLERANCE
            status = "verified: drift on this date is now zero" if resolved else \
                f"WARNING: Rs.{c['remaining_drift_on_date']:.2f} still unexplained on this date"
            print(f"  [{c['date']}] {c['bucket']} ({c['txn_id']}): removed {', '.join(c['row_ids'])} "
                  f"from {c['table']}.csv - {status}")
        print(f"\nCorrected books written to: {corrected_dir}")
        print("(The original data_dir is untouched - the corrected copy is a separate output.)")
    else:
        print("AUTO-FIXED BY THE AGENT: none of this batch's incidents were safe to auto-fix.")

    print(f"\nNEEDS HUMAN ACTION ({len(needs_human)} incident(s) - moving money or an unclear "
          f"cause, neither of which the agent is authorized to act on alone):")
    for i, ev in enumerate(needs_human, 1):
        print(f"\n  {i}. [{ev.date.date()}] {ev.bucket} ({ev.txn_id or 'no single transaction'}, "
              f"Rs.{abs(ev.delta):,.2f})")
        for j, step in enumerate(ev.resolution_steps, 1):
            print(f"     {j}. {step}")
    print("\n" + "=" * 70)


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
