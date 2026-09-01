"""
Measures detection accuracy across many synthetic batches, instead of
trusting a single anecdotal run.

For each seed: generates a fresh batch (data/generate_data.py knows which
transactions it seeded as incidents, via `ground_truth`), runs the real
audit pipeline (engine/replay.py -> engine/classify.py) exactly as
report.py does, and checks whether each seeded incident was (a) detected
at all and (b) bucketed under the correct root cause. Aggregates into a
per-type recall table plus overall precision.

Usage:
    python scripts/backtest.py [--seeds 20]
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import generate_data
from engine import classify, replay

TMP_DIR = Path(__file__).parent.parent / "data" / "_backtest_tmp"

# Maps a ground-truth noise type (from generate_data.NOISE_COUNTS) to the
# bucket name engine/classify.py is expected to assign it.
EXPECTED_BUCKET = {
    "duplicate_settlement": "duplicate_settlement",
    "stuck_refund": "stuck_refund",
    "late_fee_sweep": "fee_sweep_timing",
    "partial_refund_mismatch": "unexplained",  # by design - no rule covers this one
}


def run_one_seed(seed):
    out_dir = TMP_DIR / f"seed_{seed}"
    out_dir, ground_truth = generate_data.generate(seed=seed, out_dir=out_dir, write_csvs=True, verbose=False)

    merged = replay.run(data_dir=out_dir)
    transactions, settlements, refunds, fee_sweeps, _ = replay.load_tables(out_dir)
    events = classify.find_events(merged)
    classified = classify.classify_events(events, settlements, refunds, fee_sweeps, None)

    # Rule-classified events carry the txn_id they're about directly. A
    # genuinely "unexplained" event doesn't (correctly - the rule engine
    # really doesn't know which transaction caused it), so those are
    # matched back to a seeded incident by date instead: the generator
    # spaces every incident onto its own day (assign_noise_indices), so an
    # onset date uniquely identifies which incident produced that day's drift.
    by_txn_id = {gt["txn_id"]: gt for gt in ground_truth}
    by_onset_date = {gt["onset_date"]: gt for gt in ground_truth}

    def matching_incident(ev):
        return by_txn_id.get(ev.txn_id) or by_onset_date.get(ev.date.date())

    event_to_gt = {id(ev): matching_incident(ev) for ev in classified}
    hits_by_txn_id = {
        event_to_gt[id(ev)]["txn_id"]
        for ev in classified
        if event_to_gt[id(ev)] is not None and ev.bucket == EXPECTED_BUCKET[event_to_gt[id(ev)]["type"]]
    }

    per_incident = [
        {"seed": seed, "type": gt["type"], "txn_id": gt["txn_id"], "detected": gt["txn_id"] in hits_by_txn_id}
        for gt in ground_truth
    ]

    total_incidents = len(ground_truth)
    false_positive_candidates = [
        ev for ev in classified
        if ev.bucket != "fee_sweep_timing_resolved" and event_to_gt[id(ev)] is None
    ]

    shutil.rmtree(out_dir, ignore_errors=True)
    return per_incident, len(false_positive_candidates), total_incidents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()

    all_incidents = []
    total_false_positives = 0
    total_seeded = 0

    for seed in range(args.seeds):
        per_incident, false_positives, total_incidents = run_one_seed(seed)
        all_incidents.extend(per_incident)
        total_false_positives += false_positives
        total_seeded += total_incidents

    shutil.rmtree(TMP_DIR, ignore_errors=True)

    by_type = {}
    for row in all_incidents:
        by_type.setdefault(row["type"], []).append(row["detected"])

    print(f"Backtest across {args.seeds} seeds, {total_seeded} seeded incidents total\n")
    print(f"{'Incident type':<28} {'Detected':>10} {'Total':>8} {'Recall':>8}")
    overall_detected = 0
    for noise_type, hits in sorted(by_type.items()):
        detected = sum(hits)
        overall_detected += detected
        print(f"{noise_type:<28} {detected:>10} {len(hits):>8} {100*detected/len(hits):>7.1f}%")
    print(f"{'OVERALL':<28} {overall_detected:>10} {len(all_incidents):>8} {100*overall_detected/len(all_incidents):>7.1f}%")
    print(f"\nFalse-positive candidates (flagged, no matching seeded incident): {total_false_positives}")
    print("(Expected: 0 for duplicate_settlement/stuck_refund/fee_sweep_timing - clean")
    print(" transactions never carry drift; 'unexplained' is only ever the seeded")
    print(" partial_refund_mismatch cases by construction of this synthetic generator.)")


if __name__ == "__main__":
    main()
