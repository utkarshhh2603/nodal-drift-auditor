"""
Throughput benchmark: how fast does replay + classify run at scale?

Track 04's bar explicitly asks for "throughput plus measured accuracy."
scripts/backtest.py answers the accuracy half (100% recall, 0 false
positives) on ~460-transaction batches; this answers the throughput half
at 10,000+ transactions, since a batch that size never otherwise gets
generated or audited in this repo's normal demo flow.

Uses clean=True (no seeded incidents) - correctness is already proven by
scripts/backtest.py, so this only measures how fast the pipeline processes
volume, not whether it's accurate at volume.

Usage:
    python scripts/benchmark.py --transactions 10000
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import generate_data
from engine import classify, replay

TMP_DIR = Path(__file__).parent.parent / "data" / "_benchmark_tmp"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transactions", type=int, default=10000)
    parser.add_argument("--txns-per-day", type=int, default=25)
    args = parser.parse_args()
    num_days = -(-args.transactions // args.txns_per_day)  # ceil division
    total_txns = num_days * args.txns_per_day

    try:
        t0 = time.perf_counter()
        out_dir, _ = generate_data.generate(
            seed=0, out_dir=TMP_DIR, write_csvs=True, verbose=False, clean=True,
            num_days=num_days, txns_per_day=args.txns_per_day,
        )
        t_generate = time.perf_counter() - t0

        transactions, settlements, refunds, fee_sweeps, chargebacks, nodal_ledger = replay.load_tables(out_dir)

        t1 = time.perf_counter()
        expected = replay.compute_expected_balance(
            transactions, settlements, refunds, fee_sweeps, chargebacks, nodal_ledger["date"].tolist()
        )
        merged = expected.merge(nodal_ledger, on="date")
        merged["drift"] = (merged["expected_balance"] - merged["actual_balance"]).round(2)
        t_replay = time.perf_counter() - t1

        t2 = time.perf_counter()
        events = classify.find_events(merged)
        classified = classify.classify_events(events, settlements, refunds, fee_sweeps, chargebacks)
        t_classify = time.perf_counter() - t2

        audit_time = t_replay + t_classify
        print(f"Transactions:        {total_txns:,} ({num_days:,} days x {args.txns_per_day}/day)")
        print(f"Records:             {len(settlements):,} settlements, {len(refunds):,} refunds, "
              f"{len(fee_sweeps):,} fee sweeps, {len(chargebacks):,} chargebacks")
        print(f"Generate (setup, not part of the audit): {t_generate:.2f}s")
        print(f"Replay:              {t_replay:.2f}s")
        print(f"Classify:            {t_classify:.2f}s")
        print(f"Audit time (replay + classify): {audit_time:.2f}s")
        print(f"Throughput:          {total_txns / audit_time:,.0f} transactions/sec")
        print(f"Findings:            {len(classified)} (expected 0 - clean batch)")
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
