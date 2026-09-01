"""
Generates a synthetic nodal-account dataset for the Drift Auditor.

Produces 5 CSVs under data/generated/:
  transactions.csv   - captured payments (money enters the nodal account)
  settlements.csv    - payouts to merchants (money leaves the nodal account, net of fee)
  refunds.csv        - refunds to payers (money leaves the nodal account)
  fee_sweeps.csv     - Razorpay's fee, swept out of the nodal account
  nodal_ledger.csv   - the *actual* bank balance, ground truth, one row per day

The recorded tables (transactions/settlements/refunds/fee_sweeps) are what
finance ops *believes* happened. nodal_ledger.csv is what the bank statement
*actually* shows. The large majority of transactions reconcile exactly. A
small, deliberately seeded slice of transactions each carry one specific
real-world failure mode, so the recorded books and the bank statement
diverge - that divergence is exactly what engine/replay.py and
engine/classify.py exist to find.
"""
import csv
import random
from datetime import date, timedelta
from pathlib import Path

DEFAULT_SEED = 42
_rng = random.Random(DEFAULT_SEED)

START_DATE = date(2026, 8, 1)
NUM_DAYS = 90
TXNS_PER_DAY = 4
TOTAL_TXNS = NUM_DAYS * TXNS_PER_DAY

FEE_RATE = 0.02
SETTLEMENT_LAG_DAYS = 1
LATE_FEE_SWEEP_LAG_DAYS = 3

# How many transactions get each seeded failure mode. Kept small relative
# to TOTAL_TXNS (~5% combined) so this reads as "find the needle in the
# haystack," not "half the batch is broken."
NOISE_COUNTS = {
    "duplicate_settlement": 4,
    "stuck_refund": 4,
    "late_fee_sweep": 4,
    "partial_refund_mismatch": 4,
}

OUT_DIR = Path(__file__).parent / "generated"


def r2(x):
    return round(x, 2)


NOISE_DAY_STRIDE = 5  # >3 (LATE_FEE_SWEEP_LAG_DAYS) so a delayed sweep's
                       # resolution can never land on another incident's day
NOISE_DAY_BASE = 3


def assign_noise_indices():
    """Deterministically (seeded) picks which transaction indices get which
    failure mode. Each incident is put on its own day, evenly spaced by
    NOISE_DAY_STRIDE - two incidents landing on the same calendar day would
    make their balance-drift signals sum together, which the magnitude
    matching in engine/classify.py can't disentangle back into two causes."""
    total_noise = sum(NOISE_COUNTS.values())
    noise_days = [NOISE_DAY_BASE + i * NOISE_DAY_STRIDE for i in range(total_noise)]
    assert noise_days[-1] + LATE_FEE_SWEEP_LAG_DAYS + 2 < NUM_DAYS, \
        "NUM_DAYS too small for the configured noise volume/spacing"

    noise_types = [t for t, count in NOISE_COUNTS.items() for _ in range(count)]
    _rng.shuffle(noise_types)

    assignment = {}
    for day_offset, noise_type in zip(noise_days, noise_types):
        slot = _rng.randrange(TXNS_PER_DAY)
        assignment[day_offset * TXNS_PER_DAY + slot] = noise_type
    return assignment


def gen_normal(idx, payment_date, amount, tables, true_events):
    roll = _rng.random()
    if roll < 0.10:
        _full_refund(idx, payment_date, amount, tables, true_events, status="processed")
    elif roll < 0.20:
        _partial_refund(idx, payment_date, amount, tables, true_events, buggy=False)
    else:
        _normal_settlement(idx, payment_date, amount, tables, true_events)


def _normal_settlement(idx, payment_date, amount, tables, true_events, settlement_date=None):
    settlement_date = settlement_date or payment_date + timedelta(days=SETTLEMENT_LAG_DAYS)
    tables["transactions"].append([f"T{idx:04d}", amount, payment_date, "captured"])
    true_events.append((payment_date, amount))
    settlement_amount = r2(amount * (1 - FEE_RATE))
    fee_amount = r2(amount * FEE_RATE)
    tables["settlements"].append([f"S{idx:04d}", f"T{idx:04d}", settlement_amount, settlement_date, "settled"])
    tables["fee_sweeps"].append([f"F{idx:04d}", f"T{idx:04d}", fee_amount, settlement_date])
    true_events.append((settlement_date, -settlement_amount))
    true_events.append((settlement_date, -fee_amount))
    return settlement_amount, fee_amount, settlement_date


def _full_refund(idx, payment_date, amount, tables, true_events, status):
    tables["transactions"].append([f"T{idx:04d}", amount, payment_date, "refunded"])
    true_events.append((payment_date, amount))
    refund_date = payment_date + timedelta(days=SETTLEMENT_LAG_DAYS)
    tables["refunds"].append([f"R{idx:04d}", f"T{idx:04d}", amount, refund_date, status])
    if status == "processed":
        true_events.append((refund_date, -amount))
    # else: refund recorded but never actually debited (stuck).
    return refund_date


def _partial_refund(idx, payment_date, amount, tables, true_events, buggy):
    tables["transactions"].append([f"T{idx:04d}", amount, payment_date, "partial_refund"])
    true_events.append((payment_date, amount))
    refund_date = payment_date + timedelta(days=SETTLEMENT_LAG_DAYS)
    refund_amount = r2(amount * 0.4)
    remainder = r2(amount - refund_amount)
    correct_settlement = r2(remainder * (1 - FEE_RATE))
    correct_fee = r2(remainder * FEE_RATE)

    tables["refunds"].append([f"R{idx:04d}", f"T{idx:04d}", refund_amount, refund_date, "processed"])
    true_events.append((refund_date, -refund_amount))

    if buggy:
        # Bug: settlement computed off the full amount instead of the
        # post-refund remainder, so recorded settlement+fee don't reconcile
        # against amount-refund. The bank statement reflects what actually
        # happened (the correct math) - the records are simply wrong.
        recorded_settlement = r2(amount * (1 - FEE_RATE))
        recorded_fee = r2(amount * FEE_RATE)
    else:
        recorded_settlement, recorded_fee = correct_settlement, correct_fee

    tables["settlements"].append([f"S{idx:04d}", f"T{idx:04d}", recorded_settlement, refund_date, "settled"])
    tables["fee_sweeps"].append([f"F{idx:04d}", f"T{idx:04d}", recorded_fee, refund_date])
    true_events.append((refund_date, -correct_settlement))
    true_events.append((refund_date, -correct_fee))


def gen_duplicate_settlement(idx, payment_date, amount, tables, true_events):
    """Recorded twice by mistake (e.g. a retried settlement job that wasn't
    idempotent) - the bank only ever paid out once."""
    tables["transactions"].append([f"T{idx:04d}", amount, payment_date, "captured"])
    true_events.append((payment_date, amount))
    settlement_date = payment_date + timedelta(days=SETTLEMENT_LAG_DAYS)
    settlement_amount = r2(amount * (1 - FEE_RATE))
    fee_amount = r2(amount * FEE_RATE)
    tables["settlements"].append([f"S{idx:04d}", f"T{idx:04d}", settlement_amount, settlement_date, "settled"])
    tables["settlements"].append([f"S{idx:04d}dup", f"T{idx:04d}", settlement_amount, settlement_date, "settled"])
    tables["fee_sweeps"].append([f"F{idx:04d}", f"T{idx:04d}", fee_amount, settlement_date])
    true_events.append((settlement_date, -settlement_amount))
    true_events.append((settlement_date, -fee_amount))


def gen_stuck_refund(idx, payment_date, amount, tables, true_events):
    """Refund initiated but silently failed - it never left the bank
    account, even though ops recorded it as issued."""
    _full_refund(idx, payment_date, amount, tables, true_events, status="issued")


def gen_late_fee_sweep(idx, payment_date, amount, tables, true_events):
    """Recorded as same-day, but the bank actually debited the fee days
    later (an operational sweep delay)."""
    settlement_amount, fee_amount, settlement_date = _normal_settlement(
        idx, payment_date, amount, tables, true_events
    )
    # _normal_settlement already added a true debit of fee_amount on
    # settlement_date; replace it with the delayed one.
    true_events.pop()  # remove the on-time fee debit we just added
    actual_sweep_date = settlement_date + timedelta(days=LATE_FEE_SWEEP_LAG_DAYS)
    true_events.append((actual_sweep_date, -fee_amount))


def gen_partial_refund_mismatch(idx, payment_date, amount, tables, true_events):
    _partial_refund(idx, payment_date, amount, tables, true_events, buggy=True)


NOISE_GENERATORS = {
    "duplicate_settlement": gen_duplicate_settlement,
    "stuck_refund": gen_stuck_refund,
    "late_fee_sweep": gen_late_fee_sweep,
    "partial_refund_mismatch": gen_partial_refund_mismatch,
}


def generate(seed=DEFAULT_SEED, out_dir=None, write_csvs=True, verbose=True):
    """Builds one synthetic batch and returns (out_dir, ground_truth).

    ground_truth is a list of dicts - {txn_id, onset_date, type} - one per
    seeded incident. It's not used by the audit pipeline itself (a real
    auditor doesn't get an answer key); it exists so scripts/backtest.py
    can score engine/classify.py's output against what was actually seeded,
    across many seeds, to get a measured precision/recall rather than a
    single anecdotal demo run.
    """
    global _rng
    _rng = random.Random(seed)
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    if write_csvs:
        out_dir.mkdir(parents=True, exist_ok=True)

    noise_assignment = assign_noise_indices()
    tables = {"transactions": [], "settlements": [], "refunds": [], "fee_sweeps": []}
    true_events = []
    ground_truth = []

    idx = 0
    for day_offset in range(NUM_DAYS):
        payment_date = START_DATE + timedelta(days=day_offset)
        for _ in range(TXNS_PER_DAY):
            amount = r2(_rng.uniform(500, 45000))
            noise_type = noise_assignment.get(idx)
            if noise_type:
                NOISE_GENERATORS[noise_type](idx, payment_date, amount, tables, true_events)
                ground_truth.append({
                    "txn_id": f"T{idx:04d}",
                    "onset_date": payment_date + timedelta(days=SETTLEMENT_LAG_DAYS),
                    "type": noise_type,
                })
            else:
                gen_normal(idx, payment_date, amount, tables, true_events)
            idx += 1

    # Build the ground-truth daily bank balance from true_events. Extend a
    # few days past the last possible event date so any delayed fee sweep
    # seeded on the very last day still has room to resolve.
    end_date = START_DATE + timedelta(days=NUM_DAYS + LATE_FEE_SWEEP_LAG_DAYS + 2)
    events_by_date = {}
    for ev_date, delta in true_events:
        events_by_date[ev_date] = events_by_date.get(ev_date, 0.0) + delta

    running = 0.0
    nodal_ledger = []
    d = START_DATE
    while d <= end_date:
        running = r2(running + events_by_date.get(d, 0.0))
        nodal_ledger.append([d, running])
        d += timedelta(days=1)

    if write_csvs:
        _write_csv(out_dir / "transactions.csv", ["txn_id", "amount", "payment_date", "status"], tables["transactions"])
        _write_csv(out_dir / "settlements.csv", ["settlement_id", "txn_id", "amount", "settlement_date", "status"], tables["settlements"])
        _write_csv(out_dir / "refunds.csv", ["refund_id", "txn_id", "amount", "refund_date", "status"], tables["refunds"])
        _write_csv(out_dir / "fee_sweeps.csv", ["fee_id", "txn_id", "amount", "swept_date"], tables["fee_sweeps"])
        _write_csv(out_dir / "nodal_ledger.csv", ["date", "actual_balance"], nodal_ledger)
    if write_csvs and verbose:
        print(
            f"Generated {idx} transactions ({sum(NOISE_COUNTS.values())} seeded incidents: "
            + ", ".join(f"{k}={v}" for k, v in NOISE_COUNTS.items())
            + f"), {len(tables['settlements'])} settlements, {len(tables['refunds'])} refunds, "
            f"{len(tables['fee_sweeps'])} fee sweeps, {len(nodal_ledger)} ledger days -> {out_dir}"
        )

    return out_dir, ground_truth


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


if __name__ == "__main__":
    generate()
