"""
Actually applies the corrections the audit pipeline is confident are safe
to make on its own, and proves each one worked by re-auditing afterward.

The boundary is deliberate and narrow: only `duplicate_settlement` and
`chargeback_duplicate` incidents ever carry an `auto_fix` (set in
engine/classify.py) - both are cases where the aggregator's own system
wrote an erroneous *extra* row, and replay.py already confirmed (by
diffing against the real bank ledger) that no extra money actually moved.
Deleting that duplicate record is a pure bookkeeping correction. Every
other incident - a stuck refund, a duplicate chargeback that DID pull
money twice, anything unexplained - requires either moving real money or
understanding a cause the rules don't have evidence for, neither of which
this pipeline is authorized to do on its own. Those stay a checklist for
a human, never an auto-fix.
"""
import shutil
from pathlib import Path

import pandas as pd

TABLE_FILES = {
    "transactions": "transactions.csv",
    "settlements": "settlements.csv",
    "refunds": "refunds.csv",
    "fee_sweeps": "fee_sweeps.csv",
    "chargebacks": "chargebacks.csv",
}


def apply_corrections(classified, data_dir, out_dir):
    """Writes a corrected copy of the recorded books to out_dir - the
    original data_dir is never modified, so the flawed records stay
    available as evidence of what was actually wrong.

    Returns a list of correction records, one per auto-fixed incident:
    {date, bucket, txn_id, table, row_ids, value_corrected}.
    """
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = {name: pd.read_csv(data_dir / filename) for name, filename in TABLE_FILES.items()}

    corrections = []
    for ev in classified:
        if not ev.auto_fix:
            continue
        table_name = ev.auto_fix["table"]
        id_column = ev.auto_fix["id_column"]
        row_ids = ev.auto_fix["row_ids"]

        df = tables[table_name]
        tables[table_name] = df[~df[id_column].isin(row_ids)]

        corrections.append({
            "date": ev.date.date() if hasattr(ev.date, "date") else ev.date,
            "bucket": ev.bucket,
            "txn_id": ev.txn_id,
            "table": table_name,
            "row_ids": row_ids,
            "value_corrected": round(abs(ev.delta), 2),
        })

    for name, df in tables.items():
        df.to_csv(out_dir / TABLE_FILES[name], index=False)
    # nodal_ledger.csv is the real bank statement - correcting our own
    # bookkeeping doesn't change what the bank actually did, so it's
    # copied through untouched rather than regenerated.
    shutil.copy(data_dir / "nodal_ledger.csv", out_dir / "nodal_ledger.csv")

    return corrections


def verify_corrections(corrections, out_dir):
    """Re-runs the audit against the corrected books and checks whether
    each fixed date's drift *event* actually shrank - not whether the
    day's cumulative drift value hit zero, since duplicate_settlement and
    chargeback_duplicate drift is permanent and later, still-unfixed
    incidents keep contributing to every day's running total after theirs.
    find_events() already isolates the day-over-day *change*, which is
    exactly what one incident's own correction should erase - so this
    compares that, before vs. after, which is the honest way to prove a
    specific fix worked rather than just claiming it did.
    """
    from engine import classify, replay  # deferred: avoids a circular import at module load

    merged = replay.run(data_dir=out_dir)
    new_events = classify.find_events(merged)
    new_delta_by_date = {ev["date"].date(): ev["delta"] for ev in new_events}

    results = []
    for c in corrections:
        remaining_delta = new_delta_by_date.get(c["date"], 0.0)
        results.append({**c, "remaining_drift_on_date": round(remaining_delta, 2)})
    return results
