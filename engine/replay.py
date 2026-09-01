"""
Reconstructs the *expected* nodal account balance, per day, purely from the
recorded books (transactions/settlements/refunds/fee_sweeps) - i.e. what
finance ops's own records say should have happened. This is deliberately
naive: it trusts every settlement row with status "settled" and every
refund row regardless of status, the way a records-only view would. That
naivety is what lets it diverge from the real bank balance in exactly the
cases the auditor is meant to catch.
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "generated"


def load_tables(data_dir=None):
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    transactions = pd.read_csv(data_dir / "transactions.csv", parse_dates=["payment_date"])
    settlements = pd.read_csv(data_dir / "settlements.csv", parse_dates=["settlement_date"])
    refunds = pd.read_csv(data_dir / "refunds.csv", parse_dates=["refund_date"])
    fee_sweeps = pd.read_csv(data_dir / "fee_sweeps.csv", parse_dates=["swept_date"])
    nodal_ledger = pd.read_csv(data_dir / "nodal_ledger.csv", parse_dates=["date"])
    return transactions, settlements, refunds, fee_sweeps, nodal_ledger


def compute_expected_balance(transactions, settlements, refunds, fee_sweeps, dates):
    """Returns a DataFrame [date, expected_balance] for the given date index."""
    inflow = transactions.groupby("payment_date")["amount"].sum()
    settled_out = settlements.loc[settlements["status"] == "settled"].groupby("settlement_date")["amount"].sum()
    fee_out = fee_sweeps.groupby("swept_date")["amount"].sum()
    refund_out = refunds.groupby("refund_date")["amount"].sum()

    running = 0.0
    rows = []
    for d in dates:
        running += inflow.get(d, 0.0)
        running -= settled_out.get(d, 0.0)
        running -= fee_out.get(d, 0.0)
        running -= refund_out.get(d, 0.0)
        rows.append({"date": d, "expected_balance": round(running, 2)})
    return pd.DataFrame(rows)


def run(data_dir=None):
    transactions, settlements, refunds, fee_sweeps, nodal_ledger = load_tables(data_dir)
    dates = nodal_ledger["date"].tolist()
    expected = compute_expected_balance(transactions, settlements, refunds, fee_sweeps, dates)
    merged = expected.merge(nodal_ledger, on="date")
    merged["drift"] = (merged["expected_balance"] - merged["actual_balance"]).round(2)
    return merged


if __name__ == "__main__":
    result = run()
    print(result.to_string(index=False))
