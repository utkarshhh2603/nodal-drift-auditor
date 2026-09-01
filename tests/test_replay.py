from datetime import date

import pandas as pd

from engine.replay import compute_expected_balance
from engine.classify import find_events, classify_events


def test_compute_expected_balance_simple_capture_and_settle():
    d0, d1 = date(2026, 1, 1), date(2026, 1, 2)
    transactions = pd.DataFrame([{"txn_id": "T1", "amount": 1000.0, "payment_date": d0, "status": "captured"}])
    settlements = pd.DataFrame([{"settlement_id": "S1", "txn_id": "T1", "amount": 980.0, "settlement_date": d1, "status": "settled"}])
    refunds = pd.DataFrame(columns=["refund_id", "txn_id", "amount", "refund_date", "status"])
    fee_sweeps = pd.DataFrame([{"fee_id": "F1", "txn_id": "T1", "amount": 20.0, "swept_date": d1}])

    result = compute_expected_balance(transactions, settlements, refunds, fee_sweeps, [d0, d1])

    assert result.loc[result["date"] == d0, "expected_balance"].iloc[0] == 1000.0
    # 1000 (still in) - 980 (settled out) - 20 (fee out) = 0
    assert result.loc[result["date"] == d1, "expected_balance"].iloc[0] == 0.0


def test_duplicate_settlement_detected():
    # engine/replay.py's real pipeline produces pandas Timestamps (via
    # pd.read_csv(parse_dates=...)), so the fixture uses them too rather
    # than plain datetime.date - classify.py calls .date() on these values.
    d0, d1 = pd.Timestamp(2026, 1, 1), pd.Timestamp(2026, 1, 2)
    settlements = pd.DataFrame([
        {"settlement_id": "S1", "txn_id": "T1", "amount": 500.0, "settlement_date": d1, "status": "settled"},
        {"settlement_id": "S1dup", "txn_id": "T1", "amount": 500.0, "settlement_date": d1, "status": "settled"},
    ])
    refunds = pd.DataFrame(columns=["refund_id", "txn_id", "amount", "refund_date", "status"])
    fee_sweeps = pd.DataFrame(columns=["fee_id", "txn_id", "amount", "swept_date"])

    replay_df = pd.DataFrame([
        {"date": d0, "expected_balance": 1000.0, "actual_balance": 1000.0, "drift": 0.0},
        {"date": d1, "expected_balance": 0.0, "actual_balance": 500.0, "drift": -500.0},
    ])

    events = find_events(replay_df)
    assert len(events) == 1
    assert events[0]["date"] == d1

    classified = classify_events(events, settlements, refunds, fee_sweeps, None)
    assert len(classified) == 1
    assert classified[0].bucket == "duplicate_settlement"


def test_two_incidents_on_same_day_are_both_recovered():
    # A duplicate settlement (T1, extra Rs.500) and a stuck refund (T2,
    # Rs.300) both land on the same day, so the day's total drift (-800) is
    # the sum of two unrelated causes. Magnitude-matching the whole day
    # against a single candidate (the old approach) can't explain -800 with
    # either candidate alone and falls back to one undifferentiated
    # "unexplained" event - classify_events should instead decompose the
    # day into both real causes.
    d0, d1 = pd.Timestamp(2026, 1, 1), pd.Timestamp(2026, 1, 2)
    settlements = pd.DataFrame([
        {"settlement_id": "S1", "txn_id": "T1", "amount": 500.0, "settlement_date": d1, "status": "settled"},
        {"settlement_id": "S1dup", "txn_id": "T1", "amount": 500.0, "settlement_date": d1, "status": "settled"},
    ])
    refunds = pd.DataFrame([
        {"refund_id": "R2", "txn_id": "T2", "amount": 300.0, "refund_date": d1, "status": "issued"},
    ])
    fee_sweeps = pd.DataFrame(columns=["fee_id", "txn_id", "amount", "swept_date"])

    replay_df = pd.DataFrame([
        {"date": d0, "expected_balance": 2000.0, "actual_balance": 2000.0, "drift": 0.0},
        {"date": d1, "expected_balance": 1200.0, "actual_balance": 2000.0, "drift": -800.0},
    ])

    events = find_events(replay_df)
    classified = classify_events(events, settlements, refunds, fee_sweeps, None)

    buckets = sorted(ev.bucket for ev in classified)
    assert buckets == ["duplicate_settlement", "stuck_refund"]
    assert all(ev.bucket != "unexplained" for ev in classified)
    assert round(sum(ev.delta for ev in classified), 2) == -800.0
