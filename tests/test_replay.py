from datetime import date

import pandas as pd

from engine.replay import compute_expected_balance
from engine.classify import (
    Candidate,
    MAX_CANDIDATES_FOR_SUBSET_SEARCH,
    find_events,
    classify_events,
    _best_subset,
    _greedy_subset,
)


EMPTY_CHARGEBACKS = pd.DataFrame(columns=["chargeback_id", "txn_id", "amount", "chargeback_date", "status"])


def test_compute_expected_balance_simple_capture_and_settle():
    d0, d1 = date(2026, 1, 1), date(2026, 1, 2)
    transactions = pd.DataFrame([{"txn_id": "T1", "amount": 1000.0, "payment_date": d0, "status": "captured"}])
    settlements = pd.DataFrame([{"settlement_id": "S1", "txn_id": "T1", "amount": 980.0, "settlement_date": d1, "status": "settled"}])
    refunds = pd.DataFrame(columns=["refund_id", "txn_id", "amount", "refund_date", "status"])
    fee_sweeps = pd.DataFrame([{"fee_id": "F1", "txn_id": "T1", "amount": 20.0, "swept_date": d1}])

    result = compute_expected_balance(transactions, settlements, refunds, fee_sweeps, EMPTY_CHARGEBACKS, [d0, d1])

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

    classified = classify_events(events, settlements, refunds, fee_sweeps, EMPTY_CHARGEBACKS)
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
    classified = classify_events(events, settlements, refunds, fee_sweeps, EMPTY_CHARGEBACKS)

    buckets = sorted(ev.bucket for ev in classified)
    assert buckets == ["duplicate_settlement", "stuck_refund"]
    assert all(ev.bucket != "unexplained" for ev in classified)
    assert round(sum(ev.delta for ev in classified), 2) == -800.0


def test_suggested_fix_cites_the_specific_row_to_reverse():
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
    classified = classify_events(events, settlements, refunds, fee_sweeps, EMPTY_CHARGEBACKS)

    assert len(classified) == 1
    # The fix must name the specific duplicate row to reverse (S1dup), not
    # just describe the bucket in general terms - that's the difference
    # between a diagnosis and an actionable correcting entry.
    assert "S1dup" in classified[0].suggested_fix
    assert "S1" in classified[0].suggested_fix


def test_unexplained_event_gets_manual_review_fix_not_a_fabricated_one():
    d0, d1 = pd.Timestamp(2026, 1, 1), pd.Timestamp(2026, 1, 2)
    empty_settlements = pd.DataFrame(columns=["settlement_id", "txn_id", "amount", "settlement_date", "status"])
    empty_refunds = pd.DataFrame(columns=["refund_id", "txn_id", "amount", "refund_date", "status"])
    empty_fees = pd.DataFrame(columns=["fee_id", "txn_id", "amount", "swept_date"])
    replay_df = pd.DataFrame([
        {"date": d0, "expected_balance": 1000.0, "actual_balance": 1000.0, "drift": 0.0},
        {"date": d1, "expected_balance": 700.0, "actual_balance": 1000.0, "drift": -300.0},
    ])

    events = find_events(replay_df)
    classified = classify_events(events, empty_settlements, empty_refunds, empty_fees, EMPTY_CHARGEBACKS)

    assert len(classified) == 1
    assert classified[0].bucket == "unexplained"
    # Should honestly say a person needs to look at it, not fabricate a fix
    # for something the rule engine doesn't actually understand.
    assert "person" in classified[0].suggested_fix.lower()
    # But "needs a person" alone isn't a runbook - there should be concrete,
    # numbered steps (bank contact, what to check) behind it.
    assert len(classified[0].resolution_steps) >= 3
    assert any("bank" in step.lower() for step in classified[0].resolution_steps)


def test_stuck_refund_gets_a_concrete_runbook_not_just_a_one_liner():
    d0, d1 = pd.Timestamp(2026, 1, 1), pd.Timestamp(2026, 1, 2)
    empty_settlements = pd.DataFrame(columns=["settlement_id", "txn_id", "amount", "settlement_date", "status"])
    refunds = pd.DataFrame([
        {"refund_id": "R9", "txn_id": "T9", "amount": 500.0, "refund_date": d1, "status": "issued"},
    ])
    empty_fees = pd.DataFrame(columns=["fee_id", "txn_id", "amount", "swept_date"])
    replay_df = pd.DataFrame([
        {"date": d0, "expected_balance": 1000.0, "actual_balance": 1000.0, "drift": 0.0},
        {"date": d1, "expected_balance": 500.0, "actual_balance": 1000.0, "drift": -500.0},
    ])

    events = find_events(replay_df)
    classified = classify_events(events, empty_settlements, refunds, empty_fees, EMPTY_CHARGEBACKS)

    assert len(classified) == 1
    ev = classified[0]
    assert ev.bucket == "stuck_refund"
    assert not ev.auto_fix  # moving money always needs a human, never auto-applied
    assert len(ev.resolution_steps) >= 3
    # Steps must be concrete and specific to this incident, not boilerplate -
    # citing the actual refund ID is what makes it a runbook, not a platitude.
    assert any("R9" in step for step in ev.resolution_steps)
    assert any("processor" in step.lower() for step in ev.resolution_steps)


def test_best_subset_dispatches_to_greedy_above_the_exponential_search_limit():
    # _best_subset does an exhaustive combinations() search for small
    # candidate counts, which is exponential - MAX_CANDIDATES_FOR_SUBSET_SEARCH
    # exists so a single unusually busy day can't make classify_events hang.
    # That fallback path (_greedy_subset) had no direct test coverage even
    # though it's the one branch the 100%-recall backtest never exercises
    # (the synthetic generator never produces >8 candidates on one day).
    candidates = [Candidate("stuck_refund", float(100 * i), f"detail {i}", txn_id=f"T{i}") for i in range(1, 10)]
    assert len(candidates) > MAX_CANDIDATES_FOR_SUBSET_SEARCH

    # Target exactly matches the three largest (900+800+700=2400) - greedy's
    # largest-first strategy should find precisely that combination.
    chosen = _best_subset(candidates, 2400.0)

    assert sorted(c.amount for c in chosen) == [700.0, 800.0, 900.0]
    assert round(sum(c.amount for c in chosen), 2) == 2400.0


def test_greedy_subset_directly_on_a_partial_match():
    # No exact combination sums to the target - greedy takes the largest
    # candidates first until the remaining budget can't fit another one,
    # rather than giving up and reporting nothing.
    candidates = [Candidate("stuck_refund", amt, "d", txn_id="T") for amt in [50.0, 40.0, 30.0, 20.0, 10.0]]

    chosen = _greedy_subset(candidates, target_magnitude=95.0)

    # 50 + 40 = 90 (fits, remaining 5), next is 30 (doesn't fit under
    # remaining+tolerance=6) so it stops there.
    assert sorted(c.amount for c in chosen) == [40.0, 50.0]


def test_duplicate_chargeback_detected():
    d0, d1 = pd.Timestamp(2026, 1, 1), pd.Timestamp(2026, 1, 2)
    empty_settlements = pd.DataFrame(columns=["settlement_id", "txn_id", "amount", "settlement_date", "status"])
    empty_refunds = pd.DataFrame(columns=["refund_id", "txn_id", "amount", "refund_date", "status"])
    empty_fees = pd.DataFrame(columns=["fee_id", "txn_id", "amount", "swept_date"])
    chargebacks = pd.DataFrame([
        {"chargeback_id": "C1", "txn_id": "T1", "amount": 400.0, "chargeback_date": d1, "status": "debited"},
        {"chargeback_id": "C1dup", "txn_id": "T1", "amount": 400.0, "chargeback_date": d1, "status": "debited"},
    ])
    replay_df = pd.DataFrame([
        {"date": d0, "expected_balance": 1000.0, "actual_balance": 1000.0, "drift": 0.0},
        {"date": d1, "expected_balance": 200.0, "actual_balance": 600.0, "drift": -400.0},
    ])

    events = find_events(replay_df)
    classified = classify_events(events, empty_settlements, empty_refunds, empty_fees, chargebacks)

    assert len(classified) == 1
    assert classified[0].bucket == "chargeback_duplicate"
    assert "C1dup" in classified[0].suggested_fix
    assert "C1" in classified[0].suggested_fix
