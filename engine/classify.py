"""
Turns the day-by-day drift series from engine/replay.py into a list of
discrete drift *events* (a day where the drift changed, rather than every
day it remains nonzero) and assigns each a root-cause bucket using
rule-based checks against the recorded books. Anything that doesn't match
a known pattern is left as "unexplained" for engine/llm_explain.py to take
a pass at.
"""
from dataclasses import dataclass, field
from typing import Optional

TOLERANCE = 1.0
REVERSAL_WINDOW_DAYS = 10


@dataclass
class DriftEvent:
    date: object
    delta: float
    bucket: str
    confidence: str
    detail: str = ""


def find_events(replay_df):
    """Collapses the cumulative drift series into discrete change events."""
    events = []
    prev_drift = 0.0
    for _, row in replay_df.iterrows():
        delta = round(row["drift"] - prev_drift, 2)
        if abs(delta) > TOLERANCE:
            events.append({"date": row["date"], "delta": delta})
        prev_drift = row["drift"]
    return events


def _amounts_close(a, b, tol=1.0):
    return abs(a - b) <= tol


def classify_events(events, settlements, refunds, fee_sweeps, drift_by_date):
    results = []
    # Maps a reversal event's date -> explanatory detail, so the day a
    # delayed fee sweep finally lands isn't reported as a second, separate
    # "unexplained" mystery on top of the fee_sweep_timing event that
    # already accounts for it.
    consumed = {}
    for ev in events:
        d, delta = ev["date"], ev["delta"]
        if d in consumed:
            results.append(DriftEvent(d, delta, "fee_sweep_timing_resolved", "high", consumed[d]))
            continue
        magnitude = abs(delta)

        # 1. Duplicate settlement: >1 "settled" row for the same txn on this day.
        day_settlements = settlements[(settlements["settlement_date"] == d) & (settlements["status"] == "settled")]
        dup_groups = day_settlements.groupby("txn_id").size()
        dup_txns = dup_groups[dup_groups > 1]
        matched = False
        for txn_id in dup_txns.index:
            rows = day_settlements[day_settlements["txn_id"] == txn_id]
            extra_amount = rows["amount"].iloc[0] * (len(rows) - 1)
            if _amounts_close(extra_amount, magnitude):
                results.append(DriftEvent(
                    d, delta, "duplicate_settlement", "high",
                    f"{txn_id}: {len(rows)} 'settled' rows recorded on {d.date()}, "
                    f"bank shows only one payout - likely a non-idempotent retry.",
                ))
                matched = True
                break
        if matched:
            continue

        # 2. Stuck refund: a refund row not in 'processed' status on this day.
        day_refunds = refunds[(refunds["refund_date"] == d) & (refunds["status"] != "processed")]
        matched = False
        for _, rrow in day_refunds.iterrows():
            if _amounts_close(rrow["amount"], magnitude):
                results.append(DriftEvent(
                    d, delta, "stuck_refund", "high",
                    f"{rrow['txn_id']}: refund recorded as '{rrow['status']}' on {d.date()} "
                    f"but never actually debited from the bank - likely a silently failed refund.",
                ))
                matched = True
                break
        if matched:
            continue

        # 3. Fee sweep timing mismatch: a fee_sweep on this day whose amount
        # matches, AND the drift reverses by the same magnitude within the
        # reversal window (i.e. it's transient, not a permanent hole).
        day_fees = fee_sweeps[fee_sweeps["swept_date"] == d]
        candidate_fee = None
        for _, frow in day_fees.iterrows():
            if _amounts_close(frow["amount"], magnitude):
                candidate_fee = frow
                break
        if candidate_fee is not None:
            reversal = _find_reversal(events, d, -delta, REVERSAL_WINDOW_DAYS)
            if reversal is not None:
                results.append(DriftEvent(
                    d, delta, "fee_sweep_timing", "high",
                    f"{candidate_fee['txn_id']}: fee recorded as swept on {d.date()}, "
                    f"but the bank debit lagged until {reversal['date'].date()} - "
                    f"an operational sweep delay, not a real loss.",
                ))
                consumed[reversal["date"]] = (
                    f"{candidate_fee['txn_id']}: balance recovered as the delayed "
                    f"fee sweep flagged on {d.date()} finally landed."
                )
                continue

        # 4. Unexplained - hand off to the LLM layer.
        results.append(DriftEvent(
            d, delta, "unexplained", "low",
            f"No recorded event on {d.date()} fully explains a drift of ~Rs.{magnitude:.2f}.",
        ))
    return results


def _find_reversal(events, after_date, target_delta, window_days):
    for ev in events:
        if ev["date"] <= after_date:
            continue
        if (ev["date"] - after_date).days > window_days:
            continue
        if _amounts_close(ev["delta"], target_delta):
            return ev
    return None
