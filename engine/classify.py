"""
Turns the day-by-day drift series from engine/replay.py into a list of
discrete drift *events* (a day where the drift changed, rather than every
day it remains nonzero) and assigns each a root-cause bucket using
rule-based checks against the recorded books.

A single day's total drift can be caused by more than one incident at once
(two duplicate settlements landing the same day, a stuck refund alongside
a fee timing blip, etc.) - production data won't politely space incidents
onto separate days the way the synthetic generator's default settings do.
So each day's candidates are enumerated first, then matched against the
day's total delta via subset-sum: if some combination of candidates
explains the whole delta, each is reported as its own incident; if only
part of it is explained, the matched candidates are still reported and the
untouched remainder is reported as a smaller, more precise "unexplained"
residual - instead of one undifferentiated blob covering everything.
"""
from dataclasses import dataclass
from itertools import combinations

TOLERANCE = 1.0
REVERSAL_WINDOW_DAYS = 10
MAX_CANDIDATES_FOR_SUBSET_SEARCH = 8  # 2**8 = 256 combinations, plenty fast


@dataclass
class DriftEvent:
    date: object
    delta: float
    bucket: str
    confidence: str
    detail: str = ""
    txn_id: str = ""  # "" for a genuinely unexplained event - it isn't tied to one
    suggested_fix: str = ""
    # None unless engine/remediate.py can safely apply this correction on
    # its own: {"table": "settlements"|"chargebacks", "id_column": ...,
    # "row_ids": [...]}. Only ever set for a pure bookkeeping error the
    # agent made in its own records (an extra duplicate row) - never for
    # anything that requires actually moving money, which needs a human's
    # authorization the agent doesn't have.
    auto_fix: dict = None


@dataclass
class Candidate:
    bucket: str
    amount: float  # always positive; sign is inferred from the day's delta
    detail: str
    txn_id: str = ""
    reversal_date: object = None  # for fee_sweep_timing: date the reversal was confirmed on
    suggested_fix: str = ""
    auto_fix: dict = None


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


def classify_events(events, settlements, refunds, fee_sweeps, chargebacks):
    results = []
    # Maps a reversal event's date -> explanatory detail, so the day a
    # delayed fee sweep finally lands isn't reported as a second, separate
    # "unexplained" mystery on top of the fee_sweep_timing event that
    # already accounts for it.
    consumed = {}

    NO_AUTO_FIX = "This needs a person to look into directly - there isn't enough information here to safely guess what happened."

    for ev in events:
        d, delta = ev["date"], ev["delta"]
        if d in consumed:
            txn_id, detail, fix = consumed[d]
            results.append(DriftEvent(d, delta, "fee_sweep_timing_resolved", "high", detail, txn_id, fix))
            continue

        sign = -1 if delta < 0 else 1
        candidates = _gather_candidates(d, delta, sign, settlements, refunds, fee_sweeps, chargebacks, events)
        chosen = _best_subset(candidates, abs(delta))

        if not chosen:
            results.append(DriftEvent(
                d, delta, "unexplained", "low",
                f"On {d.date()}, the money in the account doesn't match what the records say it "
                f"should (off by about Rs.{abs(delta):,.2f}) - and none of the usual explanations "
                f"(a duplicate payout, a failed refund, a late fee, a duplicate chargeback) "
                f"account for it.",
                suggested_fix=NO_AUTO_FIX,
            ))
            continue

        matched_total = sum(c.amount for c in chosen)
        for c in chosen:
            event_delta = sign * c.amount
            if c.bucket == "fee_sweep_timing":
                consumed[c.reversal_date] = (
                    c.txn_id,
                    f"This is the moment the late fee payment (flagged on {d.date()}) finally "
                    f"came through - the books are back in balance now.",
                    "Nothing to do - this was already resolved by the late payment flagged earlier.",
                )
            results.append(DriftEvent(
                d, event_delta, c.bucket, "high", c.detail, c.txn_id, c.suggested_fix, c.auto_fix,
            ))

        residual = round(abs(delta) - matched_total, 2)
        if residual > TOLERANCE:
            results.append(DriftEvent(
                d, sign * residual, "unexplained", "low",
                f"Part of the money missing on {d.date()} - Rs.{matched_total:,.2f} of it - is "
                f"explained above. The remaining Rs.{residual:,.2f} doesn't match any of the "
                f"usual explanations.",
                suggested_fix=NO_AUTO_FIX,
            ))

    return results


def _gather_candidates(d, delta, sign, settlements, refunds, fee_sweeps, chargebacks, events):
    candidates = []

    # Duplicate settlement: >1 "settled" row for the same txn on this day.
    # This can only exist as a candidate if a genuine duplicate row exists -
    # it's not something an ordinary clean transaction ever produces, so
    # every one found here is inherently a real anomaly signal.
    day_settlements = settlements[(settlements["settlement_date"] == d) & (settlements["status"] == "settled")]
    dup_groups = day_settlements.groupby("txn_id").size()
    for txn_id in dup_groups[dup_groups > 1].index:
        rows = day_settlements[day_settlements["txn_id"] == txn_id]
        extra_amount = round(rows["amount"].iloc[0] * (len(rows) - 1), 2)
        original_id = rows["settlement_id"].iloc[0]
        duplicate_ids = ", ".join(rows["settlement_id"].iloc[1:])
        candidates.append(Candidate(
            "duplicate_settlement", extra_amount,
            f"This payment ({txn_id}) was recorded as paid out to the merchant twice on "
            f"{d.date()}, but the bank only actually sent the money once - most likely an "
            f"automatic retry created a duplicate entry instead of noticing the first one "
            f"already went through.",
            txn_id=txn_id,
            suggested_fix=f"Delete the duplicate payout entry ({duplicate_ids}) from the records, "
                f"keeping only {original_id}. First double-check against the actual bank "
                f"statement that the merchant really was paid only once.",
            # Safe to auto-apply: this only deletes a row the aggregator's
            # own system wrote in error - it doesn't move any money, since
            # replay.py already confirmed (by diffing against the real bank
            # ledger) that the bank only ever paid out once.
            auto_fix={"table": "settlements", "id_column": "settlement_id",
                      "row_ids": list(rows["settlement_id"].iloc[1:])},
        ))

    # Stuck refund: a refund row not in 'processed' status on this day. Same
    # logic - a clean refund is always 'processed', so a non-processed row
    # is inherently anomalous, not something to disambiguate further.
    day_refunds = refunds[(refunds["refund_date"] == d) & (refunds["status"] != "processed")]
    for _, rrow in day_refunds.iterrows():
        candidates.append(Candidate(
            "stuck_refund", round(rrow["amount"], 2),
            f"A refund for {rrow['txn_id']} was started on {d.date()} and marked as done, but "
            f"the money never actually left the bank account - the refund silently failed "
            f"somewhere along the way.",
            txn_id=rrow["txn_id"],
            suggested_fix=f"Try the refund again ({rrow['refund_id']}), and this time don't mark "
                f"it complete until you can see the money actually leave the account.",
        ))

    # Fee sweep timing: unlike the two checks above, an ordinary fee_sweep
    # row looks completely unremarkable on its own - most transactions
    # settling on this same day will have one, whether or not anything is
    # wrong. So a fee_sweep is only accepted as a candidate once a matching
    # reversal is independently confirmed later in the drift series; without
    # that check, on a busy day two unrelated clean fees can coincidentally
    # sum close to the real target and get subset-matched by pure chance.
    day_fees = fee_sweeps[fee_sweeps["swept_date"] == d]
    for _, frow in day_fees.iterrows():
        amount = round(frow["amount"], 2)
        reversal = _find_reversal(events, d, -(sign * amount), REVERSAL_WINDOW_DAYS)
        if reversal is None:
            continue
        candidates.append(Candidate(
            "fee_sweep_timing", amount,
            f"Razorpay's fee for {frow['txn_id']} was supposed to come out of the account on "
            f"{d.date()}, but the bank didn't actually take the money until "
            f"{reversal['date'].date()} - a short delay, not a real loss.",
            txn_id=frow["txn_id"],
            reversal_date=reversal["date"],
            suggested_fix="Nothing to do - the books already balanced themselves out once the "
                f"late payment came through on {reversal['date'].date()}.",
        ))

    # Duplicate chargeback: >1 row recorded for the same disputed
    # transaction on this day. Same logic as duplicate settlement - a
    # genuine second row only exists here because of a real logging
    # mistake, so it's inherently a real anomaly signal, not something
    # that needs a reversal check the way an ordinary fee does.
    day_chargebacks = chargebacks[chargebacks["chargeback_date"] == d]
    cb_dup_groups = day_chargebacks.groupby("txn_id").size()
    for txn_id in cb_dup_groups[cb_dup_groups > 1].index:
        rows = day_chargebacks[day_chargebacks["txn_id"] == txn_id]
        extra_amount = round(rows["amount"].iloc[0] * (len(rows) - 1), 2)
        original_id = rows["chargeback_id"].iloc[0]
        duplicate_ids = ", ".join(rows["chargeback_id"].iloc[1:])
        candidates.append(Candidate(
            "chargeback_duplicate", extra_amount,
            f"A chargeback dispute for {txn_id} was logged twice on {d.date()}, but the bank "
            f"only actually pulled the disputed money out of the account once - most likely "
            f"the card network's dispute notice got processed twice.",
            txn_id=txn_id,
            suggested_fix=f"Delete the duplicate chargeback entry ({duplicate_ids}) from the "
                f"records, keeping only {original_id}. First confirm with the bank statement "
                f"that the disputed amount was only pulled once.",
            # Same reasoning as duplicate_settlement above - deleting our
            # own erroneous duplicate row, not moving money.
            auto_fix={"table": "chargebacks", "id_column": "chargeback_id",
                      "row_ids": list(rows["chargeback_id"].iloc[1:])},
        ))

    return candidates


def _best_subset(candidates, target_magnitude):
    """Finds the subset of candidates whose amounts sum closest to (and
    within tolerance of, if possible) target_magnitude. Prefers an exact
    match; falls back to the closest under-shoot so at least partial credit
    is given rather than discarding real matches because of one extra
    unrelated candidate on a busy day."""
    if not candidates:
        return []
    if len(candidates) > MAX_CANDIDATES_FOR_SUBSET_SEARCH:
        # Unusual, very busy day - fall back to greedy largest-first rather
        # than an exponential search.
        return _greedy_subset(candidates, target_magnitude)

    best = []
    best_sum = 0.0
    for r in range(len(candidates), 0, -1):
        for combo in combinations(candidates, r):
            total = round(sum(c.amount for c in combo), 2)
            if _amounts_close(total, target_magnitude):
                return list(combo)
            if total < target_magnitude + TOLERANCE and total > best_sum:
                best, best_sum = list(combo), total
    return best


def _greedy_subset(candidates, target_magnitude):
    chosen = []
    remaining = target_magnitude
    for c in sorted(candidates, key=lambda c: -c.amount):
        if c.amount <= remaining + TOLERANCE:
            chosen.append(c)
            remaining -= c.amount
    return chosen


def _find_reversal(events, after_date, target_delta, window_days):
    for ev in events:
        if ev["date"] <= after_date:
            continue
        if (ev["date"] - after_date).days > window_days:
            continue
        if _amounts_close(ev["delta"], target_delta):
            return ev
    return None
