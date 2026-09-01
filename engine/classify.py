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


@dataclass
class Candidate:
    bucket: str
    amount: float  # always positive; sign is inferred from the day's delta
    detail: str
    txn_id: str = ""
    reversal_date: object = None  # for fee_sweep_timing: date the reversal was confirmed on
    suggested_fix: str = ""


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

    NO_AUTO_FIX = "No auto-generated fix - route to manual review."

    for ev in events:
        d, delta = ev["date"], ev["delta"]
        if d in consumed:
            txn_id, detail, fix = consumed[d]
            results.append(DriftEvent(d, delta, "fee_sweep_timing_resolved", "high", detail, txn_id, fix))
            continue

        sign = -1 if delta < 0 else 1
        candidates = _gather_candidates(d, delta, sign, settlements, refunds, fee_sweeps, events)
        chosen = _best_subset(candidates, abs(delta))

        if not chosen:
            results.append(DriftEvent(
                d, delta, "unexplained", "low",
                f"No recorded event on {d.date()} fully explains a drift of ~Rs.{abs(delta):.2f}.",
                suggested_fix=NO_AUTO_FIX,
            ))
            continue

        matched_total = sum(c.amount for c in chosen)
        for c in chosen:
            event_delta = sign * c.amount
            if c.bucket == "fee_sweep_timing":
                consumed[c.reversal_date] = (
                    c.txn_id,
                    f"{c.txn_id}: balance recovered as the delayed "
                    f"fee sweep flagged on {d.date()} finally landed.",
                    "No correcting entry needed - this was already the resolution of "
                    f"the fee sweep flagged on {d.date()}.",
                )
            results.append(DriftEvent(d, event_delta, c.bucket, "high", c.detail, c.txn_id, c.suggested_fix))

        residual = round(abs(delta) - matched_total, 2)
        if residual > TOLERANCE:
            results.append(DriftEvent(
                d, sign * residual, "unexplained", "low",
                f"Rs.{matched_total:.2f} of the Rs.{abs(delta):.2f} drift on {d.date()} is "
                f"explained above; Rs.{residual:.2f} remains unaccounted for.",
                suggested_fix=NO_AUTO_FIX,
            ))

    return results


def _gather_candidates(d, delta, sign, settlements, refunds, fee_sweeps, events):
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
            f"{txn_id}: {len(rows)} 'settled' rows recorded on {d.date()}, "
            f"bank shows only one payout - likely a non-idempotent retry.",
            txn_id=txn_id,
            suggested_fix=f"Reverse settlement {duplicate_ids} (Rs.{extra_amount:,.2f}) - "
                f"duplicate of {original_id}. Confirm the bank never actually paid it out twice first.",
        ))

    # Stuck refund: a refund row not in 'processed' status on this day. Same
    # logic - a clean refund is always 'processed', so a non-processed row
    # is inherently anomalous, not something to disambiguate further.
    day_refunds = refunds[(refunds["refund_date"] == d) & (refunds["status"] != "processed")]
    for _, rrow in day_refunds.iterrows():
        candidates.append(Candidate(
            "stuck_refund", round(rrow["amount"], 2),
            f"{rrow['txn_id']}: refund recorded as '{rrow['status']}' on {d.date()} "
            f"but never actually debited from the bank - likely a silently failed refund.",
            txn_id=rrow["txn_id"],
            suggested_fix=f"Re-trigger refund {rrow['refund_id']} (Rs.{rrow['amount']:,.2f}) through the "
                f"payment processor; confirm the bank debit actually posts before marking it processed.",
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
            f"{frow['txn_id']}: fee recorded as swept on {d.date()}, but the bank debit "
            f"lagged until {reversal['date'].date()} - an operational sweep delay, not a real loss.",
            txn_id=frow["txn_id"],
            reversal_date=reversal["date"],
            suggested_fix="No correcting entry needed - the balance recovers automatically "
                f"once the delayed sweep posts (confirmed landing {reversal['date'].date()}).",
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
