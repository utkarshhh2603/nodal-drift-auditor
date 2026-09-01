"""
Optional LLM pass, scoped narrowly to the "unexplained" bucket from
engine/classify.py. It never touches events the rule engine already
classified - this is the "right tool in the right place" boundary: an LLM
proposes a hypothesis for genuine anomalies, it doesn't re-derive arithmetic
the deterministic engine already got right.

Requires ANTHROPIC_API_KEY. If unset, report.py skips this step entirely
and the unexplained events are reported as-is.
"""
import json
import os


def explain_unexplained(events, transactions, settlements, refunds, fee_sweeps, chargebacks):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return events

    try:
        import anthropic
    except ImportError:
        return events

    client = anthropic.Anthropic(api_key=api_key)

    for ev in events:
        if ev.bucket != "unexplained":
            continue
        d = ev.date
        context = _rows_near_date(transactions, settlements, refunds, fee_sweeps, chargebacks, d)
        prompt = (
            "You are auditing a payment aggregator's nodal account. On "
            f"{d.date()}, the recorded books and the real bank balance "
            f"diverged by roughly Rs.{abs(ev.delta):.2f}, and no rule-based "
            "check (duplicate settlement, stuck refund, fee sweep timing, "
            "duplicate chargeback) explains it. Here are the recorded rows "
            f"within 2 days of that date:\n\n{json.dumps(context, indent=2, default=str)}\n\n"
            "Propose the single most likely root cause in one sentence, "
            "then state a confidence (low/medium/high). Only use the data "
            "given - do not invent transactions not listed. If nothing in "
            "the data supports a specific hypothesis, say so plainly."
        )
        try:
            response = client.messages.create(
                model="claude-sonnet-5",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            ev.detail = response.content[0].text.strip()
            ev.confidence = "llm-proposed"
        except Exception as exc:
            ev.detail = f"{ev.detail} (LLM call failed: {exc})"
    return events


def _rows_near_date(transactions, settlements, refunds, fee_sweeps, chargebacks, d, window_days=2):
    from datetime import timedelta
    lo, hi = d - timedelta(days=window_days), d + timedelta(days=window_days)

    def _slice(df, col):
        mask = (df[col] >= lo) & (df[col] <= hi)
        return df.loc[mask].to_dict(orient="records")

    return {
        "transactions": _slice(transactions, "payment_date"),
        "settlements": _slice(settlements, "settlement_date"),
        "refunds": _slice(refunds, "refund_date"),
        "fee_sweeps": _slice(fee_sweeps, "swept_date"),
        "chargebacks": _slice(chargebacks, "chargeback_date"),
    }
