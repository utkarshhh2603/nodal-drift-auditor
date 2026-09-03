# Nodal Account Drift Auditor

Razorpay AI Buildathon — Track 04, AI Finance Controller.

## The problem

Payment aggregators must hold collected funds in a **nodal/escrow account**
until they're settled to the merchant, per RBI's PA-PG guidelines. What the
recorded books say should be in that account and what the bank statement
actually shows **drift apart** — a duplicate settlement retry, a refund that
silently failed, a fee sweep that landed late, a math bug in a partial
refund, a chargeback dispute logged twice. Nobody automatically audits
this; it's usually caught (or missed) by hand.

This tool replays the recorded books to compute the *expected* balance,
diffs it against the *actual* bank balance, and classifies every drift into
a root cause — using rules first, and an LLM only for the residual it
can't explain.

## Run it

```bash
pip install -r requirements.txt
python data/generate_data.py   # writes data/generated/*.csv
python report.py               # prints incident summary + exception table, writes report.csv
```

Set `ANTHROPIC_API_KEY` to enable the LLM explanation step for unexplained
drift events. Without it, `report.py` still runs end-to-end — unexplained
events are just reported without a hypothesis.

Run tests:

```bash
pytest
```

Measure detection accuracy across many synthetic batches (not just the one
`report.py` happens to run against):

```bash
python scripts/backtest.py --seeds 25
```

Current result: **100% recall, 0 false positives across 500 seeded
incidents over 25 seeds** — see [Measured accuracy](#measured-accuracy).

Sanity-check against a batch with zero seeded incidents (proves the tool
doesn't just always find something to flag):

```bash
python data/generate_data.py --clean --out-dir data/generated_clean
python report.py --data-dir data/generated_clean
```

Build the dashboard (regenerates `dashboard/index.html` from the current
`data/generated/` + a fresh backtest run — re-run this after regenerating
data or changing `engine/*.py`):

```bash
python dashboard/build.py --repo-url https://github.com/<you>/nodal-drift-auditor
```

Then open `dashboard/index.html` directly, or serve it:

```bash
python -m http.server 8000 --directory dashboard
```

Measure throughput at scale (Track 04's bar asks for "throughput plus
measured accuracy" — this answers the throughput half, `backtest.py`
answers accuracy):

```bash
python scripts/benchmark.py --transactions 10000
```

CI (`.github/workflows/ci.yml`) runs `pytest` and `scripts/backtest.py`
on every push — the backtest script exits nonzero if recall drops below
100% or any false positive appears, so the 100%-recall claim below is
continuously checked, not a one-time assertion.

Let the agent actually fix what it safely can, and get a checklist for
everything else (see [Remediation](#remediation) for what "safely" means
here):

```bash
python report.py --apply-fixes
```

## How it works

1. **`data/generate_data.py`** — builds a synthetic nodal account: 460
   transactions over 115 days, recorded settlements/refunds/fee sweeps/
   chargebacks, and a ground-truth daily bank ledger. 440 transactions
   reconcile perfectly (including plenty of routine, correctly-recorded
   chargebacks - proving the duplicate-chargeback check doesn't just flag
   every dispute it sees); 20 (~4.3%) are each seeded with one of five
   real-world failure modes - 4 duplicate settlements, 4 stuck refunds,
   4 late fee sweeps, 4 partial-refund math bugs, 4 duplicate chargebacks
   - spaced onto their own days (and lags) so each produces an isolated,
   unambiguous signal, rather than several incidents blurring together
   into one unreadable spike. `generate()` also takes `num_days`/
   `txns_per_day` overrides (used by `scripts/benchmark.py --transactions`
   to build a much larger batch for a throughput measurement, without a
   second copy of this generator).
2. **`engine/replay.py`** — recomputes the expected daily balance purely
   from the recorded tables (captures in, settlements/fees/refunds out).
   This is deliberately naive — it trusts the records the way a
   records-only view would, which is exactly why it can drift from the
   real bank balance.
3. **`engine/classify.py`** — collapses the cumulative drift series into
   discrete events (a day the drift *changed*, not every day it stayed
   nonzero). A single day can have more than one cause at once, so every
   day's candidates (duplicate settlements, stuck refunds, duplicate
   chargebacks, fee sweeps with an independently confirmed later reversal)
   are enumerated first, then matched against that day's total drift via
   subset-sum — each matched candidate is reported as its own incident,
   and any untouched remainder is reported as a smaller, more precise
   "unexplained" residual instead of one undifferentiated blob. A fee_sweep
   is only ever treated as a candidate once a matching reversal is
   confirmed elsewhere in the series — an ordinary, non-anomalous fee looks
   identical to a real one on its own, so without that check two unrelated
   clean fees on a busy day can coincidentally sum close enough to a
   target to get falsely matched.
4. **`engine/llm_explain.py`** — only touches the "unexplained" bucket.
   Given the raw recorded rows near that date, it proposes a hypothesis
   and a confidence level. It never re-derives arithmetic the rule engine
   already got right — that boundary is the "right tool in the right
   place" story for this track's AI-judgment criterion.
5. **`report.py`** — orchestrates the pipeline, prints the incident summary,
   exception table, and per-incident suggested fix, and writes `report.csv`
   (`--data-dir` points it at any batch, not just the default one).
   `--apply-fixes` additionally invokes `engine/remediate.py` - see
   [Remediation](#remediation).

## What "the bar" looks like here

- **Incidents detected + ₹ value**: split into "value at risk" (duplicate
  settlements, stuck refunds, duplicate chargebacks, unexplained — needs a
  human/action) vs. "self-resolving timing noise" (delayed fee sweeps —
  flagged but not a real loss). A flat "day match rate" isn't used as the
  headline number:
  nodal drift is cumulative, so once introduced it persists on every later
  day until reconciled, which makes day-level "% matched" degrade
  mechanically as more incidents accumulate rather than reflect audit
  quality.
- **Exception table**: every drift event, its ₹ delta, root-cause bucket,
  and confidence — including the ones the rule engine and LLM both fail to
  explain. That residual is reported honestly, not hidden.
- **Audit trail**: every classified event cites the specific transaction
  and recorded row(s) it's based on.
- **Suggested fix in plain language, not just diagnosis**: both the "what
  happened" and "the fix" are written as something a non-technical reviewer
  can read at a glance — e.g. *"This payment was recorded as paid out to
  the merchant twice, but the bank only actually sent the money once"* /
  *"Delete the duplicate payout entry, keeping only the original — first
  double-check against the bank statement."* — while still citing the exact
  row (settlement/refund ID) so it stays specific and auditable, not vague.
  `fee_sweep_timing` says plainly "nothing to do — it already balanced
  itself out," and `unexplained` says "this needs a person to look into
  directly," honestly, rather than a fabricated fix for something the rule
  engine doesn't actually understand.
- **Two incident types get applied automatically, the rest stay gated for
  a human** — see [Remediation](#remediation). The dividing line is
  whether real money would move: deleting the aggregator's own erroneous
  duplicate row is safe to automate, re-triggering a refund or authorizing
  a payout reversal is not, and the pipeline never blurs that line.
  Matches the track's own bar ("every money action explainable, bounded
  and gated").

## Measured accuracy

`scripts/backtest.py` generates a fresh synthetic batch per seed, runs the
real pipeline against it, and checks each seeded incident against the
generator's own ground truth (never seen by the audit code itself):

```
Incident type                  Detected    Total   Recall
chargeback_duplicate                100      100   100.0%
duplicate_settlement                100      100   100.0%
late_fee_sweep                      100      100   100.0%
partial_refund_mismatch             100      100   100.0%
stuck_refund                        100      100   100.0%
OVERALL                             500      500   100.0%

False-positive candidates: 0
```
(25 seeds, 500 total seeded incidents.) `partial_refund_mismatch` recall
means "correctly left unexplained for the LLM" — that failure mode has no
rule-based signature by design, so 100% there means the rule engine never
falsely claims to explain it with the wrong cause. This is also the exact
check CI runs on every push (`.github/workflows/ci.yml`), gated by
`scripts/backtest.py`'s exit code.

**Clean-batch sanity check** (`python data/generate_data.py --clean` +
`python report.py --data-dir ...`, real output, not hypothetical):

```
Transactions audited:   460
Days audited:            128  (0 with a book-vs-bank mismatch)
Incidents detected:      0
  Value at risk (action needed):   Rs.0.00
  Self-resolving timing noise:     Rs.0.00  (delayed fee sweeps, not a real loss)
Detection: 0/0 incidents explained same-day by rules; 0 handed to LLM review
```
Zero incidents on a batch seeded with zero incidents — the tool doesn't
just always find something to flag.

## Remediation

`report.py --apply-fixes` doesn't stop at diagnosis — it actually applies
the corrections that are safe to make on its own, via `engine/remediate.py`.

**The boundary, and why it's drawn there:** `duplicate_settlement` and
`chargeback_duplicate` are the only two incident types that ever carry an
`auto_fix` (set in `engine/classify.py`, at the exact place the specific
duplicate row IDs are already known). Both are cases where the
aggregator's own system wrote an erroneous *extra* record - and
`replay.py` already proved, by diffing against the real bank ledger, that
no extra money actually moved. Deleting that duplicate row is a pure
bookkeeping correction. Every other incident - a stuck refund (money
needs to actually move, via the payment processor), an unexplained
residual (the cause isn't even confirmed) - requires authorization this
pipeline doesn't have, so it stays a numbered checklist for a human
instead of an auto-fix. The line is drawn at "does this move real money,"
not at "is this easy."

**What actually happens:** `engine/remediate.py` copies the recorded
books to `<data-dir>/corrected/`, removes exactly the flagged duplicate
rows (the original data is never modified), then re-runs the audit
against the corrected copy and confirms each fixed date's drift is
genuinely gone - not just marked fixed. Real output from the default
batch:

```
AUTO-FIXED BY THE AGENT (8 incident(s), Rs.150,017.66 corrected)
  [2026-08-14] chargeback_duplicate (T0013): removed C0013dup from chargebacks.csv - verified: drift on this date is now zero
  [2026-10-04] duplicate_settlement (T0254): removed S0254dup from settlements.csv - verified: drift on this date is now zero
  ... (6 more)

Corrected books written to: data/generated/corrected

NEEDS HUMAN ACTION (8 incident(s) - moving money or an unclear cause, neither of which the agent is authorized to act on alone):
  1. [2026-08-10] stuck_refund (T0035, Rs.38,804.70)
     Try the refund again (R0035), and this time don't mark it complete until you can see the money actually leave the account.
  ... (7 more)
```
Re-running `report.py --data-dir data/generated/corrected` afterward
shows exactly 12 incidents instead of 20, and `duplicate_settlement`/
`chargeback_duplicate` are completely absent from the bucket counts -
`tests/test_remediate.py` asserts this holds (both that fixed incidents
vanish and that untouched incidents survive unchanged) on a freshly
generated batch, not just the default one.

## Throughput

`scripts/benchmark.py` generates a large, unseeded batch and times the
audit pipeline against it — Track 04's bar asks for "throughput plus
measured accuracy," and the two are deliberately kept separate: this
measures speed, `scripts/backtest.py` measures correctness, so a large
run here doesn't have to double as a correctness claim.

```
Transactions:        10,000 (400 days x 25/day)
Records:             9,182 settlements, 1,579 refunds, 9,182 fee sweeps, 823 chargebacks
Replay:              0.02s
Classify:            0.01s
Audit time (replay + classify): 0.03s
Throughput:          342,772 transactions/sec
```
At 100,000 transactions: 0.21s audit time, ~469,000 transactions/sec -
throughput holds (even improves slightly, from pandas' vectorized
groupby/sum operations amortizing better at volume) rather than degrading.

## Dashboard

`dashboard/index.html` (built by `dashboard/build.py` from `dashboard/template.html`
+ the live pipeline output) is a self-contained, static one-page site: hero
stats, a clickable exception list with a live detail panel per incident,
the pipeline explained in four steps, and the backtested recall table.
Auto-fixable incidents carry a "✓ auto-fixed" tag in the list and an
"✓ AUTO-FIXED BY THE AGENT" badge in the detail panel (with the fix
section relabeled "What the agent did"); everything else shows
"ACTION REQUIRED — HUMAN" instead — matching `engine/remediate.py`'s
boundary exactly, computed from the same `auto_fix` field, not a separate
guess. No build tooling required — it's plain HTML/CSS/JS with the data
embedded as JSON at build time. Regenerate it any time the underlying
data or classification logic changes; the committed `index.html` is a
snapshot, not a live view.

## Known limitations / next steps

- Synthetic data only.
- `llm_explain.py` calls the LLM once per unexplained event with no
  batching or caching — fine at this scale, would need batching for a
  larger ledger.
- `dashboard/index.html` is a static snapshot of one pipeline run — it
  doesn't re-run the audit live in the browser.
