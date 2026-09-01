# Nodal Account Drift Auditor

Razorpay AI Buildathon — Track 04, AI Finance Controller.

## The problem

Payment aggregators must hold collected funds in a **nodal/escrow account**
until they're settled to the merchant, per RBI's PA-PG guidelines. What the
recorded books say should be in that account and what the bank statement
actually shows **drift apart** — a duplicate settlement retry, a refund that
silently failed, a fee sweep that landed late, a math bug in a partial
refund. Nobody automatically audits this; it's usually caught (or missed)
by hand.

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

Current result: **100% recall, 0 false positives across 400 seeded
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

## How it works

1. **`data/generate_data.py`** — builds a synthetic nodal account: 360
   transactions over 90 days, recorded settlements/refunds/fee sweeps, and
   a ground-truth daily bank ledger. 344 transactions reconcile perfectly;
   16 (~4.4%) are each seeded with one of four real-world failure modes —
   4 duplicate settlements, 4 stuck refunds, 4 late fee sweeps, 4 partial-
   refund math bugs — spaced onto their own days so each produces an
   isolated, unambiguous signal, rather than several incidents blurring
   together into one unreadable spike.
2. **`engine/replay.py`** — recomputes the expected daily balance purely
   from the recorded tables (captures in, settlements/fees/refunds out).
   This is deliberately naive — it trusts the records the way a
   records-only view would, which is exactly why it can drift from the
   real bank balance.
3. **`engine/classify.py`** — collapses the cumulative drift series into
   discrete events (a day the drift *changed*, not every day it stayed
   nonzero). A single day can have more than one cause at once, so every
   day's candidates (duplicate settlements, stuck refunds, fee sweeps with
   an independently confirmed later reversal) are enumerated first, then
   matched against that day's total drift via subset-sum — each matched
   candidate is reported as its own incident, and any untouched remainder
   is reported as a smaller, more precise "unexplained" residual instead
   of one undifferentiated blob. A fee_sweep is only ever treated as a
   candidate once a matching reversal is confirmed elsewhere in the
   series — an ordinary, non-anomalous fee looks identical to a real one
   on its own, so without that check two unrelated clean fees on a busy
   day can coincidentally sum close enough to a target to get falsely
   matched.
4. **`engine/llm_explain.py`** — only touches the "unexplained" bucket.
   Given the raw recorded rows near that date, it proposes a hypothesis
   and a confidence level. It never re-derives arithmetic the rule engine
   already got right — that boundary is the "right tool in the right
   place" story for this track's AI-judgment criterion.
5. **`report.py`** — orchestrates the pipeline, prints the incident summary,
   exception table, and per-incident suggested fix, and writes `report.csv`
   (`--data-dir` points it at any batch, not just the default one).

## What "the bar" looks like here

- **Incidents detected + ₹ value**: split into "value at risk" (duplicate
  settlements, stuck refunds, unexplained — needs a human/action) vs.
  "self-resolving timing noise" (delayed fee sweeps — flagged but not a
  real loss). A flat "day match rate" isn't used as the headline number:
  nodal drift is cumulative, so once introduced it persists on every later
  day until reconciled, which makes day-level "% matched" degrade
  mechanically as more incidents accumulate rather than reflect audit
  quality.
- **Exception table**: every drift event, its ₹ delta, root-cause bucket,
  and confidence — including the ones the rule engine and LLM both fail to
  explain. That residual is reported honestly, not hidden.
- **Audit trail**: every classified event cites the specific transaction
  and recorded row(s) it's based on.
- **Suggested fix, not just diagnosis**: each incident carries a specific
  correcting action citing the exact row — "Reverse settlement S0173dup —
  duplicate of S0173" or "Re-trigger refund R0013" — not generic per-bucket
  advice. `fee_sweep_timing` gets "no action needed" (it self-resolves) and
  `unexplained` gets "route to manual review," honestly, rather than a
  fabricated fix for something the rule engine doesn't actually understand.
  This is still diagnosis-and-recommendation, not autonomous execution —
  gated for a human to apply, matching the track's own bar ("every money
  action explainable, bounded and gated").

## Measured accuracy

`scripts/backtest.py` generates a fresh synthetic batch per seed, runs the
real pipeline against it, and checks each seeded incident against the
generator's own ground truth (never seen by the audit code itself):

```
Incident type                  Detected    Total   Recall
duplicate_settlement                100      100   100.0%
late_fee_sweep                      100      100   100.0%
partial_refund_mismatch             100      100   100.0%
stuck_refund                        100      100   100.0%
OVERALL                             400      400   100.0%

False-positive candidates: 0
```
(25 seeds, 400 total seeded incidents.) `partial_refund_mismatch` recall
means "correctly left unexplained for the LLM" — that failure mode has no
rule-based signature by design, so 100% there means the rule engine never
falsely claims to explain it with the wrong cause.

**Clean-batch sanity check** (`python data/generate_data.py --clean` +
`python report.py --data-dir ...`, real output, not hypothetical):

```
Transactions audited:   360
Days audited:            96  (0 with a book-vs-bank mismatch)
Incidents detected:      0
  Value at risk (action needed):   Rs.0.00
  Self-resolving timing noise:     Rs.0.00  (delayed fee sweeps, not a real loss)
Detection: 0/0 incidents explained same-day by rules; 0 handed to LLM review
```
Zero incidents on a batch seeded with zero incidents — the tool doesn't
just always find something to flag.

## Dashboard

`dashboard/index.html` (built by `dashboard/build.py` from `dashboard/template.html`
+ the live pipeline output) is a self-contained, static one-page site: hero
stats, a clickable exception list with a live detail panel per incident,
the pipeline explained in four steps, and the backtested recall table.
No build tooling required — it's plain HTML/CSS/JS with the data embedded
as JSON at build time. Regenerate it any time the underlying data or
classification logic changes; the committed `index.html` is a snapshot,
not a live view.

## Known limitations / next steps

- Synthetic data only.
- `llm_explain.py` calls the LLM once per unexplained event with no
  batching or caching — fine at this scale, would need batching for a
  larger ledger.
- `dashboard/index.html` is a static snapshot of one pipeline run — it
  doesn't re-run the audit live in the browser.
