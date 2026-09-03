"""
Renders dashboard/index.html from the live pipeline output.

Reads dashboard/template.html (the static design), fills in a JSON blob
built from report.compute_report() (the same computation report.py's CLI
prints) plus a fast backtest run, and writes the result to
dashboard/index.html. Run this after regenerating data/generated/ or
changing engine/*.py so the dashboard reflects the current numbers instead
of a stale snapshot.

Usage:
    python dashboard/build.py [--backtest-seeds 15]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import report
from scripts.backtest import run_one_seed

BUCKET_LABELS = {
    "duplicate_settlement": "Duplicate settlement",
    "stuck_refund": "Stuck refund",
    "fee_sweep_timing": "Fee sweep timing",
    "fee_sweep_timing_resolved": "Fee sweep resolved",
    "chargeback_duplicate": "Duplicate chargeback",
    "unexplained": "Unexplained",
}


def build_incidents_json(r):
    incidents = []
    for ev in r["incidents"]:
        incidents.append({
            "date": ev.date.date().isoformat(),
            "bucket": ev.bucket,
            "bucket_label": BUCKET_LABELS.get(ev.bucket, ev.bucket),
            "delta": round(ev.delta, 2),
            "txn_id": ev.txn_id,
            "confidence": ev.confidence,
            "detail": ev.detail,
            # Real, incident-specific correcting action (cites the exact row
            # to reverse/re-trigger) - set in engine/classify.py, not
            # generic per-bucket boilerplate.
            "fix": ev.suggested_fix,
            # True only for a pure bookkeeping correction (an extra
            # duplicate row) that engine/remediate.py can apply on its own -
            # never for anything that requires actually moving money.
            "auto_fixable": bool(ev.auto_fix),
        })
    incidents.sort(key=lambda i: i["date"])
    return incidents


def run_backtest(seeds):
    tmp_dir = ROOT / "data" / "_backtest_tmp"
    by_type = {}
    total_fp = 0
    total_seeded = 0
    for seed in range(seeds):
        per_incident, fps, total = run_one_seed(seed)
        total_fp += fps
        total_seeded += total
        for row in per_incident:
            by_type.setdefault(row["type"], []).append(row["detected"])
    shutil.rmtree(tmp_dir, ignore_errors=True)

    rows = []
    overall_detected = 0
    overall_total = 0
    for noise_type, hits in sorted(by_type.items()):
        detected = sum(hits)
        overall_detected += detected
        overall_total += len(hits)
        rows.append({
            "type": noise_type,
            "detected": detected,
            "total": len(hits),
            "recall": round(100 * detected / len(hits), 1),
        })
    return {
        "seeds": seeds,
        "total_seeded": total_seeded,
        "false_positives": total_fp,
        "overall_recall": round(100 * overall_detected / overall_total, 1) if overall_total else 0,
        "by_type": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest-seeds", type=int, default=15)
    parser.add_argument("--repo-url", default="#", help="GitHub repo URL to wire up the page's links to")
    args = parser.parse_args()

    r = report.compute_report()
    auto_fixed = [ev for ev in r["incidents"] if ev.auto_fix]
    data = {
        "github_url": args.repo_url,
        "summary": {
            "transactions_audited": r["transactions_audited"],
            "total_days": r["total_days"],
            "drift_days": r["drift_days"],
            "incident_count": len(r["incidents"]),
            "value_at_risk": round(r["value_at_risk"], 2),
            "self_resolving_value": round(r["self_resolving_value"], 2),
            "rule_classified": r["rule_classified"],
            "llm_classified": r["llm_classified"],
            "auto_fixed_count": len(auto_fixed),
            "auto_fixed_value": round(sum(abs(ev.delta) for ev in auto_fixed), 2),
        },
        "incidents": build_incidents_json(r),
        "backtest": run_backtest(args.backtest_seeds),
    }

    template_path = Path(__file__).parent / "template.html"
    out_path = Path(__file__).parent / "index.html"
    template = template_path.read_text(encoding="utf-8")
    injected = template.replace(
        "/*__DATA__*/",
        f"const DATA = {json.dumps(data, indent=2)};",
    )
    out_path.write_text(injected, encoding="utf-8")
    print(f"Wrote {out_path} ({len(data['incidents'])} incidents, "
          f"{data['backtest']['overall_recall']}% backtested recall)")


if __name__ == "__main__":
    main()
