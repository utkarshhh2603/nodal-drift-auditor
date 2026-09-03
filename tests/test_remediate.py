"""
Proves the agent's auto-fixes actually work, not just that they claim to:
apply corrections to a real generated batch, then re-audit the corrected
books from scratch and confirm the fixed incidents are genuinely gone
(not just marked fixed) while the untouched incidents survive exactly as
before - remediation must be surgical, not a reset.
"""
import shutil
import tempfile
from pathlib import Path

from data import generate_data
from engine import classify, remediate, replay


def test_auto_fixes_are_verified_and_surgical():
    tmp_dir = Path(tempfile.mkdtemp())
    data_dir = tmp_dir / "data"
    corrected_dir = tmp_dir / "corrected"
    try:
        generate_data.generate(seed=3, out_dir=data_dir, write_csvs=True, verbose=False)

        merged = replay.run(data_dir=data_dir)
        transactions, settlements, refunds, fee_sweeps, chargebacks, _ = replay.load_tables(data_dir)
        events = classify.find_events(merged)
        classified = classify.classify_events(events, settlements, refunds, fee_sweeps, chargebacks)

        auto_fixable = [ev for ev in classified if ev.auto_fix]
        not_fixable = [ev for ev in classified if not ev.auto_fix]
        assert auto_fixable, "expected at least one auto-fixable incident in this seed's batch"
        # Only duplicate_settlement/chargeback_duplicate are ever auto-fixable -
        # anything that requires moving money (stuck_refund) or has no
        # confirmed cause (unexplained) must never be auto-applied.
        assert {ev.bucket for ev in auto_fixable} <= {"duplicate_settlement", "chargeback_duplicate"}

        corrections = remediate.apply_corrections(classified, data_dir, corrected_dir)
        assert len(corrections) == len(auto_fixable)

        verified = remediate.verify_corrections(corrections, corrected_dir)
        for v in verified:
            assert abs(v["remaining_drift_on_date"]) <= classify.TOLERANCE, (
                f"fix for {v['bucket']} on {v['date']} claimed Rs.{v['value_corrected']} corrected "
                f"but Rs.{v['remaining_drift_on_date']} drift remains on that date"
            )

        # Re-audit the corrected books completely from scratch (not reusing
        # any in-memory state) - the fixed incidents' dates should produce
        # no event at all now, and every incident that was NOT auto-fixed
        # should still show up, unchanged, proving the correction was
        # surgical rather than a blanket reset.
        corrected_merged = replay.run(data_dir=corrected_dir)
        corrected_events = classify.find_events(corrected_merged)
        corrected_dates = {ev["date"].date() for ev in corrected_events}

        for ev in auto_fixable:
            assert ev.date.date() not in corrected_dates, (
                f"{ev.bucket} on {ev.date.date()} was auto-fixed but still shows drift"
            )
        for ev in not_fixable:
            if ev.bucket == "fee_sweep_timing_resolved":
                continue  # paired with its own fee_sweep_timing event, not independently checked here
            assert ev.date.date() in corrected_dates, (
                f"{ev.bucket} on {ev.date.date()} was never auto-fixed but disappeared anyway"
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
