"""
Sanity check: a batch with zero seeded incidents should audit to zero
findings. Every other test/demo in this repo seeds incidents on purpose,
so nothing so far actually proves the tool doesn't just always find
something to flag - this closes that gap.
"""
import shutil
import tempfile
from pathlib import Path

from data import generate_data
from engine import classify, replay


def test_clean_batch_yields_zero_incidents():
    tmp_dir = Path(tempfile.mkdtemp()) / "clean_batch"
    try:
        out_dir, ground_truth = generate_data.generate(
            seed=1, out_dir=tmp_dir, write_csvs=True, verbose=False, clean=True
        )
        assert ground_truth == []

        merged = replay.run(data_dir=out_dir)
        events = classify.find_events(merged)

        assert events == [], (
            f"Expected zero drift on a clean batch, found {len(events)} day(s) with drift: {events}"
        )
    finally:
        shutil.rmtree(tmp_dir.parent, ignore_errors=True)
