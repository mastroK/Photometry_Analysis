"""
Regression check against validation/baselines/*.json snapshots -- re-runs each
saved reference session through pipeline.run_session() with the SAME default
parameters used to capture the baseline, and verifies the output is
byte-for-byte identical (via array hash) plus a few summary stats for a
human-readable diff.

Exists specifically to protect FP1/FP2/WCL results while the SM cohort's
carrier-frequency/cross-talk methodology gets rebuilt: every change to shared
code (pipeline.py, config/params.py, preprocessing/, alignment/, behavior/,
qc/) should be additive-only (new optional parameters defaulting to the old
behavior), and this script is the actual verification that held, not just a
promise -- run it after every such change, before committing.

Usage:
    python validation/check_regression_baseline.py
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import run_session  # noqa: E402

BASELINE_DIR = Path(__file__).parent / "baselines"

SESSION_DIRS = {
    "WCL23_060223": "/Volumes/Neurobio/MICROSCOPE/Kevin/3-Experiments/2-Behavior/3-nosepoke_WL/1-Raw data/FP1/060223/WCL23",
}


def array_hash(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def check_one(baseline_path):
    with open(baseline_path) as f:
        baseline = json.load(f)

    session_dir = SESSION_DIRS[baseline["session"]]
    result = run_session(session_dir, hemisphere=baseline["hemisphere"], align_event="side_in")

    current = dict(
        n_trials=len(result["trial_table"]),
        n_peth_trials=len(result["peth_trial_table"]),
        zscore_hash=array_hash(result["zscore"]),
        zscore_mean=float(np.nanmean(result["zscore"])),
        zscore_std=float(np.nanstd(result["zscore"])),
        all_zscore_windows_hash=array_hash(result["all_zscore_windows"]),
        all_zscore_windows_shape=list(result["all_zscore_windows"].shape),
        peak_z_side_in_mean=float(result["trial_table"]["peak_z_side_in"].mean()),
        auc_side_in_mean=float(result["trial_table"]["auc_side_in"].mean()),
        was_rewarded_sum=int(result["trial_table"]["was_rewarded"].sum()),
    )

    print(f"\n=== {baseline['session']} ({baseline['hemisphere']}) ===")
    all_match = True
    for key in current:
        match = current[key] == baseline[key]
        all_match &= match
        status = "OK" if match else "MISMATCH"
        print(f"  [{status}] {key}: baseline={baseline[key]!r} current={current[key]!r}")

    if all_match:
        print(f"PASS: {baseline['session']} output unchanged from baseline")
    else:
        print(f"FAIL: {baseline['session']} output DIVERGED from baseline -- investigate before committing")
    return all_match


def main():
    baseline_files = sorted(BASELINE_DIR.glob("*.json"))
    if not baseline_files:
        print(f"No baseline files found in {BASELINE_DIR}")
        sys.exit(1)

    results = [check_one(p) for p in baseline_files]
    print(f"\n{sum(results)}/{len(results)} baseline(s) passed")
    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
