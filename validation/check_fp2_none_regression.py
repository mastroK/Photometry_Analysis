"""
One-off regression check for run_model_series_comparison.py's generalization
(cohort_label param, truncation/manifest wiring -- see the plan this was
built from). Confirms the generalized main() reproduces the existing
FP2_none baseline (outputs_fixed/model_series_comparison/results/) exactly
when called with all-default params, WITHOUT writing into that directory
(it has no manifest.json, so write_run_manifest's overwrite guard can't
protect it -- see run_model_series_comparison.py's OUT_DIR default).

Usage:
    python validation/check_fp2_none_regression.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_model_series_comparison import main  # noqa: E402

BASELINE_DIR = Path("outputs_fixed/model_series_comparison/results")
CHECK_DIR = Path("outputs_fixed/model_series_comparison_fp2_none_regression_check/results")


def main_check():
    main(cohort_label="FP2_none", out_dir=CHECK_DIR, fig_dir=Path("figures_fixed_model_series_fp2_none_regression_check"))

    baseline_csv = pd.read_csv(BASELINE_DIR / "encoding_glm_model_comparison.csv")
    check_csv = pd.read_csv(CHECK_DIR / "encoding_glm_model_comparison.csv")

    csv_ok = baseline_csv.equals(check_csv)
    print(f"encoding_glm_model_comparison.csv identical: {csv_ok}")
    if not csv_ok:
        merged = baseline_csv.merge(check_csv, on=["mouse", "model", "event"], suffixes=("_baseline", "_check"))
        diff = merged[~np.isclose(merged["r2_mean_baseline"], merged["r2_mean_check"])]
        print("Rows differing in r2_mean:")
        print(diff)

    with open(BASELINE_DIR / "summary_stats.json") as f:
        baseline_stats = json.load(f)
    with open(CHECK_DIR / "summary_stats.json") as f:
        check_stats = json.load(f)
    stats_ok = baseline_stats == check_stats
    print(f"summary_stats.json identical: {stats_ok}")
    if not stats_ok:
        print("baseline:", json.dumps(baseline_stats, indent=2))
        print("check:", json.dumps(check_stats, indent=2))

    print(f"\n{'PASS' if csv_ok and stats_ok else 'FAIL'}")


if __name__ == "__main__":
    main_check()
