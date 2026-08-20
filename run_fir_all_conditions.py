"""
Run the FIR deconvolution GLM (models.fir_glm / run_fir_glm.py's existing,
established machinery -- same hemisphere-resolution convention originally
used for FP1/FP2 in tasks #29/#37) fresh for all 5 FP1/FP2 conditions, so
the comprehensive report has FIR alongside the default/expanded time-
resolved GLM for every condition.

Usage:
    python run_fir_all_conditions.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from run_expanded_glm_analysis import COHORTS
from run_fir_glm import run_fir_glm

OUT_DIR = Path("figures_fixed_expanded_glm")


def session_dirs_for_cohort(master_csv, qc_report_csv, exclude_pairs):
    """session_dirs restricted to (mouse, date) pairs actually present in
    master_csv -- NOT every row in qc_report_csv. A session can pass QC
    (QC_PASS=True) and still be correctly absent from the master for other
    curation reasons (e.g. none_cohort_master.csv's 8 sessions removed for
    being mislabeled DCZ/saline, not actually "none" condition -- confirmed
    directly: qc_report has 46 rows for FP1_none, master has only 38).
    Blindly using qc_report_csv's full row set would silently pull those
    wrongly-labeled sessions back into the FIR pool.
    """
    master = pd.read_csv(master_csv, dtype={"date": str})
    qc = pd.read_csv(qc_report_csv, dtype={"date": str})
    keep_pairs = set(zip(master["mouse"], master["date"])) - exclude_pairs
    qc = qc[qc.apply(lambda r: (r["mouse"], r["date"]) in keep_pairs, axis=1)]
    return [Path(p) for p in qc["session_dir"]]


def main(only_labels=None):
    for label, master_csv, qc_report_csv, exclude_pairs in COHORTS:
        if only_labels is not None and label not in only_labels:
            continue
        print(f"\n{'=' * 70}\n{label}: FIR deconvolution GLM\n{'=' * 70}")
        session_dirs = session_dirs_for_cohort(master_csv, qc_report_csv, exclude_pairs)
        print(f"{len(session_dirs)} sessions")
        fit = run_fir_glm(session_dirs, output_dir=OUT_DIR)

        # run_fir_glm's own output filename (pooled_{n}sessions_{group_col}_
        # fir_kernels) doesn't include the cohort label and could collide
        # across conditions with the same session count -- rename to a
        # cohort-prefixed name immediately so nothing gets silently
        # overwritten or ambiguous in the final report.
        mice = [info["mouse"] for info in fit["session_info"]]
        n_sessions = len(fit["session_info"])
        stale_stem = OUT_DIR / f"pooled_{n_sessions}sessions_reward_seq_3_fir_kernels"
        new_stem = OUT_DIR / f"{label.lower()}_fir_kernels"
        for ext in (".png", ".svg"):
            stale_path = stale_stem.with_suffix(ext)
            if stale_path.exists():
                stale_path.rename(new_stem.with_suffix(ext))
                print(f"Renamed {stale_path.name} -> {new_stem.with_suffix(ext).name}")

        # Persist fold_kernels/lag_time_s (not just the plotted figure) --
        # needed to overlay multiple conditions' FIR kernels on the same
        # axes in the cross-condition comparison report.
        kernel_arrays = {f"kernel__{name}": arr for name, arr in fit["fold_kernels"].items()}
        np.savez(OUT_DIR / f"{label.lower()}_fir_fold_kernels.npz",
                  lag_time_s=fit["lag_time_s"], **kernel_arrays)
        print(f"Saved {label.lower()}_fir_fold_kernels.npz "
              f"({len(fit['fold_kernels'])} features, R^2={fit['cv_results']['r2_mean']:.4f})")


if __name__ == "__main__":
    main()
