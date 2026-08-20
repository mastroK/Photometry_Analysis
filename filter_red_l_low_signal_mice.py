"""
Derived copy of outputs_fixed/rpe_analysis_sm_red_l/ excluding regular_mice
that showed essentially zero FIR-explainable red_l signal (analysis3b_fir_
glm_per_mouse_regular_mice.csv: SM2L r2=0.003, SM3MN r2=0.005, SM3MR
r2=0.0005 -- all near the noise floor, vs. 0.05-0.10 for the other 5 regular
mice). carrier_valid (the channel-inclusion gate) only certifies a real 223Hz
oscillation exists above the noise floor; it says nothing about whether that
channel, once correctly demodulated, actually encodes task-relevant signal --
these three mice pass that gate but carry no usable red_l signal, so pooling
them in dilutes/dominates group-level results with noise (e.g. SM3MR's
"significant" beta_RPE, p=2.8e-7, r2=0.008 -- almost certainly significant-by-
-sample-size, not a real effect).

Excludes ONLY from the regular_mice group -- jrgeco_only (SM1B, SM2B) both
show real signal (r2=0.056, 0.144) and are untouched.

Writes a full sibling dataset (pooled_trial_table.parquet, peth_windows.npz,
fir_pooled.npz, fir_column_names.pkl) so run_sm_glm_fir_analysis.py /
rpe_analysis_stats_SM.py / rpe_analysis_figures_SM.py all work UNCHANGED
against the new --data-dir, same derived-copy pattern already used for the
red_l group relabeling itself.

Usage:
    python filter_red_l_low_signal_mice.py
"""

import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path("outputs_fixed/rpe_analysis_sm_red_l")
OUT_DIR = Path("outputs_fixed/rpe_analysis_sm_red_l_excl_low_signal")

EXCLUDE_MICE = {"SM2L", "SM3MN", "SM3MR"}
EXCLUDE_FROM_GROUP = "regular_mice"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tt = pd.read_parquet(SRC_DIR / "pooled_trial_table.parquet")
    drop = (tt["hemisphere"] == EXCLUDE_FROM_GROUP) & (tt["mouse"].isin(EXCLUDE_MICE))
    keep = ~drop.to_numpy()
    print(f"trial_table: dropping {drop.sum()}/{len(tt)} trials "
          f"({', '.join(sorted(EXCLUDE_MICE))} from {EXCLUDE_FROM_GROUP})")
    tt.loc[keep].reset_index(drop=True).to_parquet(OUT_DIR / "pooled_trial_table.parquet")

    with np.load(SRC_DIR / "peth_windows.npz") as f:
        zscore_windows, peth_time = f["zscore_windows"], f["peth_time"]
    assert zscore_windows.shape[0] == len(tt), "peth_windows rows must match trial_table rows"
    np.savez(OUT_DIR / "peth_windows.npz", zscore_windows=zscore_windows[keep], peth_time=peth_time)
    print(f"peth_windows: {keep.sum()}/{len(keep)} rows kept")

    with np.load(SRC_DIR / "fir_pooled.npz", allow_pickle=True) as f:
        fir = {k: f[k] for k in f.files}
    fir_drop = (fir["hemisphere"] == EXCLUDE_FROM_GROUP) & np.isin(fir["mouse"], list(EXCLUDE_MICE))
    fir_keep = ~fir_drop
    print(f"fir_pooled: dropping {fir_drop.sum()}/{len(fir_drop)} samples")
    fir_filtered = {k: v[fir_keep] for k, v in fir.items()}
    np.savez(OUT_DIR / "fir_pooled.npz", **fir_filtered)

    shutil.copyfile(SRC_DIR / "fir_column_names.pkl", OUT_DIR / "fir_column_names.pkl")

    tt_kept = tt.loc[keep]
    print(f"\nFinal session/trial counts by group:")
    print(tt_kept.groupby("hemisphere")["mouse"].value_counts())
    print(f"\nSaved filtered dataset to {OUT_DIR}")


if __name__ == "__main__":
    main()
