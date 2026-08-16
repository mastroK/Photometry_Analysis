"""
Onset-latency and decay-time-constant analysis, run on an already-pooled RPE
dataset (rpe_analysis_prep.py's output -- no raw reprocessing). Reuses
alignment.kinetics.compute_group_kinetics on the SAME pooled_trial_table.parquet
+ peth_windows.npz already built for the within-animal RPE/GLM/FIR analyses
(rpe_analysis_stats.py).

Computed per (mouse, was_rewarded) -- separately for reward and omission
trials, both because that's the natural condition split for this task and
because it keeps opposite-signed responses from being pooled together (see
alignment/kinetics.py's module docstring). Uses KINETICS_METRIC_WINDOW_S
(config.params), which is DELIBERATELY WIDER than DECISION_WINDOW_S -- see
that constant's comment for why (DECISION_WINDOW_S alone made decay fitting
fail on 21/22 real mouse x condition groups, confirmed directly on this same
pooled dataset before this window was chosen).

Output: two per-mouse CSVs (reward, omission), same indexed-by-mouse shape as
rpe_analysis_stats.py's analysis*_per_mouse.csv tables, so run_age_comparison.py
can read them with compare_metric unchanged.

Usage:
    python rpe_analysis_kinetics.py [data_dir] [out_dir]
"""

from pathlib import Path

import numpy as np
import pandas as pd

from alignment.kinetics import compute_group_kinetics
from config.params import KINETICS_METRIC_WINDOW_S

DATA_DIR = Path("outputs_fixed/rpe_analysis_pooled")
OUT_DIR = Path("outputs_fixed/rpe_analysis_pooled/results")


def load_pooled_peth(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    trial_table = pd.read_parquet(data_dir / "pooled_trial_table.parquet")
    peth = np.load(data_dir / "peth_windows.npz")
    return trial_table, peth["zscore_windows"], peth["peth_time"]


def main(data_dir=DATA_DIR, out_dir=OUT_DIR, metric_window_s=KINETICS_METRIC_WINDOW_S):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trial_table, zscore_windows, peth_time = load_pooled_peth(data_dir)
    print(f"Pooled dataset: {len(trial_table)} trials, {trial_table['mouse'].nunique()} mice, "
          f"metric_window_s={metric_window_s}")

    by_mouse_reward = compute_group_kinetics(
        zscore_windows, trial_table, peth_time, metric_window_s,
        group_cols=("mouse", "was_rewarded"),
    )
    print("\n=== Onset/decay kinetics, per mouse x reward/omission ===")
    print(by_mouse_reward.to_string())

    n_total = len(by_mouse_reward)
    n_skipped = int(by_mouse_reward["skip_reason"].notna().sum())
    print(f"\n{n_skipped}/{n_total} (mouse, reward) groups had the decay fit skipped:")
    if n_skipped:
        print(by_mouse_reward.loc[by_mouse_reward["skip_reason"].notna(), "skip_reason"].value_counts().to_string())

    by_mouse_reward.to_csv(out_dir / "analysis5_kinetics_by_mouse_and_reward.csv")

    for was_rewarded, label in [(True, "reward"), (False, "omission")]:
        sub = by_mouse_reward.xs(was_rewarded, level="was_rewarded")
        out_path = out_dir / f"analysis5_kinetics_{label}_per_mouse.csv"
        sub.to_csv(out_path)
        n_ok = int(sub["skip_reason"].isna().sum())
        print(f"\nSaved {label} table ({n_ok}/{len(sub)} mice with a successful decay fit) to {out_path}")

    print(f"\nSaved all kinetics results to {out_dir}")
    return by_mouse_reward


if __name__ == "__main__":
    import sys
    data_dir_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR
    out_dir_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT_DIR
    main(data_dir=data_dir_arg, out_dir=out_dir_arg)
