"""
Pooled RPE/GLM/FIR data preparation for the SM (PV-dualphotometry) cohort --
adapted from rpe_analysis_prep.py for two differences from FP1/FP2/WCL:

1. Bilateral hemispheres: SM sessions can validly contribute BOTH green_r and
   green_l (see run_sm_bilateral_batch.py's bilateral-validity report),
   unlike FP1/FP2's one-hemisphere-per-mouse lookup. So this takes explicit
   (session_dir, hemisphere) pairs rather than resolving hemisphere from a
   per-mouse/per-session lookup, and stamps `hemisphere` as a column on the
   pooled trial table (alongside mouse/date) for the downstream RPE_signed *
   C(hemisphere) robustness check in rpe_analysis_stats_SM.py.
2. Forced-choice (100-0) trial exclusion: behavior.trial_table.build_trial_table
   already flags each trial's `is_forced_block` (a real 80/20 bandit trial
   has neither reward prob at 0 or 1). That flag must NOT be used to drop
   rows before run_session computes its sequential features (word/lag
   sequences, Q-learning fit) -- those need the complete chronological trial
   sequence, forced blocks included, or every real trial's N-back/Q-value
   context downstream of a forced block would be silently wrong. So exclusion
   happens HERE, right after run_session returns, before PETH/FIR assembly --
   never inside trial_table construction itself.

Saves, under outputs_fixed/rpe_analysis_sm/ (mirrors rpe_analysis_prep.py's
FP1/FP2 output shape exactly, plus a `hemisphere` column):
  - peth_windows.npz          -- side_in-aligned z-score PETH windows + peth_time
  - pooled_trial_table.parquet -- per-trial covariates, row-aligned with peth_windows.npz
  - fir_pooled.npz            -- FIR design-matrix pieces (y, Phi, groups, mouse, hemisphere)
  - fir_column_names.pkl      -- FIR column_names + n_lags, for reshape_kernels

Usage:
    python rpe_analysis_prep_SM.py outputs_fixed/sm_bilateral_hemisphere_report.csv
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config.params import FINAL_SAMPLE_FREQ_HZ
from models.fir_glm import (
    DEFAULT_GROUP_COLUMN,
    DEFAULT_LAG_SECONDS,
    build_event_impulses,
    build_shifted_design_matrix,
    build_task_mask_and_groups,
)
from pipeline import run_session

OUT_DIR = Path("outputs_fixed/rpe_analysis_sm")
HEMISPHERES = ("green_r", "green_l")


def session_hemisphere_pairs_from_report(report_path, hemispheres=HEMISPHERES):
    """Read run_sm_bilateral_batch.py's bilateral-validity report and return
    every (session_dir, hemisphere) pair that passed its independent
    per-hemisphere gate -- one session can contribute 0, 1, or 2 pairs.
    """
    report = pd.read_csv(report_path)
    pairs = []
    for _, row in report.iterrows():
        for hemisphere in hemispheres:
            if row.get(f"{hemisphere}_valid"):
                pairs.append((Path(row["session_dir"]), hemisphere))
    return pairs


def build_pooled_dataset(session_hemisphere_pairs, group_col=DEFAULT_GROUP_COLUMN,
                          n_lags_seconds=DEFAULT_LAG_SECONDS, out_dir=OUT_DIR):
    loaded = []
    n_forced_total, n_trials_total = 0, 0
    for session_dir, hemisphere in session_hemisphere_pairs:
        session_dir = Path(session_dir)
        mouse, date = session_dir.name, session_dir.parent.name
        print(f"--- {mouse} {date} (hemisphere={hemisphere}) ---")
        try:
            result = run_session(session_dir, hemisphere=hemisphere, align_event="side_in")
        except Exception as exc:
            print(f"WARNING: skipping {session_dir} ({mouse} {date}, {hemisphere}): {exc}")
            continue

        # Forced-choice (100-0) exclusion -- see module docstring for why this
        # happens here, not inside trial_table construction.
        trial_table = result["trial_table"]
        peth_trial_table = result["peth_trial_table"]
        all_zscore_windows = result["all_zscore_windows"]

        n_forced = int(trial_table["is_forced_block"].sum())
        n_forced_total += n_forced
        n_trials_total += len(trial_table)
        if n_forced:
            print(f"  Dropping {n_forced}/{len(trial_table)} forced-choice (100-0) trials")

        trial_table = trial_table[~trial_table["is_forced_block"]].reset_index(drop=True)
        peth_keep = ~peth_trial_table["is_forced_block"].to_numpy()
        peth_trial_table = peth_trial_table[peth_keep].reset_index(drop=True)
        all_zscore_windows = all_zscore_windows[peth_keep]

        if len(peth_trial_table) == 0:
            print(f"  WARNING: skipping {mouse} {date} ({hemisphere}) -- no real (non-forced) trials remain")
            continue

        result = dict(result)  # shallow copy, don't mutate run_session's own dict
        result["trial_table"] = trial_table
        result["peth_trial_table"] = peth_trial_table
        result["all_zscore_windows"] = all_zscore_windows
        loaded.append((mouse, date, hemisphere, result))

    print(f"\nDropped {n_forced_total}/{n_trials_total} forced-choice trials across all loaded sessions")

    if not loaded:
        raise RuntimeError("No sessions successfully processed")

    all_groups = set()
    for _, _, _, result in loaded:
        all_groups.update(result["trial_table"][group_col].dropna().unique())
    group_values = sorted(all_groups)
    n_lags = int(round(n_lags_seconds * FINAL_SAMPLE_FREQ_HZ))

    peth_time = None
    window_frames, table_frames = [], []
    y_parts, phi_parts, group_parts, mouse_parts, hemisphere_parts = [], [], [], [], []
    column_names = None
    group_offset = 0

    for mouse, date, hemisphere, result in loaded:
        if peth_time is None:
            peth_time = result["peth_time"]
        elif not np.array_equal(peth_time, result["peth_time"]):
            raise ValueError(f"{mouse} {date} ({hemisphere}) has a different peth_time grid")

        session_table = result["peth_trial_table"].copy()
        session_table["mouse"] = mouse
        session_table["date"] = date
        session_table["hemisphere"] = hemisphere
        table_frames.append(session_table)
        window_frames.append(result["all_zscore_windows"])

        trial_table = result["trial_table"]
        continuous_signal = np.asarray(result["zscore"], dtype=float)
        n_samples = len(continuous_signal)
        impulses = build_event_impulses(trial_table, n_samples, group_col=group_col, group_values=group_values)
        Phi, cols = build_shifted_design_matrix(impulses, n_lags)
        if column_names is None:
            column_names = cols
        mask, groups = build_task_mask_and_groups(trial_table, n_samples, n_lags)

        y_parts.append(continuous_signal[mask])
        phi_parts.append(Phi[mask])
        group_parts.append(groups[mask] + group_offset)
        mouse_parts.append(np.full(int(mask.sum()), mouse))
        hemisphere_parts.append(np.full(int(mask.sum()), hemisphere))
        group_offset += len(trial_table)

    zscore_windows = np.vstack(window_frames)
    pooled_trial_table = pd.concat(table_frames, ignore_index=True)
    y_pooled = np.concatenate(y_parts)
    Phi_pooled = np.vstack(phi_parts)
    groups_pooled = np.concatenate(group_parts)
    mouse_pooled = np.concatenate(mouse_parts)
    hemisphere_pooled = np.concatenate(hemisphere_parts)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "peth_windows.npz", zscore_windows=zscore_windows, peth_time=peth_time)
    pooled_trial_table.to_parquet(out_dir / "pooled_trial_table.parquet")
    np.savez(
        out_dir / "fir_pooled.npz", y=y_pooled, Phi=Phi_pooled, groups=groups_pooled,
        mouse=mouse_pooled, hemisphere=hemisphere_pooled,
    )
    with open(out_dir / "fir_column_names.pkl", "wb") as f:
        pickle.dump(dict(column_names=column_names, n_lags=n_lags), f)

    n_sessions = pooled_trial_table[["mouse", "date", "hemisphere"]].drop_duplicates().shape[0]
    print(f"\nSaved pooled dataset to {out_dir}:")
    print(f"  PETH: {len(pooled_trial_table)} trials, windows {zscore_windows.shape}, "
          f"{n_sessions} (session, hemisphere) pairs")
    print(f"  FIR: {len(y_pooled)} samples across {group_offset} trials, "
          f"{len(np.unique(mouse_pooled))} mice")
    print(pooled_trial_table.groupby("mouse")["hemisphere"].value_counts())


if __name__ == "__main__":
    import sys
    report_path = sys.argv[1] if len(sys.argv) > 1 else "outputs_fixed/sm_bilateral_hemisphere_report.csv"
    out_dir_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT_DIR
    pairs = session_hemisphere_pairs_from_report(report_path)
    print(f"Loaded {len(pairs)} valid (session, hemisphere) pair(s) from {report_path}")
    build_pooled_dataset(pairs, out_dir=out_dir_arg)
