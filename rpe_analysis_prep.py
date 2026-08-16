"""
One-off data preparation for the within-animal RPE analyses requested on top
of the pooled GLM report (see conversation for full spec): a single expensive
pass over all "none"-condition sessions, reusing pipeline.run_session exactly
as models.glm_data.build_pooled_glm_dataset and models.fir_glm.
build_pooled_fir_dataset already do, so every downstream statistical test can
run fast and locally afterward with no further raw-data reprocessing.

Saves, under outputs_fixed/rpe_analysis/:
  - peth_windows.npz          -- side_in-aligned z-score PETH windows + peth_time
  - pooled_trial_table.parquet -- per-trial covariates (mouse, date, chose_right,
                                   was_rewarded, Q_left, Q_right, Q_diff,
                                   Behavioral_State, word/lag columns, ...),
                                   row-aligned with peth_windows.npz
  - fir_pooled.npz            -- FIR design-matrix pieces (y, Phi, groups, mouse)
  - fir_column_names.pkl      -- FIR column_names + n_lags, for reshape_kernels

Usage:
    python rpe_analysis_prep.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config.params import DEFAULT_HEMISPHERE, FINAL_SAMPLE_FREQ_HZ
from config.session_metadata import get_mouse_hemisphere, load_mouse_hemisphere
from io_utils.raw_loader import parse_session_id
from models.fir_glm import (
    DEFAULT_GROUP_COLUMN,
    DEFAULT_LAG_SECONDS,
    build_event_impulses,
    build_shifted_design_matrix,
    build_task_mask_and_groups,
)
from pipeline import run_session
from qc.channel_selection import load_session_hemisphere_overrides
from run_glm_analysis import DEFAULT_HEMISPHERE_LOOKUP, DEFAULT_SESSION_OVERRIDES

OUT_DIR = Path("outputs_fixed/rpe_analysis")


def build_pooled_dataset(session_dirs, hemisphere_lookup_path=DEFAULT_HEMISPHERE_LOOKUP,
                          session_overrides_path=DEFAULT_SESSION_OVERRIDES,
                          group_col=DEFAULT_GROUP_COLUMN, n_lags_seconds=DEFAULT_LAG_SECONDS,
                          out_dir=OUT_DIR):
    hemisphere_lookup = load_mouse_hemisphere(hemisphere_lookup_path)
    session_overrides = load_session_hemisphere_overrides(session_overrides_path)

    def hemisphere_for(mouse, date):
        if (mouse, date) in session_overrides:
            return session_overrides[(mouse, date)]
        return get_mouse_hemisphere(hemisphere_lookup, mouse, DEFAULT_HEMISPHERE)

    loaded = []
    for session_dir in session_dirs:
        mouse, date = parse_session_id(session_dir)
        hemisphere = hemisphere_for(mouse, date)
        print(f"--- {mouse} {date} (hemisphere={hemisphere}) ---")
        try:
            result = run_session(session_dir, hemisphere=hemisphere, align_event="side_in")
        except Exception as exc:
            print(f"WARNING: skipping {session_dir} ({mouse} {date}): {exc}")
            continue
        loaded.append((mouse, date, result))

    if not loaded:
        raise RuntimeError("No sessions successfully processed")

    all_groups = set()
    for _, _, result in loaded:
        all_groups.update(result["trial_table"][group_col].dropna().unique())
    group_values = sorted(all_groups)
    n_lags = int(round(n_lags_seconds * FINAL_SAMPLE_FREQ_HZ))

    peth_time = None
    window_frames, table_frames = [], []
    y_parts, phi_parts, group_parts, mouse_parts = [], [], [], []
    column_names = None
    group_offset = 0

    for mouse, date, result in loaded:
        if peth_time is None:
            peth_time = result["peth_time"]
        elif not np.array_equal(peth_time, result["peth_time"]):
            raise ValueError(f"{mouse} {date} has a different peth_time grid")

        session_table = result["peth_trial_table"].copy()
        session_table["mouse"] = mouse
        session_table["date"] = date
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
        group_offset += len(trial_table)

    zscore_windows = np.vstack(window_frames)
    pooled_trial_table = pd.concat(table_frames, ignore_index=True)
    y_pooled = np.concatenate(y_parts)
    Phi_pooled = np.vstack(phi_parts)
    groups_pooled = np.concatenate(group_parts)
    mouse_pooled = np.concatenate(mouse_parts)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "peth_windows.npz", zscore_windows=zscore_windows, peth_time=peth_time)
    pooled_trial_table.to_parquet(out_dir / "pooled_trial_table.parquet")
    np.savez(out_dir / "fir_pooled.npz", y=y_pooled, Phi=Phi_pooled, groups=groups_pooled, mouse=mouse_pooled)
    with open(out_dir / "fir_column_names.pkl", "wb") as f:
        pickle.dump(dict(column_names=column_names, n_lags=n_lags), f)

    print(f"\nSaved pooled dataset to {out_dir}:")
    print(f"  PETH: {len(pooled_trial_table)} trials, windows {zscore_windows.shape}")
    print(f"  FIR: {len(y_pooled)} samples across {group_offset} trials, "
          f"{len(np.unique(mouse_pooled))} mice")


if __name__ == "__main__":
    import sys
    session_list_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/none_session_dirs.txt"
    out_dir_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT_DIR
    with open(session_list_path) as f:
        session_dirs = [line.strip() for line in f if line.strip()]
    build_pooled_dataset(session_dirs, out_dir=out_dir_arg)
