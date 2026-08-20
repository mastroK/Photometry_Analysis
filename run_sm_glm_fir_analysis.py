"""
Time-resolved GLM + FIR deconvolution GLM for the SM (PV-dualphotometry)
cohort's pooled green-channel (GCaMP/pyramidal) dataset.

Unlike run_glm_analysis.py/run_fir_glm.py (which rebuild the pooled dataset
from session_dirs + a per-session hemisphere resolver -- a model that assumes
exactly ONE valid hemisphere per session), this consumes the ALREADY-POOLED
outputs of rpe_analysis_prep_SM.py directly:
  - peth_windows.npz / pooled_trial_table.parquet -> fit_time_resolved_glm
  - fir_pooled.npz / fir_column_names.pkl         -> fit_fir_glm

That prep step already handles SM's bilateral (session, hemisphere) pairs
(both green_r and green_l can be valid at once) and forced-choice trial
exclusion, so there is nothing left to re-derive here -- this script only
fits + plots.

Run rpe_analysis_prep_SM.py first (against
outputs_fixed/sm_corrected_channel_report.csv) to produce its inputs.

Usage:
    python run_sm_glm_fir_analysis.py [--data-dir outputs_fixed/rpe_analysis_sm]
        [--output-dir figures_fixed_sm] [--formula "..."]
        [--group-col reward_seq_3] [--n-splits 10] [--test-size 0.2]
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config.params import DEFAULT_TIME_RESOLVED_GLM_FORMULA
from models.fir_glm import DEFAULT_N_SPLITS, DEFAULT_TEST_SIZE, fit_fir_glm, reshape_kernels
from models.glm_encoding import fit_time_resolved_glm
from viz.fir_plots import plot_fir_kernels
from viz.glm_plots import plot_glm_coefficients

DEFAULT_DATA_DIR = Path("outputs_fixed/rpe_analysis_sm")
DEFAULT_OUTPUT_DIR = Path("figures_fixed_sm")


def run_time_resolved_glm(data_dir, output_dir, formula=DEFAULT_TIME_RESOLVED_GLM_FORMULA):
    """Fits the time-resolved GLM SEPARATELY per hemisphere (green_r, green_l)
    -- two independent fits, not one pooled fit with hemisphere as a nuisance
    term -- so each hemisphere gets its own coefficient trajectory rather than
    a single trajectory averaged across both fibers.
    """
    with np.load(data_dir / "peth_windows.npz") as f:
        zscore_windows, peth_time = f["zscore_windows"], f["peth_time"]
    trial_table = pd.read_parquet(data_dir / "pooled_trial_table.parquet")

    results = {}
    for hemisphere in sorted(trial_table["hemisphere"].unique()):
        hemi_mask = (trial_table["hemisphere"] == hemisphere).to_numpy()
        hemi_windows = zscore_windows[hemi_mask]
        hemi_table = trial_table[hemi_mask].reset_index(drop=True)

        print(f"\n=== SM time-resolved GLM (side_in, {hemisphere} only): "
              f"{hemi_windows.shape[0]} trials, {hemi_table['mouse'].nunique()} mice ===")
        beta_df = fit_time_resolved_glm(hemi_windows, peth_time, hemi_table, formula=formula)

        fig = plot_glm_coefficients(peth_time, beta_df, formula, align_event="side_in")
        out_stem = output_dir / f"sm_glm_side_in_coefficients_{hemisphere}"
        fig.savefig(out_stem.with_suffix(".png"), dpi=150)
        fig.savefig(out_stem.with_suffix(".svg"))
        print(f"Saved {out_stem.with_suffix('.png')} and {out_stem.with_suffix('.svg')}")
        results[hemisphere] = beta_df
    return results


def run_fir_deconvolution(data_dir, output_dir, group_col=None,
                           n_splits=DEFAULT_N_SPLITS, test_size=DEFAULT_TEST_SIZE):
    """Fits the FIR deconvolution GLM SEPARATELY per hemisphere -- two
    independent RidgeCV fits (each with its own CV/kernels), not one pooled
    fit with a hemisphere nuisance term, so each hemisphere gets its own
    kernels. Simpler and safer than adding a Hemisphere interaction to the
    FIR design matrix itself (which would require doubling every feature
    column), and gives directly comparable, independently-cross-validated
    R^2 per hemisphere as a bonus.
    """
    with np.load(data_dir / "fir_pooled.npz") as f:
        y, Phi, groups, mouse, hemisphere = f["y"], f["Phi"], f["groups"], f["mouse"], f["hemisphere"]
    with open(data_dir / "fir_column_names.pkl", "rb") as f:
        meta = pickle.load(f)
    column_names, n_lags = meta["column_names"], meta["n_lags"]

    results = {}
    for hemi in sorted(np.unique(hemisphere)):
        hemi_mask = hemisphere == hemi
        y_h, Phi_h, groups_h, mouse_h = y[hemi_mask], Phi[hemi_mask], groups[hemi_mask], mouse[hemi_mask]
        # rpe_analysis_prep_SM.py's build_pooled_dataset already applies the
        # task-mask (build_task_mask_and_groups) before saving y/Phi/groups,
        # so every saved sample is already in-mask -- mask is all-True here,
        # not a second filtering step.
        mask = np.ones(len(y_h), dtype=bool)
        n_mice = len(np.unique(mouse_h))

        print(f"\n=== SM FIR deconvolution GLM ({hemi} only): {len(y_h)} samples, {n_mice} mice ===")
        model, cv_results = fit_fir_glm(y_h, Phi_h, mask, groups_h, n_splits=n_splits, test_size=test_size)

        print(f"Lag window: +/-{n_lags / 18.5185:.2f}s ({n_lags} samples each side)")
        print(f"Design matrix: {len(column_names)} columns")
        print(f"Best alpha (final RidgeCV fit): {cv_results['best_alpha']:.4g}")
        print(f"Out-of-sample R^2 ({n_splits}-fold GroupShuffleSplit by trial): "
              f"{cv_results['r2_mean']:.4f} +/- {cv_results['r2_std']:.4f}")
        print(f"Out-of-sample MSE: {cv_results['mse_mean']:.4f} +/- {cv_results['mse_std']:.4f}")

        fold_kernels, lag_time_s = reshape_kernels(cv_results["fold_betas"], column_names, n_lags)
        group_label = group_col or "pooled groups"
        title_prefix = f"SM cohort, {n_mice} mice, {hemi} only ({group_label})"
        fig = plot_fir_kernels(fold_kernels, lag_time_s, title_prefix=title_prefix)
        out_stem = output_dir / f"sm_pooled_{n_mice}mice_{hemi}_fir_kernels"
        fig.savefig(out_stem.with_suffix(".png"), dpi=150)
        fig.savefig(out_stem.with_suffix(".svg"))
        print(f"Saved {out_stem.with_suffix('.png')} and {out_stem.with_suffix('.svg')}")
        results[hemi] = (model, cv_results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formula", default=DEFAULT_TIME_RESOLVED_GLM_FORMULA)
    parser.add_argument("--group-col", default=None,
                         help="Only used for the FIR figure title (the grouping itself is baked into "
                              "fir_pooled.npz by rpe_analysis_prep_SM.py's DEFAULT_GROUP_COLUMN)")
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_time_resolved_glm(args.data_dir, args.output_dir, formula=args.formula)
    run_fir_deconvolution(args.data_dir, args.output_dir, group_col=args.group_col,
                          n_splits=args.n_splits, test_size=args.test_size)


if __name__ == "__main__":
    main()
