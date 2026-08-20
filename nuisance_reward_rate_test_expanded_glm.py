"""
Extends the nuisance-regressor check (nuisance_reward_rate_test.py) from the
single RPE_signed term to the FULL EXPANDED_TIME_RESOLVED_GLM_FORMULA -- the
formula task #64 (SM expanded run) would use. Tests whether adding local
reward rate as a covariate shifts the coefficient TRAJECTORIES (beta(t)
across the whole peri-event window) of the terms most likely to actually be
confounded with it: Reward_lag1/2/3, RPE, RPE_abs, belief_p_right, and the
switch-dynamics terms -- all of which are, at some level, other encodings of
recent reward history, unlike the earlier single-number post_amp~RPE_signed
check.

Rebuilds the pooled dataset fresh from raw sessions (build_pooled_glm_dataset
-- same as run_expanded_glm_analysis.py originally used), since the earlier
RPE-analysis pooled parquet predates the expanded-formula columns
(RPE/belief_p_right/switch-dynamics) being wired into pipeline.run_session.

Usage:
    python nuisance_reward_rate_test_expanded_glm.py [cohort_label]
    # cohort_label one of run_expanded_glm_analysis.COHORTS' labels, default FP2_none
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config.params import EXPANDED_TIME_RESOLVED_GLM_FORMULA
from models.glm_data import build_pooled_glm_dataset
from models.glm_encoding import fit_time_resolved_glm
from nuisance_reward_rate_test import add_local_reward_rate
from run_expanded_glm_analysis import COHORTS, session_hemisphere_lookup

TERMS_OF_INTEREST = [
    "Reward_lag1", "Reward_lag2", "Reward_lag3",
    "RPE", "RPE_abs", "Choice:RPE",
    "belief_p_right", "true_switch", "trials_since_switch", "trials_since_switch_expdecay",
    "detected_switch", "first_win_after_switch", "first_loss_after_switch",
]


def compare_trajectories(fit_without, fit_with, terms):
    rows = []
    for term in terms:
        col = f"{term}_beta"
        if col not in fit_without.columns or col not in fit_with.columns:
            continue
        b_without = fit_without[col].to_numpy()
        b_with = fit_with[col].to_numpy()
        ok = np.isfinite(b_without) & np.isfinite(b_with)
        if ok.sum() < 3:
            continue
        corr = np.corrcoef(b_without[ok], b_with[ok])[0, 1]
        peak_idx = np.nanargmax(np.abs(b_without))
        peak_time = fit_without.index[peak_idx]
        peak_without = b_without[peak_idx]
        peak_with = b_with[peak_idx]
        pct_change_at_peak = 100 * (peak_with - peak_without) / abs(peak_without) if peak_without != 0 else np.nan
        max_abs_diff = np.max(np.abs(b_with[ok] - b_without[ok]))
        rows.append(dict(
            term=term, corr_beta_t=corr, peak_time_s=peak_time,
            beta_at_peak_without=peak_without, beta_at_peak_with=peak_with,
            pct_change_at_peak=pct_change_at_peak, max_abs_diff_across_t=max_abs_diff,
        ))
    return pd.DataFrame(rows).set_index("term")


def main(cohort_label="FP2_none"):
    cohort = next(c for c in COHORTS if c[0] == cohort_label)
    _, master_csv, qc_report_csv, exclude_pairs = cohort
    session_dirs, hemisphere_for_session = session_hemisphere_lookup(master_csv, qc_report_csv, exclude_pairs)
    print(f"{cohort_label}: {len(session_dirs)} sessions")

    peth_time, zscore_windows, trial_table = build_pooled_glm_dataset(
        session_dirs, align_event="side_in", hemisphere_for_session=lambda d: hemisphere_for_session[d])
    print(f"Pooled: {len(trial_table)} trials, {trial_table['mouse'].nunique()} mice")

    trial_table = add_local_reward_rate(trial_table)
    n_missing = trial_table["local_reward_rate"].isna().sum()
    print(f"{n_missing}/{len(trial_table)} trials missing local_reward_rate (dropped by listwise deletion downstream)")

    print("\nFitting expanded formula WITHOUT local_reward_rate...")
    fit_without = fit_time_resolved_glm(zscore_windows, peth_time, trial_table, formula=EXPANDED_TIME_RESOLVED_GLM_FORMULA)

    print("Fitting expanded formula WITH local_reward_rate...")
    formula_with = EXPANDED_TIME_RESOLVED_GLM_FORMULA + " + local_reward_rate"
    fit_with = fit_time_resolved_glm(zscore_windows, peth_time, trial_table, formula=formula_with)

    pd.set_option("display.width", 160)
    print(f"\n=== {cohort_label}: expanded-formula terms, with vs without local_reward_rate covariate ===")
    comparison = compare_trajectories(fit_without, fit_with, TERMS_OF_INTEREST)
    print(comparison.to_string())

    if "local_reward_rate_beta" in fit_with.columns:
        lr = fit_with["local_reward_rate_beta"].to_numpy()
        lr_p = fit_with["local_reward_rate_pvalue"].to_numpy()
        frac_sig = np.nanmean(lr_p < 0.05)
        print(f"\nlocal_reward_rate's own coefficient: mean beta={np.nanmean(lr):.4f}, "
              f"frac of time points significant (p<0.05)={frac_sig:.2f}, "
              f"peak |beta| time={fit_with.index[np.nanargmax(np.abs(lr))]:.3f}s")

    out_path = Path(f"outputs_fixed/nuisance_expanded_glm_{cohort_label}.csv")
    comparison.to_csv(out_path)
    print(f"\nSaved {out_path}")

    fit_with_path = Path(f"outputs_fixed/nuisance_expanded_glm_{cohort_label}_fit_with.csv")
    fit_with.to_csv(fit_with_path)
    print(f"Saved full with-covariate fit (all terms x peth_time) to {fit_with_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "FP2_none")
