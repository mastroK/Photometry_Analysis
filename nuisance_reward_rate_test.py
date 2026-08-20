"""
Prototype test: does adding local reward rate as an explicit nuisance
regressor change the within-animal RPE effect (Analysis 1: post_amp ~
RPE_signed)? Run against FP2_none's already-pooled dataset -- no raw
reprocessing needed, since the regressor only requires trial timing
(photometry_side_in_index) and outcome (was_rewarded), both already in
pooled_trial_table.parquet.

For each mouse, fits both:
    post_amp ~ RPE_signed
    post_amp ~ RPE_signed + local_reward_rate
and compares beta_RPE (magnitude, sign, significance) between the two, plus
reports local_reward_rate's own coefficient (the size of the slow/tonic
component, if any). local_reward_rate is the count of rewarded trials
within +/-30s (60s window, matching BASELINE_WINDOW_SEC) of each trial's
side-in time, computed per-session (never leaking across sessions).

Usage:
    python nuisance_reward_rate_test.py [data_dir]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import wilcoxon

from config.params import BASELINE_WINDOW_SEC, FINAL_SAMPLE_FREQ_HZ
from rpe_analysis_stats import add_derived_columns, load_pooled_data

DATA_DIR = Path("outputs_fixed/rpe_analysis_fp2")
WINDOW_SEC = BASELINE_WINDOW_SEC  # 60s, matching the baseline window whose confound this targets


def add_local_reward_rate(tt, window_sec=WINDOW_SEC):
    """Per-trial count of rewarded trials within +/- window_sec/2 of this
    trial's side-in time, computed independently per (mouse, date) session
    -- never pooling across sessions. Vectorized via searchsorted per group.
    """
    half = window_sec / 2.0
    tt = tt.copy()
    tt["event_time_s"] = tt["photometry_side_in_index"] / FINAL_SAMPLE_FREQ_HZ
    rates = np.full(len(tt), np.nan)

    for (_, _), g in tt.groupby(["mouse", "date"]):
        idx = g.index.to_numpy()
        order = np.argsort(g["event_time_s"].to_numpy())
        times_sorted = g["event_time_s"].to_numpy()[order]
        rewarded_sorted = g["was_rewarded"].to_numpy()[order].astype(float)
        cum = np.concatenate([[0.0], np.cumsum(rewarded_sorted)])
        lo = np.searchsorted(times_sorted, times_sorted - half, side="left")
        hi = np.searchsorted(times_sorted, times_sorted + half, side="right")
        counts_sorted = cum[hi] - cum[lo]
        counts = np.empty_like(counts_sorted)
        counts[order] = counts_sorted
        rates[idx] = counts

    tt["local_reward_rate"] = rates
    return tt


def compare_with_without(tt):
    rows = []
    for mouse, g in tt.groupby("mouse"):
        m_without = smf.ols("post_amp ~ RPE_signed", data=g).fit()
        m_with = smf.ols("post_amp ~ RPE_signed + local_reward_rate", data=g).fit()
        rows.append(dict(
            mouse=mouse, n_trials=len(g),
            beta_rpe_without=m_without.params["RPE_signed"], p_rpe_without=m_without.pvalues["RPE_signed"],
            r2_without=m_without.rsquared,
            beta_rpe_with=m_with.params["RPE_signed"], p_rpe_with=m_with.pvalues["RPE_signed"],
            beta_reward_rate=m_with.params["local_reward_rate"], p_reward_rate=m_with.pvalues["local_reward_rate"],
            r2_with=m_with.rsquared,
        ))
    df = pd.DataFrame(rows).set_index("mouse")
    df["pct_change_beta_rpe"] = 100 * (df["beta_rpe_with"] - df["beta_rpe_without"]) / df["beta_rpe_without"].abs()
    return df


def main(data_dir=DATA_DIR):
    trial_table, zscore_windows, peth_time, fir, fir_meta = load_pooled_data(data_dir)
    tt = add_derived_columns(trial_table, zscore_windows, peth_time)
    tt = add_local_reward_rate(tt)

    valid = tt["local_reward_rate"].notna()
    print(f"{valid.sum()}/{len(tt)} trials with a valid local_reward_rate")
    tt = tt.loc[valid].reset_index(drop=True)

    df = compare_with_without(tt)
    pd.set_option("display.width", 160)
    print("\n=== beta_RPE with vs without local_reward_rate nuisance term ===")
    print(df.to_string())

    stat_w, p_w = wilcoxon(df["beta_rpe_without"])
    stat_wi, p_wi = wilcoxon(df["beta_rpe_with"])
    stat_rr, p_rr = wilcoxon(df["beta_reward_rate"])
    print(f"\nWilcoxon beta_RPE (without reward_rate term): W={stat_w:.3f}, p={p_w:.4f}, "
          f"median={df['beta_rpe_without'].median():.4f}, {(df['beta_rpe_without']<0).sum()}/{len(df)} negative")
    print(f"Wilcoxon beta_RPE (with reward_rate term):    W={stat_wi:.3f}, p={p_wi:.4f}, "
          f"median={df['beta_rpe_with'].median():.4f}, {(df['beta_rpe_with']<0).sum()}/{len(df)} negative")
    print(f"Wilcoxon beta_reward_rate (nuisance term itself): W={stat_rr:.3f}, p={p_rr:.4f}, "
          f"median={df['beta_reward_rate'].median():.6f}, {(df['beta_reward_rate']<0).sum()}/{len(df)} negative")
    print(f"\nMedian %% change in beta_RPE after adding reward_rate term: {df['pct_change_beta_rpe'].median():.1f}%")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR)
