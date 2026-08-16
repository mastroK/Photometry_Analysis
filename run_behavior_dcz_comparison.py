"""
Behavioral (not photometry) comparison: does DCZ change task performance or
the fitted sticky Q-learning parameters, independent of the neural signal?

Refits the sticky Q-learning model (external.bandit_state_model) fresh per
session directly from each condition's already-extracted pooled_trial_table
(chose_right/was_rewarded) -- no raw photometry or behavior file access
needed, since the model only depends on the trial-level choice/reward
sequence already sitting in outputs_fixed/rpe_analysis_<cohort>_<condition>/
pooled_trial_table.parquet.

Usage:
    python run_behavior_dcz_comparison.py
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from config.params import BANDIT_MIN_TRIALS, BANDIT_N_STARTS
from external.bandit_state_model import multistart_fit

FP1_MICE = ["WCL23", "WCL24", "WCL27", "WCL28", "WCL29"]
FP2_MICE = ["WCL30", "WCL31", "WCL32", "WCL33"]
OUT_DIR = "outputs_fixed/behavior_dcz_comparison"


def fit_sessions(tt, min_trials=BANDIT_MIN_TRIALS, n_starts=BANDIT_N_STARTS):
    """Refit sticky-Q per (mouse, date), return one row per session."""
    rows = []
    for (mouse, date), g in tt.groupby(["mouse", "date"]):
        g = g.sort_values("side_in_poke_index") if "side_in_poke_index" in g.columns else g
        choices = g["chose_right"].dropna().to_numpy().astype(int)
        rewards = g["was_rewarded"].dropna().to_numpy().astype(int)
        n_trials = len(choices)
        row = dict(mouse=mouse, date=date, n_trials=n_trials)
        if n_trials < min_trials:
            row.update(alpha=np.nan, beta=np.nan, kappa=np.nan, nll=np.nan)
        else:
            best = multistart_fit(choices, rewards, n_starts)
            row.update(alpha=best.x[0], beta=best.x[1], kappa=best.x[2], nll=best.fun)
        row["reward_rate"] = np.nanmean(rewards) if n_trials else np.nan
        row["switch_rate"] = g["switched"].mean() if "switched" in g.columns else np.nan
        if "Choice_Deviation" in g.columns:
            row["choice_deviation"] = pd.to_numeric(g["Choice_Deviation"], errors="coerce").mean()
        rows.append(row)
    return pd.DataFrame(rows)


def win_stay_lose_switch(tt):
    """Per-session win-stay / lose-switch probabilities."""
    rows = []
    for (mouse, date), g in tt.groupby(["mouse", "date"]):
        g = g.reset_index(drop=True)
        rewarded = g["was_rewarded"].to_numpy()
        switched = g["switched"].to_numpy()
        # trial t's switched refers to whether trial t differs from t-1;
        # win-stay/lose-switch is conditioned on trial t-1's outcome.
        prev_rewarded = np.concatenate([[np.nan], rewarded[:-1].astype(float)])
        eligible_stay = prev_rewarded == 1
        eligible_switch = prev_rewarded == 0
        win_stay = np.nan
        lose_switch = np.nan
        if eligible_stay.sum() > 0:
            win_stay = 1.0 - np.nanmean(switched[eligible_stay].astype(float))
        if eligible_switch.sum() > 0:
            lose_switch = np.nanmean(switched[eligible_switch].astype(float))
        rows.append(dict(mouse=mouse, date=date, win_stay_prob=win_stay, lose_switch_prob=lose_switch))
    return pd.DataFrame(rows)


def per_mouse_summary(cohort_lower, condition, mice):
    tt = pd.read_parquet(f"outputs_fixed/rpe_analysis_{cohort_lower}_{condition}/pooled_trial_table.parquet")
    fits = fit_sessions(tt)
    wsls = win_stay_lose_switch(tt)
    merged = fits.merge(wsls, on=["mouse", "date"])
    per_mouse = merged.groupby("mouse").agg(
        n_sessions=("date", "count"),
        n_trials=("n_trials", "sum"),
        alpha=("alpha", "mean"),
        beta=("beta", "mean"),
        kappa=("kappa", "mean"),
        reward_rate=("reward_rate", "mean"),
        switch_rate=("switch_rate", "mean"),
        choice_deviation=("choice_deviation", "mean"),
        win_stay_prob=("win_stay_prob", "mean"),
        lose_switch_prob=("lose_switch_prob", "mean"),
    )
    return per_mouse.loc[mice], merged


def paired_compare(cohort_label, mice, dcz_summary, sal_summary):
    metrics = ["alpha", "beta", "kappa", "reward_rate", "switch_rate", "choice_deviation", "win_stay_prob", "lose_switch_prob"]
    print(f"\n=== {cohort_label} (n={len(mice)}) behavior: DCZ vs saline, paired ===")
    rows = []
    for metric in metrics:
        dcz_vals = dcz_summary.loc[mice, metric]
        sal_vals = sal_summary.loc[mice, metric]
        valid = dcz_vals.notna() & sal_vals.notna()
        dcz_v, sal_v = dcz_vals[valid], sal_vals[valid]
        if len(dcz_v) < 2:
            continue
        diff = dcz_v - sal_v
        stat, p = wilcoxon(dcz_v, sal_v)
        n_pos = int((diff > 0).sum())
        print(f"  {metric}: DCZ_median={dcz_v.median():+.4f} saline_median={sal_v.median():+.4f} "
              f"n_increased={n_pos}/{len(dcz_v)} p={p:.4f}")
        for m in dcz_v.index:
            print(f"      {m}: DCZ={dcz_summary.loc[m,metric]:+.4f} saline={sal_summary.loc[m,metric]:+.4f} "
                  f"diff={dcz_summary.loc[m,metric]-sal_summary.loc[m,metric]:+.4f}")
        rows.append(dict(metric=metric, dcz_median=dcz_v.median(), saline_median=sal_v.median(),
                          n_increased=n_pos, n=len(dcz_v), wilcoxon_p=p))
    return pd.DataFrame(rows)


def main():
    import pathlib
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    all_comparisons = []
    for cohort_label, cohort_lower, mice in [("FP1", "fp1", FP1_MICE), ("FP2", "fp2", FP2_MICE)]:
        dcz_summary, dcz_sessions = per_mouse_summary(cohort_lower, "dcz", mice)
        sal_summary, sal_sessions = per_mouse_summary(cohort_lower, "saline", mice)
        dcz_summary.to_csv(f"{OUT_DIR}/{cohort_lower}_dcz_per_mouse.csv")
        sal_summary.to_csv(f"{OUT_DIR}/{cohort_lower}_saline_per_mouse.csv")
        dcz_sessions.to_csv(f"{OUT_DIR}/{cohort_lower}_dcz_per_session.csv", index=False)
        sal_sessions.to_csv(f"{OUT_DIR}/{cohort_lower}_saline_per_session.csv", index=False)

        print(f"\n--- {cohort_label} DCZ per-mouse ---")
        print(dcz_summary.to_string())
        print(f"\n--- {cohort_label} saline per-mouse ---")
        print(sal_summary.to_string())

        comparison = paired_compare(cohort_label, mice, dcz_summary, sal_summary)
        comparison["cohort"] = cohort_label
        comparison.to_csv(f"{OUT_DIR}/{cohort_lower}_behavior_comparison.csv", index=False)
        all_comparisons.append(comparison)

    pd.concat(all_comparisons, ignore_index=True).to_csv(f"{OUT_DIR}/all_behavior_comparison.csv", index=False)
    print(f"\nSaved all results to {OUT_DIR}/")


if __name__ == "__main__":
    main()
