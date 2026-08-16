"""
FP1 (young, 51-80 days at recording) vs FP2 (old, 147-167 days) per-mouse
comparison of the within-animal RPE metrics, per the plan agreed with the
user: age is perfectly confounded with cohort in this dataset (every FP1
mouse is young, every FP2 mouse is old, no exceptions), so this is reported
as a cohort difference consistent with an age effect, not a proven age
effect. Uses only the per-mouse result tables already computed for the
FP1/FP2/pooled RPE reports plus the pooled trial table -- no raw data
reprocessing.

Usage:
    python run_age_comparison.py
"""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

RESULTS_DIR = "outputs_fixed/rpe_analysis_pooled/results"
OUT_DIR = "outputs_fixed/age_comparison"
FP1_MICE = ["WCL23", "WCL24", "WCL25", "WCL26", "WCL27", "WCL28", "WCL29"]
FP2_MICE = ["WCL30", "WCL31", "WCL32", "WCL33"]
EXCLUDE_LOW_FIT = ["WCL25", "WCL26"]
N_BOOTSTRAP = 10000
RANDOM_STATE = np.random.RandomState(0)


def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    gt = sum((xi > y).sum() for xi in x)
    lt = sum((xi < y).sum() for xi in x)
    return (gt - lt) / (len(x) * len(y))


def bootstrap_median_diff_ci(x, y, n_boot=N_BOOTSTRAP, alpha=0.05):
    x, y = np.asarray(x), np.asarray(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        bx = RANDOM_STATE.choice(x, size=len(x), replace=True)
        by = RANDOM_STATE.choice(y, size=len(y), replace=True)
        diffs[i] = np.median(bx) - np.median(by)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def compare_metric(df, col, label, exclude=None):
    d = df if exclude is None else df[~df.index.isin(exclude)]
    fp1_vals = d.loc[d.index.isin(FP1_MICE), col].dropna().to_numpy()
    fp2_vals = d.loc[d.index.isin(FP2_MICE), col].dropna().to_numpy()
    if len(fp1_vals) < 2 or len(fp2_vals) < 2:
        return dict(metric=label, n_fp1=len(fp1_vals), n_fp2=len(fp2_vals),
                    median_fp1=np.nan, median_fp2=np.nan, median_diff=np.nan,
                    ci_lo=np.nan, ci_hi=np.nan, cliffs_delta=np.nan,
                    mwu_stat=np.nan, mwu_p=np.nan, excluded=exclude is not None)
    stat, p = mannwhitneyu(fp1_vals, fp2_vals, alternative="two-sided")
    delta = cliffs_delta(fp1_vals, fp2_vals)
    ci_lo, ci_hi = bootstrap_median_diff_ci(fp1_vals, fp2_vals)
    return dict(
        metric=label, n_fp1=len(fp1_vals), n_fp2=len(fp2_vals),
        median_fp1=float(np.median(fp1_vals)), median_fp2=float(np.median(fp2_vals)),
        median_diff=float(np.median(fp1_vals) - np.median(fp2_vals)),
        ci_lo=ci_lo, ci_hi=ci_hi, cliffs_delta=float(delta),
        mwu_stat=float(stat), mwu_p=float(p), excluded=exclude is not None,
    )


def main():
    r1 = pd.read_csv(f"{RESULTS_DIR}/analysis1_rpe_regression.csv", index_col=0)
    r2 = pd.read_csv(f"{RESULTS_DIR}/analysis2_signed_vs_unsigned.csv", index_col=0)
    r3a = pd.read_csv(f"{RESULTS_DIR}/analysis3a_encoding_glm_per_mouse.csv", index_col=0)
    r3b = pd.read_csv(f"{RESULTS_DIR}/analysis3b_fir_glm_per_mouse.csv", index_col=0)
    r5_reward = pd.read_csv(f"{RESULTS_DIR}/analysis5_kinetics_reward_per_mouse.csv", index_col=0)
    r5_omission = pd.read_csv(f"{RESULTS_DIR}/analysis5_kinetics_omission_per_mouse.csv", index_col=0)

    metric_specs = [
        (r1, "beta_rpe", "beta_RPE (post_amp ~ RPE_signed)"),
        (r1, "interaction_coef", "outcome x Q(chosen) interaction coefficient"),
        (r3a, "peak_r2", "Encoding-GLM peak R2"),
        (r3b, "r2_mean", "FIR out-of-sample R2"),
        (r2, "delta_r2", "Signed vs unsigned RPE delta R2"),
        # Onset/decay kinetics (rpe_analysis_kinetics.py, alignment/kinetics.py).
        # Fit from trial-averaged PETHs, so some mice have no fittable decay
        # (see that module + config.params.KINETICS_METRIC_WINDOW_S for why) --
        # compare_metric already drops NaNs and requires >=2 per group, so this
        # comparison just runs on whichever mice DID get a successful fit.
        (r5_reward, "onset_latency_s", "Reward onset latency (s)"),
        (r5_reward, "decay_tau_s", "Reward decay tau (s)"),
        (r5_omission, "onset_latency_s", "Omission onset latency (s)"),
        (r5_omission, "decay_tau_s", "Omission decay tau (s)"),
    ]

    rows = []
    for df, col, label in metric_specs:
        rows.append(compare_metric(df, col, label, exclude=None))
        rows.append(compare_metric(df, col, label, exclude=EXCLUDE_LOW_FIT))

    comparison = pd.DataFrame(rows)
    comparison.to_csv(f"{OUT_DIR}/cohort_comparison.csv", index=False)
    print("=== FP1 (young, n=7) vs FP2 (old, n=4) per-mouse comparison ===\n")
    print(comparison.to_string(index=False))

    # --- confound audit ---
    tt = pd.read_parquet("outputs_fixed/rpe_analysis_pooled/pooled_trial_table.parquet")
    hemi = pd.read_csv("config/session_hemisphere_overrides.csv")
    hemi["cohort"] = hemi["Mouse ID"].apply(lambda m: "FP1" if m in FP1_MICE else "FP2")

    trial_counts = tt.groupby("mouse").size().rename("n_trials")
    session_counts = tt.groupby("mouse")["date"].nunique().rename("n_sessions")
    task_perf = tt.groupby("mouse").agg(
        reward_rate=("was_rewarded", "mean"),
        switch_rate=("switched", "mean"),
        rolling_accuracy=("Rolling_Accuracy", "mean"),
    )
    confound = pd.concat([trial_counts, session_counts, task_perf], axis=1)
    confound["cohort"] = confound.index.map(lambda m: "FP1" if m in FP1_MICE else "FP2")
    confound.to_csv(f"{OUT_DIR}/confound_audit_per_mouse.csv")

    print("\n=== Confound audit: trial/session counts + task performance ===\n")
    print(confound.to_string())

    print("\n=== Hemisphere/channel composition by cohort ===\n")
    print(hemi.groupby(["cohort", "Hemisphere"]).size().unstack(fill_value=0).to_string())

    for group_label, mice in [("FP1", FP1_MICE), ("FP2", FP2_MICE)]:
        gc = confound.loc[confound.index.isin(mice), ["n_trials", "n_sessions", "reward_rate", "switch_rate", "rolling_accuracy"]]
        stat, p = mannwhitneyu(
            confound.loc[confound.index.isin(FP1_MICE), "n_trials"],
            confound.loc[confound.index.isin(FP2_MICE), "n_trials"],
            alternative="two-sided",
        )
    print(f"\nTrial-count MWU (FP1 vs FP2): stat={stat:.1f}, p={p:.4f}")

    print(f"\nSaved comparison to {OUT_DIR}/cohort_comparison.csv")
    print(f"Saved confound audit to {OUT_DIR}/confound_audit_per_mouse.csv")


if __name__ == "__main__":
    main()
