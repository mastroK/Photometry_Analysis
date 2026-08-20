"""
Same nuisance-regressor test as nuisance_reward_rate_test.py, but for SM
green_l broken out by age_bin (P70..P170) rather than by mouse alone --
SM mice are recorded across many age bins over time, so a single per-mouse
fit (as used for FP1/FP2) would average over the exact age-dependent
variation we want to see. age_bin isn't in the pooled trial table, so it's
joined in from outputs_fixed/sm_corrected_channel_report.csv on (mouse, date).

Usage:
    python nuisance_reward_rate_test_sm_by_age.py
"""

from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import wilcoxon

from nuisance_reward_rate_test import add_local_reward_rate
from rpe_analysis_stats import add_derived_columns, load_pooled_data

DATA_DIR = Path("outputs_fixed/rpe_analysis_sm")
CHANNEL_REPORT = Path("outputs_fixed/sm_corrected_channel_report.csv")
MIN_TRIALS = 50


def main():
    trial_table, zscore_windows, peth_time, fir, fir_meta = load_pooled_data(DATA_DIR)
    tt = trial_table[trial_table["hemisphere"] == "green_l"].reset_index(drop=True)
    zscore_windows = zscore_windows[trial_table["hemisphere"].to_numpy() == "green_l"]

    tt = add_derived_columns(tt, zscore_windows, peth_time)
    tt = add_local_reward_rate(tt)
    tt = tt.dropna(subset=["local_reward_rate"]).reset_index(drop=True)

    report = pd.read_csv(CHANNEL_REPORT, dtype={"date": str})[["mouse", "date", "age_bin"]].drop_duplicates()
    tt["date"] = tt["date"].astype(str)
    tt = tt.merge(report, on=["mouse", "date"], how="left")
    missing_age = tt["age_bin"].isna().sum()
    if missing_age:
        print(f"WARNING: {missing_age}/{len(tt)} trials had no age_bin match, dropping")
        tt = tt.dropna(subset=["age_bin"]).reset_index(drop=True)

    rows = []
    for (mouse, age_bin), g in tt.groupby(["mouse", "age_bin"]):
        if len(g) < MIN_TRIALS:
            continue
        m_without = smf.ols("post_amp ~ RPE_signed", data=g).fit()
        m_with = smf.ols("post_amp ~ RPE_signed + local_reward_rate", data=g).fit()
        rows.append(dict(
            mouse=mouse, age_bin=age_bin, n_trials=len(g),
            beta_rpe_without=m_without.params["RPE_signed"], p_rpe_without=m_without.pvalues["RPE_signed"],
            beta_rpe_with=m_with.params["RPE_signed"], p_rpe_with=m_with.pvalues["RPE_signed"],
            beta_reward_rate=m_with.params["local_reward_rate"], p_reward_rate=m_with.pvalues["local_reward_rate"],
        ))
    df = pd.DataFrame(rows)
    df["pct_change_beta_rpe"] = 100 * (df["beta_rpe_with"] - df["beta_rpe_without"]) / df["beta_rpe_without"].abs()

    pd.set_option("display.width", 160)
    print(f"\n{len(df)} (mouse, age_bin) groups with >= {MIN_TRIALS} trials\n")
    print(df.sort_values(["age_bin", "mouse"]).to_string(index=False))

    print("\n=== Per age_bin summary (reward_rate coefficient) ===")
    summary = df.groupby("age_bin").agg(
        n_groups=("beta_reward_rate", "size"),
        mean_beta_reward_rate=("beta_reward_rate", "mean"),
        frac_significant=("p_reward_rate", lambda p: (p < 0.05).mean()),
        frac_positive=("beta_reward_rate", lambda b: (b > 0).mean()),
        median_pct_change_beta_rpe=("pct_change_beta_rpe", "median"),
    )
    print(summary.to_string())

    stat, p = wilcoxon(df["beta_rpe_without"])
    print(f"\nOverall beta_RPE (without term), all {len(df)} (mouse,age_bin) groups pooled: "
          f"Wilcoxon W={stat:.3f}, p={p:.4f}, median={df['beta_rpe_without'].median():.4f}, "
          f"{(df['beta_rpe_without']<0).sum()}/{len(df)} negative")
    stat_rr, p_rr = wilcoxon(df["beta_reward_rate"])
    print(f"Overall beta_reward_rate, all groups pooled: Wilcoxon W={stat_rr:.3f}, p={p_rr:.4f}, "
          f"median={df['beta_reward_rate'].median():.6f}, {(df['beta_reward_rate']>0).sum()}/{len(df)} positive")


if __name__ == "__main__":
    main()
