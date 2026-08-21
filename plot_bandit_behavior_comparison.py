"""
Compare the fitted sticky Q-learning behavioral parameters (alpha, beta,
kappa -- see external/bandit_state_model.py, produced by
run_bandit_behavior_comparison.py) across reward-probability condition
(FP1/FP2 = 90/10, SM = 80/20) and condition x age.

Reads outputs_fixed/bandit_behavior_comparison/bandit_params_pooled.csv.

Age is reported on two different bases and never coerced onto one scale
without saying so: FP1/FP2 have exact age_days at recording (DOB on file);
SM only has the RA-assigned P70..P170 age_bin category (no DOB on file), so
SM's "age" in the combined age figure is the bin's nominal day number
(P70 -> 70), an approximation, not each mouse's exact age.

4 SM sessions are 100%-forced-choice training days (reward_probs_observed
only {0.0, 1.0} -- frac_forced_trials == 1.0), not real 80/20-bandit
sessions -- excluded from every comparison here (see EXCLUDE_FORCED).

Usage:
    python plot_bandit_behavior_comparison.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from external.bandit_state_model import boundary_flags

IN_CSV = Path("outputs_fixed/bandit_behavior_comparison/bandit_params_pooled.csv")
OUT_DIR = Path("figures_fixed_bandit_behavior_comparison")

COHORT_ORDER = ["FP1_none", "FP1_retrained_none", "FP1_dcz", "FP2_none", "FP2_dcz", "SM"]
PARAMS = ["alpha", "beta", "kappa"]
PARAM_LABELS = {
    "alpha": r"$\alpha$ (learning rate)",
    "beta": r"$\beta$ (inverse temperature)",
    "kappa": r"$\kappa$ (stickiness)",
}
CONDITION_PALETTE = {"90/10": "#4C72B0", "80/20": "#DD8452"}


def load_clean():
    df = pd.read_csv(IN_CSV)
    n_forced = int((df["frac_forced_trials"] >= 1.0).sum())
    df = df[df["frac_forced_trials"] < 1.0].copy()
    print(f"Excluded {n_forced} fully-forced-choice session(s) (not real bandit trials)")

    df["age_numeric_days"] = df["age_days"]
    is_sm = df["age_bin"].notna()
    df.loc[is_sm, "age_numeric_days"] = df.loc[is_sm, "age_bin"].str.lstrip("P").astype(float)
    df["age_is_exact"] = ~is_sm

    flags = df.apply(lambda row: boundary_flags([row["alpha"], row["beta"], row["kappa"]]), axis=1)
    flags_df = pd.DataFrame(list(flags), index=df.index)
    for param in PARAMS:
        df[f"{param}_at_bound"] = flags_df[f"{param}_at_bound"]
    return df


def plot_by_condition(df, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, param in zip(axes, PARAMS):
        sns.boxplot(data=df, x="cohort", y=param, order=COHORT_ORDER, hue="condition",
                    palette=CONDITION_PALETTE, dodge=False, showfliers=False, ax=ax)
        sns.stripplot(data=df, x="cohort", y=param, order=COHORT_ORDER, color="black",
                      size=3, alpha=0.5, jitter=True, ax=ax)
        pinned = df[df[f"{param}_at_bound"]]
        if len(pinned):
            sns.stripplot(data=pinned, x="cohort", y=param, order=COHORT_ORDER, color="red",
                          marker="x", size=6, jitter=True, ax=ax, label="_nolegend_")
        if param == "beta":
            ax.set_yscale("log")
        ax.set_title(PARAM_LABELS[param])
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        if ax is not axes[0]:
            ax.get_legend().remove()
    axes[0].legend(title="condition", loc="upper left")
    fig.suptitle("Sticky Q-learning behavioral parameters by cohort / reward condition\n"
                 "(session-level fits; box = FP1/FP2 90/10 vs SM 80/20; red x = boundary-pinned fit, "
                 f"BETA_BOUNDS upper={49.9})")
    fig.tight_layout()
    fig.savefig(out_dir / "params_by_condition.png", dpi=150)
    plt.close(fig)


def plot_by_age(df, out_dir):
    plot_df = df[df["cohort"].isin(["FP1_none", "FP2_none", "SM"])]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, param in zip(axes, PARAMS):
        log_scale = param == "beta"
        for condition, sub in plot_df.groupby("condition"):
            pinned_mask = sub[f"{param}_at_bound"]
            ax.scatter(sub.loc[~pinned_mask, "age_numeric_days"], sub.loc[~pinned_mask, param], label=condition,
                       color=CONDITION_PALETTE[condition], alpha=0.6, s=25,
                       marker="o" if sub["age_is_exact"].iloc[0] else "^")
            if pinned_mask.any():
                ax.scatter(sub.loc[pinned_mask, "age_numeric_days"], sub.loc[pinned_mask, param],
                           color="red", marker="x", s=40, label="_nolegend_")
            if sub["mouse"].nunique() >= 2:
                y = np.log10(sub[param]) if log_scale else sub[param]
                slope, intercept, r, p, _ = stats.linregress(sub["age_numeric_days"], y)
                rho, p_spear = stats.spearmanr(sub["age_numeric_days"], sub[param])
                xs = np.linspace(sub["age_numeric_days"].min(), sub["age_numeric_days"].max(), 50)
                fit_ys = slope * xs + intercept
                ax.plot(xs, 10 ** fit_ys if log_scale else fit_ys, color=CONDITION_PALETTE[condition],
                        linestyle="--", label=f"{condition} Spearman rho={rho:.2f}, p={p_spear:.3f}")
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(PARAM_LABELS[param])
        ax.set_xlabel("age (days; exact for FP, P-bin nominal for SM)")
        ax.legend(fontsize=7)
    fig.suptitle("Behavioral parameters vs age, by condition (none/undrugged sessions only)\n"
                 "circle = FP exact age_days, triangle = SM nominal age_bin")
    fig.tight_layout()
    fig.savefig(out_dir / "params_by_age_condition.png", dpi=150)
    plt.close(fig)


def plot_dcz_effect(df, out_dir):
    dcz_df = df[df["cohort"].isin(["FP1_none", "FP1_dcz", "FP2_none", "FP2_dcz"])].copy()
    dcz_df["family"] = dcz_df["cohort_family"]
    dcz_df["drug"] = np.where(dcz_df["cohort"].str.endswith("_dcz"), "DCZ", "none")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, param in zip(axes, PARAMS):
        sns.boxplot(data=dcz_df, x="family", y=param, hue="drug", showfliers=False, ax=ax)
        sns.stripplot(data=dcz_df, x="family", y=param, hue="drug", dodge=True, color="black",
                      size=3, alpha=0.5, ax=ax, legend=False)
        if param == "beta":
            ax.set_yscale("log")
        ax.set_title(PARAM_LABELS[param])
        ax.set_xlabel("")
    fig.suptitle("Behavioral parameters: DCZ vs none, within FP1/FP2 (same 90/10 schedule, same mice)")
    fig.tight_layout()
    fig.savefig(out_dir / "params_dcz_effect.png", dpi=150)
    plt.close(fig)


def summarize_stats(df):
    print("\n=== Per-mouse mean parameters (all sessions, forced-training excluded) ===")
    per_mouse = df.groupby(["mouse", "cohort", "condition"])[PARAMS].mean().reset_index()
    print(per_mouse.to_string(index=False))

    print("\n=== Condition comparison (90/10 vs 80/20), 'none'-condition sessions only, per-mouse means ===")
    none_df = df[df["cohort"].isin(["FP1_none", "FP2_none", "FP1_retrained_none", "SM"])]
    per_mouse_none = none_df.groupby(["mouse", "condition"])[PARAMS].mean().reset_index()
    for param in PARAMS:
        g90 = per_mouse_none.loc[per_mouse_none.condition == "90/10", param]
        g80 = per_mouse_none.loc[per_mouse_none.condition == "80/20", param]
        u, p = stats.mannwhitneyu(g90, g80, alternative="two-sided")
        print(f"  {param}: 90/10 mice mean={g90.mean():.3f} (n={len(g90)}), "
              f"80/20 mice mean={g80.mean():.3f} (n={len(g80)}), Mann-Whitney p={p:.4f}")

    print("\n=== Age comparison at fixed 90/10 condition: FP1 (young) vs FP2 (old), 'none' sessions, per-mouse means ===")
    fp_none = df[df["cohort"].isin(["FP1_none", "FP2_none"])]
    per_mouse_fp = fp_none.groupby(["mouse", "cohort_family"])[PARAMS + ["age_numeric_days"]].mean().reset_index()
    print(per_mouse_fp.to_string(index=False))
    for param in PARAMS:
        y = per_mouse_fp.loc[per_mouse_fp.cohort_family == "FP1", param]
        o = per_mouse_fp.loc[per_mouse_fp.cohort_family == "FP2", param]
        u, p = stats.mannwhitneyu(y, o, alternative="two-sided")
        print(f"  {param}: FP1(young) mean={y.mean():.3f} (n={len(y)}), "
              f"FP2(old) mean={o.mean():.3f} (n={len(o)}), Mann-Whitney p={p:.4f}")

    print("\n=== Age trend within SM (80/20, continuous P70-P170), per-session Spearman "
          "(rank-based -- robust to the beta-boundary-pinned fits) ===")
    print("  NOTE: sessions are repeated-measures within only 5 mice -- these p-values likely "
          "overstate significance (pseudoreplication); treat as descriptive, not a clean per-animal test.")
    sm_df = df[df.cohort == "SM"]
    for param in PARAMS:
        rho, p = stats.spearmanr(sm_df["age_numeric_days"], sm_df[param])
        print(f"  {param} ~ age_bin: Spearman rho={rho:.3f}, p={p:.4f}, n_sessions={len(sm_df)}")

    print("\n=== DCZ effect within FP1/FP2 (same mice, same 90/10 schedule), per-mouse means ===")
    for family in ("FP1", "FP2"):
        fam_df = df[(df.cohort_family == family) & (df.cohort.isin([f"{family}_none", f"{family}_dcz"]))]
        per_mouse_drug = fam_df.groupby(["mouse", "cohort"])[PARAMS].mean().reset_index()
        wide = per_mouse_drug.pivot(index="mouse", columns="cohort", values=PARAMS)
        print(f"\n  {family}:")
        for param in PARAMS:
            none_col, dcz_col = f"{family}_none", f"{family}_dcz"
            paired = wide[param][[none_col, dcz_col]].dropna()
            if len(paired) < 2:
                print(f"    {param}: insufficient paired mice (n={len(paired)})")
                continue
            stat, p = stats.wilcoxon(paired[none_col], paired[dcz_col])
            print(f"    {param}: none mean={paired[none_col].mean():.3f}, DCZ mean={paired[dcz_col].mean():.3f}, "
                  f"n_mice={len(paired)}, Wilcoxon p={p:.4f}")


def main():
    df = load_clean()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_by_condition(df, OUT_DIR)
    plot_by_age(df, OUT_DIR)
    plot_dcz_effect(df, OUT_DIR)
    summarize_stats(df)
    print(f"\nWrote figures to {OUT_DIR}/")


if __name__ == "__main__":
    main()
