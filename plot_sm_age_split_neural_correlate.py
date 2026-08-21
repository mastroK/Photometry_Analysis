"""
Neural correlate of the behavioral kappa (stickiness) decline with age in
SM, using the already-pooled young/old data from run_sm_age_split_comparison.py
(outputs_fixed/model_series_comparison_sm_age_split/{young,old}/results/).

IMPORTANT CORRECTION: run_sm_age_split_comparison.py originally compared
1_main_effects (Z ~ Reward + Port + Port:Reward) against 2b_word_l1
(Z ~ C(word_l1)), on the mistaken assumption that word_l1 encodes the
PREVIOUS trial's choice+outcome -- the direct neural analogue of kappa's
"stick" term. It does not: behavior/word_encoding.py::add_word_labels
builds word_l{level} from the window slice(t-level+1, t+1), so word_l1's
window is just [t] -- THIS trial's own choice+outcome, not the previous
one. That makes 2b_word_l1 algebraically equivalent to 1_main_effects
(same information, different encoding: C(word_l1)'s 4 categories are
exactly the Port x Reward cross), which is exactly why their CV R^2 came
back bit-for-bit identical -- not a bug, a wrong choice of model.

The correct neural analogue of "does the previous trial's own identity
still shape today's activity, on top of what's happening this trial" is
the INCREMENTAL R^2 from word_l1 (window [t]) to word_l2 (window
[t-1, t]) -- i.e. what word_l2 explains beyond word_l1. This script
recomputes both directly from the cached pooled arrays (no photometry
reload needed) and compares that increment between young and old SM mice.

Usage:
    python plot_sm_age_split_neural_correlate.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from models.glm_encoding import cross_validate_window_glm

RESULTS_ROOT = Path("outputs_fixed/model_series_comparison_sm_age_split")
OUT_DIR = Path("figures_fixed_bandit_behavior_comparison")


def compute_word_increment_table():
    rows = []
    for label in ("young", "old"):
        tt = pd.read_parquet(RESULTS_ROOT / label / "results" / "pooled_trial_table_in.parquet")
        npz = np.load(RESULTS_ROOT / label / "results" / "pooled_zscore_windows.npz")
        zs, peth_time = npz["zscore_in"], npz["peth_time_in"]
        for mouse in sorted(tt["mouse"].unique()):
            idx = tt.index[tt.mouse == mouse].to_numpy()
            sub = tt.loc[idx].reset_index(drop=True)
            sub_zs = zs[idx]
            cv1 = cross_validate_window_glm(sub_zs, peth_time, sub, "C(word_l1)")
            cv2 = cross_validate_window_glm(sub_zs, peth_time, sub, "C(word_l2)")
            rows.append(dict(
                age_group=label, mouse=mouse, n_trials=cv1["n_trials"],
                r2_word_l1=cv1["r2_mean"], r2_word_l2=cv2["r2_mean"],
                increment=cv2["r2_mean"] - cv1["r2_mean"],
            ))
    return pd.DataFrame(rows)


def plot_increment(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    age_palette = {"young": "#55A868", "old": "#8172B2"}
    sns.boxplot(data=df, x="age_group", y="increment", order=["young", "old"], palette=age_palette,
                showfliers=False, ax=ax)
    sns.stripplot(data=df, x="age_group", y="increment", order=["young", "old"], hue="mouse",
                  size=8, alpha=0.8, jitter=True, ax=ax)
    ax.axhline(0, color="gray", linestyle=":")
    u, p = stats.mannwhitneyu(df.loc[df.age_group == "young", "increment"],
                               df.loc[df.age_group == "old", "increment"], alternative="two-sided")
    ax.set_title(f"Neural: word_l2 - word_l1 incremental R2\n(per-mouse, Mann-Whitney p={p:.3f}, "
                 f"n={df.age_group.eq('young').sum()} young / {df.age_group.eq('old').sum()} old mice)")
    ax.set_ylabel(r"$\Delta R^2$ (previous-trial increment)")
    ax.set_xlabel("")
    ax.legend(fontsize=7, title="mouse")

    ax = axes[1]
    for mouse in df["mouse"].unique():
        sub = df[df.mouse == mouse].set_index("age_group").reindex(["young", "old"])
        ax.plot([0, 1], sub["increment"], "-o", label=mouse, alpha=0.8)
    ax.axhline(0, color="gray", linestyle=":")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["young", "old"])
    ax.set_title("Per-mouse young -> old (lines connect same mouse)")
    ax.set_ylabel(r"$\Delta R^2$ (previous-trial increment)")

    fig.suptitle("Neural correlate check: does the PREVIOUS trial's own identity still shape today's\n"
                 "mPFC signal, beyond what today's own trial explains -- and does that shrink with age like kappa did?")
    fig.tight_layout()
    fig.savefig(out_dir / "sm_neural_word_increment_by_age.png", dpi=150)
    plt.close(fig)


def main():
    df = compute_word_increment_table()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "sm_neural_word_increment_by_age.csv"
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))
    print(f"\nyoung mean increment: {df.loc[df.age_group=='young','increment'].mean():.4f}")
    print(f"old mean increment:   {df.loc[df.age_group=='old','increment'].mean():.4f}")
    u, p = stats.mannwhitneyu(df.loc[df.age_group == "young", "increment"],
                               df.loc[df.age_group == "old", "increment"], alternative="two-sided")
    print(f"Mann-Whitney (mouse-level, n={df.age_group.eq('young').sum()} vs {df.age_group.eq('old').sum()}): p={p:.4f}")
    plot_increment(df, OUT_DIR)
    print(f"\nWrote {csv_path} and figure to {OUT_DIR}/")


if __name__ == "__main__":
    main()
