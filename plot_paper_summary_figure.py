"""
Combined summary figure for the paper: behavioral (kappa, lose-switch-prob)
young-vs-old panels alongside the neural (word_l2 LR/RL/RR) young-vs-old
trace panels, with cluster-permutation p-values (cluster_permutation_word_l2.py)
annotated directly on the neural panels.

Behavioral panels reuse plot_bandit_behavior_comparison.load_clean's SM
young/old split (age_numeric_days < SM_AGE_CUTOFF_DAYS) and per-mouse means;
lose-switch/win-stay are recomputed from the cached green_l pooled trial
table (behavior columns are channel-independent -- any hemisphere's pooled
table gives the same chose_right/was_rewarded/switched values for a given
session).

Usage:
    python plot_paper_summary_figure.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from cluster_permutation_word_l2 import (
    CLEAN_TERMS,
    PAIRED_MICE,
    cluster_permutation_test,
    paired_diffs,
)
from plot_bandit_behavior_comparison import SM_AGE_CUTOFF_DAYS, load_clean
from models.glm_encoding import fit_time_resolved_glm

GREEN_L_ROOT = Path("outputs_fixed/model_series_comparison_sm_age_split")
OUT_DIR = Path("figures_fixed_bandit_behavior_comparison")
NEURAL_TERMS_TO_SHOW = ["LR", "RL", "RR"]
AGE_COLORS = {"young": "#55A868", "old": "#8172B2"}


def load_pooled(root, label):
    tt = pd.read_parquet(root / label / "results" / "pooled_trial_table_in.parquet")
    npz = np.load(root / label / "results" / "pooled_zscore_windows.npz")
    return tt, npz["zscore_in"], npz["peth_time_in"]


def behavioral_per_mouse(param):
    df = load_clean()
    sm_df = df[df["cohort"] == "SM"].copy()
    sm_df["age_group"] = np.where(sm_df["age_numeric_days"] < SM_AGE_CUTOFF_DAYS, "young", "old")
    return sm_df.groupby(["mouse", "age_group"])[param].mean().reset_index()


def lose_switch_per_mouse():
    rows = []
    for label in ("young", "old"):
        tt, _, _ = load_pooled(GREEN_L_ROOT, label)
        for mouse, sub in tt.groupby("mouse"):
            prev_reward = sub["1_Reward"].astype("float")
            switched = sub["switched"].astype("float")
            has_prev = prev_reward.notna() & switched.notna()
            pr, sw = prev_reward[has_prev], switched[has_prev]
            lose_switch_prob = sw[pr == 0].mean()
            rows.append(dict(age_group=label, mouse=mouse, lose_switch_prob=lose_switch_prob))
    return pd.DataFrame(rows)


def plot_paired(ax, per_mouse_df, value_col, title):
    wide = per_mouse_df.pivot(index="mouse", columns="age_group", values=value_col)
    for mouse, row in wide.iterrows():
        xs, ys = [], []
        if pd.notna(row.get("young")):
            xs.append(0); ys.append(row["young"])
        if pd.notna(row.get("old")):
            xs.append(1); ys.append(row["old"])
        ax.plot(xs, ys, "-o", alpha=0.8, label=mouse)
    paired = wide[["young", "old"]].dropna() if {"young", "old"}.issubset(wide.columns) else pd.DataFrame()
    if len(paired) >= 2:
        stat, p = stats.wilcoxon(paired["young"], paired["old"])
        title += f"\npaired Wilcoxon n={len(paired)} mice, p={p:.3f}"
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["young", "old"])
    ax.set_title(title, fontsize=10)


def main():
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])

    ax_kappa = fig.add_subplot(gs[0, 0])
    plot_paired(ax_kappa, behavioral_per_mouse("kappa"), "kappa", r"Behavior: $\kappa$ (stickiness)")
    ax_kappa.set_ylabel(r"$\kappa$")

    ax_ls = fig.add_subplot(gs[0, 1])
    plot_paired(ax_ls, lose_switch_per_mouse(), "lose_switch_prob", "Behavior: lose-switch probability")
    ax_ls.set_ylabel("P(switch | previous trial unrewarded)")
    ax_ls.legend(fontsize=7, title="mouse", loc="upper left", bbox_to_anchor=(1.02, 1))

    ax_note = fig.add_subplot(gs[0, 2])
    ax_note.axis("off")
    ax_note.text(0, 0.9, "This photometry subset (n=4-5 mice) is\nunderpowered on its own -- the full\n"
                 "behavioral cohort carries the significance\nclaim for both effects (see main text).",
                 fontsize=10, va="top", wrap=True)

    tt_young, zs_young, peth_time = load_pooled(GREEN_L_ROOT, "young")
    tt_old, zs_old, peth_time_old = load_pooled(GREEN_L_ROOT, "old")
    fit_young = fit_time_resolved_glm(zs_young, peth_time, tt_young, formula="Z ~ C(word_l2)", min_retained_frac=0.5)
    fit_old = fit_time_resolved_glm(zs_old, peth_time_old, tt_old, formula="Z ~ C(word_l2)", min_retained_frac=0.5)

    diffs, peth_time_pm = paired_diffs(GREEN_L_ROOT, terms=NEURAL_TERMS_TO_SHOW, mice=PAIRED_MICE)

    for i, term in enumerate(NEURAL_TERMS_TO_SHOW):
        ax = fig.add_subplot(gs[1, i])
        col, se_col = f"C(word_l2)[T.{term}]_beta", f"C(word_l2)[T.{term}]_se"
        for label, fit in (("young", fit_young), ("old", fit_old)):
            x = fit.index.to_numpy()
            y, se = fit[col], fit[se_col]
            ax.plot(x, y, color=AGE_COLORS[label], label=label, linewidth=1.5)
            ax.fill_between(x, y - se, y + se, color=AGE_COLORS[label], alpha=0.2)
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.axvline(0, color="black", linestyle=":", linewidth=0.8)

        mass, p, t = cluster_permutation_test(diffs[term])
        sig_note = f"cluster p={p:.3f} (n=4 mice, min achievable=0.0625)"
        ax.set_title(f"Neural: C(word_l2)[T.{term}]\n{sig_note}", fontsize=9)
        if i == 0:
            ax.legend(fontsize=8)
            ax.set_ylabel("beta (pooled fit)")
        ax.set_xlabel("time from side_in (s)")

    fig.suptitle("SM aging: behavior (kappa decline, increased lose-switch) alongside candidate neural\n"
                 "correlate (stronger previous-trial word_l2 encoding in old mice, green_l channel)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "paper_summary_figure.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'paper_summary_figure.png'}")

    kappa_df = behavioral_per_mouse("kappa")
    ls_df = lose_switch_per_mouse()
    kappa_df.to_csv(OUT_DIR / "sm_kappa_per_mouse_young_old.csv", index=False)
    ls_df.to_csv(OUT_DIR / "sm_lose_switch_per_mouse_young_old.csv", index=False)
    print(f"Wrote {OUT_DIR / 'sm_kappa_per_mouse_young_old.csv'} and "
          f"{OUT_DIR / 'sm_lose_switch_per_mouse_young_old.csv'}")


if __name__ == "__main__":
    main()
