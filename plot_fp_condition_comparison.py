"""
Specificity control for the SM aging finding: FP1_none (young, 90/10) vs
FP2_none (old, 90/10) is a cross-sectional cohort comparison (DIFFERENT
mice, not the same animals tracked across age like SM) where the earlier
behavioral fit already found NO significant kappa difference (mixedlm_
bandit_behavior.py: FP1 vs FP2 kappa p=0.93). The hypothesis under test:
if the word_l2/Q_diff_bin neural pattern found in SM genuinely tracks a
real behavioral change (not a generic aging or channel artifact), then a
cohort with NO behavioral change (FP1/FP2 at 90/10) should show NO parallel
neural divergence either -- same analyses, same code, opposite prediction.

Reuses already-cached pooled data from the FP1_none/FP2_none truncated runs
(outputs_fixed/model_series_comparison_fp{1,2}_none_truncated/results/) --
no photometry reload needed, same truncate_at_side_out=True/
min_retained_frac=0.5 convention as the SM age-split analysis.

FP1/FP2 have no fixed whole-cohort channel like SM's red_l/green_l/green_r
-- hemisphere is resolved per mouse (config/mouse_hemisphere.csv) by the
existing run_model_series_comparison.py pipeline upstream of this script;
this just consumes whichever channel each mouse's own pooled rows already
reflect.

Since FP1 (7 mice) and FP2 (4 mice) are DIFFERENT animals, the cluster test
here is the UNPAIRED version (cluster_permutation_unpaired.py, exact
C(11,4)=330-permutation null), not SM's paired sign-flip test.

Usage:
    python plot_fp_condition_comparison.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from cluster_permutation_unpaired import unpaired_cluster_permutation_test
from cluster_permutation_word_l2 import CLEAN_TERMS
from models.glm_encoding import fit_time_resolved_glm
from run_model_series_comparison import _add_qdiff_bins, _qdiff_bin_edges

ROOTS = {
    "FP1_none": Path("outputs_fixed/model_series_comparison_fp1_none_truncated/results"),
    "FP2_none": Path("outputs_fixed/model_series_comparison_fp2_none_truncated/results"),
}
GROUP_COLORS = {"FP1_none": "#4C72B0", "FP2_none": "#C44E52"}  # young/old at 90/10 -- distinct from SM's green/purple
GROUP_LABELS = {"FP1_none": "FP1 (young, 90/10)", "FP2_none": "FP2 (old, 90/10)"}
BANDIT_CSV = Path("outputs_fixed/bandit_behavior_comparison/bandit_params_pooled.csv")
OUT_DIR = Path("figures_fixed_bandit_behavior_comparison")
RESULTS_DIR = Path("outputs_fixed/fp_condition_comparison")
NEURAL_TERMS_TO_SHOW = ["LR", "RL", "RR"]


def load_pooled(cohort):
    tt = pd.read_parquet(ROOTS[cohort] / "pooled_trial_table_in.parquet")
    npz = np.load(ROOTS[cohort] / "pooled_zscore_windows.npz")
    return tt, npz["zscore_in"], npz["peth_time_in"]


def fit_mouse(tt, zs, peth_time, mouse, formula):
    idx = tt.index[tt.mouse == mouse].to_numpy()
    sub_tt = tt.loc[idx].reset_index(drop=True)
    return fit_time_resolved_glm(zs[idx], peth_time, sub_tt, formula=formula, min_retained_frac=0.5)


# ---------------------------------------------------------------- behavior

def kappa_per_mouse():
    df = pd.read_csv(BANDIT_CSV)
    df = df[df["cohort"].isin(["FP1_none", "FP2_none"])]
    return df.groupby(["mouse", "cohort"])["kappa"].mean().reset_index()


def lose_switch_per_mouse():
    rows = []
    for cohort in ROOTS:
        tt, _, _ = load_pooled(cohort)
        for mouse, sub in tt.groupby("mouse"):
            prev_reward = sub["1_Reward"].astype("float")
            switched = sub["switched"].astype("float")
            has_prev = prev_reward.notna() & switched.notna()
            pr, sw = prev_reward[has_prev], switched[has_prev]
            rows.append(dict(cohort=cohort, mouse=mouse, lose_switch_prob=sw[pr == 0].mean()))
    return pd.DataFrame(rows)


def plot_unpaired(ax, per_mouse_df, value_col, title):
    for cohort in ROOTS:
        sub = per_mouse_df[per_mouse_df.cohort == cohort]
        x = 0 if cohort == "FP1_none" else 1
        ax.scatter([x] * len(sub), sub[value_col], color=GROUP_COLORS[cohort], s=60, alpha=0.8)
    a = per_mouse_df.loc[per_mouse_df.cohort == "FP1_none", value_col]
    b = per_mouse_df.loc[per_mouse_df.cohort == "FP2_none", value_col]
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["FP1 (young)", "FP2 (old)"])
    ax.set_title(f"{title}\nMann-Whitney n={len(a)} vs {len(b)} mice, p={p:.3f}", fontsize=10)


# ---------------------------------------------------------------- neural: word_l2

def word_l2_group_fits():
    fits = {}
    for cohort in ROOTS:
        tt, zs, peth_time = load_pooled(cohort)
        fits[cohort] = fit_time_resolved_glm(zs, peth_time, tt, formula="Z ~ C(word_l2)", min_retained_frac=0.5)
    return fits, peth_time


def word_l2_per_mouse_betas(terms=CLEAN_TERMS):
    per_mouse = {"FP1_none": {}, "FP2_none": {}}
    peth_time = None
    for cohort in ROOTS:
        tt, zs, pt = load_pooled(cohort)
        peth_time = pt
        for mouse in sorted(tt.mouse.unique()):
            fit = fit_mouse(tt, zs, pt, mouse, "Z ~ C(word_l2)")
            per_mouse[cohort][mouse] = {term: fit[f"C(word_l2)[T.{term}]_beta"].to_numpy() for term in terms}
    return per_mouse, peth_time


def run_word_l2_cluster_tests(per_mouse, terms=CLEAN_TERMS):
    results = {}
    for term in terms:
        beta_a = np.array([v[term] for v in per_mouse["FP1_none"].values()])
        beta_b = np.array([v[term] for v in per_mouse["FP2_none"].values()])
        mass, p, t, n_perms = unpaired_cluster_permutation_test(beta_a, beta_b)
        results[term] = dict(mass=mass, p=p, n_perms=n_perms)
        print(f"  {term}: max cluster mass={mass:.2f}  p={p:.4f}  (exact, {n_perms} permutations)")
    return results


# ---------------------------------------------------------------- neural: Q_diff_bin

def qdiff_bin_group_fits():
    fits = {}
    for cohort in ROOTS:
        tt, zs, peth_time = load_pooled(cohort)
        tt = tt.copy()
        tt["Q_diff_bin"] = np.nan
        for mouse, sub_mask in tt.groupby("mouse").groups.items():
            edges = _qdiff_bin_edges(tt.loc[sub_mask, "Q_diff"])
            tt.loc[sub_mask, "Q_diff_bin"] = _add_qdiff_bins(tt.loc[sub_mask], edges)["Q_diff_bin"]
        fits[cohort] = fit_time_resolved_glm(zs, peth_time, tt, formula="Z ~ C(Q_diff_bin)", min_retained_frac=0.5)
    return fits, peth_time


def plot_qdiff_bins(fits, peth_time, out_dir):
    bins = ["0", "1", "2", "3", "4"]
    bin_labels = {"0": "extreme (bin 0)\n(= Intercept)", "1": "bin 1", "2": "middle (bin 2)",
                  "3": "bin 3", "4": "extreme (bin 4)"}
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharey=True)
    for ax, b in zip(axes, bins):
        col = f"C(Q_diff_bin)[T.{b}]_beta" if b != "0" else "Intercept_beta"
        se_col = f"C(Q_diff_bin)[T.{b}]_se" if b != "0" else "Intercept_se"
        for cohort, fit in fits.items():
            x = fit.index.to_numpy()
            y, se = fit[col], fit[se_col]
            ax.plot(x, y, color=GROUP_COLORS[cohort], label=GROUP_LABELS[cohort], linewidth=1.5)
            ax.fill_between(x, y - se, y + se, color=GROUP_COLORS[cohort], alpha=0.2)
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.axvline(0, color="black", linestyle=":", linewidth=0.8)
        ax.set_title(bin_labels[b], fontsize=9)
    axes[0].legend(fontsize=8)
    fig.suptitle("FP1 (young, 90/10) vs FP2 (old, 90/10): Q_diff_bin coefficients -- prediction: "
                 "NO middle-bin divergence (no behavioral difference to explain)")
    fig.tight_layout()
    fig.savefig(out_dir / "fp_qdiff_bins_young_vs_old.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- combined figure

def plot_combined(kappa_df, ls_df, word_l2_fits, cluster_results, out_dir):
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])

    ax_kappa = fig.add_subplot(gs[0, 0])
    plot_unpaired(ax_kappa, kappa_df, "kappa", r"Behavior: $\kappa$ (stickiness)")
    ax_kappa.set_ylabel(r"$\kappa$")

    ax_ls = fig.add_subplot(gs[0, 1])
    plot_unpaired(ax_ls, ls_df, "lose_switch_prob", "Behavior: lose-switch probability")
    ax_ls.set_ylabel("P(switch | previous trial unrewarded)")

    ax_note = fig.add_subplot(gs[0, 2])
    ax_note.axis("off")
    ax_note.text(0, 0.9, "Specificity control: FP1 (7 mice) vs FP2\n(4 mice), DIFFERENT animals, both at\n"
                 "90/10. Prediction: no behavioral OR\nneural age difference here, unlike SM's\n"
                 "80/20 within-animal aging series.", fontsize=10, va="top", wrap=True)

    for i, term in enumerate(NEURAL_TERMS_TO_SHOW):
        ax = fig.add_subplot(gs[1, i])
        col, se_col = f"C(word_l2)[T.{term}]_beta", f"C(word_l2)[T.{term}]_se"
        for cohort, fit in word_l2_fits.items():
            x = fit.index.to_numpy()
            y, se = fit[col], fit[se_col]
            ax.plot(x, y, color=GROUP_COLORS[cohort], label=GROUP_LABELS[cohort], linewidth=1.5)
            ax.fill_between(x, y - se, y + se, color=GROUP_COLORS[cohort], alpha=0.2)
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.axvline(0, color="black", linestyle=":", linewidth=0.8)
        r = cluster_results[term]
        ax.set_title(f"Neural: C(word_l2)[T.{term}]\ncluster p={r['p']:.3f} (exact, {r['n_perms']} perms)",
                     fontsize=9)
        if i == 0:
            ax.legend(fontsize=7)
            ax.set_ylabel("beta (pooled fit)")
        ax.set_xlabel("time from side_in (s)")

    fig.suptitle("FP1 vs FP2 (90/10 schedule, no behavioral kappa difference): specificity control\n"
                 "for the SM 80/20 aging finding", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "fp_condition_comparison_summary.png", dpi=150)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Behavioral: kappa ===")
    kappa_df = kappa_per_mouse()
    print(kappa_df.to_string(index=False))
    a = kappa_df.loc[kappa_df.cohort == "FP1_none", "kappa"]
    b = kappa_df.loc[kappa_df.cohort == "FP2_none", "kappa"]
    kappa_p = stats.mannwhitneyu(a, b, alternative="two-sided")[1]
    print(f"Mann-Whitney p={kappa_p:.4f}\n")
    kappa_df.to_csv(RESULTS_DIR / "kappa_per_mouse.csv", index=False)

    print("=== Behavioral: lose-switch-prob ===")
    ls_df = lose_switch_per_mouse()
    print(ls_df.to_string(index=False))
    a = ls_df.loc[ls_df.cohort == "FP1_none", "lose_switch_prob"]
    b = ls_df.loc[ls_df.cohort == "FP2_none", "lose_switch_prob"]
    ls_p = stats.mannwhitneyu(a, b, alternative="two-sided")[1]
    print(f"Mann-Whitney p={ls_p:.4f}\n")
    ls_df.to_csv(RESULTS_DIR / "lose_switch_per_mouse.csv", index=False)

    print("=== Neural: word_l2 group-pooled fits + unpaired cluster permutation ===")
    word_l2_fits, _ = word_l2_group_fits()
    per_mouse_betas, _ = word_l2_per_mouse_betas()
    cluster_results = run_word_l2_cluster_tests(per_mouse_betas)
    pd.DataFrame([
        dict(term=term, max_cluster_mass=r["mass"], p_value=r["p"], n_permutations=r["n_perms"])
        for term, r in cluster_results.items()
    ]).to_csv(RESULTS_DIR / "word_l2_cluster_permutation_results.csv", index=False)

    print("\n=== Neural: Q_diff_bin ===")
    qdiff_fits, peth_time = qdiff_bin_group_fits()
    plot_qdiff_bins(qdiff_fits, peth_time, OUT_DIR)

    plot_combined(kappa_df, ls_df, word_l2_fits, cluster_results, OUT_DIR)

    pd.DataFrame([
        dict(comparison="kappa_mannwhitney", p_value=kappa_p, n_fp1_mice=len(a), n_fp2_mice=len(b)),
        dict(comparison="lose_switch_mannwhitney", p_value=ls_p, n_fp1_mice=len(a), n_fp2_mice=len(b)),
    ]).to_csv(RESULTS_DIR / "behavioral_summary.csv", index=False)

    print(f"\nWrote figures to {OUT_DIR}/ and results CSVs to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
