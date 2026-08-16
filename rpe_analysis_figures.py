"""
Strip-plot figures for the within-animal RPE analyses (rpe_analysis_stats.py
results). One point per mouse throughout, per the user's request that every
claim be shown at the animal level rather than pooled. Cohort-agnostic --
annotations and reference lines are computed from the actual results/summary
stats passed in, not hardcoded to any one cohort's numbers.

Usage:
    python rpe_analysis_figures.py [results_dir] [fig_dir] [--pooled-encoding-r2 X] [--pooled-fir-r2 X]
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon

sns.set_theme(style="ticks", context="talk")


def _mouse_colors(mice):
    palette = sns.color_palette("tab10", len(mice))
    return dict(zip(sorted(mice), palette))


def _strip(ax, values, labels, mouse_colors, ylabel, title, hline=0.0, annotate=None):
    x = np.arange(len(values))
    colors = [mouse_colors[m] for m in labels]
    ax.scatter(x, values, c=colors, s=110, zorder=3, edgecolor="white", linewidth=1.2)
    ax.axhline(hline, color="0.4", lw=1, ls="--", zorder=1)
    median = np.median(values)
    ax.axhline(median, color="0.2", lw=2, zorder=2, label=f"median={median:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13)
    ax.legend(frameon=False, fontsize=10, loc="best")
    sns.despine(ax=ax)
    if annotate:
        ax.text(0.02, 0.02, annotate, transform=ax.transAxes, fontsize=10,
                va="bottom", ha="left", color="0.3")


def fig_rpe_regression(results_dir, fig_dir, mouse_colors, summary_stats):
    df = pd.read_csv(results_dir / "analysis1_rpe_regression.csv", index_col=0)
    n = len(df)
    n_neg = int((df["beta_rpe"] < 0).sum())
    p = summary_stats["analysis1"]["wilcoxon_p"]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    _strip(ax, df["beta_rpe"], df.index, mouse_colors, r"$\beta_{RPE}$ (post_amp ~ RPE$_{signed}$)",
           "Within-animal RPE regression",
           annotate=f"Wilcoxon p={p:.3f}, {n_neg}/{n} mice negative")
    fig.tight_layout()
    fig.savefig(fig_dir / "rpe_beta_per_mouse.png", dpi=150)
    fig.savefig(fig_dir / "rpe_beta_per_mouse.svg")
    plt.close(fig)


def fig_signed_vs_unsigned(results_dir, fig_dir, mouse_colors, summary_stats):
    df = pd.read_csv(results_dir / "analysis2_signed_vs_unsigned.csv", index_col=0)
    n = len(df)
    n_pos = summary_stats["analysis2"]["n_positive"]
    p = summary_stats["analysis2"]["wilcoxon_p"]
    sig_note = "" if p < 0.05 else " (n.s.)"
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    _strip(ax, df["delta_r2"], df.index, mouse_colors, r"$\Delta R^2$ (signed $-$ unsigned)",
           "Signed vs. unsigned RPE (5-fold CV)",
           annotate=f"Wilcoxon p={p:.2f}{sig_note}, {n_pos}/{n} mice positive")
    fig.tight_layout()
    fig.savefig(fig_dir / "rpe_signed_vs_unsigned.png", dpi=150)
    fig.savefig(fig_dir / "rpe_signed_vs_unsigned.svg")
    plt.close(fig)


def fig_temporal_specificity(results_dir, fig_dir, mouse_colors):
    df = pd.read_csv(results_dir / "analysis4_temporal_specificity.csv", index_col=0)
    mice = list(df.index)
    n = len(mice)
    n_post_sig = int((df["interaction_p_post"] < 0.05).sum())
    n_pre_sig = int((df["interaction_p_pre"] < 0.05).sum())

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for m in mice:
        color = mouse_colors[m]
        ax.plot([0, 1], [df.loc[m, "interaction_coef_post"], df.loc[m, "interaction_coef_pre"]],
                color=color, alpha=0.6, lw=1.5, zorder=2)
        ax.scatter([0], [df.loc[m, "interaction_coef_post"]], color=color, s=110, zorder=3,
                   edgecolor="white", linewidth=1.2, label=m)
        ax.scatter([1], [df.loc[m, "interaction_coef_pre"]], color=color, s=110, zorder=3,
                   edgecolor="white", linewidth=1.2)
    ax.axhline(0, color="0.4", lw=1, ls="--", zorder=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Post-event\n(0 to +1s)", "Pre-event\n(-1 to 0s)"])
    ax.set_ylabel(r"outcome $\times$ Q(chosen) interaction $\beta$")
    ax.set_title("Temporal specificity of value modulation", fontsize=13)
    ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.text(0.02, 0.02, f"Post: {n_post_sig}/{n} significant\nPre: {n_pre_sig}/{n} significant",
            transform=ax.transAxes, fontsize=10, va="bottom", ha="left", color="0.3")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "rpe_temporal_specificity.png", dpi=150)
    fig.savefig(fig_dir / "rpe_temporal_specificity.svg")
    plt.close(fig)


def fig_between_animal_r2(results_dir, fig_dir, mouse_colors, pooled_encoding_r2=None, pooled_fir_r2=None):
    enc = pd.read_csv(results_dir / "analysis3a_encoding_glm_per_mouse.csv", index_col=0)
    fir = pd.read_csv(results_dir / "analysis3b_fir_glm_per_mouse.csv", index_col=0)
    if pooled_encoding_r2 is None:
        pooled_encoding_r2 = enc["peak_r2"].median()
    if pooled_fir_r2 is None:
        pooled_fir_r2 = fir["r2_mean"].median()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    _strip(axes[0], enc["peak_r2"], enc.index, mouse_colors, r"Peak $R^2$ (encoding GLM)",
           "Encoding GLM: per-mouse peak $R^2$", hline=pooled_encoding_r2,
           annotate=f"dashed line: pooled peak R² ≈ {pooled_encoding_r2:.2f}")

    ax = axes[1]
    x = np.arange(len(fir))
    colors = [mouse_colors[m] for m in fir.index]
    ax.errorbar(x, fir["r2_mean"], yerr=fir["r2_std"], fmt="none", ecolor="0.5", capsize=4, zorder=2)
    ax.scatter(x, fir["r2_mean"], c=colors, s=110, zorder=3, edgecolor="white", linewidth=1.2)
    ax.axhline(pooled_fir_r2, color="0.4", lw=1, ls="--", zorder=1,
               label=f"pooled out-of-sample R² = {pooled_fir_r2:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(fir.index, rotation=30, ha="right")
    ax.set_ylabel(r"Out-of-sample $R^2$ (FIR model)")
    ax.set_title("FIR GLM: per-mouse out-of-sample $R^2$", fontsize=13)
    ax.legend(frameon=False, fontsize=10, loc="best")
    sns.despine(ax=ax)
    weakest = fir["r2_mean"].idxmin()
    fewest_sessions = fir["n_trials"].idxmin() if "n_trials" in fir.columns else None
    note = "Error bars: SD across CV folds."
    if weakest == fewest_sessions:
        note += f"\n{weakest} (fewest trials) shows the weakest fit."
    ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=9.5, va="bottom", ha="left", color="0.3")

    fig.tight_layout()
    fig.savefig(fig_dir / "rpe_between_animal_r2.png", dpi=150)
    fig.savefig(fig_dir / "rpe_between_animal_r2.svg")
    plt.close(fig)


def make_all_figures(results_dir, fig_dir, pooled_encoding_r2=None, pooled_fir_r2=None):
    results_dir, fig_dir = Path(results_dir), Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "summary_stats.json") as f:
        summary_stats = json.load(f)

    r1 = pd.read_csv(results_dir / "analysis1_rpe_regression.csv", index_col=0)
    mice = list(r1.index)
    mouse_colors = _mouse_colors(mice)

    fig_rpe_regression(results_dir, fig_dir, mouse_colors, summary_stats)
    fig_signed_vs_unsigned(results_dir, fig_dir, mouse_colors, summary_stats)
    fig_temporal_specificity(results_dir, fig_dir, mouse_colors)
    fig_between_animal_r2(results_dir, fig_dir, mouse_colors, pooled_encoding_r2, pooled_fir_r2)
    print(f"Saved 4 figures to {fig_dir}")


if __name__ == "__main__":
    import sys
    results_dir_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs_fixed/rpe_analysis/results")
    fig_dir_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("figures_fixed")
    make_all_figures(results_dir_arg, fig_dir_arg)
