"""
Figures for the SM (PV-dualphotometry) cohort's within-animal RPE analyses --
reuses the four cohort-agnostic strip-plot figures from rpe_analysis_figures.py
unchanged, plus one new figure specific to this cohort's bilateral-hemisphere
design: a per-mouse green_r-vs-green_l beta_RPE comparison, sourced from
rpe_analysis_stats_SM.py's hemisphere_breakdown.csv / analysis1b_hemisphere_
interaction.csv (see that module's docstring for why this is a supplementary,
non-gating check rather than a change to the pooled per-mouse statistic).

Usage:
    python rpe_analysis_figures_SM.py [results_dir] [fig_dir] [--pooled-encoding-r2 X] [--pooled-fir-r2 X]
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from rpe_analysis_figures import (
    _mouse_colors,
    fig_between_animal_r2,
    fig_rpe_regression,
    fig_signed_vs_unsigned,
    fig_temporal_specificity,
)

sns.set_theme(style="ticks", context="talk")


def fig_hemisphere_breakdown(results_dir, fig_dir, mouse_colors):
    breakdown = pd.read_csv(results_dir / "hemisphere_breakdown.csv")
    interaction = pd.read_csv(results_dir / "analysis1b_hemisphere_interaction.csv", index_col=0)

    pivot = breakdown.pivot(index="mouse", columns="hemisphere", values="beta_rpe")
    mice = list(pivot.index)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for mouse in mice:
        color = mouse_colors[mouse]
        has_r = "green_r" in pivot.columns and pd.notna(pivot.loc[mouse].get("green_r"))
        has_l = "green_l" in pivot.columns and pd.notna(pivot.loc[mouse].get("green_l"))
        if has_r and has_l:
            ax.plot([0, 1], [pivot.loc[mouse, "green_r"], pivot.loc[mouse, "green_l"]],
                    color=color, alpha=0.6, lw=1.5, zorder=2)
            ax.scatter([0], [pivot.loc[mouse, "green_r"]], color=color, s=110, zorder=3,
                       edgecolor="white", linewidth=1.2, label=mouse)
            ax.scatter([1], [pivot.loc[mouse, "green_l"]], color=color, s=110, zorder=3,
                       edgecolor="white", linewidth=1.2)
        elif has_r:
            ax.scatter([0], [pivot.loc[mouse, "green_r"]], color=color, s=110, zorder=3,
                       edgecolor="white", linewidth=1.2, marker="D", label=f"{mouse} (green_r only)")
        elif has_l:
            ax.scatter([1], [pivot.loc[mouse, "green_l"]], color=color, s=110, zorder=3,
                       edgecolor="white", linewidth=1.2, marker="D", label=f"{mouse} (green_l only)")

    ax.axhline(0, color="0.4", lw=1, ls="--", zorder=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["green_r", "green_l"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel(r"$\beta_{RPE}$ (within that hemisphere only)")
    ax.set_title("Per-mouse RPE effect: green_r vs. green_l (descriptive)", fontsize=13)
    ax.legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
    sns.despine(ax=ax)

    n_fittable = int(interaction["interaction_p"].notna().sum())
    n_sig = int((interaction["interaction_p"] < 0.05).sum())
    n_both = int((breakdown.groupby("mouse")["hemisphere"].nunique() == 2).sum())
    ax.text(
        0.02, 0.02,
        f"{n_both}/{len(mice)} mice have both hemispheres valid\n"
        f"{n_sig}/{n_fittable} show a significant RPE x hemisphere interaction (p<0.05)",
        transform=ax.transAxes, fontsize=9.5, va="bottom", ha="left", color="0.3",
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "rpe_hemisphere_breakdown.png", dpi=150)
    fig.savefig(fig_dir / "rpe_hemisphere_breakdown.svg")
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
    fig_hemisphere_breakdown(results_dir, fig_dir, mouse_colors)
    print(f"Saved 5 figures to {fig_dir}")


if __name__ == "__main__":
    import sys
    results_dir_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs_fixed/rpe_analysis_sm/results")
    fig_dir_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("figures_fixed_sm")
    make_all_figures(results_dir_arg, fig_dir_arg)
