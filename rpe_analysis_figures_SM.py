"""
Figures for the SM (PV-dualphotometry) cohort's within-animal RPE analyses.
Per explicit user direction, green_r and green_l are never pooled -- rpe_
analysis_stats_SM.py now saves each of the four analyses TWICE, once per
hemisphere, as *_{hemisphere}.csv. This module calls the four cohort-agnostic
strip-plot figure functions from rpe_analysis_figures.py (unchanged, shared
with FP1/FP2) ONCE PER HEMISPHERE via a small adapter (_hemisphere_view)
that stages that hemisphere's suffixed CSVs under the plain filenames those
shared functions expect, in a hemisphere-specific subdirectory -- so the
shared code itself needs no hemisphere-awareness and stays untouched. Each
hemisphere's 4 figures land in fig_dir/{hemisphere}/. Also produces one
direct green_r-vs-green_l beta_RPE comparison figure, sourced from the two
per-hemisphere analysis1 CSVs plus analysis1b_hemisphere_interaction.csv
(see rpe_analysis_stats_SM.py's docstring: informational, not a pooling
gate -- hemispheres are already fully separate above regardless).

Usage:
    python rpe_analysis_figures_SM.py [results_dir] [fig_dir] [--pooled-encoding-r2 X] [--pooled-fir-r2 X]
"""

import json
import shutil
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

PER_HEMISPHERE_FILES = (
    "analysis1_rpe_regression", "analysis2_signed_vs_unsigned",
    "analysis3a_encoding_glm_per_mouse", "analysis3b_fir_glm_per_mouse",
    "analysis4_temporal_specificity",
)


def _hemisphere_view(results_dir, hemisphere):
    """Stage a hemisphere's *_{hemisphere}.csv files under the plain
    filenames rpe_analysis_figures.py's shared fig_* functions expect, in a
    dedicated subdirectory -- avoids touching that shared, FP1/FP2-used
    module at all.
    """
    view_dir = results_dir / f"_view_{hemisphere}"
    view_dir.mkdir(exist_ok=True)
    for stem in PER_HEMISPHERE_FILES:
        src = results_dir / f"{stem}_{hemisphere}.csv"
        if src.exists():
            shutil.copyfile(src, view_dir / f"{stem}.csv")
    return view_dir


def fig_hemisphere_comparison(results_dir, fig_dir, mouse_colors):
    r_df = pd.read_csv(results_dir / "analysis1_rpe_regression_green_r.csv", index_col=0)
    l_df = pd.read_csv(results_dir / "analysis1_rpe_regression_green_l.csv", index_col=0)
    interaction = pd.read_csv(results_dir / "analysis1b_hemisphere_interaction.csv", index_col=0)

    pivot = pd.DataFrame({"green_r": r_df["beta_rpe"], "green_l": l_df["beta_rpe"]})
    mice = sorted(set(pivot.index))

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
    n_both = int(pivot[["green_r", "green_l"]].notna().all(axis=1).sum())
    ax.text(
        0.02, 0.02,
        f"{n_both}/{len(mice)} mice have both hemispheres valid\n"
        f"{n_sig}/{n_fittable} show a significant RPE x hemisphere interaction (p<0.05)\n"
        "(hemispheres reported fully separately above -- this is a direct comparison, not a pooled result)",
        transform=ax.transAxes, fontsize=9.5, va="bottom", ha="left", color="0.3",
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "rpe_hemisphere_comparison.png", dpi=150)
    fig.savefig(fig_dir / "rpe_hemisphere_comparison.svg")
    plt.close(fig)


def make_all_figures(results_dir, fig_dir, pooled_encoding_r2=None, pooled_fir_r2=None):
    results_dir, fig_dir = Path(results_dir), Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "summary_stats.json") as f:
        summary_stats_by_hemisphere = json.load(f)

    # All mice across both hemispheres share one consistent color mapping,
    # so a mouse plotted in the green_r figures is the same color in green_l.
    all_mice = sorted({
        mouse
        for hemisphere in summary_stats_by_hemisphere
        for mouse in pd.read_csv(results_dir / f"analysis1_rpe_regression_{hemisphere}.csv", index_col=0).index
    })
    mouse_colors = _mouse_colors(all_mice)

    for hemisphere, summary_stats in summary_stats_by_hemisphere.items():
        view_dir = _hemisphere_view(results_dir, hemisphere)
        hemi_fig_dir = fig_dir / hemisphere
        hemi_fig_dir.mkdir(parents=True, exist_ok=True)

        fig_rpe_regression(view_dir, hemi_fig_dir, mouse_colors, summary_stats)
        fig_signed_vs_unsigned(view_dir, hemi_fig_dir, mouse_colors, summary_stats)
        fig_temporal_specificity(view_dir, hemi_fig_dir, mouse_colors)
        fig_between_animal_r2(view_dir, hemi_fig_dir, mouse_colors, pooled_encoding_r2, pooled_fir_r2)
        print(f"Saved 4 figures to {hemi_fig_dir}")

    fig_hemisphere_comparison(results_dir, fig_dir, mouse_colors)
    print(f"Saved rpe_hemisphere_comparison to {fig_dir}")


if __name__ == "__main__":
    import sys
    results_dir_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs_fixed/rpe_analysis_sm/results")
    fig_dir_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("figures_fixed_sm")
    make_all_figures(results_dir_arg, fig_dir_arg)
