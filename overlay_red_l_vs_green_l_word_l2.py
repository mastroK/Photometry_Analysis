"""
Overlay red_l vs green_l's pooled coefficient trajectories (same 5 mice:
SM1L, SM1N, SM2N, SM2R, SM3FR) on shared axes, term by term, plus R^2, for
any model already run through run_model_series_comparison_sm_red_l.py.
Reads the beta_df.csv files plot_pooled already saved for each channel --
no refitting.

Usage:
    python overlay_red_l_vs_green_l_word_l2.py [model_name ...]
    # model_name defaults to 2b_word_l2; pass one or more MODEL_SPECS names
    # (e.g. 3c_reward_qchosen_qdiff 3d_reward_qchosen_rpeabs_qdiff) to
    # overlay those instead/as well.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RED_L_FIG_DIR = Path("figures_fixed_model_series_sm_red_l/final")
GREEN_L_FIG_DIR = Path("figures_fixed_model_series_sm_green_l/final")
OUT_DIR = RED_L_FIG_DIR

COLOR_RED = "#B8763E"
COLOR_GREEN = "#3B5B74"


def make_overlay(model_name, event="side_in"):
    red_path = RED_L_FIG_DIR / f"encoding_{model_name}_{event}_beta_df.csv"
    green_path = GREEN_L_FIG_DIR / f"encoding_{model_name}_{event}_beta_df.csv"
    df_red = pd.read_csv(red_path, index_col=0)
    df_green = pd.read_csv(green_path, index_col=0)
    peth_time = df_red.index.to_numpy(dtype=float)

    terms_red = {c[:-5] for c in df_red.columns if c.endswith("_beta")}
    terms_green = {c[:-5] for c in df_green.columns if c.endswith("_beta")}
    terms = sorted(terms_red & terms_green, key=lambda t: (t != "Intercept", t))
    missing = (terms_red | terms_green) - (terms_red & terms_green)
    if missing:
        print(f"  NOTE: terms not common to both channels, skipped: {sorted(missing)}")

    ncols = 3
    nrows = int(np.ceil((len(terms) + 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for i, term in enumerate(terms):
        ax = axes_flat[i]
        for df, color, label in [(df_green, COLOR_GREEN, "green_l"), (df_red, COLOR_RED, "red_l")]:
            beta, se = df[f"{term}_beta"].to_numpy(), df[f"{term}_se"].to_numpy()
            ax.plot(peth_time, beta, color=color, lw=1.8, label=label)
            ax.fill_between(peth_time, beta - 1.96 * se, beta + 1.96 * se, color=color, alpha=0.18, lw=0)
        ax.axhline(0, color="0.6", lw=0.8, ls="--")
        ax.axvline(0, color="0.3", lw=0.8, ls=":")
        ax.set_title(term, fontsize=10)
        ax.set_xlabel("Time from side-in (s)", fontsize=8)
        ax.set_ylabel(r"$\beta$", fontsize=8)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=8, frameon=False)

    r2_ax = axes_flat[len(terms)]
    for df, color, label in [(df_green, COLOR_GREEN, "green_l"), (df_red, COLOR_RED, "red_l")]:
        r2_ax.plot(peth_time, df["r_squared"].to_numpy(), color=color, lw=1.8, label=label)
    r2_ax.axvline(0, color="0.3", lw=0.8, ls=":")
    r2_ax.set_title("Model fit ($R^2$)", fontsize=10)
    r2_ax.set_xlabel("Time from side-in (s)", fontsize=8)
    r2_ax.tick_params(labelsize=8)
    r2_ax.legend(fontsize=8, frameon=False)

    for ax in axes_flat[len(terms) + 1:]:
        ax.axis("off")

    fig.suptitle(f"red_l vs green_l -- {model_name} (5 mice: SM1L, SM1N, SM2N, SM2R, SM3FR)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUT_DIR / f"overlay_red_l_vs_green_l_{model_name}"
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)
    print(f"Saved {out_path.with_suffix('.png')}")


if __name__ == "__main__":
    models = sys.argv[1:] if len(sys.argv) > 1 else ["2b_word_l2"]
    for model_name in models:
        make_overlay(model_name)
