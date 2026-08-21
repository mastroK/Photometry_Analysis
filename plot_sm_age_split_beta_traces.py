"""
Time-resolved C(word_l2) coefficient traces (beta(t), not a single pooled
CV R^2 scalar) for SM green_l, young vs old, overlaid directly for shape
comparison -- the follow-up to plot_sm_age_split_neural_correlate.py's
scalar word_l1->word_l2 incremental-R^2 check, which found no clear age
difference. A shape difference (timing of the peak, sign flip, a term that
only differentiates in one age group) can be invisible to a single
CV-window scalar but visible in the actual trajectory -- this is that
direct comparison, reusing the already-cached pooled arrays from
run_sm_age_split_comparison.py (no photometry reload needed).

word_l2's window is [t-1, t] (see behavior/word_encoding.py::add_word_labels
and the correction in plot_sm_age_split_neural_correlate.py) -- i.e. this
trial's own identity conditioned on the previous trial's, the direct neural
analogue of kappa's 1-back "stick" term.

min_retained_frac=0.5 matches run_sm_age_split_comparison.py's own
truncate_at_side_out=True convention (used to build this cached data).

Usage:
    python plot_sm_age_split_beta_traces.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.glm_encoding import fit_time_resolved_glm

RESULTS_ROOT = Path("outputs_fixed/model_series_comparison_sm_age_split")
OUT_DIR = Path("figures_fixed_bandit_behavior_comparison")
FORMULA = "Z ~ C(word_l2)"
AGE_COLORS = {"young": "#55A868", "old": "#8172B2"}


def load_pooled(label):
    tt = pd.read_parquet(RESULTS_ROOT / label / "results" / "pooled_trial_table_in.parquet")
    npz = np.load(RESULTS_ROOT / label / "results" / "pooled_zscore_windows.npz")
    return tt, npz["zscore_in"], npz["peth_time_in"]


def main():
    fits = {}
    n_trials_t0 = {}
    for label in ("young", "old"):
        tt, zs, peth_time = load_pooled(label)
        fit = fit_time_resolved_glm(zs, peth_time, tt, formula=FORMULA, min_retained_frac=0.5)
        fits[label] = fit
        n_trials_t0[label] = fit["n_trials"].iloc[0]
        print(f"{label}: {tt['mouse'].nunique()} mice, {len(tt)} trials, "
              f"{fit['n_trials'].iloc[0]} trials at t0")

    beta_terms = sorted({
        col[:-len("_beta")] for col in fits["young"].columns
        if col.endswith("_beta") and col != "Intercept_beta"
    })
    print(f"terms: {beta_terms}")

    n_cols = 4
    n_rows = int(np.ceil((len(beta_terms) + 1) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.2 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    for ax, term in zip(axes, beta_terms + ["r_squared"]):
        for label, fit in fits.items():
            if term == "r_squared":
                y = fit["r_squared"]
                se = None
            else:
                y = fit[f"{term}_beta"]
                se = fit[f"{term}_se"]
            x = fit.index.to_numpy()
            ax.plot(x, y, color=AGE_COLORS[label], label=label, linewidth=1.5)
            if se is not None:
                ax.fill_between(x, y - se, y + se, color=AGE_COLORS[label], alpha=0.2)
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.axvline(0, color="black", linestyle=":", linewidth=0.8)
        ax.set_title(term, fontsize=9)

    for ax in axes[len(beta_terms) + 1:]:
        ax.axis("off")
    axes[0].legend(fontsize=8)
    fig.suptitle("SM green_l, side_in-aligned: C(word_l2) time-resolved coefficients, young vs old\n"
                 "(shaded = SE; word_l2 window = [t-1, t], the previous-trial-conditioned term)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sm_word_l2_beta_traces_young_vs_old.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'sm_word_l2_beta_traces_young_vs_old.png'}")


if __name__ == "__main__":
    main()
