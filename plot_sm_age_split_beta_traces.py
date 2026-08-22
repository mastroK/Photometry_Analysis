"""
Time-resolved C(word_l2) coefficient traces (beta(t)) for SM, young vs old,
restricted to the 7 "clean" (current-trial-rewarded) terms -- the other 8
are truncation-noise-dominated at late timepoints, see
cluster_permutation_word_l2.py's module docstring -- with each panel's
title annotated by its own cluster-permutation p-value (paired sign-flip
test, cluster_permutation_word_l2.py) instead of a separate R^2 panel.

R^2 was dropped from this figure on request: fit_time_resolved_glm's R^2
is in-sample, single-timepoint (no window-averaging, no cross-validation),
so it reads as "bad" (0.01-0.06) for reasons unrelated to whether young and
old actually differ -- it was never a significance measure. The cluster
p-value (computed on the actual beta(t) trace, the thing being visually
compared here) is the right number to report a magnitude/significance
claim from; R^2 remains available via plot_sm_age_split_neural_correlate.py
for anyone who wants the scalar-window CV version instead.

Usage:
    python plot_sm_age_split_beta_traces.py               # green_l
    python plot_sm_age_split_beta_traces.py red_l          # red_l replication
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cluster_permutation_word_l2 import CLEAN_TERMS, PAIRED_MICE, cluster_permutation_test, paired_diffs
from models.glm_encoding import fit_time_resolved_glm

OUT_DIR = Path("figures_fixed_bandit_behavior_comparison")
FORMULA = "Z ~ C(word_l2)"
AGE_COLORS = {"young": "#55A868", "old": "#8172B2"}
MIN_ACHIEVABLE_P = 1 / 2 ** len(PAIRED_MICE)


def load_pooled(root, label):
    tt = pd.read_parquet(root / label / "results" / "pooled_trial_table_in.parquet")
    npz = np.load(root / label / "results" / "pooled_zscore_windows.npz")
    return tt, npz["zscore_in"], npz["peth_time_in"]


def main():
    hemisphere = sys.argv[1] if len(sys.argv) > 1 else "green_l"
    root = Path(f"outputs_fixed/model_series_comparison_sm_age_split{'_' + hemisphere if hemisphere != 'green_l' else ''}")

    fits = {}
    for label in ("young", "old"):
        tt, zs, peth_time = load_pooled(root, label)
        fits[label] = fit_time_resolved_glm(zs, peth_time, tt, formula=FORMULA, min_retained_frac=0.5)
        print(f"{label}: {tt['mouse'].nunique()} mice, {len(tt)} trials")

    diffs, _ = paired_diffs(root, terms=CLEAN_TERMS, mice=PAIRED_MICE)
    cluster_p = {}
    for term in CLEAN_TERMS:
        _, p, _ = cluster_permutation_test(diffs[term])
        cluster_p[term] = p
        print(f"  {term}: cluster p={p:.4f}")

    n_cols = 4
    n_rows = int(np.ceil((len(CLEAN_TERMS) + 1) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.2 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    for ax in axes[1:len(CLEAN_TERMS)]:
        ax.sharex(axes[0])  # link the time-series panels only -- NOT the categorical p-value bar chart

    for ax, term in zip(axes, CLEAN_TERMS):
        for label, fit in fits.items():
            y, se = fit[f"C(word_l2)[T.{term}]_beta"], fit[f"C(word_l2)[T.{term}]_se"]
            x = fit.index.to_numpy()
            ax.plot(x, y, color=AGE_COLORS[label], label=label, linewidth=1.5)
            ax.fill_between(x, y - se, y + se, color=AGE_COLORS[label], alpha=0.2)
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.axvline(0, color="black", linestyle=":", linewidth=0.8)
        sig = "*" if cluster_p[term] <= MIN_ACHIEVABLE_P else ""
        ax.set_title(f"{term}{sig}\ncluster p={cluster_p[term]:.3f}", fontsize=9)

    p_ax = axes[len(CLEAN_TERMS)]
    p_ax.bar(CLEAN_TERMS, [cluster_p[t] for t in CLEAN_TERMS], color="#55A868")
    p_ax.axhline(0.05, color="black", linestyle="--", linewidth=1, label="p=0.05")
    p_ax.axhline(MIN_ACHIEVABLE_P, color="red", linestyle=":", linewidth=1,
                 label=f"min achievable (n={len(PAIRED_MICE)} mice)")
    p_ax.set_ylim(0, 1.05)
    p_ax.set_title("cluster p-value by term", fontsize=9)
    p_ax.legend(fontsize=6)
    p_ax.tick_params(axis="x", rotation=45)

    for ax in axes[len(CLEAN_TERMS) + 1:]:
        ax.axis("off")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"SM {hemisphere}, side_in-aligned: C(word_l2) time-resolved coefficients, young vs old\n"
                 "(shaded = SE; cluster p from paired sign-flip permutation test, not R^2)")
    fig.tight_layout()
    out_path = OUT_DIR / f"sm_{hemisphere}_word_l2_beta_traces_young_vs_old.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
