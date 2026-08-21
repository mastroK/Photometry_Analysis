"""
Tests the "middle quadrant" extension of the age hypothesis: if old mice
have better access to recent-trial information (see
plot_sm_age_split_beta_traces.py / cluster_permutation_word_l2.py's word_l2
finding) and can therefore handle VALUE AMBIGUITY differently, the age
difference in neural encoding should be concentrated in trials where
Q_diff (the fitted sticky-Q model's value difference between options) is
near zero -- the middle Q_diff_bin -- rather than in the clear-cut extreme
bins, where there's little ambiguity for better information-access to help
resolve.

Q_diff_bin (5 quantile bins, MODEL3_QDIFF_N_BINS, see
run_model_series_comparison.py::_qdiff_bin_edges/_add_qdiff_bins) is
normally computed per mouse from that mouse's own side_in Q_diff
distribution. Here it's computed per mouse from the COMBINED young+old
Q_diff values (both cached pooled tables already have a real Q_diff column
straight from the sticky-Q fit -- computed once per session in
external.bandit_state_adapter, upstream of this young/old split), so bin 2
means the same "how ambiguous was this trial" thing in both age groups,
then applied separately to each group's rows before pooling across the 4
paired mice + SM3FR (old only).

Usage:
    python plot_sm_age_split_qdiff_bins.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.glm_encoding import fit_time_resolved_glm
from run_model_series_comparison import _add_qdiff_bins, _qdiff_bin_edges

RESULTS_ROOT = Path("outputs_fixed/model_series_comparison_sm_age_split")
OUT_DIR = Path("figures_fixed_bandit_behavior_comparison")
FORMULA = "Z ~ C(Q_diff_bin)"
AGE_COLORS = {"young": "#55A868", "old": "#8172B2"}
BIN_LABELS = {"0": "extreme (bin 0)", "1": "bin 1", "2": "middle (bin 2)", "3": "bin 3", "4": "extreme (bin 4)"}


def load_pooled(label):
    tt = pd.read_parquet(RESULTS_ROOT / label / "results" / "pooled_trial_table_in.parquet")
    npz = np.load(RESULTS_ROOT / label / "results" / "pooled_zscore_windows.npz")
    return tt, npz["zscore_in"], npz["peth_time_in"]


def add_age_consistent_qdiff_bins(tt_young, tt_old):
    """Per mouse, compute Q_diff_bin edges from that mouse's COMBINED
    young+old Q_diff values, then apply separately to each group -- so bin
    identity is age-consistent (bin 2 = "near-zero Q_diff" in both groups),
    not re-quantiled per group (which would make bin 2 mean "median of
    young" and "median of old" separately, defeating the comparison).
    """
    tt_young, tt_old = tt_young.copy(), tt_old.copy()
    tt_young["Q_diff_bin"] = np.nan
    tt_old["Q_diff_bin"] = np.nan

    all_mice = set(tt_young["mouse"]) | set(tt_old["mouse"])
    for mouse in all_mice:
        y_mask = tt_young["mouse"] == mouse
        o_mask = tt_old["mouse"] == mouse
        combined_qdiff = pd.concat([tt_young.loc[y_mask, "Q_diff"], tt_old.loc[o_mask, "Q_diff"]])
        edges = _qdiff_bin_edges(combined_qdiff)
        if y_mask.any():
            tt_young.loc[y_mask, "Q_diff_bin"] = _add_qdiff_bins(tt_young.loc[y_mask], edges)["Q_diff_bin"]
        if o_mask.any():
            tt_old.loc[o_mask, "Q_diff_bin"] = _add_qdiff_bins(tt_old.loc[o_mask], edges)["Q_diff_bin"]
    return tt_young, tt_old


def main():
    tt_young, zs_young, peth_time = load_pooled("young")
    tt_old, zs_old, peth_time_old = load_pooled("old")
    assert np.array_equal(peth_time, peth_time_old)

    tt_young, tt_old = add_age_consistent_qdiff_bins(tt_young, tt_old)
    print("young Q_diff_bin counts:\n", tt_young["Q_diff_bin"].value_counts(dropna=False).sort_index())
    print("old Q_diff_bin counts:\n", tt_old["Q_diff_bin"].value_counts(dropna=False).sort_index())

    fit_young = fit_time_resolved_glm(zs_young, peth_time, tt_young, formula=FORMULA, min_retained_frac=0.5)
    fit_old = fit_time_resolved_glm(zs_old, peth_time, tt_old, formula=FORMULA, min_retained_frac=0.5)

    bins = ["0", "1", "2", "3", "4"]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharey=True)
    for ax, b in zip(axes, bins):
        col = f"C(Q_diff_bin)[T.{b}]_beta" if b != "0" else None
        se_col = f"C(Q_diff_bin)[T.{b}]_se" if b != "0" else None
        if col is None:
            # bin "0" is patsy's dropped reference level -- plot the model
            # intercept instead, since bin 0's own coefficient is folded into it.
            col, se_col = "Intercept_beta", "Intercept_se"
        for label, fit in (("young", fit_young), ("old", fit_old)):
            x = fit.index.to_numpy()
            y, se = fit[col], fit[se_col]
            ax.plot(x, y, color=AGE_COLORS[label], label=label, linewidth=1.5)
            ax.fill_between(x, y - se, y + se, color=AGE_COLORS[label], alpha=0.2)
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.axvline(0, color="black", linestyle=":", linewidth=0.8)
        ax.set_title(BIN_LABELS[b] + ("\n(= Intercept, ref level)" if b == "0" else ""), fontsize=9)
    axes[0].legend(fontsize=8)
    fig.suptitle("SM green_l, side_in: Q_diff_bin coefficients by age (bin 2 = near-zero/ambiguous Q_diff)\n"
                 "age-consistent bin edges (per mouse, from combined young+old Q_diff)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sm_qdiff_bins_young_vs_old.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'sm_qdiff_bins_young_vs_old.png'}")

    print("\nmean |old - young| beta gap by bin (over well-supported pre-event span, peth_time<0):")
    pre_mask = fit_young.index.to_numpy() < 0
    gap_rows = []
    for b in bins:
        col = f"C(Q_diff_bin)[T.{b}]_beta" if b != "0" else "Intercept_beta"
        gap = (fit_old[col] - fit_young[col])[pre_mask]
        print(f"  bin {b}: mean|gap|={gap.abs().mean():.4f}, mean gap={gap.mean():.4f}")
        gap_rows.append(dict(bin=b, mean_abs_gap=gap.abs().mean(), mean_gap=gap.mean()))
    gap_df = pd.DataFrame(gap_rows)
    gap_csv = RESULTS_ROOT / "qdiff_bin_gap_by_bin.csv"
    gap_df.to_csv(gap_csv, index=False)
    print(f"Wrote {gap_csv}")


if __name__ == "__main__":
    main()
