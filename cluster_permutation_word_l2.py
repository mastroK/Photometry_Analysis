"""
Cluster-based permutation test for the young-vs-old C(word_l2) beta(t)
traces (plot_sm_age_split_beta_traces.py's descriptive overlay, formalized).

Only the 4 mice with BOTH a young and an old session set (SM1L, SM1N, SM2N,
SM2R -- SM3FR has no young sessions, excluded) are used, as a genuinely
paired (within-mouse) design: for each of these mice, fit_time_resolved_glm
is refit SEPARATELY on that mouse's own young-only and old-only session
rows (sliced directly out of the already-cached pooled arrays -- no
photometry reload), giving one paired (young, old) beta(t) trace per mouse
per term.

Cluster statistic: at each timepoint, a paired t across the 4 mice on
(old_beta - young_beta); contiguous timepoints with |t| above
CLUSTER_FORMING_T are summed (sum of |t|) into a cluster mass; the largest
cluster mass is the per-term test statistic.

Null distribution: exact enumeration of all 2^4=16 sign-flip assignments of
the 4 mice's paired differences (the only exchangeability a paired n=4
design supports) -- an EXACT test given full enumeration, not a Monte Carlo
approximation, at the cost of the coarsest possible p-value resolution
(multiples of 1/16=0.0625).

Only run for the 7 "clean" (current-trial-rewarded) word_l2 terms
identified in plot_sm_age_split_beta_traces.py -- the other 8
(current-trial-unrewarded) terms are dominated by late-window sample-size
collapse (see that script's docstring) and a cluster test on them would
mostly be testing truncation-noise, not signal.

Usage:
    python cluster_permutation_word_l2.py               # green_l
    python cluster_permutation_word_l2.py red_l          # red_l replication
"""

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from models.glm_encoding import fit_time_resolved_glm

FORMULA = "Z ~ C(word_l2)"
CLEAN_TERMS = ["LR", "RL", "RR", "lL", "lR", "rL", "rR"]  # current-trial-rewarded, see module docstring
CLUSTER_FORMING_T = 2.0  # df=3 paired t; a conventional cluster-forming threshold, not itself a p-value
PAIRED_MICE = ("SM1L", "SM1N", "SM2N", "SM2R")  # mice with sessions in both age groups


def load_pooled(root, label):
    tt = pd.read_parquet(root / label / "results" / "pooled_trial_table_in.parquet")
    npz = np.load(root / label / "results" / "pooled_zscore_windows.npz")
    return tt, npz["zscore_in"], npz["peth_time_in"]


def fit_mouse(tt, zs, peth_time, mouse):
    idx = tt.index[tt.mouse == mouse].to_numpy()
    sub_tt = tt.loc[idx].reset_index(drop=True)
    sub_zs = zs[idx]
    return fit_time_resolved_glm(sub_zs, peth_time, sub_tt, formula=FORMULA, min_retained_frac=0.5)


def paired_diffs(root, terms=CLEAN_TERMS, mice=PAIRED_MICE):
    """Returns dict(term -> (n_mice, n_timepoints) array of old-young diffs), and peth_time."""
    tt_young, zs_young, peth_time = load_pooled(root, "young")
    tt_old, zs_old, peth_time_old = load_pooled(root, "old")
    assert np.array_equal(peth_time, peth_time_old), "young/old peth_time grids differ"

    diffs = {term: [] for term in terms}
    for mouse in mice:
        fit_young = fit_mouse(tt_young, zs_young, peth_time, mouse)
        fit_old = fit_mouse(tt_old, zs_old, peth_time, mouse)
        for term in terms:
            col = f"C(word_l2)[T.{term}]_beta"
            diffs[term].append(fit_old[col].to_numpy() - fit_young[col].to_numpy())
    return {term: np.array(v) for term, v in diffs.items()}, peth_time


def cluster_mass_from_t(t_vals, threshold):
    """Largest contiguous |t|-above-threshold run's sum(|t|); 0.0 if none.
    NaN timepoints (degenerate/truncated fit) never count as suprathreshold.
    """
    supra = np.where(np.isfinite(t_vals), np.abs(t_vals), 0.0) > threshold
    best = 0.0
    run = 0.0
    for is_supra, t in zip(supra, np.nan_to_num(t_vals, nan=0.0)):
        if is_supra:
            run += abs(t)
            best = max(best, run)
        else:
            run = 0.0
    return best


def paired_t(diff_matrix):
    """diff_matrix: (n_mice, n_timepoints) -> per-timepoint paired t (nan-safe)."""
    n = diff_matrix.shape[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(diff_matrix, axis=0)
        sd = np.nanstd(diff_matrix, axis=0, ddof=1)
        n_valid = np.sum(np.isfinite(diff_matrix), axis=0)
        t = mean / (sd / np.sqrt(n_valid))
    t[n_valid < n] = np.nan  # any mouse NaN at this timepoint (fit degenerate) -> don't test it
    return t


def cluster_permutation_test(diff_matrix, threshold=CLUSTER_FORMING_T):
    n_mice = diff_matrix.shape[0]
    observed_t = paired_t(diff_matrix)
    observed_mass = cluster_mass_from_t(observed_t, threshold)

    null_masses = []
    for signs in product([1, -1], repeat=n_mice):
        flipped = diff_matrix * np.array(signs)[:, None]
        t = paired_t(flipped)
        null_masses.append(cluster_mass_from_t(t, threshold))
    null_masses = np.array(null_masses)

    p = float(np.mean(null_masses >= observed_mass))
    return observed_mass, p, observed_t


def main():
    root_name = sys.argv[1] if len(sys.argv) > 1 else "green_l"
    root = Path(f"outputs_fixed/model_series_comparison_sm_age_split{'_' + root_name if root_name != 'green_l' else ''}")
    print(f"Cluster permutation test on {root}, paired mice={PAIRED_MICE}")

    diffs, peth_time = paired_diffs(root)
    print(f"{len(PAIRED_MICE)} paired mice, {len(peth_time)} timepoints, "
          f"cluster-forming |t|>{CLUSTER_FORMING_T}, exact 2^{len(PAIRED_MICE)}=16-permutation null\n")

    rows = []
    for term, diff_matrix in diffs.items():
        mass, p, t = cluster_permutation_test(diff_matrix)
        supra = np.isfinite(t) & (np.abs(t) > CLUSTER_FORMING_T)
        if supra.any():
            t_range = f"[{peth_time[supra].min():.2f}, {peth_time[supra].max():.2f}]s"
        else:
            t_range = "none"
        rows.append(dict(term=term, hemisphere=root_name, max_cluster_mass=mass, p_value=p,
                          suprathreshold_range=t_range, n_paired_mice=len(PAIRED_MICE),
                          cluster_forming_t=CLUSTER_FORMING_T, min_achievable_p=1 / 2 ** len(PAIRED_MICE)))
        print(f"  {term}: max cluster mass={mass:.2f}  p={p:.4f}  suprathreshold range~{t_range}")

    results_df = pd.DataFrame(rows)
    out_path = root / f"cluster_permutation_word_l2_results_{root_name}.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    return results_df


if __name__ == "__main__":
    main()
