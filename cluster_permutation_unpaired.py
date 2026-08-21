"""
Unpaired (independent-groups) counterpart to cluster_permutation_word_l2.py's
paired sign-flip test -- for comparing two DIFFERENT sets of mice (e.g.
FP1_none's 7 mice vs FP2_none's 4 mice, different animals, not the same
mice tracked across age like SM's young/old split), where there is no
natural pairing to sign-flip.

Test statistic per timepoint: Welch's t (unequal-n, unequal-variance) on
per-mouse beta(t) between the two groups. Cluster mass = largest contiguous
run of |t| above CLUSTER_FORMING_T, summed. Null: exact enumeration of
every way to relabel the pooled n_a+n_b mice into groups of size n_a/n_b
(C(n_a+n_b, n_a) permutations) -- exact given the group sizes are small
enough to enumerate fully (e.g. C(11,4)=330 for 7-vs-4).

Usage (as a library -- see plot_fp_condition_comparison.py for the caller):
    from cluster_permutation_unpaired import unpaired_cluster_permutation_test
"""

from itertools import combinations

import numpy as np

from cluster_permutation_word_l2 import CLUSTER_FORMING_T, cluster_mass_from_t


def welch_t(group_a, group_b):
    """group_a/group_b: (n_mice, n_timepoints) beta(t) arrays -> per-timepoint Welch's t."""
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_a, mean_b = np.nanmean(group_a, axis=0), np.nanmean(group_b, axis=0)
        var_a, var_b = np.nanvar(group_a, axis=0, ddof=1), np.nanvar(group_b, axis=0, ddof=1)
        n_a = np.sum(np.isfinite(group_a), axis=0)
        n_b = np.sum(np.isfinite(group_b), axis=0)
        se = np.sqrt(var_a / n_a + var_b / n_b)
        t = (mean_a - mean_b) / se
    t[(n_a < group_a.shape[0]) | (n_b < group_b.shape[0])] = np.nan  # any mouse NaN at this t -> don't test
    return t


def unpaired_cluster_permutation_test(beta_a, beta_b, threshold=CLUSTER_FORMING_T, max_exact=5000):
    """beta_a: (n_a, n_timepoints), beta_b: (n_b, n_timepoints) -- per-mouse beta(t) traces
    for group A and group B respectively (rows are DIFFERENT mice, not paired).

    Returns (observed_mass, p_value, observed_t, n_permutations_used).
    Enumerates all C(n_a+n_b, n_a) group-label permutations if that count is
    <= max_exact (exact test); otherwise raises, since this project's group
    sizes (7-vs-4, 11 total) are always small enough to enumerate exactly
    and an approximate Monte Carlo fallback would silently change the
    test's meaning without the caller asking for it.
    """
    n_a, n_b = beta_a.shape[0], beta_b.shape[0]
    n_total = n_a + n_b
    from math import comb
    n_perms = comb(n_total, n_a)
    if n_perms > max_exact:
        raise ValueError(
            f"C({n_total},{n_a})={n_perms} exceeds max_exact={max_exact} -- exact enumeration "
            "not attempted; this project's group sizes should never hit this"
        )

    pooled = np.concatenate([beta_a, beta_b], axis=0)
    observed_t = welch_t(beta_a, beta_b)
    observed_mass = cluster_mass_from_t(observed_t, threshold)

    null_masses = []
    for a_idx in combinations(range(n_total), n_a):
        a_idx = set(a_idx)
        b_idx = [i for i in range(n_total) if i not in a_idx]
        a_idx = list(a_idx)
        t = welch_t(pooled[a_idx], pooled[b_idx])
        null_masses.append(cluster_mass_from_t(t, threshold))
    null_masses = np.array(null_masses)

    p = float(np.mean(null_masses >= observed_mass))
    return observed_mass, p, observed_t, n_perms
