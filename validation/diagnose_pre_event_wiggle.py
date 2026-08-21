"""
Does the pre-event (peth_time < 0) dip-peak-dip pattern in word_l2's pooled
beta/R^2 trace (seen in figures_fixed_model_series_fp2_none_truncated/
encoding_2b_word_l2_side_in.png, and confirmed present in the untruncated
figures_fixed_model_series/ version too -- not a truncation artifact)
reflect contamination from the PREVIOUS trial's own outcome-locked response
bleeding into this trial's pre-event window?

This task has no hardware inter-trial interval (established in
validation/KINETICS_VALIDATION_REPORT.md for the POST-event direction);
here we test the mirror-image question for the PRE-event side. Two
behavioral-timing candidates were already ruled out by a direct check
against the pooled trial table (medians nowhere near the wiggle's -1.7s to
-0.3s span): approach_latency_s (center_out -> side_in, median 0.32s) and
center_dwell_s (center_in -> center_out, median 0.05s). The remaining
candidate is prev_gap_s (previous trial's own side_in -> this trial's
side_in, median 2.7s).

Primary test (no duration assumption needed): split trials into
short-vs-long prev_gap_s terciles and fit the SAME word_l2 model
independently on each, on their INTACT (uncensored) pre-event windows. If
the wiggle is previous-trial contamination, the short-gap group (more
recent, less-decayed previous response) should show a stronger/different
pre-event pattern than the long-gap group (previous response had more time
to decay before this trial even started). If the two groups look the same,
that's evidence against the contamination hypothesis.

Secondary test: an explicit censoring version (NaN out peth_time in
[-prev_gap_s, min(0, -prev_gap_s + max_duration_s))) requires ASSUMING a
contamination duration -- an unbounded version (any positive elapsed time
since the previous outcome counts) was tried first and is wrong: since
median prev_gap_s (2.7s) already exceeds the window's own 2.16s pre-event
span, it flags nearly the ENTIRE pre-event window as contaminated for the
majority (77%) of trials, crashing effective N to ~38/10835 trials near
t=0 -- testing "contamination never decays," not the actual hypothesis.
Reported here at a few different plausible durations rather than one
arbitrary choice, with effective N always shown alongside.

Usage:
    python validation/diagnose_pre_event_wiggle.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.glm_encoding import fit_time_resolved_glm  # noqa: E402

DATA_DIR = Path("outputs_fixed/model_series_comparison_fp2_none_truncated/results")
OUT_DIR = Path("validation/pre_event_wiggle_check")
FORMULA = "Z ~ C(word_l2)"
CENSOR_DURATIONS_S = (1.0, 1.75, 3.0)


def compute_prev_gap_s(trial_table):
    """Returns prev_gap_s as a numpy array in trial_table's ORIGINAL row
    order (same order as the row-aligned zscore_windows array) -- sorts by
    (mouse, date, side_in_s) only to compute the shift, then maps back.
    """
    tt = trial_table.copy()
    tt["_orig_pos"] = np.arange(len(tt))
    tt_sorted = tt.sort_values(["mouse", "date", "side_in_s"])
    tt_sorted["prev_side_in_s"] = tt_sorted.groupby(["mouse", "date"])["side_in_s"].shift(1)
    tt_sorted["prev_gap_s"] = tt_sorted["side_in_s"] - tt_sorted["prev_side_in_s"]
    tt_restored = tt_sorted.sort_values("_orig_pos")
    return tt_restored["prev_gap_s"].to_numpy()


def censor_prev_trial_contamination(zscore_windows, peth_time, prev_gap_s, max_duration_s):
    """NaN out, per trial, peth_time in [-prev_gap_s, min(0, -prev_gap_s + max_duration_s))
    -- the span after the previous trial's own outcome, bounded to at most
    max_duration_s of assumed response duration, and before this trial's own
    side_in. Trials with prev_gap_s NaN (first trial of a session) get no
    censoring.
    """
    zscore_windows = np.asarray(zscore_windows, dtype=float).copy()
    peth_time = np.asarray(peth_time)
    prev_gap_s = np.asarray(prev_gap_s, dtype=float)

    contaminated_lo = -prev_gap_s
    contaminated_hi = np.minimum(0.0, -prev_gap_s + max_duration_s)
    mask = (peth_time[None, :] >= contaminated_lo[:, None]) & (peth_time[None, :] < contaminated_hi[:, None])
    mask = np.where(np.isnan(contaminated_lo)[:, None], False, mask)
    zscore_windows[mask] = np.nan
    return zscore_windows, mask


def fit_and_summarize(zscore_windows, peth_time, trial_table, label):
    """Restricted to peth_time < 0 -- this data may come from an already
    side_out-truncated pooled cache (post-event NaNs from a DIFFERENT,
    unrelated mechanism), and a small subgroup can hit a late post-event
    timepoint where every remaining row is NaN (statsmodels then crashes on
    an empty design matrix rather than the graceful min_resid_dof path,
    which only guards a LOW-but-nonzero nobs, not exactly zero). Restricting
    to pre-event timepoints avoids that entirely and is also all this test
    needs.
    """
    pre_mask = peth_time < 0
    beta = fit_time_resolved_glm(zscore_windows[:, pre_mask], peth_time[pre_mask], trial_table, formula=FORMULA)
    print(f"  [{label}] pre-event R^2 range: {beta['r_squared'].to_numpy().min():.4f} - "
          f"{np.nanmax(beta['r_squared'].to_numpy()):.4f}, "
          f"n_trials range: {int(beta['n_trials'].to_numpy().min())}-"
          f"{int(beta['n_trials'].to_numpy().max())}")
    return beta


def main():
    trial_table = pd.read_parquet(DATA_DIR / "pooled_trial_table_in.parquet")
    z = np.load(DATA_DIR / "pooled_zscore_windows.npz")
    zscore_in, peth_time_in = z["zscore_in"], z["peth_time_in"]
    assert len(trial_table) == zscore_in.shape[0], "trial_table/zscore_in must be row-aligned"

    prev_gap_s = compute_prev_gap_s(trial_table)
    n_no_prev = int(np.isnan(prev_gap_s).sum())
    valid_gap = prev_gap_s[~np.isnan(prev_gap_s)]
    print(f"{len(trial_table)} trials total; {n_no_prev} with no previous trial (session start)")
    print(f"prev_gap_s: median={np.median(valid_gap):.3f}s, "
          f"terciles={np.percentile(valid_gap, [33.3, 66.7]).round(3)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Primary test: split by prev_gap_s tercile, no duration assumption ---
    print("\n=== PRIMARY TEST: short-gap vs long-gap tercile split (uncensored data) ===")
    t1, t2 = np.percentile(valid_gap, [33.3, 66.7])
    short_mask = prev_gap_s <= t1
    long_mask = prev_gap_s >= t2
    print(f"short-gap group (prev_gap_s <= {t1:.3f}s): {int(short_mask.sum())} trials")
    print(f"long-gap group (prev_gap_s >= {t2:.3f}s): {int(long_mask.sum())} trials")

    beta_short = fit_and_summarize(zscore_in[short_mask], peth_time_in, trial_table.loc[short_mask].reset_index(drop=True), "short-gap")
    beta_long = fit_and_summarize(zscore_in[long_mask], peth_time_in, trial_table.loc[long_mask].reset_index(drop=True), "long-gap")
    beta_short.to_csv(OUT_DIR / "beta_short_gap.csv")
    beta_long.to_csv(OUT_DIR / "beta_long_gap.csv")

    tercile_comparison = pd.DataFrame({
        "peth_time": beta_short.index.to_numpy(),
        "r2_short_gap": beta_short["r_squared"].to_numpy(),
        "n_short_gap": beta_short["n_trials"].to_numpy(),
        "r2_long_gap": beta_long["r_squared"].to_numpy(),
        "n_long_gap": beta_long["n_trials"].to_numpy(),
    })
    print(tercile_comparison.to_string(index=False))
    tercile_comparison.to_csv(OUT_DIR / "tercile_split_comparison.csv", index=False)

    # --- Secondary test: bounded censoring at a few plausible durations ---
    print("\n=== SECONDARY TEST: bounded censoring at several assumed contamination durations ===")
    beta_orig = fit_and_summarize(zscore_in, peth_time_in, trial_table, "uncensored (original)")
    beta_orig.to_csv(OUT_DIR / "beta_orig.csv")

    censored_betas = {}
    for dur in CENSOR_DURATIONS_S:
        zscore_censored, censor_mask = censor_prev_trial_contamination(zscore_in, peth_time_in, prev_gap_s, dur)
        beta_c = fit_and_summarize(zscore_censored, peth_time_in, trial_table, f"censored @ {dur}s")
        censored_betas[dur] = beta_c
        beta_c.to_csv(OUT_DIR / f"beta_censored_{dur}s.csv")

    comparison = pd.DataFrame({"peth_time": beta_orig.index.to_numpy(),
                                "r2_orig": beta_orig["r_squared"].to_numpy(),
                                "n_orig": beta_orig["n_trials"].to_numpy()})
    for dur, beta_c in censored_betas.items():
        comparison[f"r2_censored_{dur}s"] = beta_c["r_squared"].to_numpy()
        comparison[f"n_censored_{dur}s"] = beta_c["n_trials"].to_numpy()
    print(comparison.to_string(index=False))
    comparison.to_csv(OUT_DIR / "censored_duration_sweep.csv", index=False)

    print(f"\nSaved all outputs to {OUT_DIR}/")


if __name__ == "__main__":
    main()
