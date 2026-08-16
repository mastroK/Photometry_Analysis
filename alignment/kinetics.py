"""
Onset-latency and decay-time-constant extraction from trial-averaged,
event-aligned PETHs -- alongside the existing per-trial peak/AUC metrics in
alignment.windowing.compute_per_trial_event_metrics.

Why these are computed from trial-AVERAGED traces, unlike peak/AUC: peak and
AUC are a max and a numeric integral, both robust to single-trial noise.
Onset latency and decay tau require fitting a rise/fall SHAPE, which
individual dF/F trials generally don't have enough SNR for. So these are
computed per (mouse, condition) group -- average that group's per-trial
z-scored PETH windows into one mean trace, then measure onset/decay on the
mean trace.

Definitions
-----------
Onset latency: time from the aligned event (t=0) to when the trial-averaged
z-scored trace first reaches `onset_fraction` (default 0.5, i.e.
time-to-half-max) of its own peak amplitude within `metric_window_s`.
Sub-sample precision via linear interpolation between the bracketing
samples.

Decay time constant (tau): fit a single-exponential decay,
    z(t) = A * exp(-(t - t_peak) / tau) + offset
to the trace from its peak time to the end of `metric_window_s`, or until it
returns to within DECAY_RETURN_TO_BASELINE_FRAC of its own peak amplitude
(above baseline), whichever segment is SHORTER. Reported in seconds. The fit
is skipped (NaN, with a recorded reason) rather than forced when: the
post-peak segment has fewer than DECAY_MIN_POST_PEAK_SAMPLES samples,
scipy.optimize.curve_fit fails to converge, or the fitted tau is
non-positive or exceeds DECAY_MAX_TAU_RATIO x the fitted segment's own
length (implausible).

Sign handling: onset/decay are only meaningful for a trace whose response is
POSITIVE-going in `metric_window_s` -- do not take an absolute value of the
whole trace first (that would conflate a positive-going and negative-going
response into a fabricated "peak"). compute_group_kinetics handles this by
(a) grouping by condition (e.g. reward vs. omission) rather than pooling
opposite-signed responses together, AND (b) orienting each group's own mean
trace by the sign of its mean value in metric_window_s, recording whether it
flipped -- both of the approaches the spec allowed, applied together for
robustness against per-mouse sign noise.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from config.params import (
    DECAY_MAX_TAU_RATIO,
    DECAY_MIN_POST_PEAK_SAMPLES,
    DECAY_RETURN_TO_BASELINE_FRAC,
    ONSET_FRACTION,
)


def _exp_decay_model(t_rel, amplitude, tau, offset):
    """t_rel = t - t_peak, so at t_rel=0 the model equals amplitude + offset
    (anchored at the trace's own peak)."""
    return amplitude * np.exp(-t_rel / tau) + offset


def compute_onset_and_decay(
    mean_trace, peth_time, metric_window_s, onset_fraction=ONSET_FRACTION,
    return_to_baseline_frac=DECAY_RETURN_TO_BASELINE_FRAC,
    min_post_peak_samples=DECAY_MIN_POST_PEAK_SAMPLES,
    max_tau_ratio=DECAY_MAX_TAU_RATIO,
):
    """Onset latency + decay tau for ONE trial-averaged trace, already
    oriented so its response is positive-going in metric_window_s (see
    compute_group_kinetics for the grouping/orientation step -- this
    function itself does no sign handling).

    mean_trace, peth_time : (n_samples,) arrays, same convention as
        alignment.windowing.compute_event_aligned_zscore's output (0 =
        aligning event; trace already baseline-z-scored, so its own
        baseline is ~0 and onset threshold is measured directly against 0,
        not a separately-fitted offset).
    metric_window_s : (start, end) seconds relative to the event, e.g.
        config.params.DECISION_WINDOW_S.

    Returns (onset_latency_s, decay_tau_s, fit_r_squared, diagnostics) --
    diagnostics is a dict always populated (peak_time_s, peak_value,
    n_post_peak_samples, skip_reason [None if the decay fit succeeded]) so
    callers can report skip rates rather than silently losing them.
    """
    mean_trace = np.asarray(mean_trace, dtype=float)
    peth_time = np.asarray(peth_time, dtype=float)
    window_mask = (peth_time >= metric_window_s[0]) & (peth_time <= metric_window_s[1])
    t_win, y_win = peth_time[window_mask], mean_trace[window_mask]

    diagnostics = dict(peak_time_s=np.nan, peak_value=np.nan, n_post_peak_samples=0, skip_reason=None)

    if len(t_win) < 3:
        diagnostics["skip_reason"] = "metric_window_s contains fewer than 3 samples"
        return np.nan, np.nan, np.nan, diagnostics

    peak_idx = int(np.argmax(y_win))
    peak_val, peak_time = float(y_win[peak_idx]), float(t_win[peak_idx])
    diagnostics["peak_time_s"] = peak_time
    diagnostics["peak_value"] = peak_val

    if peak_val <= 0:
        diagnostics["skip_reason"] = "peak_value <= 0 after sign orientation -- no positive-going response to measure"
        return np.nan, np.nan, np.nan, diagnostics

    # --- onset latency: first crossing of onset_fraction * peak, searched
    # from the window start up to (and including) the peak ---------------
    threshold = onset_fraction * peak_val
    above = y_win[: peak_idx + 1] >= threshold
    if not above.any():
        # Shouldn't happen (the peak sample itself is >= threshold whenever
        # onset_fraction <= 1), but guard rather than crash on edge cases.
        diagnostics["skip_reason"] = "no sample reached onset_fraction * peak before the peak"
        onset_latency_s = np.nan
    else:
        k = int(np.argmax(above))  # first True
        if k == 0:
            onset_latency_s = float(t_win[0])
            diagnostics["onset_note"] = "already at/above threshold at window start -- reporting window start"
        else:
            t0, t1 = t_win[k - 1], t_win[k]
            y0, y1 = y_win[k - 1], y_win[k]
            onset_latency_s = float(t0 + (threshold - y0) * (t1 - t0) / (y1 - y0)) if y1 != y0 else float(t1)

    # --- decay tau: fit from the peak to the shorter of (window end) or
    # (first return to within return_to_baseline_frac * peak) -------------
    post_peak_mask = np.arange(len(t_win)) >= peak_idx
    post_t, post_y = t_win[post_peak_mask], y_win[post_peak_mask]

    return_thresh = return_to_baseline_frac * peak_val
    below = post_y <= return_thresh
    if below.any():
        return_idx = int(np.argmax(below))  # first True, relative to post_t/post_y
        end_idx = return_idx + 1  # inclusive of the first below-threshold sample
    else:
        end_idx = len(post_t)

    decay_t, decay_y = post_t[:end_idx], post_y[:end_idx]
    n_post_peak = len(decay_t)
    diagnostics["n_post_peak_samples"] = n_post_peak

    if n_post_peak < min_post_peak_samples:
        diagnostics["skip_reason"] = (
            f"only {n_post_peak} post-peak sample(s), fewer than "
            f"DECAY_MIN_POST_PEAK_SAMPLES={min_post_peak_samples}"
        )
        return onset_latency_s, np.nan, np.nan, diagnostics

    t_rel = decay_t - peak_time
    segment_length = float(t_rel[-1] - t_rel[0])
    tau0 = max(segment_length / 2.0, 1e-3)
    p0 = [peak_val - decay_y[-1], tau0, decay_y[-1]]

    try:
        popt, _ = curve_fit(
            _exp_decay_model, t_rel, decay_y, p0=p0, maxfev=5000,
            bounds=([-np.inf, 1e-4, -np.inf], [np.inf, np.inf, np.inf]),
        )
    except RuntimeError as exc:
        diagnostics["skip_reason"] = f"curve_fit did not converge: {exc}"
        return onset_latency_s, np.nan, np.nan, diagnostics

    amplitude_fit, tau_fit, offset_fit = popt
    if not np.isfinite(tau_fit) or tau_fit <= 0:
        diagnostics["skip_reason"] = f"fitted tau={tau_fit:.4g} is non-positive or non-finite"
        return onset_latency_s, np.nan, np.nan, diagnostics
    if tau_fit > max_tau_ratio * segment_length:
        diagnostics["skip_reason"] = (
            f"fitted tau={tau_fit:.4g}s exceeds DECAY_MAX_TAU_RATIO ({max_tau_ratio}) x "
            f"the fitted segment length ({segment_length:.4g}s) -- implausible"
        )
        return onset_latency_s, np.nan, np.nan, diagnostics

    pred = _exp_decay_model(t_rel, *popt)
    ss_res = float(np.sum((decay_y - pred) ** 2))
    ss_tot = float(np.sum((decay_y - decay_y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return onset_latency_s, float(tau_fit), r_squared, diagnostics


def compute_group_kinetics(
    zscore_windows, trial_table, peth_time, metric_window_s,
    group_cols=("mouse", "was_rewarded"), onset_fraction=ONSET_FRACTION,
    min_trials_per_group=5,
):
    """Group per-trial z-scored PETH windows by group_cols (default: mouse x
    reward/omission -- computed separately per the module docstring's sign
    handling), average into one mean trace per group, orient that mean
    trace's sign from its own mean value in metric_window_s, and run
    compute_onset_and_decay on it.

    zscore_windows : (n_trials, n_samples) array -- e.g.
        alignment.windowing.compute_event_aligned_zscore's output, or a
        pooled peth_windows.npz's "zscore_windows".
    trial_table : row-aligned with zscore_windows (same convention as every
        other consumer of these arrays in this pipeline -- pass the SAME,
        un-row-dropped trial_table/DataFrame that zscore_windows was built
        from, not a filtered copy with a reset index).

    Returns a DataFrame indexed by group_cols with: n_trials, sign_flipped,
    onset_latency_s, decay_tau_s, fit_r_squared, peak_time_s, peak_value,
    n_post_peak_samples, skip_reason (None where the decay fit succeeded).
    """
    if len(trial_table) != len(zscore_windows):
        raise ValueError(
            f"trial_table has {len(trial_table)} rows but zscore_windows has "
            f"{len(zscore_windows)} -- both must be row-aligned (pass the same "
            f"unfiltered trial_table zscore_windows was built from)"
        )

    window_mask = (peth_time >= metric_window_s[0]) & (peth_time <= metric_window_s[1])

    rows = []
    for group_key, g in trial_table.groupby(list(group_cols)):
        idx = g.index.to_numpy()
        n_trials = len(idx)
        key = group_key if isinstance(group_key, tuple) else (group_key,)
        row = dict(zip(group_cols, key))
        row["n_trials"] = n_trials

        if n_trials < min_trials_per_group:
            row.update(
                sign_flipped=np.nan, onset_latency_s=np.nan, decay_tau_s=np.nan,
                fit_r_squared=np.nan, peak_time_s=np.nan, peak_value=np.nan,
                n_post_peak_samples=0,
                skip_reason=f"only {n_trials} trial(s), fewer than min_trials_per_group={min_trials_per_group}",
            )
            rows.append(row)
            continue

        mean_trace = zscore_windows[idx].mean(axis=0)
        mean_in_window = float(mean_trace[window_mask].mean())
        sign = 1.0 if mean_in_window >= 0 else -1.0
        oriented_trace = mean_trace * sign

        onset, tau, r2, diag = compute_onset_and_decay(
            oriented_trace, peth_time, metric_window_s, onset_fraction=onset_fraction,
        )
        row.update(
            sign_flipped=(sign < 0), onset_latency_s=onset, decay_tau_s=tau, fit_r_squared=r2,
            peak_time_s=diag["peak_time_s"], peak_value=diag["peak_value"],
            n_post_peak_samples=diag["n_post_peak_samples"], skip_reason=diag["skip_reason"],
        )
        rows.append(row)

    return pd.DataFrame(rows).set_index(list(group_cols))
