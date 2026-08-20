"""
Event-aligned window extraction -- the Python equivalent of extractMatrix.m.
"""

import numpy as np

from config.params import ALIGN_EVENT_COLUMNS, FINAL_SAMPLE_FREQ_HZ


def get_event_indices(trial_table, align_event):
    """Look up the photometry-clock sample index column for one of the 4
    primary trial events ('center_in', 'side_in', 'outcome', 'side_out';
    see config.params.ALIGN_EVENT_COLUMNS), as computed by
    behavior.sync.align_behavior_to_photometry.

    Returned indices of -1 mark trials where that event has no valid
    photometry-clock match (e.g. it falls outside the recorded envelope,
    or -- for side_out -- the session ended before the animal left the
    port); callers should filter those out before extract_peth, same as
    the existing photometry_side_in_index >= 0 convention.
    """
    if align_event not in ALIGN_EVENT_COLUMNS:
        raise ValueError(f"Unknown align_event {align_event!r}; must be one of {list(ALIGN_EVENT_COLUMNS)}")
    return trial_table[ALIGN_EVENT_COLUMNS[align_event]].to_numpy()


def extract_peth(signal, event_indices, pre_samples, post_samples):
    """Fixed-width window extraction around each event index (typically
    from get_event_indices) -- alignment-agnostic: event_indices can come
    from any of the 4 primary trial events, since this function only ever
    sees plain sample indices into `signal`. Drops any window that would
    run off either end of the trace.
    """
    event_indices = np.asarray(event_indices)
    valid = (event_indices - pre_samples >= 0) & (event_indices + post_samples < len(signal))
    event_indices = event_indices[valid]
    windows = np.stack([signal[i - pre_samples : i + post_samples + 1] for i in event_indices])
    return windows


def compute_event_aligned_zscore(
    windows,
    peth_time,
    baseline_pre_s,
    baseline_post_s,
    min_baseline_std=1e-8,
    return_baseline_stats=False,
):
    """Trial-by-trial z-score of PETH windows against each trial's OWN
    pre-event baseline, i.e. for each trial/row independently:

        Z_trial(t) = (F(t) - mu_baseline) / sigma_baseline

    where mu_baseline/sigma_baseline are computed from that same trial's
    samples falling in [baseline_pre_s, baseline_post_s] (seconds relative
    to the aligning event at t=0 in `peth_time`).

    This is distinct from the CONTINUOUS rolling z-score in
    preprocessing.demodulate.compute_dff_and_zscore, which normalizes
    against a session-wide rolling window and feeds GLMs/state classifiers.
    This trial-level version exists specifically for PETHs and per-trial
    event plots, where each trial should be judged against its own
    immediate pre-event activity rather than a session-wide baseline.

    Already alignment-agnostic w.r.t. the 4 primary trial events (center_in/
    side_in/outcome/side_out, see config.params.ALIGN_EVENT_COLUMNS): it only
    ever sees `windows` (already extracted around whichever event) and
    `peth_time` (seconds relative to that event at t=0), so no align_event
    parameter is needed here -- get_event_indices/extract_peth are where the
    event choice actually happens.

    windows : (n_trials, n_samples) array, e.g. output of extract_peth.
    peth_time : (n_samples,) array of time offsets (seconds) matching the
        window's columns, with 0 = the aligning event.
    min_baseline_std : floor applied to sigma_baseline before dividing, to
        avoid exploding/undefined z-scores on trials with a near-flat
        baseline (sigma ~= 0). Trials hitting this floor are reported in
        the returned stats when return_baseline_stats=True.
    """
    windows = np.asarray(windows, dtype=float)
    peth_time = np.asarray(peth_time)

    baseline_mask = (peth_time >= baseline_pre_s) & (peth_time <= baseline_post_s)
    if not baseline_mask.any():
        raise ValueError(
            f"No PETH samples fall inside the baseline window "
            f"[{baseline_pre_s}, {baseline_post_s}] s -- check peth_time spacing."
        )

    baseline = windows[:, baseline_mask]
    mu = baseline.mean(axis=1, keepdims=True)
    sigma = baseline.std(axis=1, ddof=1, keepdims=True)

    degenerate = (sigma < min_baseline_std) | ~np.isfinite(sigma)
    sigma_safe = np.where(degenerate, min_baseline_std, sigma)

    z = (windows - mu) / sigma_safe

    if return_baseline_stats:
        stats = dict(
            baseline_mean=mu.squeeze(axis=1),
            baseline_std=sigma.squeeze(axis=1),
            n_degenerate_trials=int(degenerate.sum()),
        )
        return z, stats
    return z


def compute_per_trial_event_metrics(
    dff, trial_table, align_event, pre_samples, post_samples, peth_time,
    baseline_pre_s, baseline_post_s, metric_window_s,
):
    """Per-trial peak z-score and AUC in `metric_window_s` (seconds relative
    to the event, e.g. config.params.DECISION_WINDOW_S) after `align_event`,
    computed for EVERY trial in trial_table -- not just the subset with a
    valid photometry window for that event -- so the result stays row-
    aligned with the full (unfiltered) trial_table and can be assigned back
    as a plain new column, same convention as add_word_labels/
    add_lag_features/add_bandit_state_features. Trials where the event has
    no valid index (get_event_indices == -1) or falls outside the recorded
    envelope get NaN.

    Returns (peak, auc), each a (len(trial_table),) float array.
    """
    event_idx = get_event_indices(trial_table, align_event)
    n_trials = len(trial_table)
    peak = np.full(n_trials, np.nan)
    auc = np.full(n_trials, np.nan)

    valid = (event_idx >= 0) & (event_idx - pre_samples >= 0) & (event_idx + post_samples < len(dff))
    if not valid.any():
        return peak, auc

    windows = extract_peth(dff, event_idx[valid], pre_samples, post_samples)
    z = compute_event_aligned_zscore(windows, peth_time, baseline_pre_s, baseline_post_s)

    metric_mask = (peth_time >= metric_window_s[0]) & (peth_time <= metric_window_s[1])
    dx = float(peth_time[1] - peth_time[0])
    peak[valid] = z[:, metric_mask].max(axis=1)
    auc[valid] = np.trapz(z[:, metric_mask], dx=dx, axis=1)
    return peak, auc


def truncate_windows_after_side_out(windows, peth_time, trial_table,
                                     align_col="photometry_side_in_index",
                                     side_out_col="photometry_side_out_index",
                                     margin_s=0.0):
    """NaN out each trial's own post-side_out samples in an already-extracted,
    already-baselined PETH window array -- e.g. a side_in-aligned window
    currently runs a fixed PETH_POST_SEC past the event regardless of when
    that trial's animal actually left the port, so a "sustained" response
    late in the window can't be distinguished from post-departure/return-to-
    center activity. Applying this AFTER compute_event_aligned_zscore (not
    before) matters: baseline normalization needs the intact pre-event data,
    and truncating raw dff first would corrupt it for no reason since only
    the post-event tail is ever affected.

    windows : (n_trials, n_samples) array, e.g. output of extract_peth or
        compute_event_aligned_zscore -- row-aligned with `trial_table`.
    trial_table : row-aligned with `windows` (e.g. extract_event_peth's
        peth_trial_table), must have both `align_col` and `side_out_col`.
    margin_s : added to each trial's own dwell time before truncating, in
        case a small buffer past side_out is wanted instead of an exact cut.

    Returns a NEW float array (never mutates `windows` in place); samples
    after a trial's own (side_out - align) latency + margin_s become NaN.
    Downstream per-timepoint fits (models.glm_encoding.fit_time_resolved_glm's
    smf.ols(..., missing="drop")) already drop NaN rows cleanly per timepoint
    -- no changes needed there.
    """
    windows = np.asarray(windows, dtype=float)
    peth_time = np.asarray(peth_time)

    align_idx = trial_table[align_col].to_numpy()
    side_out_idx = trial_table[side_out_col].to_numpy()
    latency_s = (side_out_idx - align_idx) / FINAL_SAMPLE_FREQ_HZ

    # peth_time[None, :] broadcasts against latency_s[:, None] -- one keep-mask row per trial.
    keep = peth_time[None, :] <= (latency_s[:, None] + margin_s)
    return np.where(keep, windows, np.nan)
