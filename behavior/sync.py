"""
Behavior-clock <-> photometry-clock alignment. Direct translation of
processBehavior.m:56-186.
"""

import numpy as np

from behavior.trial_table import poke_times_seconds
from config.params import (
    CH_CENTERPORT,
    CH_LEFTPORT,
    CH_RIGHTPORT,
    FINAL_TIME_STEP_SEC,
    HOP_SAMPLES,
    RAW_SAMPLE_FREQ_HZ,
    XCORR_ACCEPT_THRESHOLD,
    XCORR_MAX_LAG_POKES,
)


def matlab_style_xcorr(x, y, max_lag):
    """MATLAB `xcorr(x, y, maxlag)` -> (lags, values) for lags -maxlag..maxlag.
    c(lag) = sum_n x(n+lag) * y(n)   (zero outside the overlap, like MATLAB's default 'none' scaling)
    """
    full = np.correlate(x, y, mode="full")  # zero-lag sits at index len(y)-1
    zero_lag_idx = len(y) - 1
    lags = np.arange(-max_lag, max_lag + 1)
    values = np.full(len(lags), np.nan)
    for i, lag in enumerate(lags):
        pos = zero_lag_idx + lag
        if 0 <= pos < len(full):
            values[i] = full[pos]
    return lags, values


def _final_rate_edges(raw, channel, n_final_samples, hop_samples=HOP_SAMPLES, edge=1):
    """processed.behavior.risingEdge equivalent (processNew.m:550-569),
    generalized to either rising (edge=1, poke IN) or falling (edge=-1,
    poke OUT) transitions on a single digital channel. Block-sums raw-rate
    edge flags into each ~54 ms final-rate bin.
    """
    usable_raw_samples = n_final_samples * hop_samples
    trace = raw[channel, :usable_raw_samples]
    d = np.diff(trace, prepend=trace[0]) == edge
    bins = d.reshape(n_final_samples, hop_samples).sum(axis=1) > 0
    return np.flatnonzero(bins)


def _final_rate_rising_edges(raw, n_final_samples, hop_samples=HOP_SAMPLES):
    """Rightport / Leftport rising edges (poke IN), for side-in alignment."""
    right_bins = _final_rate_edges(raw, CH_RIGHTPORT, n_final_samples, hop_samples, edge=1)
    left_bins = _final_rate_edges(raw, CH_LEFTPORT, n_final_samples, hop_samples, edge=1)
    return right_bins, left_bins


def align_behavior_to_photometry(raw, trial_table, poke_history, n_final_samples):
    """Replicates processBehavior.m:58-186.

    Returns (trial_table, align_info); trial_table gains a
    `photometry_side_in_index` column (0-based index into the FINAL
    ~18.52 Hz signal grid). align_info['time_shift'] is the fitted clock
    offset -- compare directly against a session's ground-truth
    `processed.params.timeShift` from the reference MATLAB output.
    """
    # --- behavior-clock center pokes -------------------------------------------------
    port_names = [p["portPoked"] for p in poke_history]
    time_poked = poke_times_seconds(poke_history)

    center_mask_b = np.array(["center" in name for name in port_names])
    center_times_b = time_poked[center_mask_b]

    # --- photometry-clock center pokes (raw-rate rising edges) ----------------------
    center_raw = raw[CH_CENTERPORT, :]
    rising = np.diff(center_raw) > 0                            # processBehavior.m:82
    center_indices_p = np.flatnonzero(rising)
    center_times_p = center_indices_p / RAW_SAMPLE_FREQ_HZ       # processBehavior.m:85

    # --- cross-correlate inter-poke-intervals to find the index shift ---------------
    dt_p = np.diff(center_times_p)
    dt_b = np.diff(center_times_b)
    lags, xc = matlab_style_xcorr(dt_b, dt_p, XCORR_MAX_LAG_POKES)  # processBehavior.m:95
    max_loc = int(np.nanargmax(xc))
    norm = np.sqrt(np.nansum(dt_b ** 2)) * np.sqrt(np.nansum(dt_p ** 2))
    max_val = xc[max_loc] / norm
    p_to_b_shift = lags[max_loc]  # processBehavior.m:101 (p_to_b_dIndex)

    print(f"  xcorr normalized peak = {max_val:.3f} (accept threshold {XCORR_ACCEPT_THRESHOLD})")
    if max_val < XCORR_ACCEPT_THRESHOLD:
        raise RuntimeError("Behavior<->photometry alignment failed xcorr threshold check")

    if p_to_b_shift >= 0:
        n_overlap = min(len(dt_p) + 1, len(dt_b) - p_to_b_shift + 1)
        center_times_p_adj = center_times_p[:n_overlap]
        center_times_b_adj = center_times_b[p_to_b_shift : p_to_b_shift + n_overlap]
    else:
        n_overlap = min(len(dt_p) + 1 + p_to_b_shift, len(dt_b))
        center_times_p_adj = center_times_p[-p_to_b_shift : -p_to_b_shift + n_overlap]
        center_times_b_adj = center_times_b[:n_overlap]

    # processBehavior.m:147-151, reported for inspection only:
    time_shift_mean = np.mean(center_times_p_adj - center_times_b_adj)
    time_shift_std = np.std(center_times_p_adj - center_times_b_adj)
    print(f"  fitted index shift = {p_to_b_shift}, mean time offset = {time_shift_mean:.4f} s (std {time_shift_std:.4f})")

    # processBehavior.m:182-186 -- the offset actually used to convert
    # behavior-clock event times into photometry-clock sample indices is the
    # PHOTOMETRY time of the first matched center poke (not the mean offset above).
    time_shift = center_times_p_adj[0]
    print(f"  time_shift (anchor) used for index conversion = {time_shift:.4f} s")

    trial_table = trial_table.copy()
    trial_table["matched_side_in_index"] = np.floor(
        (trial_table["side_in_time"] + time_shift) / FINAL_TIME_STEP_SEC
    ).astype(int)

    # --- snap to the nearest real rising edge on the final-rate grid -----------------
    # (processBehavior.m:189-231)
    right_rising_final, left_rising_final = _final_rate_rising_edges(raw, n_final_samples)
    right_falling_final = _final_rate_edges(raw, CH_RIGHTPORT, n_final_samples, edge=-1)
    left_falling_final = _final_rate_edges(raw, CH_LEFTPORT, n_final_samples, edge=-1)
    center_rising_final = _final_rate_edges(raw, CH_CENTERPORT, n_final_samples, edge=1)
    center_falling_final = _final_rate_edges(raw, CH_CENTERPORT, n_final_samples, edge=-1)

    n_trials = len(trial_table)
    photometry_side_in_index = np.full(n_trials, -1, dtype=int)
    photometry_side_out_index = np.full(n_trials, -1, dtype=int)

    for i, row in trial_table.iterrows():
        side_candidates = right_rising_final if row["chose_right"] else left_rising_final
        if len(side_candidates) > 0:
            nearest = side_candidates[np.argmin(np.abs(side_candidates - row["matched_side_in_index"]))]
            photometry_side_in_index[i] = nearest

        # side-out: the first falling edge on the SAME port the trial entered,
        # occurring at/after that trial's own side-in index (the animal must
        # still be in the port at side-in, so its exit comes later).
        if photometry_side_in_index[i] >= 0:
            falling_candidates = right_falling_final if row["chose_right"] else left_falling_final
            after = falling_candidates[falling_candidates >= photometry_side_in_index[i]]
            if len(after) > 0:
                photometry_side_out_index[i] = after[0]

    trial_table["photometry_side_in_index"] = photometry_side_in_index
    trial_table["photometry_side_out_index"] = photometry_side_out_index

    # --- center-in: offset from the already-resolved side-in index -----------------
    # Unlike side pokes (exactly one per trial, well-separated), the raw center
    # channel also fires on every incomplete/repeated center poke that never led
    # to a scored trial -- confirmed directly against this session (420 isTRIAL==1
    # center pokes vs only 361 completed trials), with >10% of consecutive raw
    # center edges under 0.2s apart, so nearest-edge snapping from a single
    # session-wide `time_shift` anchor (like side-in above) is ambiguous, and a
    # rank-matched lookup against ALL raw center pokes was tried and found to be
    # unreliable (produced a photometry-clock center-in-to-side-in interval with a
    # median of ~27s, vs. the true ~0.3s from the behavior clock). Instead, each
    # trial's own center-in is resolved relative to its OWN already-correct
    # side-in index: the behavior-clock center-to-side delta is reliable (matches
    # the task's <1s selection requirement), so converting it to samples and
    # subtracting from photometry_side_in_index gives an accurate estimate, which
    # is then snapped to the nearest real center-channel edge only if one exists
    # within a small tolerance (avoiding the dense-poke ambiguity above).
    CENTER_SNAP_TOLERANCE_BINS = 3
    dt_center_to_side_samples = np.round(
        (trial_table["side_in_time"] - trial_table["center_in_time"]) / FINAL_TIME_STEP_SEC
    ).to_numpy().astype(int)
    photometry_center_in_index = np.where(
        photometry_side_in_index >= 0, photometry_side_in_index - dt_center_to_side_samples, -1
    )
    if len(center_rising_final) > 0:
        for i, target in enumerate(photometry_center_in_index):
            if target < 0:
                continue
            nearest = center_rising_final[np.argmin(np.abs(center_rising_final - target))]
            if abs(nearest - target) <= CENTER_SNAP_TOLERANCE_BINS:
                photometry_center_in_index[i] = nearest
    trial_table["photometry_center_in_index"] = photometry_center_in_index

    # --- center-out: first falling edge on the center channel, bounded to this
    # trial's own [center_in, side_in) window -- the animal must fully leave the
    # center port before entering the side port, so a fall outside that window
    # cannot be this trial's own center-out. Searching unbounded (any fall
    # anywhere at/after center_in) was tried and found to occasionally skip
    # over a genuinely absent/undetected fall and grab a much later trial's
    # fall instead (up to 20s later, violating center_out <= side_in for ~26%
    # of trials in the WCL23/060223 validation session); bounding leaves
    # center_out unresolved (-1) for those trials instead of assigning a wrong
    # value.
    photometry_center_out_index = np.full(n_trials, -1, dtype=int)
    if len(center_falling_final) > 0:
        for i, center_in_idx in enumerate(photometry_center_in_index):
            side_in_idx = photometry_side_in_index[i]
            if center_in_idx < 0 or side_in_idx < 0:
                continue
            candidates = center_falling_final[
                (center_falling_final >= center_in_idx) & (center_falling_final < side_in_idx)
            ]
            if len(candidates) > 0:
                photometry_center_out_index[i] = candidates[0]
    trial_table["photometry_center_out_index"] = photometry_center_out_index

    # 'outcome' (reward delivery/feedback) has no independently-timestamped raw
    # signal in this rig -- see config.params.ALIGN_EVENT_COLUMNS for why this
    # reuses the side-in index rather than being a distinct computed column.
    trial_table["photometry_outcome_index"] = photometry_side_in_index

    # --- seconds-from-session-start convenience columns for all 4 events -----------
    for col, index_col in [
        ("center_in_s", "photometry_center_in_index"),
        ("center_out_s", "photometry_center_out_index"),
        ("side_in_s", "photometry_side_in_index"),
        ("side_out_s", "photometry_side_out_index"),
    ]:
        idx = trial_table[index_col].to_numpy()
        trial_table[col] = np.where(idx >= 0, idx * FINAL_TIME_STEP_SEC, np.nan)

    valid_all_four = (
        (photometry_center_in_index >= 0) & (photometry_center_out_index >= 0)
        & (photometry_side_in_index >= 0) & (photometry_side_out_index >= 0)
    )
    ordered = (
        (trial_table["center_in_s"] <= trial_table["center_out_s"])
        & (trial_table["center_out_s"] <= trial_table["side_in_s"])
        & (trial_table["side_in_s"] <= trial_table["side_out_s"])
    )
    n_out_of_order = int((valid_all_four & ~ordered).sum())
    if n_out_of_order:
        print(f"  WARNING: {n_out_of_order}/{int(valid_all_four.sum())} trials with all 4 events "
              "resolved are NOT in center_in <= center_out <= side_in <= side_out order")

    align_info = dict(time_shift=time_shift, xcorr_peak=max_val, index_shift=p_to_b_shift)
    return trial_table, align_info
