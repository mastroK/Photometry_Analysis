"""
Behavior-clock <-> photometry-clock alignment. Direct translation of
processBehavior.m:56-308 (xcorr fitting, event-index resolution, and the
hasAllPhotometryData trial-validity gate), against processNew_fast_kevin.m --
the confirmed reference implementation for this cohort.
"""

import numpy as np

from behavior.trial_table import poke_times_seconds
from config.params import (
    CH_CENTERPORT,
    CH_LEFTPORT,
    CH_RIGHTPORT,
    FINAL_TIME_STEP_SEC,
    HOP_SAMPLES,
    MIN_PTS_OFFSET_FULL,
    PTS_KEEP_AFTER_SAMPLES,
    PTS_KEEP_BEFORE_SAMPLES,
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

    # processBehavior.m:14-18 (extractTrials_dataTable.m) zeros trial_table's
    # own side_in_time/center_in_time to the BEHAVIOR-clock time of this same
    # first-matched center poke (first_b_poke_index), not to the session's
    # very first logged poke -- behavior.trial_table.poke_times_seconds uses
    # poke_history[0] as t=0 for every trial regardless, so t_anchor_b below
    # converts back to that same first-matched-poke epoch before combining
    # with the (photometry-clock) time_shift anchor. Confirmed empirically
    # against a real MATLAB reference session: without this correction,
    # every resolved photometry sample index is off by a large (tens-of-
    # seconds), non-constant amount -- this single line is the fix for that.
    t_anchor_b = center_times_b_adj[0]

    trial_table = trial_table.copy()
    trial_table["matched_side_in_index"] = np.floor(
        (trial_table["side_in_time"] - t_anchor_b + time_shift) / FINAL_TIME_STEP_SEC
    ).astype(int)
    trial_table["matched_center_in_index"] = np.floor(
        (trial_table["center_in_time"] - t_anchor_b + time_shift) / FINAL_TIME_STEP_SEC
    ).astype(int)

    # --- resolve raw-rate edge indices, block-summed onto the final-rate grid ------
    # (processBehavior.m:189-274)
    right_rising_final, left_rising_final = _final_rate_rising_edges(raw, n_final_samples)
    right_falling_final = _final_rate_edges(raw, CH_RIGHTPORT, n_final_samples, edge=-1)
    left_falling_final = _final_rate_edges(raw, CH_LEFTPORT, n_final_samples, edge=-1)
    side_falling_final = np.union1d(right_falling_final, left_falling_final)
    center_rising_final = _final_rate_edges(raw, CH_CENTERPORT, n_final_samples, edge=1)
    center_falling_final = _final_rate_edges(raw, CH_CENTERPORT, n_final_samples, edge=-1)

    n_trials = len(trial_table)
    matched_side_in = trial_table["matched_side_in_index"].to_numpy()
    matched_center_in = trial_table["matched_center_in_index"].to_numpy()
    chose_right = trial_table["chose_right"].to_numpy()

    def _closest(candidates, target):
        """closest.m: nearest by absolute difference, ties -> first/lowest
        index (np.argmin's own tie-break, matching MATLAB's min())."""
        return candidates[np.argmin(np.abs(candidates - target))]

    # --- photometry_center_in_index: UNCONDITIONAL nearest-match, no tolerance ------
    # (processBehavior.m:201,230-231 -- MATLAB always snaps, however far)
    photometry_center_in_index = np.full(n_trials, -1, dtype=int)
    if len(center_rising_final) > 0:
        for i in range(n_trials):
            photometry_center_in_index[i] = _closest(center_rising_final, matched_center_in[i])

    # --- photometry_side_in_index: unconditional nearest-match, branched by side ----
    # (processBehavior.m:199-200,220-229 -- already correct, kept as-is)
    photometry_side_in_index = np.full(n_trials, -1, dtype=int)
    for i in range(n_trials):
        side_candidates = right_rising_final if chose_right[i] else left_rising_final
        if len(side_candidates) > 0:
            photometry_side_in_index[i] = _closest(side_candidates, matched_side_in[i])

    # --- photometry_center_out_index: forward search from center_in, unbounded ------
    # (processBehavior.m:196,233-242 -- first falling edge >= center_in, no upper bound)
    photometry_center_out_index = np.full(n_trials, -1, dtype=int)
    is_photometry_trial = np.zeros(n_trials, dtype=bool)
    if len(center_falling_final) > 0:
        for i in range(n_trials):
            if photometry_center_in_index[i] < 0:
                continue
            after = center_falling_final[center_falling_final >= photometry_center_in_index[i]]
            if len(after) > 0:
                photometry_center_out_index[i] = after[0]
                is_photometry_trial[i] = True
            # else: no match -> WARNING in MATLAB, isPhotometryTrial stays False, index stays -1

    # --- de-dup fixup (processBehavior.m:268-274): fix the SECOND of any pair of
    # consecutive trials that resolved to the identical center_out index -----------
    center_out_valid = photometry_center_out_index >= 0
    dup = (photometry_center_out_index[:-1] == photometry_center_out_index[1:]) & center_out_valid[:-1] & center_out_valid[1:]
    dup_second_idx = np.flatnonzero(dup) + 1
    dup_second_idx = dup_second_idx[photometry_center_in_index[dup_second_idx] >= 0]
    photometry_center_out_index[dup_second_idx] = photometry_center_in_index[dup_second_idx]

    # --- photometry_side_out_index: forward search from CENTER_IN (not side_in),
    # union of both L+R falling edges (not just the trial's own chosen side) -------
    # (processBehavior.m:197,244-251)
    photometry_side_out_index = np.full(n_trials, -1, dtype=int)
    if len(side_falling_final) > 0:
        for i in range(n_trials):
            if photometry_center_in_index[i] < 0:
                continue
            after = side_falling_final[side_falling_final >= photometry_center_in_index[i]]
            if len(after) > 0:
                photometry_side_out_index[i] = after[0]
            else:
                is_photometry_trial[i] = False
                # else: no match -> WARNING in MATLAB, isPhotometryTrial=False, index stays -1

    # --- hasAllPhotometryData gate (processBehavior.m:284-308): the primary
    # trial-validity gate feeding the main reward-split PETH -- re-invalidate
    # ALL FOUR indices for any trial that isn't a real photometry trial (both
    # center-out and side-out resolved) or doesn't leave enough margin for a
    # full PETH window on either side. dropFirstDetrendWindow=1 is hardcoded
    # in every processNew* variant, so the FULL signalDetrendWindow is used
    # here (see config.params.MIN_PTS_OFFSET_FULL vs the separate, more
    # lenient MIN_PTS_OFFSET_HALF used by behavior.word_encoding's hasP gate).
    has_all_photometry_data = (
        is_photometry_trial
        & (photometry_center_in_index > (PTS_KEEP_BEFORE_SAMPLES + MIN_PTS_OFFSET_FULL))
        & (photometry_side_out_index < (n_final_samples - PTS_KEEP_AFTER_SAMPLES))
    )
    for arr in (photometry_center_in_index, photometry_center_out_index,
                photometry_side_in_index, photometry_side_out_index):
        arr[~has_all_photometry_data] = -1

    trial_table["photometry_center_in_index"] = photometry_center_in_index
    trial_table["photometry_center_out_index"] = photometry_center_out_index
    trial_table["photometry_side_in_index"] = photometry_side_in_index
    trial_table["photometry_side_out_index"] = photometry_side_out_index
    trial_table["has_all_photometry_data"] = has_all_photometry_data

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

    print(f"  hasAllPhotometryData: {int(has_all_photometry_data.sum())}/{n_trials} trials valid")

    align_info = dict(time_shift=time_shift, xcorr_peak=max_val, index_shift=p_to_b_shift)
    return trial_table, align_info
