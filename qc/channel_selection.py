"""
Per-session hemisphere/channel selection: which physical fluorescence
channel actually carries the real GCaMP signal for a given session.

A per-MOUSE lookup (config/mouse_hemisphere.csv) isn't sufficient -- the same
mouse can have a dead/wrong channel on some days and not others. Raw carrier
lock/amplitude was tried first and found unreliable (cross-channel electrical
crosstalk in this rig means even a dead channel can show a strong, near-
nominal-frequency carrier lock -- see the mouse_hemisphere.csv-era
investigation). Instead: the real GCaMP channel should show a genuine
reward-vs-no-reward differential response (real task-locked signal); a dead
or crosstalk-only channel should not. This only works once trial alignment
itself is correct (behavior.sync's anchor-epoch fix) -- comparing rewarded vs
unrewarded z-score on misaligned trials is meaningless.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from alignment.windowing import extract_peth, get_event_indices
from config.params import (
    DECISION_WINDOW_S,
    FFT_ESTIMATE_N_POINTS,
    FINAL_SAMPLE_FREQ_HZ,
    HEMISPHERE_CHANNELS,
    PETH_POST_SEC,
    PETH_PRE_SEC,
    RAW_SAMPLE_FREQ_HZ,
)
from preprocessing.demodulate import compute_dff_and_zscore, demodulate_envelope, estimate_carrier_freq

DEFAULT_CANDIDATE_HEMISPHERES = ("green_r", "green_l", "red_l")
MIN_TRIALS_FOR_DETECTION = 20
SIGNIFICANT_P_VALUE = 0.05
# Physically adjacent channels on this rig can pick up real (not just noise)
# crosstalk from whichever channel actually carries the biological signal --
# confirmed directly (e.g. WCL28's red_l shows its own significant reward-
# differential effect, ~4% smaller than green_l's, almost certainly crosstalk
# rather than an independent real signal). A bare "largest |d| wins" pick is
# unreliable when the top two candidates are this close -- require the winner
# to beat the runner-up by a real margin, or flag the session as ambiguous
# (needs a manual config/session_hemisphere_overrides.csv entry instead).
WINNER_MARGIN_RATIO = 1.5  # winner's |cohens_d| must be >= 1.5x the runner-up's

# --- SM cohort correction: targeted, per-channel carrier detection ----------
# estimate_carrier_freq's free-ranging global FFT peak search can be fooled on
# a weak channel by a stronger low-frequency drift component, or by cross-talk
# from a neighboring channel's much stronger real carrier -- confirmed
# directly on the SM (PV-dualphotometry) cohort: it silently locked onto
# ~0 Hz for channels that, on a proper wide-band scan, actually show a real
# carrier near one of this rig's two true frequencies (167 Hz "green",
# 223 Hz "red"), and separately locked one channel onto the WRONG one of the
# two real carriers (the neighboring channel's, not its own) when both are
# genuinely present in the raw signal.
#
# A first attempt at fixing this (an earlier version of this function) did an
# unconstrained global-argmax search and only afterward checked whether the
# single dominant peak happened to land near a known frequency. That is
# STILL wrong when two real carriers coexist in the same raw channel (which
# happens routinely here: red_l's own 223 Hz jRGECO carrier plus green's
# stronger 167 Hz crosstalk bleeding into the same physical channel) --
# whichever frequency has the larger amplitude wins the argmax and the
# channel's OWN, weaker-but-real carrier is silently discarded. Confirmed
# directly: SM2N/SM3MN/SM3MR's red_l channel all show a real 223 Hz peak
# (1000-1800x the noise floor -- same order of magnitude as SM2B's, a
# jRGECO-only mouse with no green expression to cross-talk from), but the
# global argmax picked 167 Hz every time because that crosstalk component is
# larger in raw amplitude.
#
# So the check must be TARGETED, not global: does THIS channel's own assigned
# frequency (channels.nominal_carrier_freq_hz) show a real peak -- amplitude
# well above this channel's own noise floor -- regardless of whether some
# OTHER frequency in the same raw signal happens to be even larger. Forcing
# demodulation at the nominal frequency alone (without an amplitude check) was
# also tried and rejected -- it can't distinguish a genuinely dead channel
# (e.g. red_r: 223 Hz amplitude only ~3-9x its own noise floor, indistinguishable
# from noise) from a real one (e.g. green_r: 167 Hz amplitude ~50-67x floor,
# weak but genuinely present) -- see validation/sm3m_channel_diagnostic notes.
KNOWN_CARRIER_FREQS_HZ = (167.0, 223.0)
CARRIER_MATCH_TOLERANCE_HZ = 5.0  # +/- Hz window searched around the target frequency
CARRIER_SEARCH_MIN_FREQ_HZ = 5.0  # exclude near-DC drift from the noise-floor estimate
NOISE_FLOOR_EXCLUDE_TOL_HZ = 8.0  # exclude bins this close to ANY known carrier from the floor estimate
# Empirical cutoff (see module docstring above): confirmed-dead channels
# (red_r) top out at ~9x their own noise floor; the weakest confirmed-real
# channel (green_r) sits at ~50x. 15x sits with a comfortable margin on both
# sides of that gap.
CARRIER_AMPLITUDE_TO_FLOOR_RATIO = 15.0


def find_true_carrier_freq(
    raw_channel, target_freq, known_freqs=KNOWN_CARRIER_FREQS_HZ, tolerance_hz=CARRIER_MATCH_TOLERANCE_HZ,
    min_amplitude_ratio=CARRIER_AMPLITUDE_TO_FLOOR_RATIO, min_search_freq=CARRIER_SEARCH_MIN_FREQ_HZ,
    floor_exclude_tol_hz=NOISE_FLOOR_EXCLUDE_TOL_HZ, n_points=None,
):
    """Targeted check: is there a real carrier at THIS channel's own
    target_freq, regardless of what else (e.g. a neighboring channel's
    stronger crosstalk) is present elsewhere in the same raw spectrum?

    Compares the peak amplitude within +/-tolerance_hz of target_freq against
    this channel's own noise floor (median spectrum amplitude, excluding
    near-DC drift and the neighborhood of every known carrier) -- NOT against
    whatever the global spectrum maximum happens to be.

    Returns dict(detected_freq_hz [refined peak location within the search
    window], amplitude, noise_floor, amplitude_to_floor_ratio,
    matched_known_freq_hz [target_freq if the ratio clears the threshold,
    else None -- i.e. genuinely no real signal at this channel's own
    frequency, not just weaker than some other component]).
    """
    n_points = n_points or FFT_ESTIMATE_N_POINTS
    chunk = raw_channel[:n_points]
    chunk = (chunk - chunk.mean()) / chunk.std()
    spectrum = np.abs(np.fft.rfft(chunk)) / n_points
    freqs = np.fft.rfftfreq(n_points, d=1.0 / RAW_SAMPLE_FREQ_HZ)

    target_mask = np.abs(freqs - target_freq) <= tolerance_hz
    target_spectrum, target_freqs = spectrum[target_mask], freqs[target_mask]
    peak_idx = int(np.argmax(target_spectrum))
    detected_freq = float(target_freqs[peak_idx])
    amplitude = float(target_spectrum[peak_idx])

    floor_mask = freqs >= min_search_freq
    for f in known_freqs:
        floor_mask &= np.abs(freqs - f) > floor_exclude_tol_hz
    noise_floor = float(np.median(spectrum[floor_mask]))

    ratio = amplitude / noise_floor if noise_floor > 0 else float("inf")
    matched = target_freq if ratio >= min_amplitude_ratio else None
    return dict(
        detected_freq_hz=detected_freq, amplitude=amplitude, noise_floor=noise_floor,
        amplitude_to_floor_ratio=ratio, matched_known_freq_hz=matched,
    )


def evaluate_channel_true_signal(raw, trial_table, hemisphere, decision_window_s=DECISION_WINDOW_S,
                                  p_value_thresh=SIGNIFICANT_P_VALUE):
    """Corrected replacement for evaluate_channel_reward_response's carrier
    handling: run find_true_carrier_freq, TARGETED at this hemisphere's own
    assigned frequency (channels.nominal_carrier_freq_hz), on its raw channel
    FIRST; only if that target frequency shows a real peak (well above this
    channel's own noise floor) does it demodulate (at the refined DETECTED
    frequency, not the raw nominal one -- more precise) and run the same
    reward-vs-unrewarded differential-response test.

    Returns a dict with two SEPARATE validity concepts, always both present:
      carrier_valid -- a real, physically-stable carrier was found at this
        channel's own frequency. Confirmed via forced-frequency spot-checks
        across 7+ SM sessions to be highly stable/repeatable per channel
        (e.g. red_r is dead 7/7 times, red_l is real 7/7 times) -- this is
        the right basis for deciding whether a (session, hemisphere) row
        belongs in downstream pooled analysis.
      reward_significant -- this ONE session's reward-vs-unrewarded
        differential response happened to reach p_value_thresh. Confirmed to
        be noisy from session to session even for a channel with an
        unambiguous, visually-obvious real signal (e.g. green_l passed in
        only 1 of 7 P170 smoke-test sessions despite a huge, consistent
        carrier in all 7) -- this is descriptive/QC information about THIS
        session's statistical power, NOT a reliable channel-validity signal,
        and should not be used to gate inclusion in pooled analysis.
    None when it couldn't be computed (carrier_valid=False, or a downstream
    step failed) -- see "error" for why.

    detected_freq_hz/amplitude/matched_known_freq_hz are always populated;
    cohens_d/p_value/n_rewarded/n_unrewarded are populated whenever
    reward_significant is not None.
    """
    channels = HEMISPHERE_CHANNELS[hemisphere]
    carrier = find_true_carrier_freq(raw[channels.signal_channel], channels.nominal_carrier_freq_hz)
    carrier_valid = carrier["matched_known_freq_hz"] is not None

    if not carrier_valid:
        return dict(carrier, carrier_valid=False, reward_significant=None,
                    error="no real carrier detected at this channel's own frequency (dead channel)")

    try:
        envelope, _ = demodulate_envelope(raw[channels.signal_channel], carrier["detected_freq_hz"])
        _, zscore, _ = compute_dff_and_zscore(raw[channels.signal_channel], carrier["detected_freq_hz"], envelope)
    except Exception as exc:
        return dict(carrier, carrier_valid=True, reward_significant=None, error=f"demodulation failed: {exc}")

    pre_samples = int(round(PETH_PRE_SEC * FINAL_SAMPLE_FREQ_HZ))
    post_samples = int(round(PETH_POST_SEC * FINAL_SAMPLE_FREQ_HZ))
    peth_time = np.arange(-pre_samples, post_samples + 1) / FINAL_SAMPLE_FREQ_HZ

    event_idx = get_event_indices(trial_table, "side_in")
    has_window = (event_idx - pre_samples >= 0) & (event_idx + post_samples < len(zscore))
    if has_window.sum() < MIN_TRIALS_FOR_DETECTION:
        return dict(carrier, carrier_valid=True, reward_significant=None,
                    error=f"only {int(has_window.sum())} trials with a valid PETH window")

    windows = extract_peth(zscore, event_idx[has_window], pre_samples, post_samples)
    sub_trial_table = trial_table[has_window].reset_index(drop=True)

    metric_mask = (peth_time >= decision_window_s[0]) & (peth_time <= decision_window_s[1])
    metric = windows[:, metric_mask].mean(axis=1)

    is_rewarded = sub_trial_table["was_rewarded"].to_numpy()
    rewarded_vals = metric[is_rewarded]
    unrewarded_vals = metric[~is_rewarded]
    if len(rewarded_vals) < 2 or len(unrewarded_vals) < 2:
        return dict(carrier, carrier_valid=True, reward_significant=None,
                    error=f"not enough rewarded ({len(rewarded_vals)})/unrewarded ({len(unrewarded_vals)}) trials")

    t_stat, p_value = stats.ttest_ind(rewarded_vals, unrewarded_vals, equal_var=False)
    pooled_std = np.sqrt((rewarded_vals.var(ddof=1) + unrewarded_vals.var(ddof=1)) / 2)
    cohens_d = float((rewarded_vals.mean() - unrewarded_vals.mean()) / pooled_std) if pooled_std > 0 else 0.0

    return dict(
        carrier, carrier_valid=True, reward_significant=bool(p_value < p_value_thresh),
        cohens_d=cohens_d, t_stat=float(t_stat), p_value=float(p_value),
        n_rewarded=int(len(rewarded_vals)), n_unrewarded=int(len(unrewarded_vals)),
    )


def evaluate_all_channels_true_signal(raw, trial_table, hemispheres=("green_r", "green_l", "red_r", "red_l"),
                                       p_value_thresh=SIGNIFICANT_P_VALUE):
    """evaluate_channel_true_signal for every candidate hemisphere in one
    session -- the SM-cohort-corrected analogue of evaluate_bilateral_
    hemispheres, using the targeted carrier search instead of trusting either
    estimate_carrier_freq or the channel's own label.

    Returns (carrier_valid, results): carrier_valid is {hemisphere: bool} --
    the stable, physical-channel-quality gate to use for downstream pooled
    analysis (see evaluate_channel_true_signal's docstring for why this,
    not per-session reward significance, is the right inclusion criterion).
    results is {hemisphere: result_dict}, which also carries each
    hemisphere's reward_significant value for reporting/QC.
    """
    results = {h: evaluate_channel_true_signal(raw, trial_table, h, p_value_thresh=p_value_thresh) for h in hemispheres}
    carrier_valid = {h: r["carrier_valid"] for h, r in results.items()}
    return carrier_valid, results


def evaluate_channel_reward_response(raw, trial_table, hemisphere, decision_window_s=DECISION_WINDOW_S):
    """Demodulate `hemisphere`'s fluorescence channel and compute a reward-
    vs-unrewarded differential-response effect size in decision_window_s
    after side_in, using trial_table's ALREADY-RESOLVED photometry_side_in_index
    (alignment only depends on the digital behavior channels, not which
    fluorescence channel is chosen, so trial_table/alignment is computed once
    per session and shared across every candidate hemisphere).

    Returns a dict: cohens_d (signed: rewarded - unrewarded, in pooled SDs),
    p_value (Welch's t-test), n_rewarded, n_unrewarded, or {"error": ...} if
    this channel can't be evaluated (e.g. too few aligned trials).
    """
    channels = HEMISPHERE_CHANNELS[hemisphere]
    try:
        measured_freq, _ = estimate_carrier_freq(raw[channels.signal_channel])
        envelope, _ = demodulate_envelope(raw[channels.signal_channel], measured_freq)
        _, zscore, _ = compute_dff_and_zscore(raw[channels.signal_channel], measured_freq, envelope)
    except Exception as exc:
        return dict(error=f"demodulation failed: {exc}")

    pre_samples = int(round(PETH_PRE_SEC * FINAL_SAMPLE_FREQ_HZ))
    post_samples = int(round(PETH_POST_SEC * FINAL_SAMPLE_FREQ_HZ))
    peth_time = np.arange(-pre_samples, post_samples + 1) / FINAL_SAMPLE_FREQ_HZ

    event_idx = get_event_indices(trial_table, "side_in")
    has_window = (event_idx - pre_samples >= 0) & (event_idx + post_samples < len(zscore))
    if has_window.sum() < MIN_TRIALS_FOR_DETECTION:
        return dict(error=f"only {int(has_window.sum())} trials with a valid PETH window")

    windows = extract_peth(zscore, event_idx[has_window], pre_samples, post_samples)
    sub_trial_table = trial_table[has_window].reset_index(drop=True)

    metric_mask = (peth_time >= decision_window_s[0]) & (peth_time <= decision_window_s[1])
    metric = windows[:, metric_mask].mean(axis=1)

    is_rewarded = sub_trial_table["was_rewarded"].to_numpy()
    rewarded_vals = metric[is_rewarded]
    unrewarded_vals = metric[~is_rewarded]
    if len(rewarded_vals) < 2 or len(unrewarded_vals) < 2:
        return dict(error=f"not enough rewarded ({len(rewarded_vals)})/unrewarded ({len(unrewarded_vals)}) trials")

    t_stat, p_value = stats.ttest_ind(rewarded_vals, unrewarded_vals, equal_var=False)
    pooled_std = np.sqrt((rewarded_vals.var(ddof=1) + unrewarded_vals.var(ddof=1)) / 2)
    cohens_d = float((rewarded_vals.mean() - unrewarded_vals.mean()) / pooled_std) if pooled_std > 0 else 0.0

    return dict(
        cohens_d=cohens_d, t_stat=float(t_stat), p_value=float(p_value),
        n_rewarded=int(len(rewarded_vals)), n_unrewarded=int(len(unrewarded_vals)),
    )


def detect_session_hemisphere(raw, trial_table, candidates=DEFAULT_CANDIDATE_HEMISPHERES,
                               winner_margin_ratio=WINNER_MARGIN_RATIO):
    """Evaluate every candidate hemisphere's reward-vs-no-reward differential
    response for this session and pick whichever shows the strongest
    (largest |cohens_d|, among those reaching SIGNIFICANT_P_VALUE) real,
    task-locked signal -- but only if it clearly beats the runner-up
    (winner_margin_ratio), since a crosstalk-affected neighbor channel can
    show its own smaller-but-still-significant effect.

    Returns (best_hemisphere_or_None, {hemisphere: result_dict}) -- None
    means either no candidate reached significance, or the top two were too
    close to call confidently; either way the session needs a manual
    config/session_hemisphere_overrides.csv entry instead.
    """
    results = {name: evaluate_channel_reward_response(raw, trial_table, name) for name in candidates}
    significant = {
        name: r for name, r in results.items()
        if "error" not in r and r["p_value"] < SIGNIFICANT_P_VALUE
    }
    if not significant:
        return None, results

    ranked = sorted(significant, key=lambda name: abs(significant[name]["cohens_d"]), reverse=True)
    best = ranked[0]
    if len(ranked) > 1:
        best_d = abs(significant[best]["cohens_d"])
        runner_up_d = abs(significant[ranked[1]]["cohens_d"])
        if runner_up_d > 0 and best_d / runner_up_d < winner_margin_ratio:
            return None, results  # too close to call -- ambiguous
    return best, results


def evaluate_bilateral_hemispheres(raw, trial_table, hemispheres=("green_r", "green_l"),
                                    p_value_thresh=SIGNIFICANT_P_VALUE):
    """Independently gate each hemisphere for a true dual-fiber rig (e.g. the
    SM cohort), where BOTH sides are real, simultaneously-recorded pyramidal
    signal rather than WCL's single-active-channel-per-mouse setup.

    Unlike detect_session_hemisphere, this does NOT pick a single winner or
    suppress a close runner-up -- both sides can and should be valid at once.
    A hemisphere is valid iff it demodulates cleanly and reaches significance
    on its own reward-vs-no-reward differential response.

    Returns (valid, results): valid is {hemisphere: bool}; results is
    {hemisphere: result_dict} from evaluate_channel_reward_response, for
    reporting/debugging.
    """
    results = {name: evaluate_channel_reward_response(raw, trial_table, name) for name in hemispheres}
    valid = {
        name: ("error" not in r and r["p_value"] < p_value_thresh)
        for name, r in results.items()
    }
    return valid, results


def load_session_hemisphere_overrides(path):
    """Load a manual per-(mouse,date) hemisphere override CSV (columns:
    Mouse ID, Date, Hemisphere -- Date in the same MMDDYY string convention
    as io_utils.raw_loader.parse_session_id), indexed by (mouse, date).
    Returns {} if `path` doesn't exist (overrides are optional -- most
    sessions should resolve via automated detection or the per-mouse
    default).
    """
    path = Path(path)
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype={"Date": str})
    return {(row["Mouse ID"], row["Date"]): row["Hemisphere"] for _, row in df.iterrows()}


def resolve_session_hemisphere(mouse, date, raw, trial_table, mouse_default,
                                overrides=None, candidates=DEFAULT_CANDIDATE_HEMISPHERES):
    """Three-tier hemisphere resolution for one session, highest priority first:
      1. A manual (mouse, date) entry in `overrides` (config/session_hemisphere_overrides.csv)
         -- the user's own known-correct answer for a specific anomalous day.
      2. Confident automated per-day detection (detect_session_hemisphere) --
         catches a dead/wrong channel on an otherwise-normal mouse.
      3. `mouse_default` (config/mouse_hemisphere.csv) -- used whenever
         per-day detection is ambiguous, since "couldn't confirm or deny"
         should trust the established per-mouse baseline, not silently
         guess.
    Returns (hemisphere, source) where source is one of "override",
    "detected", "mouse_default".
    """
    overrides = overrides or {}
    if (mouse, date) in overrides:
        return overrides[(mouse, date)], "override"

    detected, _ = detect_session_hemisphere(raw, trial_table, candidates=candidates)
    if detected is not None:
        return detected, "detected"

    return mouse_default, "mouse_default"
