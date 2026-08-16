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
    FINAL_SAMPLE_FREQ_HZ,
    HEMISPHERE_CHANNELS,
    PETH_POST_SEC,
    PETH_PRE_SEC,
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
