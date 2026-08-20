"""
Value-decomposition, model-implied belief, and switch-dynamics features for
the time-resolved GLM -- layered on top of behavior.trial_table's raw columns
and external.bandit_state_adapter.add_bandit_state_features's Q_left/Q_right/
Q_diff/Behavioral_State. Purely additive: every function here returns new
columns only, existing columns/callers (models.glm_encoding's DEFAULT
formula, rpe_analysis_*, etc.) are untouched.

True switch is GROUND TRUTH, not estimated. Unlike a naturalistic-behavior
study, left_reward_prob/right_reward_prob are logged per trial directly by
the task's own controller, so which side currently has the higher reward
probability is exactly known on every trial -- there is no need to infer the
reversal trial from behavior the way the neuroscience literature usually
must.

"Detected switch" is necessarily a judgment call (there's no single
canonical definition), so its window/threshold are named, configurable
constants rather than hardcoded -- see DETECTED_SWITCH_WINDOW_TRIALS/
DETECTED_SWITCH_THRESHOLD below and adjust if a different operational
definition is wanted.
"""

import numpy as np
import pandas as pd

# Forward-looking window (in trials) used to decide whether/when the animal
# has "detected" a true reversal -- retrospective/post-hoc labeling (uses
# trials AFTER the trial being labeled), which is fine for characterizing
# behavior after the fact but must NOT be used as a causal, real-time
# predictor.
DETECTED_SWITCH_WINDOW_TRIALS = 5
DETECTED_SWITCH_THRESHOLD = 0.6

# Post-switch exponential-decay regressor's time constant, in TRIALS (not
# seconds) -- a first-pass value, not fit; swap for a swept/selected value if
# the exponential term turns out to matter.
SWITCH_DECAY_TAU_TRIALS = 3.0


def add_value_decomposition_features(trial_table):
    """Q_chosen/Q_unchosen/Q_total/RPE/RPE_abs, derived from the already-fit
    Q_left/Q_right/Q_diff (external.bandit_state_adapter.
    add_bandit_state_features must run first) plus was_rewarded/chose_right.
    NaN wherever Q_left/Q_right are NaN (session's Q-learning fit skipped,
    below BANDIT_MIN_TRIALS).
    """
    out = trial_table.copy()
    chose_right = out["chose_right"].astype(float)
    out["Q_chosen"] = np.where(chose_right == 1, out["Q_right"], out["Q_left"])
    out["Q_unchosen"] = np.where(chose_right == 1, out["Q_left"], out["Q_right"])
    out["Q_total"] = out["Q_left"] + out["Q_right"]
    out["RPE"] = out["was_rewarded"].astype(float) - out["Q_chosen"]
    out["RPE_abs"] = out["RPE"].abs()
    return out


def add_model_belief(trial_table, bandit_fit_params):
    """Continuous 'belief' proxy: the sticky Q-learning model's OWN implied
    P(choose right) = sigmoid(beta*Q_diff + kappa*stick), evaluated at each
    trial's already-simulated Q_diff and the actual previous choice. This is
    NOT a separate Bayesian change-point/belief model -- it's just exposing
    what the already-fit sticky-logit model implies about the animal's
    per-trial value belief, at zero extra model-fitting cost. A true
    from-scratch Bayesian change-point model would be a larger, separate
    follow-on if this proxy isn't convincing.

    bandit_fit_params : the dict external.bandit_state_adapter.
        add_bandit_state_features stashes at trial_table.attrs["bandit_fit_params"]
        (has "beta"/"kappa"; None if that session's fit was skipped).
    """
    out = trial_table.copy()
    if bandit_fit_params is None:
        out["belief_p_right"] = np.nan
        return out
    beta, kappa = bandit_fit_params["beta"], bandit_fit_params["kappa"]
    prev_choice = out["chose_right"].astype(float).shift(1)
    stick = np.where(prev_choice.isna(), 0.0, np.where(prev_choice == 1, 1.0, -1.0))
    z = beta * out["Q_diff"].to_numpy(dtype=float) + kappa * stick
    out["belief_p_right"] = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
    return out


def add_switch_dynamics(trial_table, decay_tau_trials=SWITCH_DECAY_TAU_TRIALS,
                         detect_window=DETECTED_SWITCH_WINDOW_TRIALS,
                         detect_thresh=DETECTED_SWITCH_THRESHOLD):
    """true_switch (ground truth) + the post-switch feature set built on it:
    trials_since_switch (+ squared + exponential-decay versions),
    first_win_after_switch/first_loss_after_switch, detected_switch,
    switch_detection_lag.

    true_switch: current_high_side = right_reward_prob > left_reward_prob;
    true_switch is True on any trial where current_high_side differs from
    the previous trial (the first trial of a session is never a switch --
    there's no defined "previous" to switch from). Note this also fires at
    forced-choice-block <-> real-bandit-block transitions (a genuine
    task-phase change, not a within-bandit reversal) -- not filtered out
    here, since that's still a real change in the operative contingency;
    exclude on `is_forced_block` downstream if that phase mix is unwanted.

    trials_since_switch resets to 0 at each true_switch trial and counts up;
    NaN before the first true_switch in the session (no defined "since" yet).

    detected_switch: first trial at/after a true_switch where a forward
    window of chose_right matches current_high_side at >= detect_thresh
    (see module docstring -- retrospective, not causal). switch_detection_lag
    is that trial's trials_since_switch value, broadcast across the whole
    post-switch segment (constant within segment) so it's usable as a
    per-trial GLM modulator, not just a single-trial marker.
    """
    out = trial_table.reset_index(drop=True)
    n = len(out)
    current_high_side = (out["right_reward_prob"] > out["left_reward_prob"]).astype(float).to_numpy()

    true_switch = np.zeros(n, dtype=bool)
    true_switch[1:] = current_high_side[1:] != current_high_side[:-1]
    out["true_switch"] = true_switch

    switch_positions = np.flatnonzero(true_switch)
    trials_since = np.full(n, np.nan)
    detected_switch = np.zeros(n, dtype=bool)
    switch_detection_lag = np.full(n, np.nan)
    first_win_after_switch = np.zeros(n, dtype=bool)
    first_loss_after_switch = np.zeros(n, dtype=bool)

    chose_right = out["chose_right"].to_numpy(dtype=float)
    was_rewarded = out["was_rewarded"].to_numpy(dtype=float)

    for seg_i, start in enumerate(switch_positions):
        end = switch_positions[seg_i + 1] if seg_i + 1 < len(switch_positions) else n
        trials_since[start:end] = np.arange(end - start)

        seg_rewarded = was_rewarded[start:end]
        if np.any(seg_rewarded == 0):
            first_loss_after_switch[start + int(np.argmax(seg_rewarded == 0))] = True
        if np.any(seg_rewarded == 1):
            first_win_after_switch[start + int(np.argmax(seg_rewarded == 1))] = True

        target = current_high_side[start]
        for t in range(start, end):
            window = chose_right[t:min(t + detect_window, end)]
            if len(window) == 0:
                continue
            if np.mean(window == target) >= detect_thresh:
                detected_switch[t] = True
                switch_detection_lag[start:end] = t - start
                break

    out["trials_since_switch"] = trials_since
    out["trials_since_switch_sq"] = trials_since ** 2
    out["trials_since_switch_expdecay"] = np.exp(-trials_since / decay_tau_trials)
    out["detected_switch"] = detected_switch
    out["switch_detection_lag"] = switch_detection_lag
    out["first_win_after_switch"] = first_win_after_switch
    out["first_loss_after_switch"] = first_loss_after_switch
    return out
