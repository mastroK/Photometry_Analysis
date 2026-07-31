"""
Attach behavioral-state classification and Q-learning value estimates to a
session's own trial_table, using external.bandit_state_model (ported from
mastro_mouse_bandit_analysis) run fresh against THIS session's trials --
not joined from that package's external nosepoke-aging dataset, which this
cohort's mice/sessions aren't part of (per lab decision: the trial_table is
the single source of truth per session).

Since bandit_state_model's functions expect specific column names ("Session
ID", "Decision", "Reward", "Switch", "Target", ...), this module's only job
is mapping trial_table's own columns into that schema and copying the
results back -- no key-based join is needed, since the intermediate frame is
built directly from trial_table and stays row-aligned with it throughout.
"""

import numpy as np
import pandas as pd

from config.params import BANDIT_MIN_TRIALS, BANDIT_N_STARTS
from external.bandit_state_model import add_behavioral_state, add_rolling_features, fit_session_level, simulate_qvalues

NEW_COLUMNS = (
    "Q_left", "Q_right", "Q_diff",
    "Rolling_Accuracy", "Rolling_PRight", "Expected_PRight",
    "Signed_Deviation", "Choice_Deviation", "Rolling_Switch_Rate",
    "Behavioral_State",
)


def add_bandit_state_features(trial_table, session_id, min_trials=BANDIT_MIN_TRIALS, n_starts=BANDIT_N_STARTS):
    """Return a copy of trial_table with Behavioral_State, Q_left, Q_right,
    Q_diff, and the rolling-feature columns added, fit fresh against this
    session's own trials.

    Requires trial_table to already have chose_right, was_rewarded, switched,
    left_reward_prob, right_reward_prob (behavior.trial_table.build_trial_table
    + behavior.word_encoding.add_lag_features), in trial chronological order.

    If the session has fewer than min_trials trials, the sticky Q-learning
    fit is skipped (short sessions are an expected occurrence in a cohort,
    not a bug) and the new columns are filled with NaN / "Unknown".
    """
    trial_table = trial_table.reset_index(drop=True)
    n_trials = len(trial_table)

    bandit_df = pd.DataFrame({
        "Session ID": session_id,
        "Mouse ID": session_id,
        "Trial": np.arange(n_trials),
        "Decision": trial_table["chose_right"].astype(int).to_numpy(),
        "Reward": trial_table["was_rewarded"].astype(int).to_numpy(),
        "Switch": trial_table["switched"].to_numpy(dtype="float64", na_value=np.nan),
        "Target": (trial_table["right_reward_prob"] > trial_table["left_reward_prob"]).astype(int).to_numpy(),
    })

    out = trial_table.copy()

    if n_trials < min_trials:
        print(
            f"WARNING: session '{session_id}' has {n_trials} trials, below "
            f"BANDIT_MIN_TRIALS={min_trials} -- skipping sticky Q-learning fit, "
            f"filling bandit-state columns with NaN/'Unknown'"
        )
        for col in NEW_COLUMNS:
            out[col] = "Unknown" if col == "Behavioral_State" else np.nan
        out.attrs["bandit_fit_params"] = None
        return out

    params = fit_session_level(bandit_df, min_trials=min_trials, n_starts=n_starts)

    bandit_df = simulate_qvalues(bandit_df, params)
    bandit_df = add_rolling_features(bandit_df)

    # add_rolling_features' Expected_PRight hardcodes 0.8/0.2, tied to the
    # source cohort's fixed 80-20 task design -- use this session's ACTUAL
    # per-trial reward probability instead, which generalizes to any
    # configured probability block.
    bandit_df["Expected_PRight"] = trial_table["right_reward_prob"].to_numpy()
    bandit_df["Signed_Deviation"] = bandit_df["Rolling_PRight"] - bandit_df["Expected_PRight"]
    bandit_df["Choice_Deviation"] = bandit_df["Signed_Deviation"].abs()

    bandit_df = add_behavioral_state(bandit_df)

    for col in NEW_COLUMNS:
        out[col] = bandit_df[col].to_numpy()

    fit_row = params.iloc[0]
    out.attrs["bandit_fit_params"] = fit_row[
        ["alpha", "beta", "kappa", "nll", "n_trials", "group"]
    ].to_dict()
    return out
