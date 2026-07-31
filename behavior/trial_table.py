"""
Raw behavior logs (pokeHistory + stats) -> per-trial DataFrame.
Direct translation of extractTrials_dataTable.m.
"""

import numpy as np
import pandas as pd


def poke_times_seconds(poke_history):
    """Elapsed seconds since the first poke, replicating
    `etime(datevec(t), datevec(t0))` for MATLAB datenum timestamps (which,
    for two datevecs derived from the same datenum epoch, reduces to a
    plain difference in days * 86400 seconds/day).
    """
    timestamps = np.array([p["timeStamp"] for p in poke_history], dtype=float)
    return (timestamps - timestamps[0]) * 86400.0


def build_trial_table(poke_history, stats):
    """Direct translation of extractTrials_dataTable.m. Indices below are
    0-based (Python/pokeHistory-list-index), whereas the MATLAB source is
    1-based -- `centerInPokeIndex = sideInPokeIndex - 1` means "the
    immediately preceding poke event", which is base-independent, so no
    off-by-one adjustment is needed beyond staying consistently 0-based.

    center_in_poke_index is resolved from pokeHistory's own `isTRIAL` field
    (1 = a poke that starts a scored trial, 2 = the poke that completes it,
    0 = an invalid/incomplete poke -- e.g. a center poke abandoned before a
    side entry) rather than assumed positionally: for each side-in poke, take
    the most recent isTRIAL==1 poke at or before it. This is the ground-truth
    source (not every isTRIAL==1 poke leads to a completed trial -- 420
    isTRIAL==1 pokes vs 361 isTRIAL==2 completions in the WCL23/060223
    session used to validate this). The immediately-preceding-poke assertion
    below documents and checks the (currently universally true) simplifying
    case where no invalidated poke intervenes between a trial's own center-in
    and side-in.
    """
    time_poked = poke_times_seconds(poke_history)  # extractTrials_dataTable.m:14-18
    is_trial = np.array([p["isTRIAL"] for p in poke_history])

    is_left_trial = np.asarray(stats["trials"]["left"]) == 2   # extractTrials_dataTable.m:20
    is_right_trial = np.asarray(stats["trials"]["right"]) == 2  # extractTrials_dataTable.m:21

    left_trials = np.flatnonzero(is_left_trial)
    right_trials = np.flatnonzero(is_right_trial)

    combined = np.concatenate([left_trials, right_trials])
    sort_order = np.argsort(combined, kind="stable")
    side_in_poke_index = combined[sort_order]                  # extractTrials_dataTable.m:55-57

    dummy = np.concatenate([np.ones(len(left_trials)), np.zeros(len(right_trials))])
    chose_left = dummy[sort_order] == 1                        # extractTrials_dataTable.m:63-64
    chose_right = ~chose_left

    assert np.all(is_trial[side_in_poke_index] == 2), (
        "side_in_poke_index (from stats['trials']) must always land on an isTRIAL==2 "
        "(valid side-port) poke in poke_history"
    )
    center_in_candidates = np.flatnonzero(is_trial == 1)
    center_in_poke_index = center_in_candidates[
        np.searchsorted(center_in_candidates, side_in_poke_index) - 1
    ]
    assert np.array_equal(center_in_poke_index, side_in_poke_index - 1), (
        "Expected the isTRIAL==1 center-in poke immediately preceding each side-in poke to be "
        "the poke immediately before it in poke_history (no invalidated/intervening poke) -- "
        "if this fails, some trial in this session has an invalidated center poke between its "
        "true center-in and side-in, and center_in_poke_index needs a different resolution "
        "strategy for that trial"
    )

    n_trials = len(side_in_poke_index)
    left_reward_prob = np.zeros(n_trials)
    right_reward_prob = np.zeros(n_trials)
    was_rewarded = np.zeros(n_trials, dtype=bool)
    for i, poke_idx in enumerate(side_in_poke_index):
        entry = poke_history[poke_idx]
        left_reward_prob[i] = entry["leftPortStats"]["prob"]
        right_reward_prob[i] = entry["rightPortStats"]["prob"]
        was_rewarded[i] = entry["REWARD"] == 1

    trial_table = pd.DataFrame(
        {
            "side_in_poke_index": side_in_poke_index,
            "side_in_time": time_poked[side_in_poke_index],
            "center_in_poke_index": center_in_poke_index,
            "center_in_time": time_poked[center_in_poke_index],
            "chose_left": chose_left,
            "chose_right": chose_right,
            "left_reward_prob": left_reward_prob,
            "right_reward_prob": right_reward_prob,
            "was_rewarded": was_rewarded,
        }
    )
    return trial_table.sort_values("side_in_time").reset_index(drop=True)
