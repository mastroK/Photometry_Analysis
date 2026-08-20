"""
Per-trial action/outcome "word" labels -- Python port of the trial-labeling
logic in newCode/KM_processCeliaWord.m (and its non-KM twin
newCode/processCeliaWord.m + newCode/flipCeliaWord.m).

Two representations are computed for each requested window length L
(a.k.a. "levels" in the MATLAB scripts), over the trailing window of L
trials ending at (and including) the current trial:

  word_lL          -- raw choice+outcome word using R/L for side and case
                       for reward (R/L = rewarded, r/l = unrewarded),
                       oldest trial in the window first, current trial last.
                       (processCeliaWord.m:129-169 builds this same word set,
                       via recursive prefix expansion over trial-index sets
                       rather than directly -- confirmed by index arithmetic
                       that a length-L word is assigned to the trial at the
                       END of its L-trial window, matching this direct form.)

  word_lL_generic  -- the same window collapsed to stay/switch relative to
                       the window's OLDEST trial: same side as that trial ->
                       A/a (stay), different side -> B/b (switch), case still
                       from that trial's own reward outcome. This is exactly
                       what KM_processCeliaWord.m:61-88 computes directly as
                       `wordLabel`, and matches the `genericWord` output of
                       flipCeliaWord.m applied to word_lL.

Trials with fewer than L trials of history (including themselves) get None
for that level's columns, matching the MATLAB `trialCounter>=levels` guard
and the `wordLabel(1:nTrials)=""` default fill.

NOT ported here (out of scope -- see plan): the exhaustive per-word trial-index
bookkeeping and downstream per-word-condition PETH/stats plumbing in
processCeliaWord.m (words/wordTrials/conditionsList/processConditions), and
KM_processCeliaWord.m's trialsSinceSwitch/trialsToSwitch (confirmed to track
reward-probability BLOCK changes, not the animal's own choice-switch --
explicitly excluded per lab decision).
"""

import numpy as np
import pandas as pd


def _trial_letter(chose_left, was_rewarded):
    """Single-trial raw code: R/r for a right choice, L/l for a left choice;
    uppercase = rewarded, lowercase = unrewarded.
    """
    side = "L" if chose_left else "R"
    return side if was_rewarded else side.lower()


def _raw_word(chose_left_window, was_rewarded_window):
    return "".join(
        _trial_letter(cl, wr) for cl, wr in zip(chose_left_window, was_rewarded_window)
    )


def _generic_word(chose_left_window, was_rewarded_window):
    """Collapse a window to stay/switch relative to its oldest (first) trial.
    KM_processCeliaWord.m:63-86.
    """
    initial_chose_left = chose_left_window[0]
    chars = []
    for cl, wr in zip(chose_left_window, was_rewarded_window):
        stay = cl == initial_chose_left
        letter = "A" if stay else "B"
        chars.append(letter if wr else letter.lower())
    return "".join(chars)


def add_word_labels(trial_table, levels=(1, 2, 3)):
    """Return a copy of `trial_table` with word_lL / word_lL_generic columns
    added for each L in `levels`. Requires `chose_left` and `was_rewarded`
    columns (as produced by behavior.trial_table.build_trial_table), in
    trial chronological order.
    """
    trial_table = trial_table.copy()
    chose_left = trial_table["chose_left"].to_numpy()
    was_rewarded = trial_table["was_rewarded"].to_numpy()
    n_trials = len(trial_table)

    for level in levels:
        raw_col = np.full(n_trials, None, dtype=object)
        generic_col = np.full(n_trials, None, dtype=object)
        for t in range(level - 1, n_trials):
            window = slice(t - level + 1, t + 1)
            raw_col[t] = _raw_word(chose_left[window], was_rewarded[window])
            generic_col[t] = _generic_word(chose_left[window], was_rewarded[window])
        trial_table[f"word_l{level}"] = raw_col
        trial_table[f"word_l{level}_generic"] = generic_col

    return trial_table


def _lag(values, lag, n_trials):
    """values[i - lag] for each i, NaN where i - lag < 0."""
    out = np.full(n_trials, np.nan)
    if lag < n_trials:
        out[lag:] = values[: n_trials - lag]
    return out


def add_lag_features(trial_table, n_lags=3):
    """Add explicit n-1, n-2, ..., n-`n_lags` lag columns plus concatenated
    sequence-string columns, per lab convention (not a MATLAB port).

    Adds, for k in 1..n_lags:
      {k}_Reward  -- was_rewarded at trial t-k (1=rewarded, 0=unrewarded)
      {k}_Choice  -- chose_right at trial t-k (1=right, 0=left)
      {k}_Switch  -- whether trial t-k itself differed in choice side from
                     trial t-k-1 (1=switch, 0=stay)
    plus a `switched` column (trial t's own switch status relative to t-1,
    the lag-0 / unlagged version of {k}_Switch), and sequence-string columns
    (most recent trial first, e.g. "101" = [t-1, t-2, t-3]):
      reward_seq_{n_lags}, switch_seq_{n_lags}          -- "0"/"1" chars
      choice_seq_{n_lags}                                -- "0"/"1" chars
      choice_seq_{n_lags}_letters                        -- "R"/"L" chars

    All lag/sequence values are None/NaN wherever the required trial history
    doesn't exist yet (nullable Int64 columns; sequence strings are None if
    any underlying lag value in the window is missing).
    """
    trial_table = trial_table.copy()
    n_trials = len(trial_table)

    reward = trial_table["was_rewarded"].astype(int).to_numpy()
    choice_right = trial_table["chose_right"].astype(int).to_numpy()

    switched = np.full(n_trials, np.nan)
    switched[1:] = (choice_right[1:] != choice_right[:-1]).astype(float)
    trial_table["switched"] = pd.array(switched, dtype="Int64")

    lag_reward, lag_choice, lag_switch = {}, {}, {}
    for k in range(1, n_lags + 1):
        lag_reward[k] = _lag(reward, k, n_trials)
        lag_choice[k] = _lag(choice_right, k, n_trials)
        lag_switch[k] = _lag(switched, k, n_trials)
        trial_table[f"{k}_Reward"] = pd.array(lag_reward[k], dtype="Int64")
        trial_table[f"{k}_Choice"] = pd.array(lag_choice[k], dtype="Int64")
        trial_table[f"{k}_Switch"] = pd.array(lag_switch[k], dtype="Int64")

    def _seq(lag_dict, char_map=None):
        seq = np.full(n_trials, None, dtype=object)
        for i in range(n_trials):
            vals = [lag_dict[k][i] for k in range(1, n_lags + 1)]
            if any(np.isnan(v) for v in vals):
                continue
            chars = [str(int(v)) for v in vals]
            if char_map is not None:
                chars = [char_map[c] for c in chars]
            seq[i] = "".join(chars)
        return seq

    trial_table[f"reward_seq_{n_lags}"] = _seq(lag_reward)
    trial_table[f"choice_seq_{n_lags}"] = _seq(lag_choice)
    trial_table[f"choice_seq_{n_lags}_letters"] = _seq(lag_choice, char_map={"1": "R", "0": "L"})
    trial_table[f"switch_seq_{n_lags}"] = _seq(lag_switch)

    return trial_table


def add_reward_seq_2(trial_table):
    """Add a `reward_seq_2` column: the leading 2 characters of the existing
    `reward_seq_3` column (t-1, t-2 -- reward_seq_3 is most-recent-first, per
    add_lag_features' docstring, e.g. "101" = [t-1, t-2, t-3]), i.e. the
    2-trial-back-only retrospective reward-history bit-string ("00"/"01"/
    "10"/"11"), for the simplified model series' Model 2 (2-bit variant).

    Requires `reward_seq_3` (from add_lag_features) already present.
    """
    trial_table = trial_table.copy()
    trial_table["reward_seq_2"] = trial_table["reward_seq_3"].str[:2]
    return trial_table


def evaluate_word_outcomes(
    trial_table,
    group_col,
    zscore_windows=None,
    peth_trial_table=None,
    peth_time=None,
    decision_window_s=(0.0, 1.0),
    reward_window_s=(1.0, 3.0),
):
    """Group trials by a word/sequence label column (e.g. "word_l2_generic",
    "reward_seq_3") and summarize behavioral -- and, if photometry is
    supplied, physiological -- outcomes on trial t within each group.

    trial_table : full per-trial DataFrame, already run through
        add_word_labels/add_lag_features (needs `group_col`, `switched`,
        `1_Reward`, `chose_right`). Behavioral stats are computed over every
        trial with a non-null `group_col` value, regardless of photometry
        availability.
    zscore_windows, peth_trial_table, peth_time : optional trio for
        photometry outcomes. `zscore_windows` (n_peth_trials, n_samples) and
        `peth_trial_table` (same length/order, e.g. the trial-table subset
        actually used to build the PETH) must line up row-for-row;
        `peth_time` gives the seconds-from-event offset of each column.
        All three are required together, or all omitted.
    decision_window_s, reward_window_s : (start_s, end_s) sub-windows of the
        PETH used for the photometry peak-Z and AUC summaries.

    Returns a DataFrame indexed by group label, sorted by trial count
    (most common pattern first), with columns:
      n_trials, p_stay, p_switch, p_choice_right,
      n_win_stay_eligible, win_stay_prob,
      n_lose_switch_eligible, lose_switch_prob,
      and, only if photometry was supplied:
      n_photometry_trials, mean_peak_z_decision, mean_auc_decision,
      mean_peak_z_reward, mean_auc_reward.

    Win-Stay = P(trial t is a stay | trial t-1 was rewarded).
    Lose-Switch = P(trial t is a switch | trial t-1 was unrewarded).

    On MATLAB's "hasP" gate (processCeliaWord.m:101-108): that gate uses a
    HALF-window minPtsOffset vs. behavior.sync's hasAllPhotometryData's FULL
    window, but it's computed against the SAME already-persisted trialTable
    that behavior.sync's stricter gate has already zeroed
    photometryCenterInIndex/photometrySideOutIndex on for any failing trial
    -- so on that (already-gated) table, hasAllPhotometryData passing
    algebraically implies hasP passing too (full-window threshold > half-window
    threshold), and hasAllPhotometryData failing means the indices are already
    0, which fails hasP's own bound regardless. hasP's practical effect is
    therefore fully subsumed by hasAllPhotometryData on this pipeline's data --
    no separate gate is needed here; peth_trial_table/zscore_windows already
    reflect behavior.sync's gate via their own `photometry_*_index >= 0`
    filtering upstream (pipeline.py).
    """
    rows = []
    for label, grp in trial_table.groupby(group_col, dropna=True):
        switched = grp["switched"]
        p_switch = switched.astype("float").mean(skipna=True)
        p_stay = 1.0 - p_switch if pd.notna(p_switch) else np.nan

        prev_rewarded = (grp["1_Reward"] == 1).fillna(False)
        prev_unrewarded = (grp["1_Reward"] == 0).fillna(False)
        win_stay_pool = switched.loc[prev_rewarded]
        lose_switch_pool = switched.loc[prev_unrewarded]
        win_stay_prob = (win_stay_pool == 0).mean() if len(win_stay_pool) else np.nan
        lose_switch_prob = (lose_switch_pool == 1).mean() if len(lose_switch_pool) else np.nan

        rows.append(
            dict(
                group=label,
                n_trials=len(grp),
                p_stay=p_stay,
                p_switch=p_switch,
                p_choice_right=grp["chose_right"].mean(),
                n_win_stay_eligible=int(prev_rewarded.sum()),
                win_stay_prob=win_stay_prob,
                n_lose_switch_eligible=int(prev_unrewarded.sum()),
                lose_switch_prob=lose_switch_prob,
            )
        )

    summary = pd.DataFrame(rows).set_index("group").sort_values("n_trials", ascending=False)

    if zscore_windows is not None:
        if peth_trial_table is None or peth_time is None:
            raise ValueError(
                "peth_trial_table and peth_time are required together with zscore_windows"
            )
        dx = float(peth_time[1] - peth_time[0])
        dec_mask = (peth_time >= decision_window_s[0]) & (peth_time <= decision_window_s[1])
        rew_mask = (peth_time >= reward_window_s[0]) & (peth_time <= reward_window_s[1])

        photo = pd.DataFrame(
            {
                group_col: peth_trial_table[group_col].to_numpy(),
                "peak_z_decision": zscore_windows[:, dec_mask].max(axis=1),
                "auc_decision": np.trapz(zscore_windows[:, dec_mask], dx=dx, axis=1),
                "peak_z_reward": zscore_windows[:, rew_mask].max(axis=1),
                "auc_reward": np.trapz(zscore_windows[:, rew_mask], dx=dx, axis=1),
            }
        )
        photo_summary = photo.groupby(group_col, dropna=True).agg(
            n_photometry_trials=("peak_z_decision", "size"),
            mean_peak_z_decision=("peak_z_decision", "mean"),
            mean_auc_decision=("auc_decision", "mean"),
            mean_peak_z_reward=("peak_z_reward", "mean"),
            mean_auc_reward=("auc_reward", "mean"),
        )
        summary = summary.join(photo_summary, how="left")

    return summary
