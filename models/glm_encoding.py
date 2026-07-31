"""
Time-resolved and single-window GLM encoding models for mPFC GCaMP6 signals,
predicting trial-level event-aligned z-score (alignment.windowing.
compute_event_aligned_zscore) from choice, outcome, choice-x-outcome,
reward history, |Q_diff|, and Behavioral_State.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from config.params import DEFAULT_TIME_RESOLVED_GLM_FORMULA


def _build_predictor_frame(trial_table):
    """Map trial_table's raw columns onto the clean, formula-ready predictor
    names used by DEFAULT_TIME_RESOLVED_GLM_FORMULA. Needed because several
    of trial_table's own column names (e.g. "1_Reward") start with a digit
    and aren't valid patsy/Python identifiers.

    Requires trial_table to already have chose_right/was_rewarded (behavior.
    trial_table.build_trial_table), {k}_Reward for k in 1..3 (behavior.
    word_encoding.add_lag_features), and Q_diff/Behavioral_State (external.
    bandit_state_adapter.add_bandit_state_features).
    """
    return pd.DataFrame({
        "Choice": trial_table["chose_right"].astype(float),
        "Reward": trial_table["was_rewarded"].astype(float),
        "Reward_lag1": trial_table["1_Reward"].astype(float),
        "Reward_lag2": trial_table["2_Reward"].astype(float),
        "Reward_lag3": trial_table["3_Reward"].astype(float),
        "Q_diff_abs": trial_table["Q_diff"].abs().astype(float),
        "Behavioral_State": trial_table["Behavioral_State"].astype(str),
    })


def fit_time_resolved_glm(
    zscore_windows,
    peth_time,
    trial_table,
    formula=DEFAULT_TIME_RESOLVED_GLM_FORMULA,
    standardize=True,
):
    """Fit an independent OLS regression at every time sample in peth_time,
    predicting that column of zscore_windows (Z) from the predictors built
    by _build_predictor_frame (Choice, Reward, Choice:Reward, Reward_lag1-3,
    Q_diff_abs, Behavioral_State), per `formula`.

    zscore_windows : (n_trials, n_samples) trial-level event-aligned
        z-score, e.g. models.glm_data.build_pooled_glm_dataset's pooled
        output or a single session's run_session(...)["all_zscore_windows"].
    trial_table : row-aligned with zscore_windows (row i <-> zscore_windows[i]),
        e.g. the matching peth_trial_table/pooled_trial_table.
    standardize : z-score every numeric predictor column (not the categorical
        Behavioral_State dummies) before fitting, so beta_i(t) is in units of
        "z-score change per 1 SD change in predictor i" and comparable in
        magnitude across predictors of different scale -- the "standardized
        predictors" requirement.

    Trials with any missing predictor (lag warm-up at the start of a session,
    a skipped Q-learning fit) are dropped via one shared listwise-deletion
    mask -- predictors don't depend on t, so the same trials are valid at
    every time sample, unlike Z itself.

    Returns a DataFrame indexed by peth_time with, for each fitted term name
    (e.g. "Choice", "Reward", "Choice:Reward", "Q_diff_abs",
    "C(Behavioral_State)[T.Exploration]", "Intercept"):
      {term}_beta, {term}_se, {term}_tstat, {term}_pvalue
    plus r_squared, r_squared_adj, n_trials (per-timepoint fit sample size).
    """
    zscore_windows = np.asarray(zscore_windows, dtype=float)
    if len(trial_table) != zscore_windows.shape[0]:
        raise ValueError(
            f"trial_table has {len(trial_table)} rows but zscore_windows has "
            f"{zscore_windows.shape[0]} -- both must be row-aligned"
        )

    predictors = _build_predictor_frame(trial_table)
    valid = predictors.notna().all(axis=1).to_numpy()
    n_dropped = int((~valid).sum())
    if n_dropped:
        print(f"fit_time_resolved_glm: dropping {n_dropped}/{len(valid)} trials with a missing "
              "predictor (e.g. reward-history warm-up, no Q-learning fit)")
    predictors = predictors.loc[valid].reset_index(drop=True)
    zscore_windows = zscore_windows[valid]

    if standardize:
        numeric_cols = predictors.select_dtypes(include="number").columns
        predictors = predictors.copy()
        predictors[numeric_cols] = (
            (predictors[numeric_cols] - predictors[numeric_cols].mean())
            / predictors[numeric_cols].std(ddof=1)
        )

    rows = []
    for t_idx in range(zscore_windows.shape[1]):
        data = predictors.copy()
        data["Z"] = zscore_windows[:, t_idx]
        model = smf.ols(formula, data=data, missing="drop").fit()

        row = {
            "r_squared": model.rsquared,
            "r_squared_adj": model.rsquared_adj,
            "n_trials": int(model.nobs),
        }
        for term in model.params.index:
            row[f"{term}_beta"] = model.params[term]
            row[f"{term}_se"] = model.bse[term]
            row[f"{term}_tstat"] = model.tvalues[term]
            row[f"{term}_pvalue"] = model.pvalues[term]
        rows.append(row)

    return pd.DataFrame(rows, index=pd.Index(np.asarray(peth_time), name="peth_time"))


def fit_window_glm(trial_table, target_col, formula, groups=None):
    """Single-window regression predicting a scalar per-trial summary metric
    (target_col, e.g. "peak_z_side_in"/"auc_outcome" -- see
    alignment.windowing.compute_per_trial_event_metrics) from whatever terms
    `formula` references directly by trial_table column name.

    Unlike fit_time_resolved_glm, this does NOT restrict predictors to the
    fixed Choice/Reward/... encoding-model set -- pass a formula referencing
    any trial_table column, including cohort factors not available per-
    timepoint (e.g. "peak_z_side_in ~ Choice + Reward + age_days +
    C(treatment)" once Choice/Reward are added as plain columns, or any raw
    column such as Q_diff/mouse/age_days directly).

    groups : optional trial_table column name (e.g. "mouse") -- if given,
    fits smf.mixedlm(formula, data, groups=data[groups]) for a random
    intercept per group (cohort trials nested within mice) instead of a
    fixed-effects-only OLS fit.

    Returns the fitted statsmodels results object (.params, .pvalues,
    .tvalues, .rsquared/.rsquared_adj for OLS, .summary(), etc.).
    """
    data = trial_table.copy()
    if groups is not None:
        return smf.mixedlm(formula, data=data, groups=data[groups], missing="drop").fit()
    return smf.ols(formula, data=data, missing="drop").fit()
