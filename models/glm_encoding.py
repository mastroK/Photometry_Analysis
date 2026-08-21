"""
Time-resolved and single-window GLM encoding models for mPFC GCaMP6 signals,
predicting trial-level event-aligned z-score (alignment.windowing.
compute_event_aligned_zscore) from choice, outcome, choice-x-outcome,
reward history, |Q_diff|, and Behavioral_State.
"""

import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from config.params import DECISION_WINDOW_S, DEFAULT_TIME_RESOLVED_GLM_FORMULA


def _build_predictor_frame(trial_table):
    """Map trial_table's raw columns onto the clean, formula-ready predictor
    names used by DEFAULT_TIME_RESOLVED_GLM_FORMULA / EXPANDED_TIME_RESOLVED_
    GLM_FORMULA. Needed because several of trial_table's own column names
    (e.g. "1_Reward") start with a digit and aren't valid patsy/Python
    identifiers.

    Requires trial_table to already have chose_right/was_rewarded (behavior.
    trial_table.build_trial_table), {k}_Reward/{k}_Choice for k in 1..3
    (behavior.word_encoding.add_lag_features), and Q_diff/Behavioral_State
    (external.bandit_state_adapter.add_bandit_state_features).

    The expanded-formula columns (RPE/Q_chosen/Q_unchosen/belief_p_right/
    switch-dynamics -- behavior.switch_dynamics, wired into pipeline.
    run_session) are included whenever present, via .get() with a NaN
    fallback, so this stays backward compatible with any trial_table that
    predates those columns -- DEFAULT_TIME_RESOLVED_GLM_FORMULA doesn't
    reference them, so their presence/absence is a no-op for existing callers.
    """
    n = len(trial_table)
    nan_col = pd.Series(np.full(n, np.nan), index=trial_table.index)

    def _get(col, dtype=float):
        return trial_table[col].astype(dtype) if col in trial_table.columns else nan_col

    def _get_categorical(col):
        # Unlike _get, does NOT .astype(str) -- these columns (word_l*,
        # reward_seq_*) already hold plain strings/None (object dtype), and
        # casting None -> str would turn it into the literal string "None"
        # instead of leaving it NaN-like for listwise deletion to catch.
        return trial_table[col] if col in trial_table.columns else nan_col

    return pd.DataFrame({
        "Choice": trial_table["chose_right"].astype(float),
        "Port": pd.Series(np.where(trial_table["chose_right"].to_numpy(), 1.0, -1.0), index=trial_table.index),
        "Reward": trial_table["was_rewarded"].astype(float),
        "Reward_lag1": trial_table["1_Reward"].astype(float),
        "Reward_lag2": trial_table["2_Reward"].astype(float),
        "Reward_lag3": trial_table["3_Reward"].astype(float),
        "Choice_lag1": _get("1_Choice"),
        "Choice_lag2": _get("2_Choice"),
        "Choice_lag3": _get("3_Choice"),
        "Q_diff_abs": trial_table["Q_diff"].abs().astype(float),
        "Q_diff_signed": _get("Q_diff"),
        "Q_chosen": _get("Q_chosen"),
        "Q_unchosen": _get("Q_unchosen"),
        "Q_total": _get("Q_total"),
        "RPE": _get("RPE"),
        "RPE_abs": _get("RPE_abs"),
        "belief_p_right": _get("belief_p_right"),
        "true_switch": _get("true_switch"),
        "trials_since_switch": _get("trials_since_switch"),
        "trials_since_switch_sq": _get("trials_since_switch_sq"),
        "trials_since_switch_expdecay": _get("trials_since_switch_expdecay"),
        "detected_switch": _get("detected_switch"),
        "switch_detection_lag": _get("switch_detection_lag"),
        "first_win_after_switch": _get("first_win_after_switch"),
        "first_loss_after_switch": _get("first_loss_after_switch"),
        # not referenced by DEFAULT_/EXPANDED_TIME_RESOLVED_GLM_FORMULA -- opt-in
        # nuisance covariate for testing whether the baseline-vs-reward-rate
        # confound (see baseline_reward_rate_diagnostic.py) contaminates the
        # expanded formula's own history terms, not just RPE_signed.
        "local_reward_rate": _get("local_reward_rate"),
        "Behavioral_State": trial_table["Behavioral_State"].astype(str),
        "reward_seq_2": _get_categorical("reward_seq_2"),
        "reward_seq_3": _get_categorical("reward_seq_3"),
        "word_l1": _get_categorical("word_l1"),
        "word_l2": _get_categorical("word_l2"),
        "word_l3": _get_categorical("word_l3"),
        # Model 3 (run_model_series_comparison.py): per-mouse Q_diff quantile
        # bin id, computed and attached externally (not derivable from
        # trial_table alone -- see _qdiff_bin_edges/_add_qdiff_bins).
        "Q_diff_bin": _get_categorical("Q_diff_bin"),
    })


def fit_time_resolved_glm(
    zscore_windows,
    peth_time,
    trial_table,
    formula=DEFAULT_TIME_RESOLVED_GLM_FORMULA,
    standardize=True,
    min_resid_dof=10,
    min_retained_frac=None,
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

    min_resid_dof : every prior caller had a FIXED n_trials across every
    timepoint (only the one shared listwise-deletion mask above affects it),
    so a formula with too few trials relative to its parameter count would
    have been unstable at every t_idx alike -- easy to notice. Once a caller
    passes zscore_windows with per-trial NaN tails (e.g.
    pipeline.run_session's truncate_at_side_out), n_trials can shrink sharply
    at LATE timepoints only, and once it drops close to the design matrix's
    parameter count the OLS fit becomes rank-deficient/near-singular --
    confirmed in practice: beta swinging to +-1000 and r_squared snapping to
    a spurious 1.0 in the last few timepoints of a truncated word_l2 fit.
    Rather than return that degenerate fit, any timepoint whose residual
    degrees of freedom (nobs - k_params) fall below min_resid_dof gets NaN
    for every returned value instead (same graceful-failure convention as
    cross_validate_window_glm skipping a rank-deficient CV fold) -- still
    correctly reflected via n_trials/nobs at that timepoint, so
    viz.glm_plots.plot_glm_coefficients(show_n_trials=True) shows why the
    trajectory stops rather than silently plotting nonsense.

    min_retained_frac : a second, independent guard for the same truncated-
    zscore_windows scenario -- min_resid_dof only catches a timepoint once
    n_trials collapses close to the parameter count (near-total numerical
    failure). Well before that point, the fit is still "stable" but is
    silently regressing on a shrinking, self-selected subpopulation (only
    the longer-dwelling trials survive at later t), which is a
    representativeness problem, not a numerical one -- the returned curve
    looks like one continuously-supported trajectory when it isn't. When
    set (opt-in, None disables this guard), any timepoint whose n_trials
    falls below min_retained_frac * (n_trials at the first, always-fully-
    populated pre-event timepoint) gets NaN'd the same way min_resid_dof's
    branch does. Every pre-existing (non-truncated) caller has constant
    n_trials across every timepoint, so retained_frac is always 1.0 and this
    never fires even if enabled -- zero behavior change unless explicitly
    passed by a truncation-aware caller.
    """
    zscore_windows = np.asarray(zscore_windows, dtype=float)
    if len(trial_table) != zscore_windows.shape[0]:
        raise ValueError(
            f"trial_table has {len(trial_table)} rows but zscore_windows has "
            f"{zscore_windows.shape[0]} -- both must be row-aligned"
        )

    predictors = _build_predictor_frame(trial_table)
    # _build_predictor_frame always returns the full superset of columns
    # (both DEFAULT_TIME_RESOLVED_GLM_FORMULA's and EXPANDED_TIME_RESOLVED_
    # GLM_FORMULA's), so listwise deletion must only look at the columns
    # THIS formula actually references -- otherwise fitting the default
    # formula would incorrectly drop trials for being missing e.g.
    # trials_since_switch or belief_p_right, which it never uses.
    formula_tokens = set(re.findall(r"\b\w+\b", formula))
    relevant_cols = [c for c in predictors.columns if c in formula_tokens]
    valid = predictors[relevant_cols].notna().all(axis=1).to_numpy()
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
    n_degenerate = 0
    n_underrepresented = 0
    n_empty = 0
    n_trials_t0 = None
    known_terms = None
    for t_idx in range(zscore_windows.shape[1]):
        data = predictors.copy()
        data["Z"] = zscore_windows[:, t_idx]

        # A caller passing zscore_windows with per-trial NaN tails on a SMALL
        # subset (e.g. one mouse's own session rows, not the full pooled
        # cohort -- see cluster_permutation_word_l2.py) can hit a late
        # timepoint where every remaining trial is already past its own
        # truncation edge, leaving zero non-NaN Z values. smf.ols's
        # missing="drop" then builds a zero-row design matrix and crashes in
        # statsmodels' own constant-check (np.max on an empty array) before
        # model.df_resid even exists for min_resid_dof to catch. Treat this
        # the same as any other degenerate timepoint -- NaN it out -- using
        # the most recent successful fit's term names (Z's own NaN pattern
        # is timepoint-specific; the design's terms are not).
        if data["Z"].notna().sum() == 0:
            n_empty += 1
            if known_terms is None:
                raise ValueError(
                    f"fit_time_resolved_glm: every Z value is NaN at t_idx={t_idx} "
                    "(peth_time="
                    f"{np.asarray(peth_time)[t_idx]}) and no earlier timepoint fit "
                    "successfully -- cannot infer this formula's term names to NaN-fill this row"
                )
            row = {"r_squared": np.nan, "r_squared_adj": np.nan, "n_trials": 0}
            for term in known_terms:
                row[f"{term}_beta"] = np.nan
                row[f"{term}_se"] = np.nan
                row[f"{term}_tstat"] = np.nan
                row[f"{term}_pvalue"] = np.nan
            rows.append(row)
            continue

        model = smf.ols(formula, data=data, missing="drop").fit()
        known_terms = model.params.index
        n_trials = int(model.nobs)
        if n_trials_t0 is None:
            n_trials_t0 = n_trials

        is_degenerate = model.df_resid < min_resid_dof
        is_underrepresented = (
            min_retained_frac is not None and n_trials < min_retained_frac * n_trials_t0
        )
        if is_degenerate:
            n_degenerate += 1
        if is_underrepresented:
            n_underrepresented += 1

        if is_degenerate or is_underrepresented:
            row = {"r_squared": np.nan, "r_squared_adj": np.nan, "n_trials": n_trials}
            for term in model.params.index:
                row[f"{term}_beta"] = np.nan
                row[f"{term}_se"] = np.nan
                row[f"{term}_tstat"] = np.nan
                row[f"{term}_pvalue"] = np.nan
            rows.append(row)
            continue

        row = {
            "r_squared": model.rsquared,
            "r_squared_adj": model.rsquared_adj,
            "n_trials": n_trials,
        }
        for term in model.params.index:
            row[f"{term}_beta"] = model.params[term]
            row[f"{term}_se"] = model.bse[term]
            row[f"{term}_tstat"] = model.tvalues[term]
            row[f"{term}_pvalue"] = model.pvalues[term]
        rows.append(row)

    if n_degenerate:
        print(f"fit_time_resolved_glm: {n_degenerate}/{zscore_windows.shape[1]} timepoint(s) had "
              f"fewer than min_resid_dof={min_resid_dof} residual degrees of freedom "
              "(rank-deficient/near-singular fit) -- NaN'd instead of returned")
    if n_underrepresented:
        print(f"fit_time_resolved_glm: {n_underrepresented}/{zscore_windows.shape[1]} timepoint(s) "
              f"retained fewer than min_retained_frac={min_retained_frac} of the original "
              "trial count (self-selected long-dwelling subsample) -- NaN'd instead of returned")
    if n_empty:
        print(f"fit_time_resolved_glm: {n_empty}/{zscore_windows.shape[1]} timepoint(s) had "
              "zero non-NaN Z values (every trial already past its own truncation edge) -- "
              "NaN'd instead of raising")

    return pd.DataFrame(rows, index=pd.Index(np.asarray(peth_time), name="peth_time"))


def cross_validate_window_glm(zscore_windows, peth_time, trial_table, formula,
                               decision_window_s=DECISION_WINDOW_S, n_splits=5, random_state=0,
                               min_valid_frac=0.5):
    """Cross-validated R^2 for `formula`'s RHS predicting a scalar per-trial
    target: the mean Z within `decision_window_s` of the aligning event
    (same window/logic as rpe_analysis_stats.add_derived_columns' post_amp).

    fit_time_resolved_glm fits an independent in-sample OLS at every PETH
    timepoint, which has no single number to cross-validate or compare
    models by; this collapses to one scalar target (matching how post_amp/
    peak_z_* are already used elsewhere for scalar model comparisons) and
    reports genuine held-out R^2: KFold(n_splits, shuffle=True,
    random_state) + LinearRegression + r2_score per fold, mean/std across
    folds -- the same pattern as rpe_analysis_stats._cv_r2, replicated here
    (not imported) to keep this module independent of that analysis script.

    formula : just the RHS (e.g. "Reward + Port + Port:Reward" or
        "C(word_l2)") -- expanded into a numeric design matrix via
        patsy.dmatrix, the same categorical dummy-coding statsmodels' smf.ols
        would use for a full "Z ~ ..." formula.

    min_valid_frac : a caller passing zscore_windows with per-trial NaN tails
    (e.g. pipeline.run_session's truncate_at_side_out) can have a trial whose
    own side_out lands inside decision_window_s -- part of that trial's
    window is genuine post-departure NaN. A plain .mean() over the window
    turns ONE NaN sample into a NaN for the trial's entire target, and
    np.isfinite(target) then drops that trial outright -- confirmed in
    practice to silently drop 38-46% of trials per mouse (every trial whose
    dwell time falls under decision_window_s's upper bound), a non-random
    subsample biased toward longer-dwelling trials, not a random data loss.
    Using nanmean recovers a valid target from whatever samples ARE present,
    and min_valid_frac (fraction of the window that must be non-NaN) keeps a
    trial from surviving on a single sample right at its own truncation edge.
    For every pre-existing (non-truncated) caller, zscore_windows never
    contains NaN, so every trial has full window coverage and this guard
    never triggers -- zero behavior change there.

    Returns dict(r2_mean, r2_std, r2_scores, n_trials).
    """
    zscore_windows = np.asarray(zscore_windows, dtype=float)
    peth_time = np.asarray(peth_time)
    predictors = _build_predictor_frame(trial_table)

    formula_tokens = set(re.findall(r"\b\w+\b", formula))
    relevant_cols = [c for c in predictors.columns if c in formula_tokens]
    valid = predictors[relevant_cols].notna().all(axis=1).to_numpy()

    window_mask = (peth_time >= decision_window_s[0]) & (peth_time <= decision_window_s[1])
    window_data = zscore_windows[:, window_mask]
    n_valid_in_window = np.sum(~np.isnan(window_data), axis=1)
    with np.errstate(invalid="ignore"):
        target = np.nanmean(window_data, axis=1)
    enough_coverage = n_valid_in_window >= min_valid_frac * window_mask.sum()
    target = np.where(enough_coverage, target, np.nan)
    valid = valid & np.isfinite(target)

    predictors = predictors.loc[valid].reset_index(drop=True)
    target = target[valid]

    design = dmatrix(formula, predictors, return_type="dataframe")
    design = design.drop(columns=["Intercept"], errors="ignore")
    X = design.to_numpy()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    n_skipped = 0
    for train_idx, test_idx in kf.split(X):
        X_train = X[train_idx]
        # A rich categorical (e.g. an 8-level 3-trial word/sequence) with a
        # modest per-mouse trial count can leave a fold's training design
        # matrix rank-deficient or numerically near-singular (a rare
        # category thinly/unevenly represented across folds) -- plain OLS
        # then returns enormous, meaningless coefficients that blow up the
        # held-out R^2 to something like -1e26 rather than raising. Skip
        # such a fold rather than letting one pathological fold dominate
        # the mean; if every fold is unusable, r2_mean comes back NaN (the
        # same graceful-failure convention models.fir_glm.fit_fir_glm's own
        # SVD-non-convergence errors already produce for its caller).
        if np.linalg.matrix_rank(X_train) < X_train.shape[1]:
            n_skipped += 1
            continue
        model = LinearRegression().fit(X_train, target[train_idx])
        pred = model.predict(X[test_idx])
        score = r2_score(target[test_idx], pred)
        if not np.isfinite(score) or score < -10:
            n_skipped += 1
            continue
        scores.append(score)

    if n_skipped:
        print(f"cross_validate_window_glm: skipped {n_skipped}/{n_splits} fold(s) with a "
              "rank-deficient or numerically unstable fit")

    return dict(
        r2_mean=float(np.mean(scores)) if scores else float("nan"),
        r2_std=float(np.std(scores)) if scores else float("nan"),
        r2_scores=np.array(scores),
        n_trials=int(len(target)),
    )


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
