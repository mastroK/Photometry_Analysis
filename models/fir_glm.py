"""
Continuous time-shifted FIR (Finite Impulse Response) deconvolution GLM,
focused on the side_in (choice/outcome) response across reward types and
reward histories.

This module regresses the CONTINUOUS session-level photometry trace
(pipeline.run_session's `dff`/`zscore`) against a bank of time-shifted
impulse regressors anchored at side_in, one set of 2T+1 lag coefficients per
reward-history group/parametric feature. Because every group's full
+/-T-second kernel is fit jointly against the same continuous trace,
kernels for temporally-overlapping trials (e.g. quick win-stay sequences)
are deconvolved rather than confounded, and the fitted coefficients
beta_f(tau) trace out that group's own temporal kernel.

By design this model does NOT include center_in/center_out/side_out as
nuisance regressors -- per lab decision, the analysis question here is the
side_in response itself (across reward type/history), not separating it
from its neighboring events, so the design matrix stays restricted to
side_in-anchored regressors only.
"""

from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from config.params import DEFAULT_HEMISPHERE, FINAL_SAMPLE_FREQ_HZ
from io_utils.raw_loader import parse_session_id
from pipeline import run_session

DEFAULT_LAG_SECONDS = 1.0  # +/- window per event; ~19-20 bins at ~18.52 Hz
DEFAULT_ALPHAS = np.logspace(-2, 6, 30)
DEFAULT_N_SPLITS = 10
DEFAULT_TEST_SIZE = 0.2
DEFAULT_GROUP_COLUMN = "reward_seq_3"


def build_event_impulses(trial_table, n_samples, group_col=DEFAULT_GROUP_COLUMN, group_values=None):
    """Build {feature_name: (n_samples,) impulse vector} -- 0 everywhere
    except a single non-zero sample at each valid trial's side_in index.

    Discrete reward-history groups (impulse height 1.0): one
    "side_in_<value>" channel per unique non-null value of
    trial_table[group_col] -- e.g. group_col="reward_seq_3" (behavior.
    word_encoding.add_lag_features' "111"/"110"/.../"000" 3-trial-back
    win/loss patterns) gives up to 8 discrete kernels; group_col=
    "word_l3_generic" gives the AAA/aAA/aaA-style stay/switch+reward
    patterns instead. Any trial_table column works (was_rewarded,
    Behavioral_State, choice_seq_3, ...).

    Parametric features (impulse height = the trial's own continuous
    covariate, placed at side_in onset regardless of group):
      side_in_x_Qdiff  -- |Q_diff| (external.bandit_state_adapter)
      side_in_x_Choice -- chose_left (Left=1, Right=0)

    A trial contributes to a feature only where its side_in index is valid
    (>=0, i.e. resolved by behavior.sync.align_behavior_to_photometry) and,
    for parametric features, the covariate itself is finite (NaN during
    e.g. a skipped Q-learning fit).

    group_values : explicit ordered list of group_col values to build a
        channel for -- pass this (rather than leaving it to be inferred from
        this trial_table alone) when pooling multiple sessions, so every
        session's design matrix has the same "side_in_<group>" columns in
        the same order even if a given session happens not to contain every
        pattern (see build_pooled_fir_dataset). Defaults to this
        trial_table's own sorted unique non-null values.
    """
    side_in_idx = trial_table["photometry_side_in_index"].to_numpy()
    group_col_values = trial_table[group_col]
    if group_values is None:
        group_values = sorted(group_col_values.dropna().unique())

    def _place(idx, values=None, extra_valid=None):
        vec = np.zeros(n_samples)
        valid = idx >= 0
        if values is not None:
            values = np.asarray(values, dtype=float)
            valid = valid & np.isfinite(values)
        if extra_valid is not None:
            valid = valid & extra_valid
        idx_valid = idx[valid].astype(int)
        vec[idx_valid] = 1.0 if values is None else values[valid]
        return vec

    impulses = {}
    for group in group_values:
        extra_valid = (group_col_values == group).to_numpy()
        impulses[f"side_in_{group}"] = _place(side_in_idx, extra_valid=extra_valid)

    impulses["side_in_x_Qdiff"] = _place(side_in_idx, values=trial_table["Q_diff"].abs())
    impulses["side_in_x_Choice"] = _place(side_in_idx, values=trial_table["chose_left"].astype(float))

    return impulses


def build_shifted_design_matrix(impulses, n_lags):
    """Expand each (n_samples,) impulse column into 2*n_lags+1 time-shifted
    columns (lags -n_lags..+n_lags, in samples), building the FIR design
    matrix Phi of shape (n_samples, n_features*(2*n_lags+1)).

    Phi[t, feature, lag] = impulse_feature[t - lag] (zero where t-lag would
    fall outside the session, i.e. no wraparound) -- so the fitted
    coefficient at a given lag describes that feature's contribution `lag`
    samples after its own impulse (or before, for negative lag, capturing
    any anticipatory ramp).

    Returns (Phi, column_names) where column_names[j] = (feature_name,
    lag_in_samples) for Phi's j-th column, in feature-major, lag-minor order
    (matches models.fir_glm.reshape_kernels' expected layout).
    """
    feature_names = list(impulses.keys())
    n_samples = len(impulses[feature_names[0]])
    lags = np.arange(-n_lags, n_lags + 1)
    n_lag_bins = len(lags)

    Phi = np.zeros((n_samples, len(feature_names) * n_lag_bins))
    column_names = []
    for fi, name in enumerate(feature_names):
        x = impulses[name]
        for li, lag in enumerate(lags):
            shifted = np.roll(x, lag)
            if lag > 0:
                shifted[:lag] = 0.0
            elif lag < 0:
                shifted[lag:] = 0.0
            Phi[:, fi * n_lag_bins + li] = shifted
            column_names.append((name, int(lag)))
    return Phi, column_names


def build_task_mask_and_groups(trial_table, n_samples, n_lags):
    """Boolean mask of samples to include in fitting, plus a parallel
    integer trial-group id per included sample (for GroupShuffleSplit).

    Since every regressor in this model is anchored at side_in (see
    build_event_impulses), a trial's included window is exactly its own
    [side_in - n_lags, side_in + n_lags] span -- precisely the samples any
    side_in-anchored column can be non-zero at. Samples outside every
    trial's window (long ITI/baseline, or time near center_in/center_out/
    side_out with no side_in nearby) are excluded entirely -- fitting the
    FIR kernels against samples with no side_in-anchored predictor active
    would waste degrees of freedom on unconstrained baseline noise.
    """
    mask = np.zeros(n_samples, dtype=bool)
    groups = np.full(n_samples, -1, dtype=int)

    side_in = trial_table["photometry_side_in_index"].to_numpy()

    for trial_idx, center in enumerate(side_in):
        if center < 0:
            continue
        lo = max(0, center - n_lags)
        hi = min(n_samples, center + n_lags + 1)
        mask[lo:hi] = True
        groups[lo:hi] = trial_idx

    return mask, groups


def fit_fir_glm(y, Phi, mask, groups, n_splits=DEFAULT_N_SPLITS, test_size=DEFAULT_TEST_SIZE,
                alphas=DEFAULT_ALPHAS, random_state=0):
    """Fit the FIR design matrix against the continuous signal, restricted to
    `mask` (see build_task_mask_and_groups).

    Cross-validation: `n_splits`-fold GroupShuffleSplit by trial (`groups`),
    so held-out folds are whole trials, not individual samples -- samples
    from the same trial are highly autocorrelated (same underlying calcium
    transient), so a plain random sample-level split would leak information
    between train/test. Each fold fits its own RidgeCV(alphas) on the
    training samples and scores out-of-sample R^2/MSE on the held-out
    trials' samples; the final returned model is a RidgeCV refit on ALL
    included samples (its own internal generalized cross-validation picks
    the regularization strength for the reported kernel weights).

    Returns (final_model, cv_results); cv_results contains fold-level
    r2_scores/mse_scores/fold_betas (n_splits, n_columns) -- the fold_betas
    are what viz.fir_plots.plot_fir_kernels uses for the +/-SEM/CI band
    around each kernel, not just the single final-fit point estimate.
    """
    y_fit = y[mask]
    Phi_fit = Phi[mask]
    groups_fit = groups[mask]

    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    fold_r2, fold_mse, fold_betas = [], [], []
    for train_idx, test_idx in splitter.split(Phi_fit, y_fit, groups=groups_fit):
        fold_model = RidgeCV(alphas=alphas)
        fold_model.fit(Phi_fit[train_idx], y_fit[train_idx])
        pred = fold_model.predict(Phi_fit[test_idx])
        fold_r2.append(r2_score(y_fit[test_idx], pred))
        fold_mse.append(mean_squared_error(y_fit[test_idx], pred))
        fold_betas.append(fold_model.coef_)

    final_model = RidgeCV(alphas=alphas)
    final_model.fit(Phi_fit, y_fit)

    cv_results = dict(
        r2_scores=np.array(fold_r2),
        mse_scores=np.array(fold_mse),
        r2_mean=float(np.mean(fold_r2)),
        r2_std=float(np.std(fold_r2)),
        mse_mean=float(np.mean(fold_mse)),
        mse_std=float(np.std(fold_mse)),
        fold_betas=np.array(fold_betas),
        best_alpha=float(final_model.alpha_),
        n_samples_fit=int(mask.sum()),
    )
    return final_model, cv_results


def reshape_kernels(beta, column_names, n_lags):
    """Reshape a fitted coefficient vector (n_columns,) -- or a stack of them,
    (n_folds, n_columns), e.g. cv_results['fold_betas'] -- back into
    {feature_name: (2*n_lags+1,) or (n_folds, 2*n_lags+1)} temporal kernels,
    plus the shared lag_time_s axis (seconds, 0 = the feature's own event).
    """
    beta = np.asarray(beta)
    n_lag_bins = 2 * n_lags + 1
    lag_time_s = np.arange(-n_lags, n_lags + 1) / FINAL_SAMPLE_FREQ_HZ

    feature_names = []
    for name, _ in column_names:
        if name not in feature_names:
            feature_names.append(name)

    is_stacked = beta.ndim == 2
    kernels = {}
    for fi, name in enumerate(feature_names):
        cols = slice(fi * n_lag_bins, (fi + 1) * n_lag_bins)
        kernels[name] = beta[:, cols] if is_stacked else beta[cols]
    return kernels, lag_time_s


def build_and_fit_fir_glm(trial_table, continuous_signal, group_col=DEFAULT_GROUP_COLUMN,
                           n_lags_seconds=DEFAULT_LAG_SECONDS, n_splits=DEFAULT_N_SPLITS,
                           test_size=DEFAULT_TEST_SIZE, alphas=DEFAULT_ALPHAS, random_state=0):
    """End-to-end: trial_table + a session's continuous photometry trace ->
    fitted FIR deconvolution GLM. See module docstring for the full
    rationale and the individual builder/fit/reshape functions above for
    each step's details.

    Returns a dict with: model (final RidgeCV), cv_results, kernels (final
    model's point-estimate kernels), fold_kernels (per-CV-fold kernels, for
    +/-SEM bands), lag_time_s, column_names, n_lags (samples),
    n_samples_included/n_samples_total.
    """
    continuous_signal = np.asarray(continuous_signal, dtype=float)
    n_samples = len(continuous_signal)
    n_lags = int(round(n_lags_seconds * FINAL_SAMPLE_FREQ_HZ))

    impulses = build_event_impulses(trial_table, n_samples, group_col=group_col)
    Phi, column_names = build_shifted_design_matrix(impulses, n_lags)
    mask, groups = build_task_mask_and_groups(trial_table, n_samples, n_lags)

    model, cv_results = fit_fir_glm(
        continuous_signal, Phi, mask, groups,
        n_splits=n_splits, test_size=test_size, alphas=alphas, random_state=random_state,
    )

    kernels, lag_time_s = reshape_kernels(model.coef_, column_names, n_lags)
    fold_kernels, _ = reshape_kernels(cv_results["fold_betas"], column_names, n_lags)

    return dict(
        model=model,
        cv_results=cv_results,
        kernels=kernels,
        fold_kernels=fold_kernels,
        lag_time_s=lag_time_s,
        column_names=column_names,
        n_lags=n_lags,
        n_samples_included=int(mask.sum()),
        n_samples_total=n_samples,
    )


def build_pooled_fir_dataset(session_dirs, hemisphere=DEFAULT_HEMISPHERE, signal="zscore",
                              group_col=DEFAULT_GROUP_COLUMN, n_lags_seconds=DEFAULT_LAG_SECONDS,
                              max_segments=None, hemisphere_for_session=None):
    """Run every session_dir through pipeline.run_session and pool each
    session's own FIR design-matrix rows into one combined multi-session
    dataset.

    hemisphere_for_session : optional callable(session_dir) -> hemisphere_key,
        overriding the single `hemisphere` value per session -- see
        batch_processor.run_batch_sessions's identically-named parameter.
        Hemisphere is a per-session, not per-mouse, property in this cohort
        (config/session_hemisphere_overrides.csv).

    Two passes: (1) load every session (soft-failing/skipping ones that
    error -- same convention as models.glm_data.build_pooled_glm_dataset /
    batch_processor.run_batch_sessions) and collect the UNION of group_col
    values seen across ALL of them; (2) build each session's impulses against
    that fixed, shared group vocabulary (via build_event_impulses'
    group_values=), so every session's design matrix has the same
    "side_in_<group>" columns in the same order even if a given session
    happens not to contain every pattern (e.g. too few trials for a rare
    3-trial history) -- it just gets an all-zero column for that group
    instead of silently shifting every other session's column layout.

    Each session's impulses/design matrix/task mask are still built
    independently against its own n_samples (so build_shifted_design_matrix's
    np.roll shift never wraps across a session boundary), and only the
    already-masked (task-active) rows are kept and stacked -- pooling never
    concatenates raw continuous traces end-to-end. Trial group ids are
    offset per session so GroupShuffleSplit (grouped by trial) never merges
    two different sessions' trials into the same group.

    Returns (y_pooled, Phi_pooled, groups_pooled, column_names, n_lags,
    session_info, total_n_samples) -- session_info is a list of
    {mouse, date, n_samples_included, n_samples_total} dicts for the
    sessions that succeeded; total_n_samples sums n_samples across all
    successful sessions BEFORE masking (for an overall "pct included").
    """
    loaded = []
    n_failed = 0
    for session_dir in session_dirs:
        session_dir = Path(session_dir)
        mouse, date = parse_session_id(session_dir)
        session_hemisphere = hemisphere_for_session(session_dir) if hemisphere_for_session is not None else hemisphere
        try:
            result = run_session(session_dir, hemisphere=session_hemisphere, max_segments=max_segments)
        except Exception as exc:
            print(f"WARNING: skipping session {session_dir} ({mouse} {date}): {exc}")
            n_failed += 1
            continue
        loaded.append((mouse, date, result))

    if not loaded:
        raise RuntimeError(f"No sessions successfully processed out of {list(session_dirs)}")

    all_groups = set()
    for _, _, result in loaded:
        all_groups.update(result["trial_table"][group_col].dropna().unique())
    group_values = sorted(all_groups)
    n_lags = int(round(n_lags_seconds * FINAL_SAMPLE_FREQ_HZ))

    y_parts, phi_parts, group_parts = [], [], []
    column_names = None
    session_info = []
    group_offset = 0
    total_n_samples = 0

    for mouse, date, result in loaded:
        trial_table = result["trial_table"]
        continuous_signal = np.asarray(result[signal], dtype=float)
        n_samples = len(continuous_signal)

        impulses = build_event_impulses(trial_table, n_samples, group_col=group_col, group_values=group_values)
        Phi, cols = build_shifted_design_matrix(impulses, n_lags)
        if column_names is None:
            column_names = cols
        mask, groups = build_task_mask_and_groups(trial_table, n_samples, n_lags)

        y_parts.append(continuous_signal[mask])
        phi_parts.append(Phi[mask])
        group_parts.append(groups[mask] + group_offset)
        group_offset += len(trial_table)
        total_n_samples += n_samples

        session_info.append(dict(
            mouse=mouse, date=date,
            n_samples_included=int(mask.sum()), n_samples_total=n_samples,
        ))

    y_pooled = np.concatenate(y_parts)
    Phi_pooled = np.vstack(phi_parts)
    groups_pooled = np.concatenate(group_parts)
    print(f"Pooled {len(loaded)} session(s) (skipped {n_failed}), {len(y_pooled)} total FIR samples "
          f"across {group_offset} trials ({len(group_values)} '{group_col}' groups: {group_values})")
    return y_pooled, Phi_pooled, groups_pooled, column_names, n_lags, session_info, total_n_samples


def build_and_fit_pooled_fir_glm(session_dirs, hemisphere=DEFAULT_HEMISPHERE, signal="zscore",
                                  group_col=DEFAULT_GROUP_COLUMN, n_lags_seconds=DEFAULT_LAG_SECONDS,
                                  max_segments=None, n_splits=DEFAULT_N_SPLITS, test_size=DEFAULT_TEST_SIZE,
                                  alphas=DEFAULT_ALPHAS, random_state=0, hemisphere_for_session=None):
    """Multi-session counterpart of build_and_fit_fir_glm: pool session_dirs
    via build_pooled_fir_dataset (see its docstring), then fit/reshape
    exactly as the single-session path does. Returns the same dict shape as
    build_and_fit_fir_glm, plus `session_info`.
    """
    y_pooled, Phi_pooled, groups_pooled, column_names, n_lags, session_info, total_n_samples = (
        build_pooled_fir_dataset(
            session_dirs, hemisphere=hemisphere, signal=signal, group_col=group_col,
            n_lags_seconds=n_lags_seconds, max_segments=max_segments,
            hemisphere_for_session=hemisphere_for_session,
        )
    )
    mask_all = np.ones(len(y_pooled), dtype=bool)
    model, cv_results = fit_fir_glm(
        y_pooled, Phi_pooled, mask_all, groups_pooled,
        n_splits=n_splits, test_size=test_size, alphas=alphas, random_state=random_state,
    )
    kernels, lag_time_s = reshape_kernels(model.coef_, column_names, n_lags)
    fold_kernels, _ = reshape_kernels(cv_results["fold_betas"], column_names, n_lags)

    return dict(
        model=model,
        cv_results=cv_results,
        kernels=kernels,
        fold_kernels=fold_kernels,
        lag_time_s=lag_time_s,
        column_names=column_names,
        n_lags=n_lags,
        n_samples_included=len(y_pooled),
        n_samples_total=total_n_samples,
        session_info=session_info,
    )
