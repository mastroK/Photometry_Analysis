"""
Sticky Q-learning fit + rule-based behavioral-state classifier -- ported
near-verbatim from mastro_mouse_bandit_analysis (src/mastro_mouse_bandit_analysis/
qlearning.py + states.py, commit 3554989).

Vendored in rather than imported live: mastro_mouse_bandit_analysis is only
installed in a separate conda env (bandit_env), not the env this pipeline
runs under, and this project already ports external logic with explicit
source citations (see behavior/word_encoding.py, config/params.py) rather
than taking cross-project runtime dependencies.

Not ported: load_session_params/load_animal_params (read that package's
precomputed per-session/per-animal fits) -- not applicable here, since this
pipeline always fits fresh against a session's own trial_table rather than
joining against that package's external nosepoke-aging dataset (per lab
decision: the trial_table is the single source of truth per session).

All tunable constants live in config/params.py, matching this project's
existing convention of centralizing constants rather than scattering them
across algorithm modules.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config.params import (
    ALPHA_BOUND_HI,
    ALPHA_BOUND_LO,
    ALPHA_BOUNDS,
    BETA_BOUNDS,
    BETA_BOUND_THRESH,
    BIAS_DEV_THRESH,
    BIAS_HIGH_PRIGHT,
    BIAS_LOW_PRIGHT,
    EXPLOITATION_ACC_THRESH,
    EXPLOITATION_SWITCH_THRESH,
    KAPPA_BOUNDS,
    KAPPA_BOUND_THRESH,
    NLL_CHANCE_THRESH,
    ROLLING_MIN_PERIODS,
    ROLLING_WINDOW,
)

BOUNDS = [ALPHA_BOUNDS, BETA_BOUNDS, KAPPA_BOUNDS]


def qlearn_sticky_nll(params, choices, rewards):
    """Negative log-likelihood for the sticky Q-learning model.

    stick = +1 (repeat right) | -1 (repeat left) | 0 (trial 1)
    z = beta*(Q_right - Q_left) + kappa*stick
    P(right) = sigmoid(z)
    Q(c) <- Q(c) + alpha*(r - Q(c))
    """
    alpha, beta, kappa = params
    if not (0 < alpha < 1) or beta <= 0:
        return np.inf

    Q = np.array([0.5, 0.5], dtype=float)
    nll = 0.0
    prev_choice = None

    for c, r in zip(choices, rewards):
        stick = 0.0 if prev_choice is None else (1.0 if prev_choice == 1 else -1.0)
        z = beta * (Q[1] - Q[0]) + kappa * stick
        p1 = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
        p = p1 if c == 1 else (1.0 - p1)
        nll -= np.log(p + 1e-12)
        Q[c] += alpha * (r - Q[c])
        prev_choice = c

    return nll


def multistart_fit(choices, rewards, n_starts, seed_offset=0):
    """Run L-BFGS-B from n_starts random initializations, return the best OptimizeResult."""
    best = None
    for seed in range(seed_offset, seed_offset + n_starts):
        rng = np.random.default_rng(seed)
        x0 = np.array(
            [rng.uniform(0.1, 0.9), rng.uniform(0.5, 5.0), rng.uniform(-2.0, 2.0)]
        )
        res = minimize(
            qlearn_sticky_nll, x0=x0, args=(choices, rewards), method="L-BFGS-B", bounds=BOUNDS
        )
        if best is None or res.fun < best.fun:
            best = res
    return best


def boundary_flags(x):
    alpha, beta, kappa = x
    return {
        "beta_at_bound": beta >= BETA_BOUND_THRESH,
        "kappa_at_bound": abs(kappa) >= KAPPA_BOUND_THRESH,
        "alpha_at_bound": alpha <= ALPHA_BOUND_LO or alpha >= ALPHA_BOUND_HI,
    }


def label_fit_groups(params_df):
    """Add a 'group' column: non_boundary | fast_learner | non_learner.

    fast_learner = at a parameter bound but still fits well (nll below chance)
    non_learner  = at a parameter bound and fits no better than chance
    """
    any_bound = params_df[["beta_at_bound", "kappa_at_bound", "alpha_at_bound"]].any(axis=1)
    nll_per_trial = params_df["nll"] / params_df["n_trials"]

    params_df = params_df.copy()
    params_df["group"] = "non_boundary"
    params_df.loc[any_bound & (nll_per_trial < NLL_CHANCE_THRESH), "group"] = "fast_learner"
    params_df.loc[any_bound & (nll_per_trial >= NLL_CHANCE_THRESH), "group"] = "non_learner"
    return params_df


def fit_session_level(
    df,
    col_animal="Mouse ID",
    col_session="Session ID",
    col_trial="Trial",
    col_choice="Decision",
    col_outcome="Reward",
    min_trials=30,
    n_starts=20,
):
    """Fit the sticky Q-learning model independently to each session.

    Sessions with fewer than min_trials valid trials are skipped.
    """
    records = []
    for session, sdf in df.groupby(col_session):
        sdf = sdf.sort_values(col_trial)
        choices = sdf[col_choice].dropna().values.astype(int)
        rewards = sdf[col_outcome].dropna().values.astype(int)
        if len(choices) < min_trials:
            continue

        best = multistart_fit(choices, rewards, n_starts)
        records.append(
            {
                col_session: session,
                col_animal: sdf[col_animal].iloc[0],
                "alpha": best.x[0],
                "beta": best.x[1],
                "kappa": best.x[2],
                "nll": best.fun,
                "n_trials": len(choices),
                **boundary_flags(best.x),
            }
        )

    return label_fit_groups(pd.DataFrame(records))


def simulate_qvalues(df, params_session, col_session="Session ID", col_trial="Trial",
                      col_choice="Decision", col_outcome="Reward"):
    """Replay each session's trials under its fitted (alpha, beta, kappa) to
    recover per-trial Q_left, Q_right, and Q_diff = Q_right - Q_left.

    Sessions without a fitted parameter row are left with NaN Q-values. Uses
    plain numpy arrays (not iterrows) so it stays fast over millions of trials.
    """
    params_by_session = params_session.set_index(col_session)["alpha"]

    q_left = np.full(len(df), np.nan)
    q_right = np.full(len(df), np.nan)
    positions = np.arange(len(df))

    for session, sdf in df.groupby(col_session):
        if session not in params_by_session.index:
            continue
        alpha = params_by_session.loc[session]

        order = sdf[col_trial].values.argsort()
        pos = positions[df.index.get_indexer(sdf.index)][order]
        choices = sdf[col_choice].values[order]
        rewards = sdf[col_outcome].values[order]

        Q = np.array([0.5, 0.5], dtype=float)
        for i in range(len(pos)):
            q_left[pos[i]] = Q[0]
            q_right[pos[i]] = Q[1]
            c, r = choices[i], rewards[i]
            if pd.notna(c) and pd.notna(r):
                c = int(c)
                Q[c] += alpha * (r - Q[c])

    out = df.copy()
    out["Q_left"] = q_left
    out["Q_right"] = q_right
    out["Q_diff"] = out["Q_right"] - out["Q_left"]
    return out


def add_rolling_features(df, col_session="Session ID", col_choice="Decision",
                          col_outcome="Reward", col_switch="Switch", col_target="Target"):
    """Add Rolling_Accuracy, Rolling_PRight, Expected_PRight, Signed_Deviation,
    Choice_Deviation, and Rolling_Switch_Rate columns, computed per-session
    with a trailing window.

    Expected_PRight here uses the hardcoded 0.8/0.2 map tied to the source
    cohort's fixed 80-20 task design -- external.bandit_state_adapter
    overrides it with each session's actual per-trial reward probability
    before calling add_behavioral_state, so this generalizes beyond 80-20.
    """
    df = df.copy()

    df["Rolling_Accuracy"] = (
        df.groupby(col_session)[col_outcome]
        .transform(lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=ROLLING_MIN_PERIODS).mean())
    )
    df["Rolling_PRight"] = (
        df.groupby(col_session)[col_choice]
        .transform(lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=ROLLING_MIN_PERIODS).mean())
    )
    df["Expected_PRight"] = df[col_target].map({1: 0.8, 0: 0.2})
    df["Signed_Deviation"] = df["Rolling_PRight"] - df["Expected_PRight"]
    df["Choice_Deviation"] = df["Signed_Deviation"].abs()
    df["Rolling_Switch_Rate"] = (
        df.groupby(col_session)[col_switch]
        .transform(lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=ROLLING_MIN_PERIODS).mean())
    )
    return df


def add_behavioral_state(df, col_target="Target"):
    """Vectorized 4-state classifier (fast over millions of rows): Exploitation,
    Exploration, Left Bias, Right Bias.
    """
    df = df.copy()
    acc = df["Rolling_Accuracy"]
    dev = df["Choice_Deviation"]
    switch_rate = df["Rolling_Switch_Rate"]
    p_right = df["Rolling_PRight"]
    target = df[col_target]

    is_bias = dev >= BIAS_DEV_THRESH
    bias_label = np.where(
        target == 1,
        np.where(p_right <= BIAS_LOW_PRIGHT, "Left Bias", "Right Bias"),
        np.where(p_right >= BIAS_HIGH_PRIGHT, "Right Bias", "Left Bias"),
    )

    is_exploitation = (~is_bias) & (acc >= EXPLOITATION_ACC_THRESH) & (switch_rate <= EXPLOITATION_SWITCH_THRESH)

    state = np.select(
        [acc.isna() | dev.isna(), is_bias, is_exploitation],
        ["Unknown", bias_label, "Exploitation"],
        default="Exploration",
    )
    df["Behavioral_State"] = state
    return df
