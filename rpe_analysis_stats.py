"""
Within-animal RPE analyses, run on the pooled dataset built by
rpe_analysis_prep.py. See conversation for the full specification. Three
analyses, each reported per mouse (n=7) with a group-level Wilcoxon
signed-rank test rather than pooling across mice:

1. Within-animal RPE regression: post_amp ~ RPE_signed per mouse, one beta_RPE
   per mouse, Wilcoxon test that the 7 betas differ from 0. Supplementary
   nested-model comparison (outcome-only / outcome+Q_chosen additive /
   outcome*Q_chosen interaction / constrained-RPE) per mouse, since RPE_signed
   is an exact linear combination of outcome and Q_chosen and can't share a
   regression with both.
2. Signed vs unsigned RPE: 5-fold CV comparing post_amp ~ RPE_signed+Q_diff_abs
   +chose_right against post_amp ~ RPE_abs+Q_diff_abs+chose_right, per mouse.
   delta_R2 = R2_signed - R2_unsigned, Wilcoxon test + count positive.
3. Between-animal variability of the two pooled GLMs: refit the time-resolved
   encoding GLM and the FIR RidgeCV model separately per mouse (from the same
   pooled data -- no raw reprocessing), report peak R^2 / out-of-sample R^2
   per mouse.

Usage:
    python rpe_analysis_stats.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import wilcoxon
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from config.params import DECISION_WINDOW_S, DEFAULT_TIME_RESOLVED_GLM_FORMULA
from models.fir_glm import DEFAULT_N_SPLITS, DEFAULT_TEST_SIZE, DEFAULT_ALPHAS, fit_fir_glm
from models.glm_encoding import fit_time_resolved_glm

DATA_DIR = Path("outputs_fixed/rpe_analysis")
OUT_DIR = Path("outputs_fixed/rpe_analysis/results")
RANDOM_STATE = 0


def load_pooled_data(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    trial_table = pd.read_parquet(data_dir / "pooled_trial_table.parquet")
    peth = np.load(data_dir / "peth_windows.npz")
    zscore_windows, peth_time = peth["zscore_windows"], peth["peth_time"]
    fir = np.load(data_dir / "fir_pooled.npz", allow_pickle=True)
    with open(data_dir / "fir_column_names.pkl", "rb") as f:
        fir_meta = pickle.load(f)
    return trial_table, zscore_windows, peth_time, fir, fir_meta


def add_derived_columns(trial_table, zscore_windows, peth_time):
    """Returns tt with a `_orig_pos` column preserving each surviving row's
    original 0-based position in zscore_windows -- tt itself gets a fresh
    reset index (0..n-1), which does NOT align with zscore_windows once any
    rows are dropped, so callers that need to re-subset zscore_windows by
    mouse (analysis_3_per_mouse_encoding_glm) must index via `_orig_pos`,
    not tt's own index.
    """
    tt = trial_table.copy()
    tt["_orig_pos"] = np.arange(len(tt))
    tt["Q_chosen"] = np.where(tt["chose_right"], tt["Q_right"], tt["Q_left"])
    tt["RPE_signed"] = tt["was_rewarded"].astype(float) - tt["Q_chosen"]
    tt["RPE_abs"] = tt["RPE_signed"].abs()
    tt["outcome"] = tt["was_rewarded"].astype(float)
    tt["choice_side"] = tt["chose_right"].astype(float)

    mask = (peth_time >= DECISION_WINDOW_S[0]) & (peth_time <= DECISION_WINDOW_S[1])
    tt["post_amp"] = zscore_windows[:, mask].mean(axis=1)

    valid = tt[["Q_chosen", "post_amp", "Q_diff"]].notna().all(axis=1)
    n_dropped = int((~valid).sum())
    if n_dropped:
        print(f"Dropping {n_dropped}/{len(tt)} trials with missing Q-value/amplitude data")
    return tt.loc[valid].reset_index(drop=True)


# --- Analysis 1: within-animal RPE regression -------------------------------

def analysis_1_rpe_regression(tt):
    rows = []
    for mouse, g in tt.groupby("mouse"):
        m_rpe = smf.ols("post_amp ~ RPE_signed", data=g).fit()
        m0 = smf.ols("post_amp ~ outcome", data=g).fit()
        m1 = smf.ols("post_amp ~ outcome + Q_chosen", data=g).fit()
        m2 = smf.ols("post_amp ~ outcome * Q_chosen", data=g).fit()
        rows.append(dict(
            mouse=mouse, n_trials=len(g),
            beta_rpe=m_rpe.params["RPE_signed"], se_rpe=m_rpe.bse["RPE_signed"],
            t_rpe=m_rpe.tvalues["RPE_signed"], p_rpe=m_rpe.pvalues["RPE_signed"],
            r2_rpe=m_rpe.rsquared,
            r2_outcome_only=m0.rsquared, r2_outcome_plus_value=m1.rsquared,
            r2_outcome_x_value=m2.rsquared,
            interaction_coef=m2.params.get("outcome:Q_chosen", np.nan),
            interaction_p=m2.pvalues.get("outcome:Q_chosen", np.nan),
        ))
    per_mouse = pd.DataFrame(rows).set_index("mouse")

    stat, p = wilcoxon(per_mouse["beta_rpe"])
    print("\n=== Analysis 1: within-animal RPE regression (post_amp ~ RPE_signed) ===")
    print(per_mouse.to_string())
    print(f"\nWilcoxon signed-rank on {len(per_mouse)} beta_RPE values vs 0: W={stat:.3f}, p={p:.4f}")
    print(f"Median beta_RPE = {per_mouse['beta_rpe'].median():.4f}, "
          f"{(per_mouse['beta_rpe'] < 0).sum()}/{len(per_mouse)} mice negative")
    return per_mouse, dict(wilcoxon_stat=float(stat), wilcoxon_p=float(p))


# --- Analysis 2: signed vs unsigned RPE, cross-validated --------------------

def _cv_r2(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, test_idx in kf.split(X):
        model = LinearRegression().fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        scores.append(r2_score(y[test_idx], pred))
    return float(np.mean(scores)), float(np.std(scores))


def analysis_2_signed_vs_unsigned(tt):
    rows = []
    for mouse, g in tt.groupby("mouse"):
        y = g["post_amp"].to_numpy()
        X_signed = g[["RPE_signed", "Q_diff", "choice_side"]].to_numpy()
        X_unsigned = g[["RPE_abs", "Q_diff", "choice_side"]].to_numpy()
        r2_signed, sd_signed = _cv_r2(X_signed, y)
        r2_unsigned, sd_unsigned = _cv_r2(X_unsigned, y)
        rows.append(dict(
            mouse=mouse, n_trials=len(g),
            r2_signed=r2_signed, r2_unsigned=r2_unsigned,
            delta_r2=r2_signed - r2_unsigned,
        ))
    per_mouse = pd.DataFrame(rows).set_index("mouse")

    stat, p = wilcoxon(per_mouse["delta_r2"])
    n_positive = int((per_mouse["delta_r2"] > 0).sum())
    print("\n=== Analysis 2: signed vs unsigned RPE (5-fold CV out-of-sample R^2) ===")
    print(per_mouse.to_string())
    print(f"\nWilcoxon signed-rank on {len(per_mouse)} delta_R2 values vs 0: W={stat:.3f}, p={p:.4f}")
    print(f"{n_positive}/{len(per_mouse)} mice show delta_R2 > 0 (signed beats unsigned)")
    return per_mouse, dict(wilcoxon_stat=float(stat), wilcoxon_p=float(p), n_positive=n_positive)


# --- Analysis 3: between-animal variability of the two pooled GLMs ---------

def analysis_3_per_mouse_encoding_glm(tt, zscore_windows, peth_time):
    rows = []
    for mouse in sorted(tt["mouse"].unique()):
        sub_table = tt.loc[tt["mouse"] == mouse].reset_index(drop=True)
        sub_windows = zscore_windows[sub_table["_orig_pos"].to_numpy()]
        try:
            beta_df = fit_time_resolved_glm(sub_windows, peth_time, sub_table,
                                             formula=DEFAULT_TIME_RESOLVED_GLM_FORMULA)
            peak_r2 = beta_df["r_squared"].max()
            peak_t = beta_df["r_squared"].idxmax()
        except Exception as exc:
            print(f"WARNING: encoding GLM refit failed for {mouse}: {exc}")
            peak_r2, peak_t = np.nan, np.nan
        rows.append(dict(mouse=mouse, n_trials=len(sub_table), peak_r2=peak_r2, peak_time_s=peak_t))
    per_mouse = pd.DataFrame(rows).set_index("mouse")
    print("\n=== Analysis 3a: per-mouse time-resolved encoding GLM ===")
    print(per_mouse.to_string())
    return per_mouse


# --- Analysis 4: temporal specificity control (pre- vs post-event interaction) ---

def add_pre_event_amplitude(tt, zscore_windows, peth_time, window_s=(-1.0, 0.0)):
    mask = (peth_time >= window_s[0]) & (peth_time <= window_s[1])
    tt = tt.copy()
    tt["pre_amp"] = zscore_windows[tt["_orig_pos"].to_numpy()][:, mask].mean(axis=1)
    return tt


def analysis_4_temporal_specificity(tt, post_results):
    """For each mouse, refit the outcome*Q_chosen interaction model using
    pre_amp (mean z, -1 to 0s BEFORE side_in) instead of post_amp, and compare
    the interaction coefficient against Analysis 1's post-event fit. A real
    outcome-locked value-modulation effect should be absent (or much weaker/
    non-significant) pre-event, since the animal's own future outcome can't
    yet be reflected in its photometry signal before the event happens.
    """
    rows = []
    for mouse, g in tt.groupby("mouse"):
        m_pre = smf.ols("pre_amp ~ outcome * Q_chosen", data=g).fit()
        rows.append(dict(
            mouse=mouse, n_trials=len(g),
            interaction_coef_pre=m_pre.params.get("outcome:Q_chosen", np.nan),
            interaction_p_pre=m_pre.pvalues.get("outcome:Q_chosen", np.nan),
            r2_pre=m_pre.rsquared,
        ))
    pre_df = pd.DataFrame(rows).set_index("mouse")
    combined = post_results[["interaction_coef", "interaction_p", "r2_outcome_x_value"]].join(pre_df)
    combined = combined.rename(columns={
        "interaction_coef": "interaction_coef_post", "interaction_p": "interaction_p_post",
        "r2_outcome_x_value": "r2_post",
    })

    print("\n=== Analysis 4: temporal specificity (outcome x Q_chosen interaction, pre- vs post-event) ===")
    print(combined.to_string())
    n_post_sig = int((combined["interaction_p_post"] < 0.05).sum())
    n_pre_sig = int((combined["interaction_p_pre"] < 0.05).sum())
    n_pre_sig_same_sign = int(((combined["interaction_p_pre"] < 0.05) &
                                (np.sign(combined["interaction_coef_pre"]) == np.sign(combined["interaction_coef_post"]))).sum())
    n = len(combined)
    print(f"\nPost-event interaction significant (p<0.05): {n_post_sig}/{n} mice")
    print(f"Pre-event interaction significant (p<0.05): {n_pre_sig}/{n} mice "
          f"({n_pre_sig_same_sign}/{n} same sign as post-event)")
    return combined


def analysis_3_per_mouse_fir_glm(fir, fir_meta):
    y, Phi, groups, mouse = fir["y"], fir["Phi"], fir["groups"], fir["mouse"]
    rows = []
    for m in sorted(np.unique(mouse)):
        sel = mouse == m
        y_m, Phi_m, groups_m = y[sel], Phi[sel], groups[sel]
        mask_all = np.ones(len(y_m), dtype=bool)
        n_groups = len(np.unique(groups_m))
        n_splits = min(DEFAULT_N_SPLITS, max(2, n_groups // 10))
        try:
            _, cv_results = fit_fir_glm(y_m, Phi_m, mask_all, groups_m,
                                         n_splits=n_splits, test_size=DEFAULT_TEST_SIZE,
                                         alphas=DEFAULT_ALPHAS, random_state=RANDOM_STATE)
            r2_mean, r2_std = cv_results["r2_mean"], cv_results["r2_std"]
        except Exception as exc:
            print(f"WARNING: FIR refit failed for {m}: {exc}")
            r2_mean, r2_std = np.nan, np.nan
        rows.append(dict(mouse=m, n_samples=int(sel.sum()), n_trials=n_groups,
                          r2_mean=r2_mean, r2_std=r2_std, n_splits=n_splits))
    per_mouse = pd.DataFrame(rows).set_index("mouse")
    print("\n=== Analysis 3b: per-mouse FIR deconvolution GLM ===")
    print(per_mouse.to_string())
    return per_mouse


def main(data_dir=DATA_DIR, out_dir=OUT_DIR):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trial_table, zscore_windows, peth_time, fir, fir_meta = load_pooled_data(data_dir)
    tt = add_derived_columns(trial_table, zscore_windows, peth_time)

    print(f"Pooled dataset: {len(tt)} trials, {tt['mouse'].nunique()} mice")
    print(tt.groupby("mouse").size())

    r1, r1_stats = analysis_1_rpe_regression(tt)
    r2, r2_stats = analysis_2_signed_vs_unsigned(tt)
    r3a = analysis_3_per_mouse_encoding_glm(tt, zscore_windows, peth_time)
    r3b = analysis_3_per_mouse_fir_glm(fir, fir_meta)
    tt = add_pre_event_amplitude(tt, zscore_windows, peth_time)
    r4 = analysis_4_temporal_specificity(tt, r1)

    r1.to_csv(out_dir / "analysis1_rpe_regression.csv")
    r2.to_csv(out_dir / "analysis2_signed_vs_unsigned.csv")
    r3a.to_csv(out_dir / "analysis3a_encoding_glm_per_mouse.csv")
    r3b.to_csv(out_dir / "analysis3b_fir_glm_per_mouse.csv")
    r4.to_csv(out_dir / "analysis4_temporal_specificity.csv")

    import json
    with open(out_dir / "summary_stats.json", "w") as f:
        json.dump(dict(analysis1=r1_stats, analysis2=r2_stats), f, indent=2)

    print(f"\nSaved all results to {out_dir}")


if __name__ == "__main__":
    import sys
    data_dir_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR
    out_dir_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else (data_dir_arg / "results")
    main(data_dir=data_dir_arg, out_dir=out_dir_arg)
