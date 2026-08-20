"""
Time-resolved GLM using EXPANDED_TIME_RESOLVED_GLM_FORMULA (RPE/value
decomposition, model-implied belief, switch dynamics -- see
behavior/switch_dynamics.py and config/params.py) on the FP1_retrained none
and FP2 none cohorts, run alongside (not instead of) the existing
DEFAULT_TIME_RESOLVED_GLM_FORMULA fit, for direct before/after comparison.

Session lists are derived from each cohort's own already-fixed master CSV
(mouse/date/hemisphere -- reflects any manual exclusions, e.g. WCL28's
contaminated 091123/091323 dates from the FP1_retrained none correction,
which were fixed in a separate output and are excluded here explicitly since
outputs_fixed/fp1_retrained_none_cohort_master.csv itself was never
updated), joined against each cohort's QC report for session_dir.

Also runs a VIF (variance inflation factor) check on the expanded design
matrix -- the user's own flagged concern about collinearity among
RPE/Q_chosen/Reward/etc.

Usage:
    python run_expanded_glm_analysis.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from patsy import dmatrix
from statsmodels.stats.outliers_influence import variance_inflation_factor

from config.params import DEFAULT_TIME_RESOLVED_GLM_FORMULA, EXPANDED_TIME_RESOLVED_GLM_FORMULA
from models.glm_data import build_pooled_glm_dataset
from models.glm_encoding import _build_predictor_frame, fit_time_resolved_glm
from pipeline import DEFAULT_FIGURE_DIR
from viz.glm_plots import plot_glm_coefficients

OUT_DIR = Path("figures_fixed_expanded_glm")

# (cohort_label, master_csv, qc_report_csv, exclude (mouse, date) pairs)
# NOTE: FP1_none (base FP1) and FP1_retrained_none are DISTINCT cohorts --
# different session dates for overlapping mouse IDs (e.g. WCL28 appears in
# both, but FP1_none's dates are June 2023 while FP1_retrained's are
# Aug-Sep 2023) -- confirmed no overlap, so FP1_none needs no exclusions.
COHORTS = [
    ("FP1_none", "outputs_fixed/none_cohort_master.csv",
     "outputs_fixed/none_cohort_qc_report.csv", set()),
    ("FP1_dcz", "outputs_fixed/fp1_dcz_cohort_master.csv",
     "outputs_fixed/fp1_dcz_cohort_qc_report.csv", set()),
    ("FP1_retrained_none", "outputs_fixed/fp1_retrained_none_cohort_master.csv",
     "outputs_fixed/fp1_retrained_none_cohort_qc_report.csv", {("WCL28", "091123"), ("WCL28", "091323")}),
    ("FP2_none", "outputs_fixed/fp2_none_cohort_master.csv",
     "outputs_fixed/fp2_none_cohort_qc_report.csv", set()),
    ("FP2_dcz", "outputs_fixed/fp2_dcz_cohort_master.csv",
     "outputs_fixed/fp2_dcz_cohort_qc_report.csv", set()),
]


def session_hemisphere_lookup(master_csv, qc_report_csv, exclude_pairs):
    master = pd.read_csv(master_csv, dtype={"date": str})
    qc = pd.read_csv(qc_report_csv, dtype={"date": str})
    pairs = master[["mouse", "date", "hemisphere"]].drop_duplicates()
    pairs = pairs[~pairs.apply(lambda r: (r["mouse"], r["date"]) in exclude_pairs, axis=1)]
    merged = pairs.merge(qc[["mouse", "date", "session_dir"]].drop_duplicates(), on=["mouse", "date"], how="left")
    missing = merged[merged["session_dir"].isna()]
    if len(missing):
        print(f"WARNING: {len(missing)} (mouse, date) pair(s) have no session_dir in the QC report, skipping:")
        print(missing[["mouse", "date"]].to_string())
        merged = merged.dropna(subset=["session_dir"])
    session_dirs = [Path(p) for p in merged["session_dir"]]
    hemisphere_for_session = dict(zip(session_dirs, merged["hemisphere"]))
    return session_dirs, hemisphere_for_session


def compute_vif(trial_table, formula):
    """VIF on the RHS design matrix (predictors don't vary with t, so one
    VIF table covers the whole time-resolved fit, not per-timepoint).
    Builds the same clean predictor frame fit_time_resolved_glm uses
    internally -- the formula references names like RPE/Choice_lag1 that
    only exist there, not in the raw trial_table.
    """
    predictors = _build_predictor_frame(trial_table).dropna()
    rhs = formula.split("~", 1)[1]
    X = dmatrix(rhs, predictors, return_type="dataframe")
    X = X.drop(columns=["Intercept"], errors="ignore")
    X = X.loc[:, X.std() > 0]  # drop constant columns (e.g. a level never observed) -- VIF undefined for these
    vif = pd.DataFrame({
        "term": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    }).sort_values("VIF", ascending=False)
    return vif


def run_cohort(cohort_label, master_csv, qc_report_csv, exclude_pairs, out_dir=OUT_DIR):
    print(f"\n{'=' * 70}\n{cohort_label}\n{'=' * 70}")
    session_dirs, hemisphere_for_session = session_hemisphere_lookup(master_csv, qc_report_csv, exclude_pairs)
    print(f"{len(session_dirs)} sessions")

    peth_time, zscore_windows, trial_table = build_pooled_glm_dataset(
        session_dirs, align_event="side_in",
        hemisphere_for_session=lambda d: hemisphere_for_session[d],
    )

    results = {}
    for label, formula in [("default", DEFAULT_TIME_RESOLVED_GLM_FORMULA), ("expanded", EXPANDED_TIME_RESOLVED_GLM_FORMULA)]:
        print(f"\n--- {cohort_label}: {label} formula ---")
        beta_df = fit_time_resolved_glm(zscore_windows, peth_time, trial_table, formula=formula)
        fig = plot_glm_coefficients(peth_time, beta_df, formula, align_event="side_in")
        out_stem = out_dir / f"{cohort_label.lower()}_glm_{label}_side_in_coefficients"
        fig.savefig(out_stem.with_suffix(".png"), dpi=150)
        fig.savefig(out_stem.with_suffix(".svg"))
        # Persist the underlying beta(t)/se(t) table (not just the plotted
        # figure) -- needed to overlay multiple conditions' coefficient
        # trajectories on the same axes in the cross-condition comparison
        # report, which a PNG alone can't support.
        beta_df.to_csv(out_dir / f"{cohort_label.lower()}_glm_{label}_beta_df.csv")
        print(f"Saved {out_stem.with_suffix('.png')} and beta_df CSV")
        results[label] = beta_df

    print(f"\n--- {cohort_label}: VIF on expanded design matrix ---")
    vif = compute_vif(trial_table, EXPANDED_TIME_RESOLVED_GLM_FORMULA)
    print(vif.to_string(index=False))
    vif.to_csv(out_dir / f"{cohort_label.lower()}_expanded_vif.csv", index=False)

    return dict(trial_table=trial_table, results=results, vif=vif, n_sessions=len(session_dirs))


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for cohort_label, master_csv, qc_report_csv, exclude_pairs in COHORTS:
        all_results[cohort_label] = run_cohort(cohort_label, master_csv, qc_report_csv, exclude_pairs, out_dir=out_dir)
    return all_results


if __name__ == "__main__":
    main()
