"""
Mixed-effects versions of the sticky Q-learning behavioral-parameter
comparisons in plot_bandit_behavior_comparison.py, with a random intercept
per mouse -- the session-level Mann-Whitney/Spearman tests there treat every
session as an independent observation, which overstates significance when
(as here) 10-40 sessions come from the same 5-11 mice. A random intercept
per mouse absorbs each animal's own baseline level, so the fixed-effect
p-values below test the condition/age/drug effect against between-MOUSE
variation, not between-SESSION variation.

beta is fit on log10(beta) (statsmodels formula api, so this is written
directly into the formula strings below) -- see plot_bandit_behavior_comparison.py's
own note that raw beta is heavily right-skewed by a handful of
boundary-pinned (near BETA_BOUNDS upper=49.9) fits; log-transforming avoids
those points dominating the fit the same way they dominated the naive
Pearson regression there.

5 SM mice / 11 FP mice is a small number of groups for a random-intercept
model -- variance-component estimates are imprecise and convergence
warnings are possible; both are reported rather than suppressed.

Usage:
    python mixedlm_bandit_behavior.py
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from plot_bandit_behavior_comparison import load_clean

PARAM_FORMULA_TERM = {"alpha": "alpha", "beta": "log10_beta", "kappa": "kappa"}


def _fit_and_report(df, formula, groups_col, label):
    print(f"\n--- {label} ---")
    print(f"  formula: {formula}   groups: {groups_col}   n_obs={len(df)}, n_groups={df[groups_col].nunique()}")
    try:
        model = smf.mixedlm(formula, data=df, groups=df[groups_col])
        result = model.fit(reml=True)
    except Exception as exc:
        print(f"  FIT FAILED: {exc}")
        return None
    if not result.converged:
        print("  WARNING: model did not converge -- treat below as indicative only")
    fe = result.fe_params
    se = result.bse_fe
    pvals = result.pvalues
    for term in fe.index:
        if term == "Intercept":
            continue
        print(f"  {term}: coef={fe[term]:+.4f}, se={se[term]:.4f}, p={pvals[term]:.4f}")
    print(f"  mouse random-intercept variance: {result.cov_re.iloc[0, 0]:.5f}  "
          f"(residual variance: {result.scale:.5f})")
    return result


def condition_effect(df):
    """param ~ condition, random intercept per mouse. 'none'/undrugged sessions only."""
    sub = df[df["cohort"].isin(["FP1_none", "FP2_none", "FP1_retrained_none", "SM"])].copy()
    sub["log10_beta"] = np.log10(sub["beta"])
    print("\n" + "=" * 70)
    print("MIXED MODEL A: condition effect (90/10 vs 80/20), random intercept per mouse")
    print("=" * 70)
    for param, term in PARAM_FORMULA_TERM.items():
        _fit_and_report(sub, f"{term} ~ C(condition)", "mouse", f"{param} ~ condition")


def condition_by_age(df):
    """param ~ condition * age, random intercept per mouse. 'none'/undrugged sessions only,
    FP1_retrained_none included to widen the within-FP1-condition age range beyond the
    cross-sectional FP1-vs-FP2 gap.
    """
    sub = df[df["cohort"].isin(["FP1_none", "FP2_none", "FP1_retrained_none", "SM"])].copy()
    sub["log10_beta"] = np.log10(sub["beta"])
    print("\n" + "=" * 70)
    print("MIXED MODEL B: condition x age interaction, random intercept per mouse")
    print("(age is exact age_days for FP, nominal P-bin day for SM -- see load_clean)")
    print("=" * 70)
    for param, term in PARAM_FORMULA_TERM.items():
        _fit_and_report(sub, f"{term} ~ C(condition) * age_numeric_days", "mouse", f"{param} ~ condition * age")


def sm_age_trend(df):
    """param ~ age, within SM only, random intercept per mouse -- the direct mixed-model
    counterpart to plot_bandit_behavior_comparison.py's session-level Spearman test.
    """
    sub = df[df["cohort"] == "SM"].copy()
    sub["log10_beta"] = np.log10(sub["beta"])
    print("\n" + "=" * 70)
    print("MIXED MODEL C: age trend within SM (80/20), random intercept per mouse")
    print("=" * 70)
    for param, term in PARAM_FORMULA_TERM.items():
        _fit_and_report(sub, f"{term} ~ age_numeric_days", "mouse", f"SM {param} ~ age")


def dcz_effect(df):
    """param ~ drug, within FP1 and FP2 separately (same mice, same 90/10 schedule),
    random intercept per mouse.
    """
    print("\n" + "=" * 70)
    print("MIXED MODEL D: DCZ effect within FP1/FP2, random intercept per mouse")
    print("=" * 70)
    for family in ("FP1", "FP2"):
        sub = df[(df["cohort_family"] == family) & (df["cohort"].isin([f"{family}_none", f"{family}_dcz"]))].copy()
        sub["log10_beta"] = np.log10(sub["beta"])
        sub["drug"] = np.where(sub["cohort"].str.endswith("_dcz"), "DCZ", "none")
        for param, term in PARAM_FORMULA_TERM.items():
            _fit_and_report(sub, f"{term} ~ C(drug)", "mouse", f"{family} {param} ~ drug")


def main():
    df = load_clean()
    condition_effect(df)
    condition_by_age(df)
    sm_age_trend(df)
    dcz_effect(df)


if __name__ == "__main__":
    main()
