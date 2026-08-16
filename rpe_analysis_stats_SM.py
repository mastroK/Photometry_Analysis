"""
Within-animal RPE analyses for the SM (PV-dualphotometry) cohort, run on the
pooled dataset built by rpe_analysis_prep_SM.py. Same four analyses as
rpe_analysis_stats.py (FP1/FP2/WCL), same one-point-per-mouse Wilcoxon
convention -- every claim shown at the animal level, not pooled across mice.

SM-specific addition: this cohort's mice can contribute BOTH green_r and
green_l hemisphere extractions per session (a true dual-fiber rig, unlike
FP1/FP2's single active channel per mouse). Per-mouse models here pool every
valid (session, hemisphere) row into that mouse's own regression/refit,
exactly like multiple sessions already pool -- hemisphere is uncorrelated
with RPE_signed (a behavioral, not photometric, construct), so this cannot
bias beta_rpe; at worst it adds conservative residual variance. This is a
SUPPLEMENT, not a replacement, for that convention:
  - analysis_1b_hemisphere_interaction: per-mouse post_amp ~ RPE_signed *
    C(hemisphere), reporting the interaction p-value, to confirm the RPE
    effect isn't being driven by or masked by one hemisphere. Mice with only
    one valid hemisphere can't fit this (report NaN, noted explicitly).
  - hemisphere_breakdown: descriptive per-mouse, per-hemisphere n_sessions/
    n_trials/beta_rpe table, pure transparency -- doesn't touch the n=n_mice
    test itself.

Usage:
    python rpe_analysis_stats_SM.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import wilcoxon

from rpe_analysis_stats import (
    add_derived_columns,
    add_pre_event_amplitude,
    analysis_1_rpe_regression,
    analysis_2_signed_vs_unsigned,
    analysis_3_per_mouse_encoding_glm,
    analysis_3_per_mouse_fir_glm,
    analysis_4_temporal_specificity,
    load_pooled_data,
)

DATA_DIR = Path("outputs_fixed/rpe_analysis_sm")
OUT_DIR = Path("outputs_fixed/rpe_analysis_sm/results")


def analysis_1b_hemisphere_interaction(tt):
    """Per-mouse post_amp ~ RPE_signed * C(hemisphere): does the RPE effect
    depend on which hemisphere it's measured in? Non-gating -- reported
    alongside Analysis 1's pooled-hemisphere per-mouse betas, not instead of
    them.
    """
    rows = []
    for mouse, g in tt.groupby("mouse"):
        n_hemispheres = g["hemisphere"].nunique()
        if n_hemispheres < 2:
            rows.append(dict(
                mouse=mouse, n_trials=len(g), n_hemispheres=n_hemispheres,
                hemispheres=",".join(sorted(g["hemisphere"].unique())),
                interaction_coef=np.nan, interaction_p=np.nan,
                note="only one hemisphere valid for this mouse -- interaction not fittable",
            ))
            continue
        m = smf.ols("post_amp ~ RPE_signed * C(hemisphere)", data=g).fit()
        interaction_terms = [t for t in m.params.index if ":" in t]
        term = interaction_terms[0] if interaction_terms else None
        rows.append(dict(
            mouse=mouse, n_trials=len(g), n_hemispheres=n_hemispheres,
            hemispheres=",".join(sorted(g["hemisphere"].unique())),
            interaction_coef=m.params.get(term, np.nan) if term else np.nan,
            interaction_p=m.pvalues.get(term, np.nan) if term else np.nan,
            note="",
        ))
    per_mouse = pd.DataFrame(rows).set_index("mouse")
    n_fittable = int(per_mouse["interaction_p"].notna().sum())
    n_sig = int((per_mouse["interaction_p"] < 0.05).sum())
    print("\n=== Analysis 1b: RPE x hemisphere interaction (supplementary, non-gating) ===")
    print(per_mouse.to_string())
    print(f"\n{n_sig}/{n_fittable} mice (of {len(per_mouse)} total) show a significant "
          f"RPE x hemisphere interaction (p<0.05) -- a real effect here would suggest the "
          f"pooled Analysis-1 beta_rpe is not equally supported by both hemispheres")
    return per_mouse


def hemisphere_breakdown(tt):
    """Descriptive per-mouse, per-hemisphere table: how many (session,
    hemisphere) sub-observations fed each mouse's collapsed Analysis-1 point,
    and what beta_RPE looks like within just that hemisphere. Pure
    transparency -- does not feed into or change any Wilcoxon test.
    """
    rows = []
    for (mouse, hemisphere), g in tt.groupby(["mouse", "hemisphere"]):
        n_sessions = g["date"].nunique()
        if len(g) >= 5:
            m = smf.ols("post_amp ~ RPE_signed", data=g).fit()
            beta_rpe, p_rpe = m.params["RPE_signed"], m.pvalues["RPE_signed"]
        else:
            beta_rpe, p_rpe = np.nan, np.nan
        rows.append(dict(
            mouse=mouse, hemisphere=hemisphere, n_sessions=n_sessions, n_trials=len(g),
            beta_rpe=beta_rpe, p_rpe=p_rpe,
        ))
    breakdown = pd.DataFrame(rows).set_index(["mouse", "hemisphere"])
    print("\n=== Per-mouse, per-hemisphere breakdown (descriptive only) ===")
    print(breakdown.to_string())
    return breakdown


def main(data_dir=DATA_DIR, out_dir=OUT_DIR):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trial_table, zscore_windows, peth_time, fir, fir_meta = load_pooled_data(data_dir)
    tt = add_derived_columns(trial_table, zscore_windows, peth_time)

    print(f"Pooled dataset: {len(tt)} trials, {tt['mouse'].nunique()} mice")
    print(tt.groupby(["mouse", "hemisphere"]).size())

    r1, r1_stats = analysis_1_rpe_regression(tt)
    r1b = analysis_1b_hemisphere_interaction(tt)
    breakdown = hemisphere_breakdown(tt)
    r2, r2_stats = analysis_2_signed_vs_unsigned(tt)
    r3a = analysis_3_per_mouse_encoding_glm(tt, zscore_windows, peth_time)
    r3b = analysis_3_per_mouse_fir_glm(fir, fir_meta)
    tt = add_pre_event_amplitude(tt, zscore_windows, peth_time)
    r4 = analysis_4_temporal_specificity(tt, r1)

    r1.to_csv(out_dir / "analysis1_rpe_regression.csv")
    r1b.to_csv(out_dir / "analysis1b_hemisphere_interaction.csv")
    breakdown.to_csv(out_dir / "hemisphere_breakdown.csv")
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
