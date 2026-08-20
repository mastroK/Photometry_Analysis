"""
Within-animal RPE analyses for the SM (PV-dualphotometry) cohort, run on the
pooled dataset built by rpe_analysis_prep_SM.py. Same four analyses as
rpe_analysis_stats.py (FP1/FP2/WCL), same one-point-per-mouse Wilcoxon
convention -- every claim shown at the animal level, not pooled across mice.

SM-specific handling: this cohort's mice can contribute BOTH green_r and
green_l hemisphere extractions per session (a true dual-fiber rig, unlike
FP1/FP2's single active channel per mouse). Per explicit user direction,
green_r and green_l are NEVER collapsed/pooled together in any analysis here
-- every one of the four analyses is run TWICE, once per hemisphere, on that
hemisphere's own trial subset, producing fully separate result tables (not a
single pooled result plus a supplementary interaction check). This mirrors
run_sm_glm_fir_analysis.py's identical per-hemisphere-separate treatment of
the time-resolved GLM and FIR kernels.
  - analysis_1b_hemisphere_interaction: kept as an additional, informational
    cross-check (does the RPE effect's magnitude/sign differ significantly
    between the two SEPARATELY-reported hemispheres?), not as a gate on
    whether hemispheres get pooled -- they don't, regardless of this result.

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
    depend on which hemisphere it's measured in? Purely informational --
    hemispheres are reported fully separately regardless of this result (see
    module docstring); this only flags whether the two separate results
    should be expected to look similar or different for a given mouse.
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


def run_hemisphere(tt_all, zscore_windows, peth_time, fir, fir_meta, hemisphere, out_dir):
    """Run all four analyses restricted to ONE hemisphere's trials -- the
    complete, independent result set for that hemisphere, not a subset
    view of a pooled fit. tt_all must already have _orig_pos (add_derived_
    columns) computed against the FULL (unfiltered) zscore_windows -- that
    column is preserved correctly under boolean filtering, so zscore_windows
    itself is passed through unfiltered and analysis_3_per_mouse_encoding_glm
    re-indexes via _orig_pos, not position.
    """
    print(f"\n{'=' * 70}\n{hemisphere}\n{'=' * 70}")
    tt = tt_all[tt_all["hemisphere"] == hemisphere].reset_index(drop=True)
    print(f"{len(tt)} trials, {tt['mouse'].nunique()} mice")

    fir_mask = fir["hemisphere"] == hemisphere
    fir_h = {k: fir[k][fir_mask] for k in fir.files}

    r1, r1_stats = analysis_1_rpe_regression(tt)
    r2, r2_stats = analysis_2_signed_vs_unsigned(tt)
    r3a = analysis_3_per_mouse_encoding_glm(tt, zscore_windows, peth_time)
    r3b = analysis_3_per_mouse_fir_glm(fir_h, fir_meta)
    tt = add_pre_event_amplitude(tt, zscore_windows, peth_time)
    r4 = analysis_4_temporal_specificity(tt, r1)

    suffix = f"_{hemisphere}"
    r1.to_csv(out_dir / f"analysis1_rpe_regression{suffix}.csv")
    r2.to_csv(out_dir / f"analysis2_signed_vs_unsigned{suffix}.csv")
    r3a.to_csv(out_dir / f"analysis3a_encoding_glm_per_mouse{suffix}.csv")
    r3b.to_csv(out_dir / f"analysis3b_fir_glm_per_mouse{suffix}.csv")
    r4.to_csv(out_dir / f"analysis4_temporal_specificity{suffix}.csv")

    return dict(analysis1=r1_stats, analysis2=r2_stats)


def main(data_dir=DATA_DIR, out_dir=OUT_DIR):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trial_table, zscore_windows, peth_time, fir, fir_meta = load_pooled_data(data_dir)
    tt_all = add_derived_columns(trial_table, zscore_windows, peth_time)

    print(f"Pooled dataset: {len(tt_all)} trials, {tt_all['mouse'].nunique()} mice")
    print(tt_all.groupby(["mouse", "hemisphere"]).size())

    # Informational only -- see analysis_1b_hemisphere_interaction's
    # docstring; does not gate or change the per-hemisphere results below.
    r1b = analysis_1b_hemisphere_interaction(tt_all)
    r1b.to_csv(out_dir / "analysis1b_hemisphere_interaction.csv")

    summary_stats = {}
    for hemisphere in sorted(tt_all["hemisphere"].unique()):
        summary_stats[hemisphere] = run_hemisphere(
            tt_all, zscore_windows, peth_time, fir, fir_meta, hemisphere, out_dir
        )

    import json
    with open(out_dir / "summary_stats.json", "w") as f:
        json.dump(summary_stats, f, indent=2)

    print(f"\nSaved all results to {out_dir} (separately per hemisphere)")


if __name__ == "__main__":
    import sys
    data_dir_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR
    out_dir_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else (data_dir_arg / "results")
    main(data_dir=data_dir_arg, out_dir=out_dir_arg)
