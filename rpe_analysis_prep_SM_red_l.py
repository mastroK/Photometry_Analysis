"""
Pool red_l data for the SM cohort, split into two GROUPS instead of two
hemispheres: confirmed jRGECO-only mice (SM1B, SM2B -- no GCaMP, so red_l's
223Hz carrier there is unambiguously real jRGECO signal, not crosstalk) vs.
every other ("regular", dual-expression) mouse with a valid red_l carrier.

Reuses rpe_analysis_prep_SM.py's pooling machinery UNCHANGED: builds the
pooled dataset with every row's real extraction channel forced to "red_l",
then relabels the `hemisphere` column (and fir_pooled.npz's parallel
`hemisphere` array) from the literal string "red_l" to the group label
per-mouse. run_sm_glm_fir_analysis.py and rpe_analysis_stats_SM.py both
already treat "hemisphere" as a fully generic 2-level grouping column (no
green_r/green_l-specific logic in their actual fitting code), so both run
unmodified against this relabeled output -- same fully-separate-groups
treatment as green_r vs green_l, just with different group semantics.

Usage:
    python rpe_analysis_prep_SM_red_l.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from rpe_analysis_prep_SM import build_pooled_dataset

REPORT_PATH = "outputs_fixed/sm_corrected_channel_report.csv"
OUT_DIR = Path("outputs_fixed/rpe_analysis_sm_red_l")

JRGECO_ONLY_MICE = {"SM1B", "SM2B"}
GROUP_LABELS = {True: "jrgeco_only", False: "regular_mice"}


def red_l_session_pairs(report_path=REPORT_PATH):
    report = pd.read_csv(report_path)
    valid = report[report["red_l_carrier_valid"] == True]  # noqa: E712
    return [(Path(row["session_dir"]), "red_l") for _, row in valid.iterrows()]


def relabel_to_groups(out_dir=OUT_DIR):
    tt = pd.read_parquet(out_dir / "pooled_trial_table.parquet")
    assert (tt["hemisphere"] == "red_l").all(), "expected every pooled row to be red_l before relabeling"
    tt["hemisphere"] = tt["mouse"].isin(JRGECO_ONLY_MICE).map(GROUP_LABELS)
    tt.to_parquet(out_dir / "pooled_trial_table.parquet")

    with np.load(out_dir / "fir_pooled.npz") as f:
        fir = dict(f)
    assert (fir["hemisphere"] == "red_l").all(), "expected every FIR row to be red_l before relabeling"
    fir["hemisphere"] = np.array([GROUP_LABELS[m in JRGECO_ONLY_MICE] for m in fir["mouse"]])
    np.savez(out_dir / "fir_pooled.npz", **fir)

    print(f"Relabeled hemisphere column: {tt['hemisphere'].value_counts().to_dict()}")


def main():
    pairs = red_l_session_pairs()
    print(f"{len(pairs)} red_l-valid sessions "
          f"({sum(1 for p in pairs if p[0].name in JRGECO_ONLY_MICE)} jRGECO-only, "
          f"{sum(1 for p in pairs if p[0].name not in JRGECO_ONLY_MICE)} regular)")
    build_pooled_dataset(pairs, out_dir=OUT_DIR)
    relabel_to_groups(OUT_DIR)


if __name__ == "__main__":
    main()
