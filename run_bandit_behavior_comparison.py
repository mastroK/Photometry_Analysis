"""
Fit the sticky Q-learning behavioral choice model (external/bandit_state_model.py --
alpha=learning rate, beta=inverse temperature, kappa=stickiness, fit by MLE
directly on each session's own left/right choices and rewards) to every
session across FP1_none/FP1_dcz/FP1_retrained_none/FP2_none/FP2_dcz and SM,
and pool the per-session fitted parameters into one table for a condition
(90/10 vs 80/20) and condition x age comparison.

Behavior-only: builds trial_table straight from the raw pokeHistory/stats
.mat files, skipping photometry load/demodulation entirely (this model only
needs the trial-level choice/reward sequence) -- see
run_behavior_dcz_comparison.py for the existing precedent of this pattern.

FP1/FP2 session lists come from run_expanded_glm_analysis.COHORTS (the
already-QC'd, already-exclusion-applied master/qc-report pairing used
throughout this project) via session_hemisphere_lookup. SM sessions come
from outputs_fixed/sm_corrected_channel_report.csv, deduplicated to one
fit per unique (mouse, date) -- red_l/green_l/green_r are channel-selection
views of the SAME underlying session, confirmed via
run_model_series_comparison_sm_red_l.get_sm_red_l_session_dirs's own
per-hemisphere filtering over this same report.

Age: FP1/FP2 have per-mouse DOB (config/cohort_metadata_FP1.csv,
cohort_metadata_FP2.csv) -> exact age_days at recording
(config/session_metadata.compute_age_days). SM has no DOB on file
(config/cohort_metadata_SM.csv's Date of Birth column is blank for every
mouse, Age_Bin_Source="folder") -- SM age is only available as the
P70..P170 age_bin category already stamped in sm_corrected_channel_report.csv
by run_sm_corrected_batch.py's RA-folder-based discovery. So the age axis is
exact-days for FP and a coarser age-bin for SM; these are reported
side-by-side, not coerced onto one shared numeric scale.

Usage:
    python run_bandit_behavior_comparison.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from behavior.trial_table import build_trial_table
from behavior.word_encoding import add_lag_features
from config.params import LAG_N
from config.session_metadata import compute_age_days, load_cohort_metadata
from external.bandit_state_adapter import add_bandit_state_features
from io_utils.raw_loader import discover_behavior_files, load_behavior_raw, parse_session_id
from run_expanded_glm_analysis import COHORTS, session_hemisphere_lookup
from run_manifest import write_run_manifest

OUT_DIR = Path("outputs_fixed/bandit_behavior_comparison")

SM_CHANNEL_REPORT = Path("outputs_fixed/sm_corrected_channel_report.csv")
SM_MICE = ("SM1L", "SM1N", "SM2N", "SM2R", "SM3FR")
SM_HEMISPHERES = ("red_l", "green_l", "green_r")

FP1_METADATA_CSV = Path("config/cohort_metadata_FP1.csv")
FP2_METADATA_CSV = Path("config/cohort_metadata_FP2.csv")


def fit_one_session(session_dir):
    """Build trial_table from raw behavior only (no photometry) and fit the
    sticky Q-learning model. Returns the fit_params dict, or None if the
    session has too few trials (add_bandit_state_features's own
    BANDIT_MIN_TRIALS guard, logged there) or the behavior files are missing.
    """
    mouse, date = parse_session_id(session_dir)
    try:
        poke_history_file, stats_file = discover_behavior_files(session_dir)
        poke_history, stats = load_behavior_raw(poke_history_file, stats_file)
    except Exception as exc:
        print(f"WARNING: skipping {mouse} {date} ({session_dir}): {exc}")
        return None

    trial_table = build_trial_table(poke_history, stats)
    trial_table = add_lag_features(trial_table, n_lags=LAG_N)
    trial_table = add_bandit_state_features(trial_table, session_id=f"{mouse}_{date}")

    fit_params = trial_table.attrs["bandit_fit_params"]
    if fit_params is None:
        return None

    frac_forced = float(trial_table["is_forced_block"].mean()) if "is_forced_block" in trial_table.columns else 0.0
    reward_probs = pd.unique(np.concatenate([
        trial_table["left_reward_prob"].to_numpy(), trial_table["right_reward_prob"].to_numpy(),
    ]))
    return {
        "mouse": mouse, "date": date,
        **fit_params,
        "frac_forced_trials": frac_forced,
        "reward_probs_observed": sorted(round(float(p), 2) for p in reward_probs),
    }


def collect_fp():
    rows = []
    for cohort_label, master_csv, qc_report_csv, exclude_pairs in COHORTS:
        session_dirs, _ = session_hemisphere_lookup(master_csv, qc_report_csv, exclude_pairs)
        cohort_family = "FP1" if cohort_label.startswith("FP1") else "FP2"
        metadata_csv = FP1_METADATA_CSV if cohort_family == "FP1" else FP2_METADATA_CSV
        metadata = load_cohort_metadata(metadata_csv)

        print(f"\n{cohort_label}: {len(session_dirs)} sessions")
        for session_dir in session_dirs:
            fit = fit_one_session(session_dir)
            if fit is None:
                continue
            dob = metadata.loc[fit["mouse"], "Date of Birth"]
            fit.update({
                "cohort": cohort_label, "cohort_family": cohort_family,
                "condition": "90/10", "age_days": compute_age_days(dob, fit["date"]), "age_bin": None,
            })
            rows.append(fit)
    return rows


def collect_sm():
    report = pd.read_csv(SM_CHANNEL_REPORT, dtype={"date": str})
    valid_mask = report["mouse"].isin(SM_MICE) & report[[f"{h}_carrier_valid" for h in SM_HEMISPHERES]].any(axis=1)
    sessions = report[valid_mask].drop_duplicates(subset=["mouse", "date"])

    print(f"\nSM: {len(sessions)} unique sessions (any-hemisphere-valid, {SM_MICE})")
    rows = []
    for _, row in sessions.iterrows():
        fit = fit_one_session(Path(row["session_dir"]))
        if fit is None:
            continue
        fit.update({
            "cohort": "SM", "cohort_family": "SM",
            "condition": "80/20", "age_days": None, "age_bin": row["age_bin"],
        })
        rows.append(fit)
    return rows


def main(out_dir=OUT_DIR):
    params = {"sm_mice": list(SM_MICE), "fp_cohorts": [c[0] for c in COHORTS]}
    write_run_manifest(out_dir, params=params, script="run_bandit_behavior_comparison.main")

    rows = collect_fp() + collect_sm()
    df = pd.DataFrame(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "bandit_params_pooled.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}: {len(df)} sessions, {df['mouse'].nunique()} mice")
    print(df.groupby("cohort")[["alpha", "beta", "kappa"]].agg(["mean", "count"]))
    return df


if __name__ == "__main__":
    main()
