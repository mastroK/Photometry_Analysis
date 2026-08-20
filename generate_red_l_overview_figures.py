"""
Generate per-session overview figures (pipeline.run_session's plot_session_
overview) for red_l sessions in one group of the pooled SM red_l dataset --
the bulk-prep path (rpe_analysis_prep_SM.py's load_session_shared/
extract_hemisphere) deliberately skips figure generation for speed, so these
don't exist yet for the sessions that actually feed run_sm_glm_fir_analysis.py.

Reruns run_session per session (compute_bandit_state=False -- word labels
used by the overview figure don't need the Q-learning fit, only trial
outcome/choice history) with output_dir pointed at overview_figures/ inside
the same results folder as the rest of the red_l outputs.

Usage:
    python generate_red_l_overview_figures.py [group]
    # group one of "regular_mice" (default), "jrgeco_only"
"""

import sys
from pathlib import Path

import pandas as pd

from pipeline import run_session

DATA_DIR = Path("outputs_fixed/rpe_analysis_sm_red_l")
CHANNEL_REPORT = Path("outputs_fixed/sm_corrected_channel_report.csv")
OUT_DIR = DATA_DIR / "overview_figures"


def sessions_for_group(group):
    tt = pd.read_parquet(DATA_DIR / "pooled_trial_table.parquet")
    sessions = tt[tt["hemisphere"] == group][["mouse", "date"]].drop_duplicates()
    report = pd.read_csv(CHANNEL_REPORT, dtype={"date": str})[["mouse", "date", "session_dir"]].drop_duplicates()
    sessions["date"] = sessions["date"].astype(str)
    merged = sessions.merge(report, on=["mouse", "date"], how="left")
    missing = merged[merged["session_dir"].isna()]
    if len(missing):
        print(f"WARNING: {len(missing)} session(s) have no session_dir in the channel report, skipping:")
        print(missing[["mouse", "date"]].to_string(index=False))
        merged = merged.dropna(subset=["session_dir"])
    return sorted(Path(p) for p in merged["session_dir"])


def main(group="regular_mice"):
    session_dirs = sessions_for_group(group)
    print(f"{len(session_dirs)} {group} sessions to render\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ok, failed = 0, []
    for i, session_dir in enumerate(session_dirs, 1):
        mouse, date = session_dir.name, session_dir.parent.name
        print(f"--- [{i}/{len(session_dirs)}] {mouse} {date} ---")
        try:
            run_session(session_dir, hemisphere="red_l", output_dir=OUT_DIR, compute_bandit_state=False,
                        force_nominal_carrier_freq=True)
            ok += 1
        except Exception as exc:
            print(f"  FAILED: {exc}")
            failed.append((mouse, date, str(exc)))

    print(f"\nDone: {ok}/{len(session_dirs)} figures saved to {OUT_DIR}")
    if failed:
        print(f"{len(failed)} failures:")
        for mouse, date, err in failed:
            print(f"  {mouse} {date}: {err}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "regular_mice")
