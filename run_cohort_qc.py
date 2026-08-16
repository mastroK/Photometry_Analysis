"""
Evaluate qc.session_qc.evaluate_session_qc() over every session in one or
more cohort root(s) and write a cohort-level QC report CSV that a human can
review/override (via the manual_include column) before batch_processor.py
or run_fir_glm.py are pointed at it with --qc-report.

Usage:
    python run_cohort_qc.py /path/to/cohort_root [/path/to/other_cohort_root ...] \\
        [--hemisphere green_r|red_l] [--max-segments N] [--output outputs/cohort_qc_report.csv]
"""

import argparse
import itertools
from pathlib import Path

import pandas as pd

from batch_processor import discover_sessions
from config.params import DEFAULT_HEMISPHERE, HEMISPHERE_CHANNELS
from qc.session_qc import evaluate_session_qc

DEFAULT_OUTPUT = Path("outputs/cohort_qc_report.csv")

_CHECK_COLUMNS = ["sync_pass", "trials_pass", "balance_pass", "signal_range_pass", "missing_pass"]


def run_cohort_qc_for_sessions(session_dirs, hemisphere=DEFAULT_HEMISPHERE, max_segments=None,
                                output_path=DEFAULT_OUTPUT, hemisphere_for_session=None):
    """Evaluate an explicit list of session directories (rather than
    discovering them by walking a cohort_root -- see run_cohort_qc below for
    that path) and write/update the QC report CSV.

    hemisphere_for_session : optional callable(session_dir) -> hemisphere_key,
        overriding the single `hemisphere` value per session -- used by
        run_condition_batch.py to apply a per-(mouse,date) hemisphere lookup
        instead of one fixed value for every session in the batch.
        Hemisphere is NOT just a per-mouse property in this cohort -- this
        rig underwent a session-wide cutover mid-June 2023 (every mouse's
        active channel switched from green_r to green_l on the same date),
        so a callable keyed only by mouse can't express it; see
        config/session_hemisphere_overrides.csv.
    """
    output_path = Path(output_path)
    session_dirs = list(session_dirs)

    rows = []
    for session_dir in session_dirs:
        session_hemisphere = hemisphere_for_session(session_dir) if hemisphere_for_session is not None else hemisphere
        rows.append(evaluate_session_qc(session_dir, hemisphere=session_hemisphere, max_segments=max_segments))
    new_df = pd.DataFrame(rows)
    new_df["manual_include"] = new_df["QC_PASS"]

    if output_path.exists():
        old_df = pd.read_csv(output_path, dtype={"date": str})
        old_df["date"] = old_df["date"].astype(str)
        new_df["date"] = new_df["date"].astype(str)
        old_manual = old_df.set_index(["mouse", "date"])["manual_include"]
        for idx in new_df.index:
            key = (new_df.at[idx, "mouse"], new_df.at[idx, "date"])
            if key in old_manual.index:
                new_df.at[idx, "manual_include"] = old_manual.loc[key]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(output_path, index=False)

    n_total = len(new_df)
    n_pass = int(new_df["QC_PASS"].sum())
    print(f"\nQC summary: {n_pass}/{n_total} session(s) passed all checks -> {output_path}")
    failing = new_df[~new_df["QC_PASS"]]
    if len(failing):
        print("Failing sessions:")
        for _, row in failing.iterrows():
            if row["error"] is not None and not pd.isna(row["error"]):
                print(f"  {row['mouse']} {row['date']}: error -- {row['error']}")
                continue
            failed_checks = [c for c in _CHECK_COLUMNS if not row[c]]
            print(f"  {row['mouse']} {row['date']}: failed {', '.join(failed_checks)}")

    return new_df


def run_cohort_qc(cohort_roots, hemisphere=DEFAULT_HEMISPHERE, max_segments=None, output_path=DEFAULT_OUTPUT):
    """Discover every session under cohort_root/<date>/<mouse>/ for each of
    cohort_roots (batch_processor.discover_sessions) and evaluate them all
    with a single fixed `hemisphere` -- see run_cohort_qc_for_sessions for
    the explicit-session-list / per-mouse-hemisphere variant used by
    run_condition_batch.py.
    """
    session_dirs = list(itertools.chain.from_iterable(discover_sessions(root) for root in cohort_roots))
    return run_cohort_qc_for_sessions(
        session_dirs, hemisphere=hemisphere, max_segments=max_segments, output_path=output_path,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cohort_root", type=Path, nargs="+",
                         help="One or more root directories containing <date>/<mouse>/ session folders")
    parser.add_argument("--hemisphere", choices=list(HEMISPHERE_CHANNELS), default=DEFAULT_HEMISPHERE)
    parser.add_argument("--max-segments", type=int, default=None,
                         help="Limit each session to its first N raw segments (for a quick test run)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                         help=f"Output path for the QC report CSV (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    run_cohort_qc(
        args.cohort_root, hemisphere=args.hemisphere,
        max_segments=args.max_segments, output_path=args.output,
    )


if __name__ == "__main__":
    main()
