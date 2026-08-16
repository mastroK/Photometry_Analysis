"""
Run the pipeline over an RA-curated condition subset rather than walking/
QC-filtering an entire raw-data share: parse processed_<mouse>_<date>.mat
filenames out of an individual_days/<condition>/ folder (see 2-Output/
FP1_processed_data/individual_days/{none,DCZ,saline}), locate the matching
raw session directories, QC them fresh, and export a per-trial master
DataFrame tagged with `condition`.

Hemisphere is resolved PER SESSION, not per mouse and not one fixed
--hemisphere flag for the whole batch. Confirmed directly against this
cohort's raw data: this rig underwent a session-wide cutover in mid-June
2023 where every mouse's active fluorescence channel switched from green_r
to green_l (not a per-mouse property at all -- e.g. WCL23 is green_r through
061223 and green_l from 061523 onward) -- see
config/session_hemisphere_overrides.csv, which is checked first, per
(mouse, date). config/mouse_hemisphere.csv (a coarser per-mouse-only default)
is only a fallback for any session missing from the override file.

Usage:
    python run_condition_batch.py /path/to/individual_days/none \\
        "/path/to/1-Raw data/FP1" \\
        --metadata config/cohort_metadata_FP1.csv \\
        --output outputs/none_cohort_master.csv \\
        --qc-output outputs/none_cohort_qc_report.csv \\
        [--condition-label none] [--max-segments N]
"""

import argparse
from pathlib import Path

from batch_processor import run_batch_sessions
from config.params import DEFAULT_HEMISPHERE
from config.session_metadata import get_mouse_hemisphere, load_mouse_hemisphere
from io_utils.raw_loader import discover_sessions_from_processed_dir, parse_session_id
from qc.channel_selection import load_session_hemisphere_overrides
from run_cohort_qc import run_cohort_qc_for_sessions

DEFAULT_HEMISPHERE_LOOKUP = Path(__file__).parent / "config" / "mouse_hemisphere.csv"
DEFAULT_SESSION_OVERRIDES = Path(__file__).parent / "config" / "session_hemisphere_overrides.csv"


def run_condition_batch(processed_dir, raw_root, metadata_path, output_path, qc_output_path,
                         condition_label=None, max_segments=None,
                         hemisphere_lookup_path=DEFAULT_HEMISPHERE_LOOKUP,
                         session_overrides_path=DEFAULT_SESSION_OVERRIDES):
    """See module docstring. condition_label defaults to processed_dir's own
    folder name (e.g. "none") if not given explicitly.
    """
    processed_dir = Path(processed_dir)
    raw_root = Path(raw_root)
    if condition_label is None:
        condition_label = processed_dir.name

    session_dirs, missing = discover_sessions_from_processed_dir(processed_dir, raw_root)
    print(f"Discovered {len(session_dirs)} session(s) from {processed_dir} "
          f"({len(missing)} processed file(s) had no matching raw session dir)")
    if not session_dirs:
        raise RuntimeError(f"No matching raw sessions found for {processed_dir} under {raw_root}")

    hemisphere_lookup = load_mouse_hemisphere(hemisphere_lookup_path)
    session_overrides = load_session_hemisphere_overrides(session_overrides_path)

    session_dates = {session_dir: parse_session_id(session_dir) for session_dir in session_dirs}

    def hemisphere_for_session(session_dir):
        mouse, date = session_dates[session_dir]
        if (mouse, date) in session_overrides:
            return session_overrides[(mouse, date)]
        return get_mouse_hemisphere(hemisphere_lookup, mouse, DEFAULT_HEMISPHERE)

    print("Per-session hemisphere:")
    for session_dir in session_dirs:
        mouse, date = session_dates[session_dir]
        source = "override" if (mouse, date) in session_overrides else "mouse default"
        print(f"  {mouse} {date}: {hemisphere_for_session(session_dir)} ({source})")

    print(f"\n=== Fresh QC for '{condition_label}' ({len(session_dirs)} sessions) ===")
    qc_df = run_cohort_qc_for_sessions(
        session_dirs, max_segments=max_segments, output_path=qc_output_path,
        hemisphere_for_session=hemisphere_for_session,
    )

    print(f"\n=== Batch export for '{condition_label}' ===")
    master_df = run_batch_sessions(
        session_dirs, metadata_path, output_path, max_segments=max_segments,
        qc_report_path=qc_output_path, hemisphere_for_session=hemisphere_for_session,
        condition_label=condition_label,
    )
    return master_df, qc_df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("processed_dir", type=Path,
                         help="RA individual_days/<condition>/ folder of processed_<mouse>_<date>.mat files")
    parser.add_argument("raw_root", type=Path,
                         help="Raw data cohort root containing <date>/<mouse>/ session folders (e.g. .../1-Raw data/FP1)")
    parser.add_argument("--metadata", type=Path, required=True, help="Cohort metadata sheet (.csv or .xlsx)")
    parser.add_argument("--output", type=Path, required=True, help="Output path for the master DataFrame (.parquet or .csv)")
    parser.add_argument("--qc-output", type=Path, required=True,
                         help="Output path for the fresh QC report CSV scoped to this session list")
    parser.add_argument("--condition-label", default=None,
                         help="Value stamped in the master DataFrame's `condition` column "
                              "(default: processed_dir's own folder name, e.g. 'none')")
    parser.add_argument("--max-segments", type=int, default=None,
                         help="Limit each session to its first N raw segments (for a quick test run)")
    parser.add_argument("--hemisphere-lookup", type=Path, default=DEFAULT_HEMISPHERE_LOOKUP,
                         help=f"Per-mouse hemisphere fallback CSV (default: {DEFAULT_HEMISPHERE_LOOKUP})")
    parser.add_argument("--session-overrides", type=Path, default=DEFAULT_SESSION_OVERRIDES,
                         help=f"Per-(mouse,date) hemisphere override CSV, checked first (default: {DEFAULT_SESSION_OVERRIDES})")
    args = parser.parse_args()

    run_condition_batch(
        args.processed_dir, args.raw_root, metadata_path=args.metadata, output_path=args.output,
        qc_output_path=args.qc_output, condition_label=args.condition_label,
        max_segments=args.max_segments, hemisphere_lookup_path=args.hemisphere_lookup,
        session_overrides_path=args.session_overrides,
    )


if __name__ == "__main__":
    main()
