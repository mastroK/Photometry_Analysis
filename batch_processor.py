"""
Batch-run pipeline.run_session over every session in a cohort directory,
attach per-mouse metadata (DOB, DREADD Treatment/Viral_Expression, and
age-at-recording), and export one row-per-trial master DataFrame across the
whole cohort.

Usage:
    python batch_processor.py /path/to/cohort_root --metadata cohort_metadata.csv --output master_df.parquet

Where cohort_root contains <date>/<mouse>/ session folders (e.g.
cohort_root/060223/WCL23/, matching io_utils.raw_loader.parse_session_id's
documented <cohort>/<date>/<mouse>/ convention -- cohort_root here IS "the
cohort", one level above the date folders).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config.params import ALIGN_EVENT_COLUMNS, DEFAULT_ALIGN_EVENT, DEFAULT_HEMISPHERE, HEMISPHERE_CHANNELS
from config.session_metadata import compute_age_days, get_mouse_metadata, load_cohort_metadata
from io_utils.raw_loader import parse_session_id
from pipeline import run_session


def discover_sessions(cohort_root):
    """Yield session directories under cohort_root/<date>/<mouse>/ that
    contain a PHOTO/ subfolder.
    """
    cohort_root = Path(cohort_root)
    for date_dir in sorted(p for p in cohort_root.iterdir() if p.is_dir()):
        for mouse_dir in sorted(p for p in date_dir.iterdir() if p.is_dir()):
            if (mouse_dir / "PHOTO").is_dir():
                yield mouse_dir


def run_batch_sessions(session_dirs, metadata_path, output_path, hemisphere=DEFAULT_HEMISPHERE,
                        max_segments=None, align_event=DEFAULT_ALIGN_EVENT):
    """Run an explicit list of session directories through run_session(),
    attach cohort metadata, and write the concatenated per-trial result to
    output_path (.parquet or .csv, inferred from suffix).

    Pulls result["trial_table"] (not peth_trial_table) -- trial_table already
    carries every merged column (word/lag features, Behavioral_State/Q-values
    from external.bandit_state_adapter, and the per-event peak_z_*/auc_*
    photometry summaries from alignment.windowing.compute_per_trial_event_metrics
    for all 4 trial events, computed regardless of which one `align_event`
    picks); peth_trial_table is additionally restricted to trials with a full
    dF/F window for just the ONE chosen align_event, which the master
    DataFrame doesn't need since it only stores the scalar per-trial metrics,
    not raw window arrays.

    A session that fails (e.g. too few trials, alignment below the xcorr
    acceptance threshold) is logged and skipped rather than aborting the
    whole batch.
    """
    metadata_df = load_cohort_metadata(metadata_path)
    output_path = Path(output_path)

    frames = []
    n_ok, n_failed = 0, 0
    failed_sessions = []

    for session_dir in session_dirs:
        mouse, date = parse_session_id(session_dir)
        try:
            result = run_session(session_dir, hemisphere=hemisphere, max_segments=max_segments,
                                  align_event=align_event)
        except Exception as exc:
            print(f"WARNING: skipping session {session_dir} ({mouse} {date}): {exc}")
            n_failed += 1
            failed_sessions.append((mouse, date, str(exc)))
            continue

        session_df = result["trial_table"].copy()
        session_df["mouse"] = mouse
        session_df["date"] = date
        session_df["hemisphere"] = hemisphere

        try:
            meta = get_mouse_metadata(metadata_df, mouse)
            session_df["dob"] = meta["Date of Birth"]
            session_df["treatment"] = meta["Treatment"]
            session_df["viral_expression"] = meta["Viral_Expression"]
            session_df["age_days"] = compute_age_days(meta["Date of Birth"], date)
        except KeyError as exc:
            print(f"WARNING: {exc} -- {mouse} {date} will have NaN metadata columns")
            session_df["dob"] = pd.NaT
            session_df["treatment"] = None
            session_df["viral_expression"] = None
            session_df["age_days"] = np.nan

        frames.append(session_df)
        n_ok += 1

    if not frames:
        raise RuntimeError(f"No sessions successfully processed out of {list(session_dirs)}")

    master_df = pd.concat(frames, ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        master_df.to_parquet(output_path)
    else:
        master_df.to_csv(output_path, index=False)

    print(f"Processed {n_ok} session(s), skipped {n_failed}. "
          f"{len(master_df)} total trials -> {output_path}")
    if failed_sessions:
        print("Skipped sessions:")
        for mouse, date, reason in failed_sessions:
            print(f"  {mouse} {date}: {reason}")
    return master_df


def run_batch(cohort_root, metadata_path, output_path, hemisphere=DEFAULT_HEMISPHERE,
              max_segments=None, align_event=DEFAULT_ALIGN_EVENT):
    """Discover every session under cohort_root/<date>/<mouse>/ and run them
    all through run_batch_sessions(). See run_batch_sessions for the merge/
    export behavior; use that function directly to target an explicit subset
    of sessions instead of walking a whole cohort tree.
    """
    return run_batch_sessions(
        discover_sessions(cohort_root), metadata_path, output_path,
        hemisphere=hemisphere, max_segments=max_segments, align_event=align_event,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort_root", type=Path, help="Root directory containing <date>/<mouse>/ session folders")
    parser.add_argument("--metadata", type=Path, required=True, help="Cohort metadata sheet (.csv or .xlsx)")
    parser.add_argument("--output", type=Path, required=True, help="Output path for the master DataFrame (.parquet or .csv)")
    parser.add_argument("--hemisphere", choices=list(HEMISPHERE_CHANNELS), default=DEFAULT_HEMISPHERE)
    parser.add_argument("--max-segments", type=int, default=None, help="Limit each session to its first N raw segments (for a quick test run)")
    parser.add_argument("--align-to", choices=list(ALIGN_EVENT_COLUMNS), default=DEFAULT_ALIGN_EVENT,
                         help="Which trial event peth_trial_table/the reward-split figure is aligned to per session "
                              "(default: %(default)s) -- the master DataFrame's peak_z_*/auc_* columns cover all 4 "
                              "events regardless of this choice")
    args = parser.parse_args()

    run_batch(
        args.cohort_root,
        metadata_path=args.metadata,
        output_path=args.output,
        hemisphere=args.hemisphere,
        max_segments=args.max_segments,
        align_event=args.align_to,
    )


if __name__ == "__main__":
    main()
