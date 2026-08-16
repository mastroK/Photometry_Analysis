"""
Per-session hemisphere/channel detection report: for every session in an RA
individual_days/<condition>/ folder, check whether each candidate
fluorescence channel shows a genuine reward-vs-no-reward differential
response (see qc/channel_selection.py for why this replaced a raw-signal
carrier-lock heuristic that turned out to be unreliable), and report which
channel -- if any -- can be picked with confidence.

This is a REPORT, not a batch-processing step: review its output, then copy
any confident (mouse, date) rows you want to lock in into
config/session_hemisphere_overrides.csv (or just trust the per-mouse default
in config/mouse_hemisphere.csv for sessions this reports as ambiguous).

Usage:
    python run_hemisphere_detection.py /path/to/individual_days/none \\
        "/path/to/1-Raw data/FP1" \\
        --output outputs/none_hemisphere_detection.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from behavior.sync import align_behavior_to_photometry
from behavior.trial_table import build_trial_table
from config.params import DEFAULT_HEMISPHERE, HEMISPHERE_CHANNELS
from io_utils.raw_loader import discover_behavior_files, discover_sessions_from_processed_dir, load_behavior_raw, load_raw_photometry, parse_session_id
from preprocessing.demodulate import demodulate_envelope, estimate_carrier_freq
from qc.channel_selection import DEFAULT_CANDIDATE_HEMISPHERES, detect_session_hemisphere


def detect_one_session(session_dir, candidates=DEFAULT_CANDIDATE_HEMISPHERES, max_segments=None):
    session_dir = Path(session_dir)
    mouse, date = parse_session_id(session_dir)
    row = dict(mouse=mouse, date=date, session_dir=str(session_dir), detected=None, error=None)

    try:
        raw = load_raw_photometry(session_dir / "PHOTO", max_segments=max_segments)
        ref_channel = HEMISPHERE_CHANNELS[DEFAULT_HEMISPHERE]
        measured_freq, _ = estimate_carrier_freq(raw[ref_channel.signal_channel])
        envelope, _ = demodulate_envelope(raw[ref_channel.signal_channel], measured_freq)

        poke_history_file, stats_file = discover_behavior_files(session_dir)
        poke_history, stats = load_behavior_raw(poke_history_file, stats_file)
        trial_table = build_trial_table(poke_history, stats)
        trial_table, _ = align_behavior_to_photometry(raw, trial_table, poke_history, n_final_samples=len(envelope))

        detected, results = detect_session_hemisphere(raw, trial_table, candidates=candidates)
        row["detected"] = detected
        for name, r in results.items():
            row[f"{name}_cohens_d"] = r.get("cohens_d")
            row[f"{name}_p_value"] = r.get("p_value")
            row[f"{name}_error"] = r.get("error")
    except Exception as exc:
        row["error"] = str(exc)

    return row


def run_hemisphere_detection(processed_dir, raw_root, output_path,
                              candidates=DEFAULT_CANDIDATE_HEMISPHERES, max_segments=None):
    processed_dir = Path(processed_dir)
    raw_root = Path(raw_root)
    output_path = Path(output_path)

    session_dirs, missing = discover_sessions_from_processed_dir(processed_dir, raw_root)
    print(f"Discovered {len(session_dirs)} session(s) from {processed_dir} "
          f"({len(missing)} processed file(s) had no matching raw session dir)")

    rows = [detect_one_session(session_dir, candidates=candidates, max_segments=max_segments)
            for session_dir in session_dirs]
    report = pd.DataFrame(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)

    n_confident = int(report["detected"].notna().sum())
    n_error = int(report["error"].notna().sum())
    print(f"\n{n_confident}/{len(report)} session(s) resolved with confidence, "
          f"{len(report) - n_confident - n_error} ambiguous, {n_error} errored -> {output_path}")
    disagreements = report[report["detected"].notna()].copy()
    if len(disagreements):
        print("\nConfidently-detected sessions:")
        print(disagreements[["mouse", "date", "detected"]].to_string(index=False))

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("processed_dir", type=Path,
                         help="RA individual_days/<condition>/ folder of processed_<mouse>_<date>.mat files")
    parser.add_argument("raw_root", type=Path,
                         help="Raw data cohort root containing <date>/<mouse>/ session folders")
    parser.add_argument("--output", type=Path, required=True, help="Output path for the detection report CSV")
    parser.add_argument("--candidates", nargs="+", default=list(DEFAULT_CANDIDATE_HEMISPHERES),
                         choices=list(HEMISPHERE_CHANNELS), help="Candidate hemispheres to evaluate per session")
    parser.add_argument("--max-segments", type=int, default=None,
                         help="Limit each session to its first N raw segments (for a quick test run)")
    args = parser.parse_args()

    run_hemisphere_detection(
        args.processed_dir, args.raw_root, args.output,
        candidates=args.candidates, max_segments=args.max_segments,
    )


if __name__ == "__main__":
    main()
