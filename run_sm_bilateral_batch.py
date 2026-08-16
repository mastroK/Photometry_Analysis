"""
Bilateral hemisphere batch processing for the SM (PV-dualphotometry) cohort.

Unlike WCL/FP1/FP2 (one real fiber per mouse, picked via
config/session_hemisphere_overrides.csv), this rig has TWO real fibers
(green_r, green_l) recorded simultaneously -- both potentially valid
pyramidal (GCaMP) signal. So instead of picking a single winner per session,
each hemisphere is independently gated (qc.channel_selection.
evaluate_bilateral_hemispheres) and BOTH sides that pass are exported,
tagged by their own `hemisphere` column -- one mouse/session can contribute
0, 1, or 2 rows to the pooled dataset depending on which side(s) actually
show a real reward-locked response that day.

Session discovery is per age folder (P70..P170, Sean's own organization),
with the age-bin label stamped into `condition_label` (repurposing the same
mechanism run_condition_batch.py uses for drug condition) -- purely for
provenance/future age-stratified work; this script pools every age folder
together for the bilateral report and per-hemisphere master CSVs.

Usage:
    python run_sm_bilateral_batch.py \\
        "/Volumes/Neurobio/MICROSCOPE/Kevin/3-Experiments/2-Behavior/3-nosepoke_SM/2-Output/PV-dualphotometry/individual_days" \\
        "/Volumes/Neurobio/MICROSCOPE/Kevin/3-Experiments/2-Behavior/3-nosepoke_SM/1-raw data/PV-dualphotometry" \\
        --metadata config/cohort_metadata_SM.csv \\
        --output-dir outputs_fixed \\
        --bilateral-report outputs_fixed/sm_bilateral_hemisphere_report.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from batch_processor import run_batch_sessions
from behavior.sync import align_behavior_to_photometry
from behavior.trial_table import build_trial_table
from config.params import DEFAULT_HEMISPHERE, HEMISPHERE_CHANNELS
from io_utils.raw_loader import (
    discover_behavior_files,
    discover_sessions_from_processed_dir,
    load_behavior_raw,
    load_raw_photometry,
    parse_session_id,
)
from preprocessing.demodulate import demodulate_envelope, estimate_carrier_freq
from qc.channel_selection import evaluate_bilateral_hemispheres
from run_cohort_qc import run_cohort_qc_for_sessions

AGE_FOLDERS = ["P70", "P80", "P90", "P100", "P110", "P120", "P130", "P140", "P150", "P160", "P170"]
HEMISPHERES = ("green_r", "green_l")


def evaluate_one_session(session_dir, max_segments=None):
    """Load raw + trial_table once for this session and independently gate
    both candidate hemispheres. Errors (e.g. failed xcorr sync, no raw
    files) are recorded per-session rather than raised, matching this
    codebase's existing soft-fail convention (run_hemisphere_detection.py).
    """
    mouse, date = parse_session_id(session_dir)
    row = dict(mouse=mouse, date=date, session_dir=str(session_dir), error=None)
    for h in HEMISPHERES:
        row[f"{h}_valid"] = False

    try:
        raw = load_raw_photometry(session_dir / "PHOTO", max_segments=max_segments)
        ref_channel = HEMISPHERE_CHANNELS[DEFAULT_HEMISPHERE]
        measured_freq, _ = estimate_carrier_freq(raw[ref_channel.signal_channel])
        envelope, _ = demodulate_envelope(raw[ref_channel.signal_channel], measured_freq)

        poke_history_file, stats_file = discover_behavior_files(session_dir)
        poke_history, stats = load_behavior_raw(poke_history_file, stats_file)
        trial_table = build_trial_table(poke_history, stats)
        trial_table, _ = align_behavior_to_photometry(
            raw, trial_table, poke_history, n_final_samples=len(envelope)
        )

        valid, results = evaluate_bilateral_hemispheres(raw, trial_table, hemispheres=HEMISPHERES)
        for h in HEMISPHERES:
            row[f"{h}_valid"] = valid[h]
            row[f"{h}_cohens_d"] = results[h].get("cohens_d")
            row[f"{h}_p_value"] = results[h].get("p_value")
            row[f"{h}_error"] = results[h].get("error")
    except Exception as exc:
        row["error"] = str(exc)

    return row


def build_bilateral_report(processed_root, raw_root, age_folders=AGE_FOLDERS, max_segments=None):
    processed_root = Path(processed_root)
    raw_root = Path(raw_root)

    age_by_session = {}
    for age in age_folders:
        session_dirs, missing = discover_sessions_from_processed_dir(processed_root / age, raw_root)
        print(f"{age}: {len(session_dirs)} session(s) discovered ({len(missing)} missing raw dir)")
        for session_dir in session_dirs:
            age_by_session.setdefault(session_dir, age)

    rows = []
    for session_dir, age in sorted(age_by_session.items()):
        row = evaluate_one_session(session_dir, max_segments=max_segments)
        row["age_bin"] = age
        rows.append(row)
        print(
            f"  {row['mouse']} {row['date']} ({age}): "
            f"green_r_valid={row.get('green_r_valid')} green_l_valid={row.get('green_l_valid')} "
            f"error={row.get('error')}"
        )
    return pd.DataFrame(rows)


def run_sm_bilateral_batch(
    processed_root, raw_root, metadata_path, output_dir, bilateral_report_path,
    age_folders=AGE_FOLDERS, max_segments=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bilateral_report_path = Path(bilateral_report_path)

    report = build_bilateral_report(processed_root, raw_root, age_folders=age_folders, max_segments=max_segments)
    bilateral_report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(bilateral_report_path, index=False)
    n_either = int((report["green_r_valid"] | report["green_l_valid"]).sum())
    n_both = int((report["green_r_valid"] & report["green_l_valid"]).sum())
    print(
        f"\nSaved bilateral hemisphere report to {bilateral_report_path} "
        f"({len(report)} sessions; {n_either} with >=1 valid side, {n_both} with both)"
    )

    master_frames = []
    for hemisphere in HEMISPHERES:
        valid_sessions = [
            Path(row["session_dir"]) for _, row in report.iterrows() if row.get(f"{hemisphere}_valid")
        ]
        print(f"\n=== {hemisphere}: {len(valid_sessions)} valid session(s) ===")
        if not valid_sessions:
            continue

        qc_output = output_dir / f"sm_{hemisphere}_cohort_qc_report.csv"
        master_output = output_dir / f"sm_{hemisphere}_cohort_master.csv"
        run_cohort_qc_for_sessions(
            valid_sessions, hemisphere=hemisphere, max_segments=max_segments, output_path=qc_output,
        )
        master_df = run_batch_sessions(
            valid_sessions, metadata_path, master_output, hemisphere=hemisphere,
            max_segments=max_segments, qc_report_path=qc_output,
        )
        master_frames.append(master_df)

    if master_frames:
        pooled = pd.concat(master_frames, ignore_index=True)
        pooled_path = output_dir / "sm_bilateral_cohort_master.csv"
        pooled.to_csv(pooled_path, index=False)
        print(f"\nSaved pooled bilateral master ({len(pooled)} trials) to {pooled_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("processed_root", type=Path, help="individual_days/ root containing P70..P170 age folders")
    parser.add_argument("raw_root", type=Path, help="1-raw data/PV-dualphotometry root")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_fixed"))
    parser.add_argument(
        "--bilateral-report", type=Path, default=Path("outputs_fixed/sm_bilateral_hemisphere_report.csv")
    )
    parser.add_argument("--age-folders", nargs="+", default=AGE_FOLDERS)
    parser.add_argument("--max-segments", type=int, default=None)
    args = parser.parse_args()

    run_sm_bilateral_batch(
        args.processed_root, args.raw_root, args.metadata, args.output_dir,
        args.bilateral_report, age_folders=args.age_folders, max_segments=args.max_segments,
    )


if __name__ == "__main__":
    main()
