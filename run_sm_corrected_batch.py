"""
Corrected SM (PV-dualphotometry) cohort channel-validity batch: replaces
run_sm_bilateral_batch.py's evaluate_bilateral_hemispheres (which trusted
estimate_carrier_freq's free-ranging global FFT peak search) with
qc.channel_selection.evaluate_all_channels_true_signal (an unbiased wide-band
search that discovers the true carrier per raw channel, then only checks
whether it happens to land near one of this rig's two known carriers --
167 Hz "green", 223 Hz "red" -- rather than assuming either the auto-estimate
or the channel's own label is correct). See qc/channel_selection.py's module
comment for the full rationale and what went wrong with both prior attempts.

Also runs ALL FOUR channels (green_r, green_l, red_r, red_l) per session, not
just the two green ones -- now that we've confirmed neither color is cleanly
isolated on this rig, and that the "r" side is not uniformly dead, there's no
principled reason to restrict the corrected survey to green only. Downstream
analysis choices (which channel(s) to treat as the primary readout) are a
separate, later decision from this per-session validity survey.

Usage:
    python run_sm_corrected_batch.py \\
        "/Volumes/Neurobio/MICROSCOPE/Kevin/3-Experiments/2-Behavior/3-nosepoke_SM/2-Output/PV-dualphotometry/individual_days" \\
        "/Volumes/Neurobio/MICROSCOPE/Kevin/3-Experiments/2-Behavior/3-nosepoke_SM/1-raw data/PV-dualphotometry" \\
        --output-report outputs_fixed/sm_corrected_channel_report.csv
"""

import argparse
from pathlib import Path

import pandas as pd

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
from qc.channel_selection import evaluate_all_channels_true_signal

AGE_FOLDERS = ["P70", "P80", "P90", "P100", "P110", "P120", "P130", "P140", "P150", "P160", "P170"]
HEMISPHERES = ("green_r", "green_l", "red_r", "red_l")

# Confirmed jRGECO-only mice (no GCaMP) -- negative controls for the green
# channel, useful positive data points for red. Only SM1B/SM2B are confirmed;
# the user's original "SM3ML, SM3FL" pair doesn't match any of our 11
# processed-data mice (closest names are SM3MN/SM3MR/SM3FR), and SM3MN's
# green_l visually shows a real, strong GCaMP-like pattern -- inconsistent
# with it being jRGECO-only -- so it is deliberately NOT included here pending
# clarification. Flagged in the report rather than excluded from the sweep:
# these mice are still useful data points for this per-session validity survey.
JRGECO_ONLY_MICE = {"SM1B", "SM2B"}


def evaluate_one_session(session_dir, max_segments=None):
    """Load raw + trial_table once, run the corrected unbiased 4-channel
    evaluation. Errors (failed sync, no raw files, etc.) are recorded rather
    than raised, matching this codebase's existing soft-fail convention.
    """
    mouse, date = parse_session_id(session_dir)
    row = dict(mouse=mouse, date=date, session_dir=str(session_dir), jrgeco_only=mouse in JRGECO_ONLY_MICE, error=None)
    for h in HEMISPHERES:
        row[f"{h}_carrier_valid"] = False
        row[f"{h}_reward_significant"] = None

    try:
        raw = load_raw_photometry(session_dir / "PHOTO", max_segments=max_segments)
        # Alignment only depends on the digital behavior channels, so any
        # hemisphere's envelope length works as the reference for n_final_samples.
        ref_channel = HEMISPHERE_CHANNELS[DEFAULT_HEMISPHERE]
        measured_freq, _ = estimate_carrier_freq(raw[ref_channel.signal_channel])
        envelope, _ = demodulate_envelope(raw[ref_channel.signal_channel], measured_freq)

        poke_history_file, stats_file = discover_behavior_files(session_dir)
        poke_history, stats = load_behavior_raw(poke_history_file, stats_file)
        trial_table = build_trial_table(poke_history, stats)
        trial_table, _ = align_behavior_to_photometry(
            raw, trial_table, poke_history, n_final_samples=len(envelope)
        )

        carrier_valid, results = evaluate_all_channels_true_signal(raw, trial_table, hemispheres=HEMISPHERES)
        for h in HEMISPHERES:
            row[f"{h}_carrier_valid"] = carrier_valid[h]
            r = results[h]
            row[f"{h}_reward_significant"] = r.get("reward_significant")
            row[f"{h}_detected_freq_hz"] = r.get("detected_freq_hz")
            row[f"{h}_matched_known_freq_hz"] = r.get("matched_known_freq_hz")
            row[f"{h}_amplitude"] = r.get("amplitude")
            row[f"{h}_cohens_d"] = r.get("cohens_d")
            row[f"{h}_p_value"] = r.get("p_value")
            row[f"{h}_error"] = r.get("error")
    except Exception as exc:
        row["error"] = str(exc)

    return row


def build_corrected_report(processed_root, raw_root, age_folders=AGE_FOLDERS, max_segments=None):
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
            f"  {row['mouse']} {row['date']} ({age}){' [jRGECO-only]' if row['jrgeco_only'] else ''}: "
            + " ".join(f"{h}={row.get(f'{h}_carrier_valid')}" for h in HEMISPHERES)
            + f" error={row.get('error')}"
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("processed_root", type=Path)
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("--output-report", type=Path, default=Path("outputs_fixed/sm_corrected_channel_report.csv"))
    parser.add_argument("--age-folders", nargs="+", default=AGE_FOLDERS)
    parser.add_argument("--max-segments", type=int, default=None)
    args = parser.parse_args()

    report = build_corrected_report(
        args.processed_root, args.raw_root, age_folders=args.age_folders, max_segments=args.max_segments,
    )
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output_report, index=False)

    for h in HEMISPHERES:
        n_valid = int(report[f"{h}_carrier_valid"].sum())
        n_sig = int((report[f"{h}_reward_significant"] == True).sum())  # noqa: E712 (nullable bool column)
        print(f"{h}: {n_valid}/{len(report)} carrier_valid, {n_sig}/{len(report)} reward_significant")
    print(f"\nSaved corrected channel report ({len(report)} sessions) to {args.output_report}")


if __name__ == "__main__":
    main()
