"""
End-to-end fiber-photometry pipeline entry point: raw LabJack photometry +
raw behavior logs -> demodulated dF/F & rolling z-score -> behavior<->
photometry clock alignment -> reward-split PETH -> verification figure.

Usage:
    python pipeline.py /path/to/session_dir [--max-segments N] [--hemisphere green_r|red_l]
                       [--align-to center_in|side_in|outcome|side_out]

Where session_dir contains a PHOTO/ subfolder of Raw_*.mat segments plus
pokeHistory*.mat and stats*.mat (e.g. .../FP1/060223/WCL23/).
"""

import argparse
from pathlib import Path

import numpy as np

from alignment.windowing import (
    compute_event_aligned_zscore,
    compute_per_trial_event_metrics,
    extract_peth,
    get_event_indices,
)
from behavior.sync import align_behavior_to_photometry
from behavior.trial_table import build_trial_table
from behavior.word_encoding import add_lag_features, add_word_labels, evaluate_word_outcomes
from config.params import (
    ALIGN_EVENT_COLUMNS,
    ALIGN_EVENT_LABELS,
    DECISION_WINDOW_S,
    DEFAULT_ALIGN_EVENT,
    FINAL_SAMPLE_FREQ_HZ,
    FINAL_TIME_STEP_SEC,
    HEMISPHERE_CHANNELS,
    DEFAULT_HEMISPHERE,
    LAG_N,
    PETH_BASELINE_POST_EVENT_S,
    PETH_BASELINE_PRE_EVENT_S,
    PETH_POST_SEC,
    PETH_PRE_SEC,
    REWARD_WINDOW_S,
    SUMMARY_GROUP_COLUMNS,
    SUMMARY_TOP_N,
    WORD_ENCODING_LEVELS,
)
from external.bandit_state_adapter import add_bandit_state_features
from io_utils.raw_loader import discover_behavior_files, load_behavior_raw, load_raw_photometry, parse_session_id
from preprocessing.demodulate import compute_dff_and_zscore, demodulate_envelope, estimate_carrier_freq
from viz.traces import plot_session_overview

# Default location for saved figures (both .png and .svg) when the caller
# doesn't pass output_dir -- a dedicated folder rather than dumping generated
# figures alongside the pipeline's own source files.
DEFAULT_FIGURE_DIR = Path(__file__).parent / "figures"


def run_session(session_dir, max_segments=None, hemisphere=DEFAULT_HEMISPHERE, output_dir=None,
                 compute_bandit_state=True, align_event=DEFAULT_ALIGN_EVENT):
    if align_event not in ALIGN_EVENT_COLUMNS:
        raise ValueError(f"Unknown align_event {align_event!r}; must be one of {list(ALIGN_EVENT_COLUMNS)}")
    session_dir = Path(session_dir)
    photo_dir = session_dir / "PHOTO"
    mouse, date = parse_session_id(session_dir)
    channels = HEMISPHERE_CHANNELS[hemisphere]

    # --- 1. load raw photometry -------------------------------------------------
    raw = load_raw_photometry(photo_dir, max_segments=max_segments)

    # --- 2/3. carrier estimate + demodulate --------------------------------------
    measured_freq, _ = estimate_carrier_freq(raw[channels.signal_channel])
    print(f"Estimated carrier freq for signal channel: {measured_freq:.3f} Hz "
          f"(nominal {channels.nominal_carrier_freq_hz} Hz)")

    envelope, locked_freq = demodulate_envelope(raw[channels.signal_channel], measured_freq)
    print(f"Demodulated envelope: {len(envelope)} samples at {FINAL_SAMPLE_FREQ_HZ:.4f} Hz "
          f"({len(envelope) / FINAL_SAMPLE_FREQ_HZ:.1f} s), locked to {locked_freq:.2f} Hz")

    dff, zscore, baseline = compute_dff_and_zscore(raw[channels.signal_channel], measured_freq, envelope)
    time_axis = np.arange(len(envelope)) * FINAL_TIME_STEP_SEC

    # --- 4. behavior raw -> trial table -------------------------------------------
    poke_history_file, stats_file = discover_behavior_files(session_dir)
    poke_history, stats = load_behavior_raw(poke_history_file, stats_file)
    trial_table = build_trial_table(poke_history, stats)
    trial_table = add_word_labels(trial_table, levels=WORD_ENCODING_LEVELS)
    trial_table = add_lag_features(trial_table, n_lags=LAG_N)
    if compute_bandit_state:
        trial_table = add_bandit_state_features(trial_table, session_id=f"{mouse}_{date}")
    print(f"Parsed {len(trial_table)} trials "
          f"({trial_table['was_rewarded'].sum()} rewarded, "
          f"leftProb={trial_table['left_reward_prob'].iloc[0]}, "
          f"rightProb={trial_table['right_reward_prob'].iloc[0]})")

    # --- 5. align behavior clock to photometry clock ------------------------------
    trial_table, align_info = align_behavior_to_photometry(raw, trial_table, poke_history, n_final_samples=len(envelope))

    pre_samples = int(round(PETH_PRE_SEC * FINAL_SAMPLE_FREQ_HZ))
    post_samples = int(round(PETH_POST_SEC * FINAL_SAMPLE_FREQ_HZ))
    peth_time = np.arange(-pre_samples, post_samples + 1) * FINAL_TIME_STEP_SEC

    # Per-trial photometry summary (peak z-score / AUC in DECISION_WINDOW_S
    # after the event) for ALL 4 trial events, not just the one chosen for
    # the primary PETH below -- attached to the full (unfiltered) trial_table
    # so every downstream consumer (peth_trial_table, the batch master
    # DataFrame) inherits them automatically, same row-alignment convention
    # already used for word/lag/bandit-state features. 'outcome' is
    # numerically identical to 'side_in' here since it aliases the same
    # photometry index (see config.params.ALIGN_EVENT_COLUMNS).
    for evt in ALIGN_EVENT_COLUMNS:
        peak, auc = compute_per_trial_event_metrics(
            dff, trial_table, evt, pre_samples, post_samples, peth_time,
            PETH_BASELINE_PRE_EVENT_S, PETH_BASELINE_POST_EVENT_S, DECISION_WINDOW_S,
        )
        trial_table[f"peak_z_{evt}"] = peak
        trial_table[f"auc_{evt}"] = auc

    # keep only trials whose aligning event index falls inside the recorded envelope
    align_col = ALIGN_EVENT_COLUMNS[align_event]
    trial_table = trial_table[trial_table[align_col] >= 0].reset_index(drop=True)

    # --- 6. PETH aligned to the chosen trial event, split by reward --------------
    # Extract ONE set of per-trial windows (rather than separately per reward
    # group) so every trial's window stays row-aligned with its trial_table
    # entry -- needed to later group PETH windows by arbitrary word/sequence
    # labels, not just the reward split used for the overview figure.
    event_idx = get_event_indices(trial_table, align_event)
    has_peth_window = (event_idx - pre_samples >= 0) & (event_idx + post_samples < len(dff))
    peth_trial_table = trial_table[has_peth_window].reset_index(drop=True)

    all_dff_windows = extract_peth(dff, event_idx[has_peth_window], pre_samples, post_samples)
    all_zscore_windows, z_stats = compute_event_aligned_zscore(
        all_dff_windows, peth_time, PETH_BASELINE_PRE_EVENT_S, PETH_BASELINE_POST_EVENT_S,
        return_baseline_stats=True,
    )

    is_rewarded = peth_trial_table["was_rewarded"].to_numpy()
    rewarded_windows, unrewarded_windows = all_dff_windows[is_rewarded], all_dff_windows[~is_rewarded]
    rewarded_z, unrewarded_z = all_zscore_windows[is_rewarded], all_zscore_windows[~is_rewarded]

    print(f"PETH: {rewarded_windows.shape[0]} rewarded trials, "
          f"{unrewarded_windows.shape[0]} unrewarded trials")
    print(f"Trial-aligned Z-score baseline window [{PETH_BASELINE_PRE_EVENT_S}, "
          f"{PETH_BASELINE_POST_EVENT_S}] s: "
          f"{z_stats['n_degenerate_trials']} of {len(peth_trial_table)} trials hit the "
          f"near-zero-sigma floor")

    # --- 6b. word/sequence outcome summary (behavioral + photometry) -------------
    for group_col in SUMMARY_GROUP_COLUMNS:
        if group_col not in trial_table.columns:
            # e.g. "Behavioral_State" when compute_bandit_state=False (qc.session_qc
            # skips the Q-learning fit for speed and doesn't need this column).
            continue
        summary = evaluate_word_outcomes(
            trial_table, group_col,
            zscore_windows=all_zscore_windows, peth_trial_table=peth_trial_table, peth_time=peth_time,
            decision_window_s=DECISION_WINDOW_S, reward_window_s=REWARD_WINDOW_S,
        )
        print(f"\nTop {SUMMARY_TOP_N} '{group_col}' patterns by trial count:")
        print(summary.head(SUMMARY_TOP_N).to_string())

    # --- 7. plot -------------------------------------------------------------------
    fig = plot_session_overview(
        time_axis, dff, peth_time, rewarded_windows, unrewarded_windows,
        channel_label=channels.label, title_prefix=f"{mouse} {date}",
        rewarded_zscore_windows=rewarded_z,
        unrewarded_zscore_windows=unrewarded_z,
        baseline_window_s=(PETH_BASELINE_PRE_EVENT_S, PETH_BASELINE_POST_EVENT_S),
        align_event=align_event,
        all_zscore_windows=all_zscore_windows,
        word_l3_labels=peth_trial_table["word_l3"],
        word_l3_generic_labels=peth_trial_table["word_l3_generic"],
    )

    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_FIGURE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    out_stem = output_dir / f"{mouse}_{date}_{align_event}_session_overview"
    figure_path_png = out_stem.with_suffix(".png")
    figure_path_svg = out_stem.with_suffix(".svg")
    fig.savefig(figure_path_png, dpi=150)
    fig.savefig(figure_path_svg)
    print(f"Saved figure to {figure_path_png} and {figure_path_svg}")

    return dict(
        trial_table=trial_table,
        envelope=envelope,
        dff=dff,
        zscore=zscore,
        baseline=baseline,
        align_info=align_info,
        figure_path=figure_path_png,
        figure_path_png=figure_path_png,
        figure_path_svg=figure_path_svg,
        peth_time=peth_time,
        peth_trial_table=peth_trial_table,
        all_dff_windows=all_dff_windows,
        all_zscore_windows=all_zscore_windows,
        rewarded_dff_windows=rewarded_windows,
        unrewarded_dff_windows=unrewarded_windows,
        rewarded_zscore_windows=rewarded_z,
        unrewarded_zscore_windows=unrewarded_z,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path, help="Session directory (contains PHOTO/, pokeHistory*.mat, stats*.mat)")
    parser.add_argument("--max-segments", type=int, default=None, help="Limit to the first N raw segments (for a quick test run)")
    parser.add_argument("--hemisphere", choices=list(HEMISPHERE_CHANNELS), default=DEFAULT_HEMISPHERE)
    parser.add_argument("--output-dir", type=Path, default=None,
                         help=f"Where to save figures (.png + .svg) (default: {DEFAULT_FIGURE_DIR})")
    parser.add_argument("--no-bandit-state", action="store_true",
                         help="Skip the sticky Q-learning fit / behavioral-state classification step")
    parser.add_argument("--align-to", choices=list(ALIGN_EVENT_COLUMNS), default=DEFAULT_ALIGN_EVENT,
                         help="Which trial event the PETH/z-score windows are aligned to (default: %(default)s)")
    args = parser.parse_args()

    run_session(
        args.session_dir,
        max_segments=args.max_segments,
        hemisphere=args.hemisphere,
        output_dir=args.output_dir,
        compute_bandit_state=not args.no_bandit_state,
        align_event=args.align_to,
    )


if __name__ == "__main__":
    main()
