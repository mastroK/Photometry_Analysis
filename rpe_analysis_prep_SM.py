"""
Pooled RPE/GLM/FIR data preparation for the SM (PV-dualphotometry) cohort --
adapted from rpe_analysis_prep.py for two differences from FP1/FP2/WCL:

1. Bilateral hemispheres: SM sessions can validly contribute BOTH green_r and
   green_l (see run_sm_corrected_batch.py's corrected channel-validity
   report), unlike FP1/FP2's one-hemisphere-per-mouse lookup. So this takes explicit
   (session_dir, hemisphere) pairs rather than resolving hemisphere from a
   per-mouse/per-session lookup, and stamps `hemisphere` as a column on the
   pooled trial table (alongside mouse/date) for the downstream RPE_signed *
   C(hemisphere) robustness check in rpe_analysis_stats_SM.py.
2. Forced-choice (100-0) trial exclusion: behavior.trial_table.build_trial_table
   already flags each trial's `is_forced_block` (a real 80/20 bandit trial
   has neither reward prob at 0 or 1). That flag must NOT be used to drop
   rows before run_session computes its sequential features (word/lag
   sequences, Q-learning fit) -- those need the complete chronological trial
   sequence, forced blocks included, or every real trial's N-back/Q-value
   context downstream of a forced block would be silently wrong. So exclusion
   happens HERE, right after run_session returns, before PETH/FIR assembly --
   never inside trial_table construction itself.

Saves, under outputs_fixed/rpe_analysis_sm/ (mirrors rpe_analysis_prep.py's
FP1/FP2 output shape exactly, plus a `hemisphere` column):
  - peth_windows.npz          -- side_in-aligned z-score PETH windows + peth_time
  - pooled_trial_table.parquet -- per-trial covariates, row-aligned with peth_windows.npz
  - fir_pooled.npz            -- FIR design-matrix pieces (y, Phi, groups, mouse, hemisphere)
  - fir_column_names.pkl      -- FIR column_names + n_lags, for reshape_kernels

Usage:
    python rpe_analysis_prep_SM.py outputs_fixed/sm_corrected_channel_report.csv
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from alignment.windowing import compute_event_aligned_zscore, compute_per_trial_event_metrics, extract_peth, get_event_indices
from behavior.sync import align_behavior_to_photometry
from behavior.switch_dynamics import add_model_belief, add_switch_dynamics, add_value_decomposition_features
from behavior.trial_table import build_trial_table
from behavior.word_encoding import add_lag_features, add_word_labels
from config.params import (
    ALIGN_EVENT_COLUMNS,
    DECISION_WINDOW_S,
    DEFAULT_HEMISPHERE,
    FINAL_SAMPLE_FREQ_HZ,
    FINAL_TIME_STEP_SEC,
    HEMISPHERE_CHANNELS,
    LAG_N,
    PETH_BASELINE_POST_EVENT_S,
    PETH_BASELINE_PRE_EVENT_S,
    PETH_POST_SEC,
    PETH_PRE_SEC,
    WORD_ENCODING_LEVELS,
)
from external.bandit_state_adapter import add_bandit_state_features
from io_utils.raw_loader import discover_behavior_files, load_behavior_raw, load_raw_photometry, parse_session_id
from models.fir_glm import (
    DEFAULT_GROUP_COLUMN,
    DEFAULT_LAG_SECONDS,
    build_event_impulses,
    build_shifted_design_matrix,
    build_task_mask_and_groups,
)
from preprocessing.demodulate import compute_dff_and_zscore, demodulate_envelope, estimate_carrier_freq

OUT_DIR = Path("outputs_fixed/rpe_analysis_sm")
HEMISPHERES = ("green_r", "green_l")
DEFAULT_REPORT_PATH = "outputs_fixed/sm_corrected_channel_report.csv"

# Confirmed jRGECO-only mice (no GCaMP) -- see run_sm_corrected_batch.py's
# identically-named constant for the full rationale. Their green channels can
# still show carrier_valid=True (a real carrier is present -- see the
# crosstalk investigation) but that carrier is confirmed NOT to be genuine
# GCaMP content (visually confirmed to look very different from real GCaMP
# sessions), so they must be excluded from this green-channel/pyramidal
# encoding pool specifically, even though carrier_valid alone can't tell the
# two apart.
JRGECO_ONLY_MICE = {"SM1B", "SM2B"}


def session_hemisphere_pairs_from_report(report_path, hemispheres=HEMISPHERES, exclude_mice=JRGECO_ONLY_MICE):
    """Read run_sm_corrected_batch.py's corrected channel-validity report and
    return every (session_dir, hemisphere) pair that passed its targeted,
    carrier-based validity gate ({hemisphere}_carrier_valid -- a stable,
    physical-channel-quality signal, NOT the noisy per-session
    reward_significant column; see qc.channel_selection.
    evaluate_channel_true_signal's docstring for why). One session can
    contribute 0, 1, or 2 pairs. Confirmed jRGECO-only mice are excluded here
    since their green channels' real carrier is confirmed not to reflect
    genuine GCaMP content.
    """
    report = pd.read_csv(report_path)
    pairs = []
    for _, row in report.iterrows():
        if row["mouse"] in exclude_mice:
            continue
        for hemisphere in hemispheres:
            if row.get(f"{hemisphere}_carrier_valid"):
                pairs.append((Path(row["session_dir"]), hemisphere))
    return pairs


def load_session_shared(session_dir, max_segments=None):
    """Load raw + build the full trial_table ONCE per session -- the parts of
    pipeline.run_session that are hemisphere-INDEPENDENT: raw photometry
    loading, behavior parsing, word/lag features, the sticky Q-learning fit
    (+ its value-decomposition/belief/switch-dynamics derivatives), and
    behavior<->photometry alignment. Envelope length (and therefore every
    aligned photometry index in trial_table) is guaranteed identical across
    every hemisphere channel for a given session -- all 14 raw channels
    share the same array length, and demodulate_envelope's frame count is a
    function of that length plus fixed window/hop constants only, never the
    channel itself -- so computing this once and reusing it for both
    green_r and green_l (SM's bilateral rig routinely has both valid) is
    exact, not an approximation. Added because rpe_analysis_prep_SM.py's
    per-(session,hemisphere)-pair loop was reloading raw data from the
    network share TWICE for every session with both hemispheres valid
    (111/131 of them) -- a ~46% reduction in raw loads once deduplicated.

    Returns dict(raw, trial_table, poke_history, mouse, date, n_final_samples).
    """
    session_dir = Path(session_dir)
    mouse, date = parse_session_id(session_dir)
    raw = load_raw_photometry(session_dir / "PHOTO", max_segments=max_segments)

    poke_history_file, stats_file = discover_behavior_files(session_dir)
    poke_history, stats = load_behavior_raw(poke_history_file, stats_file)
    trial_table = build_trial_table(poke_history, stats)
    trial_table = add_word_labels(trial_table, levels=WORD_ENCODING_LEVELS)
    trial_table = add_lag_features(trial_table, n_lags=LAG_N)
    trial_table = add_bandit_state_features(trial_table, session_id=f"{mouse}_{date}")
    bandit_fit_params = trial_table.attrs.get("bandit_fit_params")  # capture before any further .copy()
    trial_table = add_value_decomposition_features(trial_table)
    trial_table = add_model_belief(trial_table, bandit_fit_params)
    trial_table = add_switch_dynamics(trial_table)
    print(f"  Parsed {len(trial_table)} trials ({trial_table['was_rewarded'].sum()} rewarded)")

    # Reference-only demodulation purely to get n_final_samples for
    # alignment -- identical across every hemisphere (see docstring), so
    # DEFAULT_HEMISPHERE's channel here is not tied to which hemisphere(s)
    # will actually be extracted below.
    ref_channel = HEMISPHERE_CHANNELS[DEFAULT_HEMISPHERE].signal_channel
    ref_freq, _ = estimate_carrier_freq(raw[ref_channel])
    ref_envelope, _ = demodulate_envelope(raw[ref_channel], ref_freq)
    n_final_samples = len(ref_envelope)

    trial_table, align_info = align_behavior_to_photometry(
        raw, trial_table, poke_history, n_final_samples=n_final_samples
    )
    return dict(raw=raw, trial_table=trial_table, mouse=mouse, date=date, n_final_samples=n_final_samples)


def extract_hemisphere(shared, hemisphere, align_event="side_in"):
    """The hemisphere-DEPENDENT remainder of pipeline.run_session (channel
    demodulation, dF/F/z-score, per-trial event metrics, PETH extraction) --
    everything load_session_shared doesn't already cover. No figure is
    saved (unlike run_session) -- this bulk-prep path pools hundreds of
    (session, hemisphere) pairs and nobody reviews a session-overview figure
    per pair, so skipping the render+disk write is a real, safe time saving,
    not a silent loss of anything used downstream.

    Returns a dict shaped like pipeline.run_session's return value, for the
    subset of keys build_pooled_dataset actually consumes (trial_table,
    peth_trial_table, all_zscore_windows, peth_time, zscore).
    """
    raw, mouse, date = shared["raw"], shared["mouse"], shared["date"]
    trial_table = shared["trial_table"].copy()
    channels = HEMISPHERE_CHANNELS[hemisphere]

    # Anchor demodulation at THIS channel's own nominal carrier frequency
    # rather than estimate_carrier_freq's unconstrained global FFT argmax --
    # the same crosstalk bug fixed in qc.channel_selection.find_true_carrier_freq
    # (used only by the channel-validity survey) was still live here: on SM's
    # red_l channel, green's much stronger real signal is the global spectral
    # maximum, so the free search silently locked onto ~166.7 Hz (green's
    # carrier) instead of red_l's real ~223 Hz every time, for every session
    # -- confirmed via generate_red_l_overview_figures.py's printed measured
    # frequencies (166.667 Hz, 107/107 sessions). demodulate_envelope still
    # does its own narrow-band refinement around whatever frequency it's
    # given, so anchoring here at the known-correct nominal value is strictly
    # safer than the free search and reproduces identical results for
    # green_r/green_l (where the free search already happened to find the
    # right peak, since green's own real signal dominates that channel).
    measured_freq = channels.nominal_carrier_freq_hz
    envelope, _ = demodulate_envelope(raw[channels.signal_channel], measured_freq)
    if len(envelope) != shared["n_final_samples"]:
        raise ValueError(
            f"{mouse} {date} ({hemisphere}): envelope length {len(envelope)} != "
            f"shared n_final_samples {shared['n_final_samples']} -- the hemisphere-"
            "independence assumption load_session_shared relies on doesn't hold here"
        )
    dff, zscore, _ = compute_dff_and_zscore(raw[channels.signal_channel], measured_freq, envelope)

    pre_samples = int(round(PETH_PRE_SEC * FINAL_SAMPLE_FREQ_HZ))
    post_samples = int(round(PETH_POST_SEC * FINAL_SAMPLE_FREQ_HZ))
    peth_time = np.arange(-pre_samples, post_samples + 1) * FINAL_TIME_STEP_SEC

    for evt in ALIGN_EVENT_COLUMNS:
        peak, auc = compute_per_trial_event_metrics(
            dff, trial_table, evt, pre_samples, post_samples, peth_time,
            PETH_BASELINE_PRE_EVENT_S, PETH_BASELINE_POST_EVENT_S, DECISION_WINDOW_S,
        )
        trial_table[f"peak_z_{evt}"] = peak
        trial_table[f"auc_{evt}"] = auc

    align_col = ALIGN_EVENT_COLUMNS[align_event]
    trial_table = trial_table[trial_table[align_col] >= 0].reset_index(drop=True)

    event_idx = get_event_indices(trial_table, align_event)
    has_peth_window = (event_idx - pre_samples >= 0) & (event_idx + post_samples < len(dff))
    peth_trial_table = trial_table[has_peth_window].reset_index(drop=True)

    all_dff_windows = extract_peth(dff, event_idx[has_peth_window], pre_samples, post_samples)
    all_zscore_windows = compute_event_aligned_zscore(
        all_dff_windows, peth_time, PETH_BASELINE_PRE_EVENT_S, PETH_BASELINE_POST_EVENT_S,
    )

    return dict(
        trial_table=trial_table, peth_trial_table=peth_trial_table,
        all_zscore_windows=all_zscore_windows, peth_time=peth_time, zscore=zscore,
    )


def build_pooled_dataset(session_hemisphere_pairs, group_col=DEFAULT_GROUP_COLUMN,
                          n_lags_seconds=DEFAULT_LAG_SECONDS, out_dir=OUT_DIR):
    # Group by session_dir so each session's raw+behavior load (see
    # load_session_shared) happens once, however many hemispheres it
    # contributes -- was previously one full pipeline.run_session() call
    # (a full raw reload) per (session, hemisphere) pair.
    hemispheres_by_session = {}
    for session_dir, hemisphere in session_hemisphere_pairs:
        hemispheres_by_session.setdefault(Path(session_dir), []).append(hemisphere)

    loaded = []
    n_forced_total, n_trials_total = 0, 0
    for session_dir, hemispheres in hemispheres_by_session.items():
        mouse, date = session_dir.name, session_dir.parent.name
        print(f"--- {mouse} {date} (hemispheres={hemispheres}) ---")
        try:
            shared = load_session_shared(session_dir)
        except Exception as exc:
            print(f"WARNING: skipping {session_dir} ({mouse} {date}, all hemispheres): {exc}")
            continue

        for hemisphere in hemispheres:
            try:
                result = extract_hemisphere(shared, hemisphere, align_event="side_in")
            except Exception as exc:
                print(f"WARNING: skipping {session_dir} ({mouse} {date}, {hemisphere}): {exc}")
                continue

            # Forced-choice (100-0) exclusion -- see module docstring for why
            # this happens here, not inside trial_table construction.
            trial_table = result["trial_table"]
            peth_trial_table = result["peth_trial_table"]
            all_zscore_windows = result["all_zscore_windows"]

            n_forced = int(trial_table["is_forced_block"].sum())
            n_forced_total += n_forced
            n_trials_total += len(trial_table)
            if n_forced:
                print(f"  [{hemisphere}] Dropping {n_forced}/{len(trial_table)} forced-choice (100-0) trials")

            trial_table = trial_table[~trial_table["is_forced_block"]].reset_index(drop=True)
            peth_keep = ~peth_trial_table["is_forced_block"].to_numpy()
            peth_trial_table = peth_trial_table[peth_keep].reset_index(drop=True)
            all_zscore_windows = all_zscore_windows[peth_keep]

            if len(peth_trial_table) == 0:
                print(f"  WARNING: skipping {mouse} {date} ({hemisphere}) -- no real (non-forced) trials remain")
                continue

            result = dict(result)  # shallow copy, don't mutate extract_hemisphere's own dict
            result["trial_table"] = trial_table
            result["peth_trial_table"] = peth_trial_table
            result["all_zscore_windows"] = all_zscore_windows
            loaded.append((mouse, date, hemisphere, result))

    print(f"\nDropped {n_forced_total}/{n_trials_total} forced-choice trials across all loaded sessions")

    if not loaded:
        raise RuntimeError("No sessions successfully processed")

    all_groups = set()
    for _, _, _, result in loaded:
        all_groups.update(result["trial_table"][group_col].dropna().unique())
    group_values = sorted(all_groups)
    n_lags = int(round(n_lags_seconds * FINAL_SAMPLE_FREQ_HZ))

    peth_time = None
    window_frames, table_frames = [], []
    y_parts, phi_parts, group_parts, mouse_parts, hemisphere_parts = [], [], [], [], []
    column_names = None
    group_offset = 0

    for mouse, date, hemisphere, result in loaded:
        if peth_time is None:
            peth_time = result["peth_time"]
        elif not np.array_equal(peth_time, result["peth_time"]):
            raise ValueError(f"{mouse} {date} ({hemisphere}) has a different peth_time grid")

        session_table = result["peth_trial_table"].copy()
        session_table["mouse"] = mouse
        session_table["date"] = date
        session_table["hemisphere"] = hemisphere
        table_frames.append(session_table)
        window_frames.append(result["all_zscore_windows"])

        trial_table = result["trial_table"]
        continuous_signal = np.asarray(result["zscore"], dtype=float)
        n_samples = len(continuous_signal)
        impulses = build_event_impulses(trial_table, n_samples, group_col=group_col, group_values=group_values)
        Phi, cols = build_shifted_design_matrix(impulses, n_lags)
        if column_names is None:
            column_names = cols
        mask, groups = build_task_mask_and_groups(trial_table, n_samples, n_lags)

        y_parts.append(continuous_signal[mask])
        phi_parts.append(Phi[mask])
        group_parts.append(groups[mask] + group_offset)
        mouse_parts.append(np.full(int(mask.sum()), mouse))
        hemisphere_parts.append(np.full(int(mask.sum()), hemisphere))
        group_offset += len(trial_table)

    zscore_windows = np.vstack(window_frames)
    pooled_trial_table = pd.concat(table_frames, ignore_index=True)
    y_pooled = np.concatenate(y_parts)
    Phi_pooled = np.vstack(phi_parts)
    groups_pooled = np.concatenate(group_parts)
    mouse_pooled = np.concatenate(mouse_parts)
    hemisphere_pooled = np.concatenate(hemisphere_parts)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "peth_windows.npz", zscore_windows=zscore_windows, peth_time=peth_time)
    pooled_trial_table.to_parquet(out_dir / "pooled_trial_table.parquet")
    np.savez(
        out_dir / "fir_pooled.npz", y=y_pooled, Phi=Phi_pooled, groups=groups_pooled,
        mouse=mouse_pooled, hemisphere=hemisphere_pooled,
    )
    with open(out_dir / "fir_column_names.pkl", "wb") as f:
        pickle.dump(dict(column_names=column_names, n_lags=n_lags), f)

    n_sessions = pooled_trial_table[["mouse", "date", "hemisphere"]].drop_duplicates().shape[0]
    print(f"\nSaved pooled dataset to {out_dir}:")
    print(f"  PETH: {len(pooled_trial_table)} trials, windows {zscore_windows.shape}, "
          f"{n_sessions} (session, hemisphere) pairs")
    print(f"  FIR: {len(y_pooled)} samples across {group_offset} trials, "
          f"{len(np.unique(mouse_pooled))} mice")
    print(pooled_trial_table.groupby("mouse")["hemisphere"].value_counts())


if __name__ == "__main__":
    import sys
    report_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REPORT_PATH
    out_dir_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT_DIR
    pairs = session_hemisphere_pairs_from_report(report_path)
    print(f"Loaded {len(pairs)} valid (session, hemisphere) pair(s) from {report_path}")
    build_pooled_dataset(pairs, out_dir=out_dir_arg)
