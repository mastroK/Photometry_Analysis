"""
run_model_series_comparison.py's model-series comparison (Bernardo's ladder:
Model 1 / 2b word / 3c / 3d etc.), applied to SM's red_l channel instead of
FP2_none. Reuses every piece of that module UNCHANGED (MODEL_SPECS,
LADDER_ORDER, HEAD_TO_HEAD_COMPARISONS, _pool_sessions, run_comparison,
plot_pooled) -- only the session-loading step differs, because SM red_l
needs three things FP2_none's loader (_load_all_sessions) doesn't:

1. hemisphere="red_l" (fixed, not a per-session lookup -- SM red_l is a
   single physical channel, not a per-mouse hemisphere choice).
2. force_nominal_carrier_freq=True -- SM's red_l demodulation was found to
   silently lock onto green's ~167Hz carrier instead of red_l's own 223Hz
   (estimate_carrier_freq's unconstrained global argmax; green's real
   signal is the larger spectral peak). Confirmed and fixed in
   rpe_analysis_prep_SM.py's extract_hemisphere; run_session's default path
   needs the same override here since this script calls run_session directly.
3. Forced-choice (100-0) trial exclusion -- SM has early-training blocks at
   a degenerate 100/0 reward probability (FP1/FP2 never do). Excluded here
   the same way rpe_analysis_prep_SM.py does: AFTER run_session returns
   (so the Q-learning fit/lag features still see the complete chronological
   sequence), via trial_table["is_forced_block"], applied identically to
   the side_in and side_out PETH tables (and their row-aligned zscore
   windows).

Default mouse set is the 5 regular_mice already validated as carrying real
red_l signal (rpe_analysis_sm_red_l_excl_low_signal/ -- SM2L/SM3MN/SM3MR
excluded, near-zero FIR R^2 in the corrected-frequency analysis). jrgeco_only
mice are a categorically different population (no GCaMP at all) and are
handled separately, not mixed into this per-mouse comparison, if wanted.

include_fir is not wired up here yet -- this script only builds the side_in/
side_out PETH-window data _load_all_sessions builds; the continuous-trace
(continuous_trial_table/continuous_zscore) side needed for FIR isn't
constructed. Fine for encoding-GLM-only runs (this module's main use so
far); extending to FIR would need the same is_forced_block masking applied
to the continuous trace's task mask, not just the PETH tables.

Usage:
    import run_model_series_comparison_sm_red_l as rmsc_red_l
    rmsc_red_l.main_red_l(
        model_names=['1_main_effects', '2b_word_l2', '3c_reward_qchosen_qdiff', '3d_reward_qchosen_rpeabs_qdiff'],
        out_dir='outputs_fixed/model_series_comparison_sm_red_l/results',
        fig_dir='figures_fixed_model_series_sm_red_l',
    )
"""

from pathlib import Path

import numpy as np
import pandas as pd

from behavior.word_encoding import add_reward_seq_2
from config.params import FINAL_SAMPLE_FREQ_HZ, PETH_POST_SEC, PETH_PRE_SEC
from io_utils.raw_loader import parse_session_id
from pipeline import extract_event_peth, run_session
from run_manifest import write_run_manifest
from run_model_series_comparison import _pool_sessions, plot_pooled, run_comparison

CHANNEL_REPORT = Path("outputs_fixed/sm_corrected_channel_report.csv")
DEFAULT_MICE = ("SM1L", "SM1N", "SM2N", "SM2R", "SM3FR")
DEFAULT_OUT_DIR = Path("outputs_fixed/model_series_comparison_sm_red_l/results")
DEFAULT_FIG_DIR = Path("figures_fixed_model_series_sm_red_l")


def get_sm_red_l_session_dirs(mice=DEFAULT_MICE, hemisphere="red_l"):
    report = pd.read_csv(CHANNEL_REPORT, dtype={"date": str})
    valid = report[(report[f"{hemisphere}_carrier_valid"] == True) & (report["mouse"].isin(mice))]  # noqa: E712
    return sorted(Path(p) for p in valid["session_dir"])


def _load_sm_red_l_sessions(session_dirs, hemisphere="red_l", truncate_at_side_out=False,
                             side_out_margin_s=0.0):
    """SM red_l analog of run_model_series_comparison._load_all_sessions --
    see module docstring for the three differences (hemisphere, forced
    carrier freq, forced-choice exclusion). Doesn't build the continuous-
    trace fields _load_all_sessions does (continuous_trial_table/
    continuous_zscore) -- only needed for FIR, not wired up here.

    truncate_at_side_out/side_out_margin_s : forwarded to run_session's
    side_in-aligned pass only (see pipeline.extract_event_peth's docstring)
    -- opt-in, default False. The side_out-aligned pass below is unaffected
    (and disallowed by run_session's own check, since dwell-time truncation
    is only meaningful relative to side_in).
    """
    pre_samples = int(round(PETH_PRE_SEC * FINAL_SAMPLE_FREQ_HZ))
    post_samples = int(round(PETH_POST_SEC * FINAL_SAMPLE_FREQ_HZ))

    sessions = []
    n_failed = 0
    n_forced_in_total = n_forced_out_total = n_trials_in_total = n_trials_out_total = 0
    for session_dir in session_dirs:
        session_dir = Path(session_dir)
        mouse, date = parse_session_id(session_dir)
        try:
            result = run_session(session_dir, hemisphere=hemisphere, align_event="side_in",
                                  force_nominal_carrier_freq=True,
                                  truncate_at_side_out=truncate_at_side_out,
                                  side_out_margin_s=side_out_margin_s)
        except Exception as exc:
            print(f"WARNING: skipping session {session_dir} ({mouse} {date}): {exc}")
            n_failed += 1
            continue
        try:
            _, peth_trial_table_out, _, zscore_windows_out, _ = extract_event_peth(
                result["full_trial_table"], result["dff"], "side_out",
                pre_samples, post_samples, result["peth_time"],
            )
        except Exception as exc:
            print(f"WARNING: side_out PETH extraction failed for {session_dir} ({mouse} {date}): {exc}")
            n_failed += 1
            continue

        trial_table_in = add_reward_seq_2(result["peth_trial_table"])
        zscore_windows_in = result["all_zscore_windows"]
        trial_table_out = add_reward_seq_2(peth_trial_table_out)

        n_forced_in = int(trial_table_in["is_forced_block"].sum())
        n_forced_out = int(trial_table_out["is_forced_block"].sum())
        n_forced_in_total += n_forced_in
        n_forced_out_total += n_forced_out
        n_trials_in_total += len(trial_table_in)
        n_trials_out_total += len(trial_table_out)

        keep_in = ~trial_table_in["is_forced_block"].to_numpy()
        trial_table_in = trial_table_in.loc[keep_in].reset_index(drop=True)
        zscore_windows_in = zscore_windows_in[keep_in]

        keep_out = ~trial_table_out["is_forced_block"].to_numpy()
        trial_table_out = trial_table_out.loc[keep_out].reset_index(drop=True)
        zscore_windows_out = zscore_windows_out[keep_out]

        if len(trial_table_in) == 0:
            print(f"  WARNING: skipping {mouse} {date} -- no real (non-forced) side_in trials remain")
            n_failed += 1
            continue

        sessions.append(dict(
            mouse=mouse, date=date,
            trial_table_in=trial_table_in, zscore_windows_in=zscore_windows_in, peth_time_in=result["peth_time"],
            trial_table_out=trial_table_out, zscore_windows_out=zscore_windows_out, peth_time_out=result["peth_time"],
        ))

    print(f"Loaded {len(sessions)} session(s) via a single run_session() pass each (skipped {n_failed})")
    print(f"Dropped {n_forced_in_total}/{n_trials_in_total} forced-choice side_in trials, "
          f"{n_forced_out_total}/{n_trials_out_total} forced-choice side_out trials")
    return sessions


def _save_pooled_arrays(out_dir, peth_time_in, zscore_in, trial_table_in,
                         peth_time_out, zscore_out, trial_table_out):
    """Cache the pooled (post-session-loop) arrays plot_pooled/run_comparison
    consume, so a future fit-only or plot-only fix (e.g. a new
    min_resid_dof/min_retained_frac guard) never again requires repeating the
    ~85 min raw-session reload that building these arrays costs -- exactly
    the rework this project just had to pay for twice in a row. Purely
    additive: doesn't change what main_red_l returns or computes.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "pooled_zscore_windows.npz",
              zscore_in=zscore_in, zscore_out=zscore_out,
              peth_time_in=peth_time_in, peth_time_out=peth_time_out)
    trial_table_in.to_parquet(out_dir / "pooled_trial_table_in.parquet")
    trial_table_out.to_parquet(out_dir / "pooled_trial_table_out.parquet")
    print(f"Cached pooled arrays to {out_dir} (pooled_zscore_windows.npz, "
          "pooled_trial_table_in/out.parquet)")


def main_red_l(mice=DEFAULT_MICE, hemisphere="red_l", model_names=None, out_dir=DEFAULT_OUT_DIR, fig_dir=DEFAULT_FIG_DIR,
                truncate_at_side_out=False, side_out_margin_s=0.0, min_retained_frac=None):
    """include_fir is not exposed -- always False here (see module docstring).
    model_names / out_dir / fig_dir match run_model_series_comparison.main's
    contract exactly. hemisphere defaults to "red_l" (this module's original
    purpose) but works identically for "green_l"/"green_r" -- force_nominal_
    carrier_freq=True is harmless for green channels (the free search already
    finds the right peak there) and the forced-choice exclusion applies to
    any SM hemisphere.

    truncate_at_side_out/side_out_margin_s : opt-in, default False -- NaNs
    out each trial's own post-side_out samples in the side_in-aligned PETH
    windows before fitting (see pipeline.extract_event_peth). Pass a distinct
    out_dir/fig_dir when using True so it doesn't overwrite an existing
    non-truncated run's files.

    min_retained_frac : forwarded to plot_pooled -> fit_time_resolved_glm.
    None (default) leaves this at truncate_at_side_out's own default: 0.5
    when truncate_at_side_out=True (a truncated pooled fit's later
    timepoints regress on a shrinking, self-selected long-dwelling
    subsample -- see fit_time_resolved_glm's docstring -- 0.5 stops the
    plotted/reported curve there instead of silently continuing), None
    (disabled) when truncate_at_side_out=False, since n_trials is constant
    across timepoints there and the guard would never fire anyway. Pass
    explicitly to override either default.
    """
    session_dirs = get_sm_red_l_session_dirs(mice, hemisphere=hemisphere)
    print(f"SM {hemisphere}, mice={sorted(mice)}: {len(session_dirs)} sessions")

    print("\nLoading sessions (one pipeline.run_session() pass each, reused for side_in, side_out)...")
    sessions = _load_sm_red_l_sessions(session_dirs, hemisphere=hemisphere,
                                        truncate_at_side_out=truncate_at_side_out,
                                        side_out_margin_s=side_out_margin_s)

    peth_time_in, zscore_in, trial_table_in = _pool_sessions(sessions, "in")
    peth_time_out, zscore_out, trial_table_out = _pool_sessions(sessions, "out")
    fir_sessions_by_mouse = {}  # FIR not wired up for SM red_l yet -- see module docstring

    _save_pooled_arrays(out_dir, peth_time_in, zscore_in, trial_table_in,
                        peth_time_out, zscore_out, trial_table_out)

    if min_retained_frac is None and truncate_at_side_out:
        min_retained_frac = 0.5

    write_run_manifest(
        out_dir,
        params=dict(
            mice=sorted(mice), hemisphere=hemisphere, model_names=model_names,
            truncate_at_side_out=truncate_at_side_out, side_out_margin_s=side_out_margin_s,
            min_retained_frac=min_retained_frac, n_sessions=len(session_dirs),
        ),
        script="run_model_series_comparison_sm_red_l.main_red_l",
    )

    encoding_df, fir_df, stats = run_comparison(
        trial_table_in, zscore_in, peth_time_in,
        trial_table_out, zscore_out, peth_time_out,
        fir_sessions_by_mouse, out_dir=Path(out_dir), include_fir=False, model_names=model_names,
    )

    plot_pooled(
        trial_table_in, zscore_in, peth_time_in,
        trial_table_out, zscore_out, peth_time_out,
        fir_sessions_by_mouse, fig_dir=Path(fig_dir), include_fir=False, model_names=model_names,
        min_retained_frac=min_retained_frac,
    )

    return encoding_df, fir_df, stats


if __name__ == "__main__":
    main_red_l()
