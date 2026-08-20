"""
Pool per-session PETH windows + trial-level covariates across a cohort, for
the time-resolved GLM encoding models in models/glm_encoding.py.

batch_processor.py's exported master DataFrame only carries scalar per-trial
photometry summaries (peak_z_*/auc_*, from
alignment.windowing.compute_per_trial_event_metrics) -- it never persists the
full (n_trials, n_timepoints) z-score window arrays, since the cohort CSV
export doesn't need them. A time-resolved GLM needs exactly those window
arrays, so this module reruns pipeline.run_session() per session directly
rather than reading the exported CSV.
"""

import numpy as np
import pandas as pd

from config.params import DEFAULT_ALIGN_EVENT, DEFAULT_HEMISPHERE
from io_utils.raw_loader import parse_session_id
from pipeline import run_session


def build_pooled_glm_dataset(session_dirs, align_event=DEFAULT_ALIGN_EVENT,
                              hemisphere=DEFAULT_HEMISPHERE, max_segments=None,
                              hemisphere_for_session=None, truncate_at_side_out=False,
                              side_out_margin_s=0.0):
    """Run every session_dir through run_session(align_event=...) and stack
    each session's all_zscore_windows / peth_trial_table into one pooled
    (windows, trial_table) pair, tagged with mouse/date.

    Stacking with a plain vstack/concat is safe here because every session
    shares the identical peth_time grid -- fixed by config.params.PETH_PRE_SEC/
    PETH_POST_SEC/FINAL_SAMPLE_FREQ_HZ, not session-dependent -- which is
    asserted below before pooling rather than assumed silently.

    A session that fails (e.g. too few trials, alignment below the xcorr
    acceptance threshold) is logged and skipped, same soft-fail convention as
    batch_processor.run_batch_sessions.

    hemisphere_for_session : optional callable(session_dir) -> hemisphere_key,
        overriding the single `hemisphere` value per session -- see
        batch_processor.run_batch_sessions's identically-named parameter.
        Hemisphere is a per-session, not per-mouse, property in this cohort
        (config/session_hemisphere_overrides.csv) -- pooling with one fixed
        hemisphere across a session list spanning the mid-cohort channel
        cutover would silently demodulate the wrong channel for half of it.

    truncate_at_side_out/side_out_margin_s : forwarded to run_session/
        pipeline.extract_event_peth, opt-in and default False -- see that
        function's docstring. Only meaningful for align_event="side_in".

    Returns (peth_time, zscore_windows, pooled_trial_table):
      peth_time : (n_samples,) seconds-from-event offsets, shared across the
          whole pool.
      zscore_windows : (n_pooled_trials, n_samples) trial-level event-aligned
          z-score, row i corresponds to pooled_trial_table.iloc[i].
      pooled_trial_table : concatenated peth_trial_table across sessions,
          tagged with `mouse`/`date` columns.
    """
    peth_time = None
    window_frames = []
    table_frames = []
    n_ok, n_failed = 0, 0

    for session_dir in session_dirs:
        mouse, date = parse_session_id(session_dir)
        session_hemisphere = hemisphere_for_session(session_dir) if hemisphere_for_session is not None else hemisphere
        try:
            result = run_session(session_dir, hemisphere=session_hemisphere, max_segments=max_segments,
                                  align_event=align_event, truncate_at_side_out=truncate_at_side_out,
                                  side_out_margin_s=side_out_margin_s)
        except Exception as exc:
            print(f"WARNING: skipping session {session_dir} ({mouse} {date}): {exc}")
            n_failed += 1
            continue

        if peth_time is None:
            peth_time = result["peth_time"]
        elif not np.array_equal(peth_time, result["peth_time"]):
            raise ValueError(
                f"{mouse} {date} has a different peth_time grid than earlier sessions in this "
                "pool -- pooling requires an identical PETH_PRE_SEC/PETH_POST_SEC/"
                "FINAL_SAMPLE_FREQ_HZ config across the whole cohort"
            )

        session_table = result["peth_trial_table"].copy()
        session_table["mouse"] = mouse
        session_table["date"] = date
        table_frames.append(session_table)
        window_frames.append(result["all_zscore_windows"])
        n_ok += 1

    if not window_frames:
        raise RuntimeError(f"No sessions successfully processed out of {list(session_dirs)}")

    zscore_windows = np.vstack(window_frames)
    pooled_trial_table = pd.concat(table_frames, ignore_index=True)
    print(f"Pooled {n_ok} session(s) (skipped {n_failed}), {len(pooled_trial_table)} total trials "
          f"aligned to '{align_event}'")
    return peth_time, zscore_windows, pooled_trial_table
