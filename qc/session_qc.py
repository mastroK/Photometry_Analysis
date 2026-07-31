"""
Per-session QC metrics: objective pass/fail checks on sync quality, behavioral
engagement, and photometry signal quality, computed from pipeline.run_session's
own outputs so every check uses exactly the same trial_table/zscore as the rest
of the pipeline.

Usage (see run_cohort_qc.py for the cohort-level CLI):
    from qc.session_qc import evaluate_session_qc, filter_sessions_by_qc
    row = evaluate_session_qc(session_dir)
"""

import numpy as np
import pandas as pd

from config.params import DEFAULT_HEMISPHERE
from io_utils.raw_loader import parse_session_id
from pipeline import run_session

# QC thresholds -- deliberately stricter than config.params.XCORR_ACCEPT_THRESHOLD
# (0.5), which is a hard floor enforced inside behavior.sync.align_behavior_to_photometry
# itself (raises RuntimeError below it). A session that reaches evaluate_session_qc
# successfully has already cleared that floor; XCORR_QC_THRESHOLD is an additional,
# stricter QC bar on top of it.
XCORR_QC_THRESHOLD = 0.60
MIN_TRIALS_QC = 150
P_RIGHT_LO = 0.05
P_RIGHT_HI = 0.95
ZSCORE_STD_MIN = 0.5
MAX_MISSING_FRAC = 0.05

_METRIC_COLUMNS = [
    "mouse", "date", "session_dir",
    "xcorr_peak", "sync_pass",
    "n_trials", "trials_pass",
    "p_right", "balance_pass",
    "zscore_std", "signal_range_pass",
    "missing_frac", "missing_pass",
    "QC_PASS", "error",
]


def evaluate_session_qc(session_dir, hemisphere=DEFAULT_HEMISPHERE, max_segments=None):
    """Run one session through pipeline.run_session() and compute QC metrics
    from its outputs. Never raises -- a session that fails to load/align
    returns a row with every metric NaN and QC_PASS=False rather than
    aborting a cohort-level QC run.
    """
    mouse, date = parse_session_id(session_dir)

    try:
        result = run_session(session_dir, hemisphere=hemisphere, max_segments=max_segments,
                              compute_bandit_state=False)
    except Exception as exc:
        print(f"WARNING: QC failed for {session_dir} ({mouse} {date}): {exc}")
        row = {col: np.nan for col in _METRIC_COLUMNS}
        row.update(mouse=mouse, date=date, session_dir=str(session_dir), error=str(exc), QC_PASS=False)
        return row

    trial_table = result["trial_table"]
    zscore = np.asarray(result["zscore"], dtype=float)

    xcorr_peak = result["align_info"]["xcorr_peak"]
    sync_pass = xcorr_peak >= XCORR_QC_THRESHOLD

    n_trials = len(trial_table)
    trials_pass = n_trials >= MIN_TRIALS_QC

    p_right = trial_table["chose_right"].mean()
    balance_pass = P_RIGHT_LO <= p_right <= P_RIGHT_HI

    finite = np.isfinite(zscore)
    zscore_std = np.nanstd(zscore[finite]) if finite.any() else np.nan
    signal_range_pass = bool(zscore_std > ZSCORE_STD_MIN) if np.isfinite(zscore_std) else False

    missing_frac = 1.0 - (finite.sum() / len(zscore)) if len(zscore) else 1.0
    missing_pass = missing_frac < MAX_MISSING_FRAC

    qc_pass = bool(sync_pass and trials_pass and balance_pass and signal_range_pass and missing_pass)

    return dict(
        mouse=mouse, date=date, session_dir=str(session_dir),
        xcorr_peak=xcorr_peak, sync_pass=bool(sync_pass),
        n_trials=n_trials, trials_pass=bool(trials_pass),
        p_right=p_right, balance_pass=bool(balance_pass),
        zscore_std=zscore_std, signal_range_pass=signal_range_pass,
        missing_frac=missing_frac, missing_pass=bool(missing_pass),
        QC_PASS=qc_pass, error=None,
    )


def load_qc_report(path):
    """Load a cohort QC report CSV, indexed by (mouse, date) for lookup."""
    df = pd.read_csv(path, dtype={"date": str})
    return df.set_index(["mouse", "date"])


def filter_sessions_by_qc(session_dirs, qc_report_path):
    """Filter a list of session directories against a cohort QC report,
    keeping a session iff its manual_include column (falling back to
    QC_PASS if manual_include is missing/NaN) is truthy. Sessions not found
    in the report are kept, with a warning -- matching the "warn and
    continue" convention already used for missing cohort metadata in
    batch_processor.run_batch_sessions, rather than silently/hard-excluding
    sessions nobody has QC'd yet.
    """
    session_dirs = list(session_dirs)
    report = load_qc_report(qc_report_path)

    kept = []
    for session_dir in session_dirs:
        mouse, date = parse_session_id(session_dir)
        if (mouse, date) not in report.index:
            print(f"WARNING: {mouse} {date} not found in QC report {qc_report_path} -- including by default")
            kept.append(session_dir)
            continue

        row = report.loc[(mouse, date)]
        include = row.get("manual_include")
        if include is None or (isinstance(include, float) and np.isnan(include)):
            include = row.get("QC_PASS")
        if bool(include):
            kept.append(session_dir)

    print(f"QC filter: kept {len(kept)}/{len(session_dirs)} session(s) "
          f"({len(session_dirs) - len(kept)} dropped)")
    return kept
