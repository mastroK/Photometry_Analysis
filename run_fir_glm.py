"""
Continuous time-shifted FIR (deconvolution) GLM: fits the (pooled)
continuous photometry trace across one or more sessions as a linear
combination of time-shifted impulse responses to side_in, split into
discrete channels by a reward-history/type grouping column, plus parametric
side_in modulation by |Q_diff| and Choice.

Why: this deliberately does NOT model center_in/center_out/side_out as
nuisance regressors -- per lab decision, the question here is the side_in
(choice/outcome) response itself across reward types and histories, not
separating it from neighboring events. Fitting every reward-history group's
+/-T-second kernel jointly against the same continuous trace still
deconvolves temporally-overlapping trials (e.g. back-to-back quick
win-stays) from each other. Pooling multiple sessions/mice/dates adds more
trials per history pattern. See models/fir_glm.py's module docstring and
build_pooled_fir_dataset for the full design/pooling mechanics.

Usage:
    python run_fir_glm.py session_dir [session_dir ...] \\
        [--hemisphere green_r|red_l] [--signal dff|zscore] \\
        [--group-col reward_seq_3] [--lag-seconds 1.0] \\
        [--n-splits 10] [--test-size 0.2] [--max-segments N] [--output-dir DIR]
"""

import argparse
from pathlib import Path

from config.params import DEFAULT_HEMISPHERE, HEMISPHERE_CHANNELS
from models.fir_glm import (
    DEFAULT_GROUP_COLUMN,
    DEFAULT_LAG_SECONDS,
    DEFAULT_N_SPLITS,
    DEFAULT_TEST_SIZE,
    build_and_fit_pooled_fir_glm,
)
from pipeline import DEFAULT_FIGURE_DIR
from qc.session_qc import filter_sessions_by_qc
from viz.fir_plots import plot_fir_kernels


def run_fir_glm(session_dirs, hemisphere=DEFAULT_HEMISPHERE, signal="zscore",
                 group_col=DEFAULT_GROUP_COLUMN, lag_seconds=DEFAULT_LAG_SECONDS,
                 n_splits=DEFAULT_N_SPLITS, test_size=DEFAULT_TEST_SIZE,
                 max_segments=None, output_dir=None, qc_report_path=None):
    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_FIGURE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if qc_report_path is not None:
        session_dirs = filter_sessions_by_qc(session_dirs, qc_report_path)

    fit = build_and_fit_pooled_fir_glm(
        session_dirs, hemisphere=hemisphere, signal=signal, group_col=group_col,
        n_lags_seconds=lag_seconds, max_segments=max_segments, n_splits=n_splits, test_size=test_size,
    )
    cv = fit["cv_results"]
    mice = [info["mouse"] for info in fit["session_info"]]
    dates = {info["date"] for info in fit["session_info"]}
    if len(dates) == 1:
        date_str = next(iter(dates))
        title_prefix = f"{'+'.join(mice)} {date_str} ({group_col})"
        out_name = f"pooled_{date_str}_{len(mice)}mice_{group_col}_fir_kernels"
    else:
        title_prefix = f"{len(mice)} sessions pooled ({group_col})"
        out_name = f"pooled_{len(mice)}sessions_{group_col}_fir_kernels"

    session_desc = ", ".join(f"{i['mouse']} {i['date']}" for i in fit["session_info"])
    print(f"\n=== FIR deconvolution GLM: {title_prefix} (signal={signal}) ===")
    print(f"Sessions: {session_desc}")
    print(f"Group column: {group_col}")
    print(f"Lag window: +/-{lag_seconds}s ({fit['n_lags']} samples each side, "
          f"{2 * fit['n_lags'] + 1} lag bins/feature)")
    print(f"Samples included: {fit['n_samples_included']}/{fit['n_samples_total']} "
          f"({100 * fit['n_samples_included'] / fit['n_samples_total']:.1f}%)")
    print(f"Design matrix: {len(fit['column_names'])} columns "
          f"({len(fit['kernels'])} features x {2 * fit['n_lags'] + 1} lags)")
    print(f"Best alpha (final RidgeCV fit): {cv['best_alpha']:.4g}")
    print(f"Out-of-sample R^2 ({n_splits}-fold GroupShuffleSplit by trial): "
          f"{cv['r2_mean']:.4f} +/- {cv['r2_std']:.4f}")
    print(f"Out-of-sample MSE: {cv['mse_mean']:.4f} +/- {cv['mse_std']:.4f}")

    fig = plot_fir_kernels(fit["fold_kernels"], fit["lag_time_s"], title_prefix=title_prefix)
    out_stem = output_dir / out_name
    fig.savefig(out_stem.with_suffix(".png"), dpi=150)
    fig.savefig(out_stem.with_suffix(".svg"))
    print(f"Saved {out_stem.with_suffix('.png')} and {out_stem.with_suffix('.svg')}")

    return fit


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_dirs", type=Path, nargs="+",
                         help="One or more session directories to pool (contains PHOTO/, pokeHistory*.mat, stats*.mat)")
    parser.add_argument("--hemisphere", choices=list(HEMISPHERE_CHANNELS), default=DEFAULT_HEMISPHERE)
    parser.add_argument("--signal", choices=["dff", "zscore"], default="zscore",
                         help="Which continuous trace to model (default: %(default)s)")
    parser.add_argument("--group-col", default=DEFAULT_GROUP_COLUMN,
                         help="trial_table column used to split side_in into discrete reward-history/type "
                              "channels, e.g. reward_seq_3, word_l3_generic, choice_seq_3, was_rewarded, "
                              "Behavioral_State (default: %(default)s)")
    parser.add_argument("--lag-seconds", type=float, default=DEFAULT_LAG_SECONDS,
                         help="+/- lag window (s) around side_in (default: %(default)s)")
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS,
                         help="Number of GroupShuffleSplit CV folds (default: %(default)s)")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE,
                         help="Held-out trial fraction per CV fold (default: %(default)s)")
    parser.add_argument("--max-segments", type=int, default=None,
                         help="Limit each session to its first N raw segments (for a quick test run)")
    parser.add_argument("--output-dir", type=Path, default=None,
                         help=f"Where to save the kernel figure (default: {DEFAULT_FIGURE_DIR})")
    parser.add_argument("--qc-report", type=Path, default=None,
                         help="Cohort QC report CSV (see run_cohort_qc.py) -- if given, session_dirs is filtered "
                              "through qc.session_qc.filter_sessions_by_qc before pooling")
    args = parser.parse_args()

    run_fir_glm(
        args.session_dirs, hemisphere=args.hemisphere, signal=args.signal, group_col=args.group_col,
        lag_seconds=args.lag_seconds, n_splits=args.n_splits, test_size=args.test_size,
        max_segments=args.max_segments, output_dir=args.output_dir, qc_report_path=args.qc_report,
    )


if __name__ == "__main__":
    main()
