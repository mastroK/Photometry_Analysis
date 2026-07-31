"""
Time-resolved GLM encoding-model script: pool PETH windows + trial-level
covariates across an explicit list of session directories (see
models.glm_data.build_pooled_glm_dataset), fit models.glm_encoding.
fit_time_resolved_glm at each requested alignment event, and save a
coefficient-trajectory figure (.png + .svg) per event.

Usage:
    python run_glm_analysis.py session_dir [session_dir ...] \\
        [--align-to side_in outcome] [--hemisphere green_r|red_l] \\
        [--output-dir DIR] [--formula "..."] [--max-segments N]

Note: 'outcome' currently aliases the same photometry index as 'side_in'
(config.params.ALIGN_EVENT_COLUMNS -- no separately-timestamped reward
signal exists in this rig), so fitting both today produces numerically
identical coefficient trajectories. Both are wired through so this script is
ready the moment a real distinct outcome timestamp exists.
"""

import argparse
from pathlib import Path

from config.params import DEFAULT_HEMISPHERE, DEFAULT_TIME_RESOLVED_GLM_FORMULA, HEMISPHERE_CHANNELS
from models.glm_data import build_pooled_glm_dataset
from models.glm_encoding import fit_time_resolved_glm
from pipeline import DEFAULT_FIGURE_DIR
from viz.glm_plots import plot_glm_coefficients


def run_glm_analysis(
    session_dirs,
    align_events=("side_in", "outcome"),
    hemisphere=DEFAULT_HEMISPHERE,
    max_segments=None,
    formula=DEFAULT_TIME_RESOLVED_GLM_FORMULA,
    output_dir=None,
):
    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_FIGURE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for align_event in align_events:
        print(f"\n=== Time-resolved GLM aligned to '{align_event}' ===")
        peth_time, zscore_windows, trial_table = build_pooled_glm_dataset(
            session_dirs, align_event=align_event, hemisphere=hemisphere, max_segments=max_segments,
        )
        beta_df = fit_time_resolved_glm(zscore_windows, peth_time, trial_table, formula=formula)

        fig = plot_glm_coefficients(peth_time, beta_df, formula, align_event=align_event)
        out_stem = output_dir / f"glm_{align_event}_coefficients"
        fig.savefig(out_stem.with_suffix(".png"), dpi=150)
        fig.savefig(out_stem.with_suffix(".svg"))
        print(f"Saved {out_stem.with_suffix('.png')} and {out_stem.with_suffix('.svg')}")

        results[align_event] = dict(peth_time=peth_time, beta_df=beta_df, trial_table=trial_table)

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_dirs", type=Path, nargs="+", help="One or more session directories to pool")
    parser.add_argument("--align-to", dest="align_events", nargs="+", default=["side_in", "outcome"],
                         choices=["center_in", "side_in", "outcome", "side_out"],
                         help="Which trial event(s) to fit the time-resolved GLM at (default: %(default)s)")
    parser.add_argument("--hemisphere", choices=list(HEMISPHERE_CHANNELS), default=DEFAULT_HEMISPHERE)
    parser.add_argument("--max-segments", type=int, default=None,
                         help="Limit each session to its first N raw segments (for a quick test run)")
    parser.add_argument("--formula", default=DEFAULT_TIME_RESOLVED_GLM_FORMULA,
                         help="Patsy formula string (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=None,
                         help="Where to save coefficient figures (default: pipeline.DEFAULT_FIGURE_DIR)")
    args = parser.parse_args()

    run_glm_analysis(
        args.session_dirs, align_events=args.align_events, hemisphere=args.hemisphere,
        max_segments=args.max_segments, formula=args.formula, output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
