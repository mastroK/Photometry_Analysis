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

Hemisphere is resolved PER SESSION from config/session_hemisphere_overrides.csv
by default (falling back to config/mouse_hemisphere.csv, then --hemisphere) --
NOT one fixed value for the whole pool. This cohort's rig underwent a
mid-June-2023 cutover where every mouse's active channel switched from
green_r to green_l, so a single --hemisphere flag would silently demodulate
the wrong channel for whichever sessions don't match it. Pass --hemisphere
explicitly (with no override file) only if you're deliberately pooling a set
of sessions you know share one hemisphere.

Note: 'outcome' currently aliases the same photometry index as 'side_in'
(config.params.ALIGN_EVENT_COLUMNS -- no separately-timestamped reward
signal exists in this rig), so fitting both today produces numerically
identical coefficient trajectories. Both are wired through so this script is
ready the moment a real distinct outcome timestamp exists.
"""

import argparse
from pathlib import Path

from config.params import DEFAULT_HEMISPHERE, DEFAULT_TIME_RESOLVED_GLM_FORMULA, HEMISPHERE_CHANNELS
from config.session_metadata import get_mouse_hemisphere, load_mouse_hemisphere
from io_utils.raw_loader import parse_session_id
from models.glm_data import build_pooled_glm_dataset
from models.glm_encoding import fit_time_resolved_glm
from pipeline import DEFAULT_FIGURE_DIR
from qc.channel_selection import load_session_hemisphere_overrides
from viz.glm_plots import plot_glm_coefficients

DEFAULT_HEMISPHERE_LOOKUP = Path(__file__).parent / "config" / "mouse_hemisphere.csv"
DEFAULT_SESSION_OVERRIDES = Path(__file__).parent / "config" / "session_hemisphere_overrides.csv"


def _build_hemisphere_resolver(hemisphere_lookup_path, session_overrides_path, default_hemisphere):
    hemisphere_lookup = load_mouse_hemisphere(hemisphere_lookup_path) if Path(hemisphere_lookup_path).exists() else {}
    session_overrides = load_session_hemisphere_overrides(session_overrides_path)

    def hemisphere_for_session(session_dir):
        mouse, date = parse_session_id(session_dir)
        if (mouse, date) in session_overrides:
            return session_overrides[(mouse, date)]
        return get_mouse_hemisphere(hemisphere_lookup, mouse, default_hemisphere)

    return hemisphere_for_session


def run_glm_analysis(
    session_dirs,
    align_events=("side_in", "outcome"),
    hemisphere=DEFAULT_HEMISPHERE,
    max_segments=None,
    formula=DEFAULT_TIME_RESOLVED_GLM_FORMULA,
    output_dir=None,
    hemisphere_lookup_path=DEFAULT_HEMISPHERE_LOOKUP,
    session_overrides_path=DEFAULT_SESSION_OVERRIDES,
):
    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_FIGURE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    hemisphere_for_session = _build_hemisphere_resolver(hemisphere_lookup_path, session_overrides_path, hemisphere)

    results = {}
    for align_event in align_events:
        print(f"\n=== Time-resolved GLM aligned to '{align_event}' ===")
        peth_time, zscore_windows, trial_table = build_pooled_glm_dataset(
            session_dirs, align_event=align_event, hemisphere=hemisphere, max_segments=max_segments,
            hemisphere_for_session=hemisphere_for_session,
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
    parser.add_argument("--hemisphere-lookup", type=Path, default=DEFAULT_HEMISPHERE_LOOKUP,
                         help=f"Per-mouse hemisphere fallback CSV (default: {DEFAULT_HEMISPHERE_LOOKUP})")
    parser.add_argument("--session-overrides", type=Path, default=DEFAULT_SESSION_OVERRIDES,
                         help=f"Per-(mouse,date) hemisphere override CSV, checked first (default: {DEFAULT_SESSION_OVERRIDES})")
    args = parser.parse_args()

    run_glm_analysis(
        args.session_dirs, align_events=args.align_events, hemisphere=args.hemisphere,
        max_segments=args.max_segments, formula=args.formula, output_dir=args.output_dir,
        hemisphere_lookup_path=args.hemisphere_lookup, session_overrides_path=args.session_overrides,
    )


if __name__ == "__main__":
    main()
