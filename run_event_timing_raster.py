"""
Per-trial event-timing raster: run a session through the pipeline's
behavior<->photometry alignment and plot the center_in/center_out/side_in/
side_out relationship for every trial, aligned to one chosen event.

Usage:
    python run_event_timing_raster.py session_dir \\
        [--align-to center_in|center_out|side_in|side_out] \\
        [--hemisphere green_r|red_l] [--max-segments N] [--output-dir DIR]
"""

import argparse
from pathlib import Path

from config.params import DEFAULT_HEMISPHERE, HEMISPHERE_CHANNELS
from io_utils.raw_loader import parse_session_id
from pipeline import DEFAULT_FIGURE_DIR, run_session
from viz.traces import plot_event_raster

RASTER_ALIGN_CHOICES = ("center_in", "center_out", "side_in", "side_out")


def run_event_timing_raster(session_dir, align_to="side_in", hemisphere=DEFAULT_HEMISPHERE,
                             max_segments=None, output_dir=None):
    session_dir = Path(session_dir)
    mouse, date = parse_session_id(session_dir)
    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_FIGURE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run_session(session_dir, hemisphere=hemisphere, max_segments=max_segments)
    trial_table = result["trial_table"]

    fig = plot_event_raster(trial_table, align_to=align_to, title_prefix=f"{mouse} {date}")
    out_stem = output_dir / f"{mouse}_{date}_event_raster_{align_to}"
    fig.savefig(out_stem.with_suffix(".png"), dpi=150)
    fig.savefig(out_stem.with_suffix(".svg"))
    print(f"Saved {out_stem.with_suffix('.png')} and {out_stem.with_suffix('.svg')}")
    return fig, trial_table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_dir", type=Path, help="Session directory (contains PHOTO/, pokeHistory*.mat, stats*.mat)")
    parser.add_argument("--align-to", choices=RASTER_ALIGN_CHOICES, default="side_in",
                         help="Which event the raster's t=0 is aligned to (default: %(default)s)")
    parser.add_argument("--hemisphere", choices=list(HEMISPHERE_CHANNELS), default=DEFAULT_HEMISPHERE)
    parser.add_argument("--max-segments", type=int, default=None,
                         help="Limit to the first N raw segments (for a quick test run)")
    parser.add_argument("--output-dir", type=Path, default=None,
                         help=f"Where to save the raster figure (default: {DEFAULT_FIGURE_DIR})")
    args = parser.parse_args()

    run_event_timing_raster(
        args.session_dir, align_to=args.align_to, hemisphere=args.hemisphere,
        max_segments=args.max_segments, output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
