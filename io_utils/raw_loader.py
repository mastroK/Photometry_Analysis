"""
Raw data loading: LabJack photometry segments and behavior logs
(pokeHistory + stats .mat files) straight off disk, no processing.
"""

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import scipy.io as sio

from config.params import N_RAW_CHANNELS, RAW_LOAD_N_WORKERS, RAW_SAMPLE_FREQ_HZ


def parse_session_id(session_dir):
    """Session directory convention: .../<cohort>/<date>/<mouse>/ -- returns (mouse, date)."""
    session_dir = Path(session_dir)
    return session_dir.name, session_dir.parent.name


PROCESSED_FILENAME_RE = re.compile(r"^processed_(?P<mouse>[A-Za-z0-9]+)_(?P<date>\d{6})\.mat$")


def discover_sessions_from_processed_dir(processed_dir, raw_root):
    """Parse an RA-curated individual_days/<condition>/ folder (see
    2-Output/FP1_processed_data/individual_days/{none,DCZ,saline}) of
    `processed_<mouse>_<date>.mat` files into the corresponding raw session
    directories under raw_root/<date>/<mouse>/, matching this pipeline's own
    <raw_root>/<date>/<mouse>/ convention (parse_session_id).

    Only files matching processed_<mouse>_<date>.mat (mouse + a 6-digit
    MMDDYY date) are considered -- non-matching files in the folder (helper
    .m/.asv scripts, per-mouse processed_sum_<mouse>_<condition>.mat
    aggregates) are skipped rather than raising.

    Returns (session_dirs, missing): session_dirs is the sorted list of raw
    Path objects that actually exist (safe to pass straight to
    batch_processor.run_batch_sessions / run_cohort_qc.run_cohort_qc_for_sessions
    / etc.); missing is a list of (mouse, date) pairs whose raw session
    directory could not be found under raw_root, printed as a warning rather
    than raised -- same soft-fail convention as the rest of this codebase.
    """
    processed_dir = Path(processed_dir)
    raw_root = Path(raw_root)

    session_dirs = []
    missing = []
    for path in sorted(processed_dir.glob("processed_*.mat")):
        match = PROCESSED_FILENAME_RE.match(path.name)
        if not match:
            continue
        mouse, date = match.group("mouse"), match.group("date")
        session_dir = raw_root / date / mouse
        if session_dir.is_dir():
            session_dirs.append(session_dir)
        else:
            missing.append((mouse, date))

    if missing:
        print(f"WARNING: {len(missing)} processed file(s) in {processed_dir} have no "
              f"matching raw session dir under {raw_root}:")
        for mouse, date in missing:
            print(f"  {mouse} {date}")

    return session_dirs, missing


def discover_behavior_files(session_dir):
    """Find the pokeHistory*.mat and stats*.mat files in a session directory,
    matching processBehavior.m:22-29's `dir('.')` + name-matching approach.
    """
    session_dir = Path(session_dir)
    poke_file = None
    stats_file = None
    for path in session_dir.iterdir():
        if not path.is_file():
            continue
        if "pokeHistory" in path.name:
            poke_file = path
        elif re.match(r"stats.*\.mat$", path.name):
            stats_file = path
    if poke_file is None or stats_file is None:
        raise FileNotFoundError(
            f"Could not find both pokeHistory*.mat and stats*.mat in {session_dir}"
        )
    return poke_file, stats_file


def load_raw_photometry(photo_dir, n_channels=N_RAW_CHANNELS, max_segments=None, n_workers=RAW_LOAD_N_WORKERS):
    """Concatenate Raw_*.mat segments (in filename order, same as MATLAB's
    sorted `dir('Raw_*.mat')`, processNew.m:160) and de-interleave into
    (n_channels, n_samples).

    Each segment is one flat vector where consecutive samples cycle through
    all channels first ("channel-fastest"): [ch1_s1, ch2_s1, ..., ch14_s1,
    ch1_s2, ...]. MATLAB recovers this with `reshape(vec, nChans, nSamples)`,
    which fills column-major (channel varies fastest) -- confirmed directly
    against processNew.m:212. The numpy equivalent is `order='F'`.

    Loading is over an SMB network share where each `loadmat` call costs
    ~1 s of latency regardless of the (tiny, ~28000-sample) file size, so we
    fan the loads out across threads (I/O-bound, not CPU-bound) rather than
    loading serially -- ~30-50 concurrent loads cuts wall-clock time by
    roughly 5-10x on this share.
    """
    photo_dir = Path(photo_dir)
    files = sorted(
        photo_dir.glob("Raw_*.mat"),
        key=lambda p: int(re.search(r"Raw_(\d+)\.mat", p.name).group(1)),
    )
    if not files:
        raise FileNotFoundError(f"No Raw_*.mat files found in {photo_dir}")
    if max_segments is not None:
        files = files[:max_segments]

    print(f"Loading {len(files)} raw segments from {photo_dir} ...")

    def _load_one(path):
        return sio.loadmat(path, simplify_cells=True)["temp"]

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        segments = list(pool.map(_load_one, files))

    n_pts_per_segment = len(segments[0])
    n_chans_assumed = n_pts_per_segment // RAW_SAMPLE_FREQ_HZ  # processNew.m:183 (28000/2000=14)
    if n_chans_assumed != n_channels:
        print(
            f"WARNING: segment length {n_pts_per_segment} implies "
            f"{n_chans_assumed} channels, not the configured {n_channels}"
        )

    flat = np.concatenate(segments)
    n_samples = len(flat) // n_chans_assumed
    flat = flat[: n_samples * n_chans_assumed]
    output = flat.reshape(n_chans_assumed, n_samples, order="F")  # processNew.m:212

    print(f"  -> raw array shape {output.shape} ({n_samples / RAW_SAMPLE_FREQ_HZ:.1f} s)")
    return output


def load_behavior_raw(poke_history_file, stats_file):
    poke_mat = sio.loadmat(poke_history_file, simplify_cells=True)
    stats_mat = sio.loadmat(stats_file, simplify_cells=True)
    return poke_mat["pokeHistory"], stats_mat["stats"]
