"""
Numeric diff of the Python pipeline against a real RA-processed MATLAB
reference (processed_<mouse>_<date>.mat), for verifying that fixes to
behavior/sync.py and preprocessing/demodulate.py actually restore parity with
the original MATLAB pipeline (processNew_fast_kevin.m).

Requires a local MATLAB install runnable headlessly via `matlab -batch`
(the trialTable field of a processed_*.mat is an MCOS/table object that
scipy.io can't parse directly -- export_matlab_reference.m does the real
export via MATLAB itself).

Usage:
    python validation/compare_to_matlab.py \\
        /path/to/processed_WCL23_060223.mat \\
        "/path/to/1-Raw data/FP1/060223/WCL23" \\
        [--matlab-bin /Applications/MATLAB_R2021b.app/bin/matlab]
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import scipy.io as sio

from behavior.sync import align_behavior_to_photometry
from behavior.trial_table import build_trial_table
from config.params import DEFAULT_HEMISPHERE, HEMISPHERE_CHANNELS
from io_utils.raw_loader import discover_behavior_files, load_behavior_raw, load_raw_photometry
from preprocessing.demodulate import compute_dff_and_zscore, demodulate_envelope, estimate_carrier_freq

DEFAULT_MATLAB_BIN = "/Applications/MATLAB_R2021b.app/bin/matlab"
_VALIDATION_DIR = Path(__file__).parent


def export_matlab_reference(mat_path, out_dir, matlab_bin=DEFAULT_MATLAB_BIN):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        matlab_bin, "-nodisplay", "-nosplash", "-batch",
        f"addpath('{_VALIDATION_DIR}'); export_matlab_reference('{mat_path}', '{out_dir}')",
    ]
    subprocess.run(cmd, check=True)
    return out_dir


def load_matlab_reference(out_dir):
    out_dir = Path(out_dir)
    trial_table = pd.read_csv(out_dir / "trialTable.csv")
    params = sio.loadmat(out_dir / "params_flat.mat", simplify_cells=True)
    signals = sio.loadmat(out_dir / "signals_ch1.mat", simplify_cells=True)["sig1"].ravel()
    return trial_table, params, signals


def run_python_pipeline(session_dir, hemisphere=DEFAULT_HEMISPHERE, max_segments=None):
    session_dir = Path(session_dir)
    photo_dir = session_dir / "PHOTO"
    channels = HEMISPHERE_CHANNELS[hemisphere]

    raw = load_raw_photometry(photo_dir, max_segments=max_segments)
    measured_freq, _ = estimate_carrier_freq(raw[channels.signal_channel])
    envelope, _ = demodulate_envelope(raw[channels.signal_channel], measured_freq)
    dff, zscore, _ = compute_dff_and_zscore(raw[channels.signal_channel], measured_freq, envelope)

    poke_history_file, stats_file = discover_behavior_files(session_dir)
    poke_history, stats = load_behavior_raw(poke_history_file, stats_file)
    trial_table = build_trial_table(poke_history, stats)
    trial_table, align_info = align_behavior_to_photometry(raw, trial_table, poke_history, n_final_samples=len(envelope))

    return dict(trial_table=trial_table, envelope=envelope, dff=dff, zscore=zscore, align_info=align_info)


def compare(matlab_trial_table, matlab_params, matlab_signals, python_result):
    py_tt = python_result["trial_table"]
    n = min(len(matlab_trial_table), len(py_tt))
    print(f"MATLAB trials: {len(matlab_trial_table)}, Python trials: {len(py_tt)}")

    print(f"\nMATLAB timeShift={matlab_params['timeShift']:.4f}  "
          f"Python time_shift={python_result['align_info']['time_shift']:.4f}")

    index_cols = [
        ("photometryCenterInIndex", "photometry_center_in_index"),
        ("photometryCenterOutIndex", "photometry_center_out_index"),
        ("photometrySideInIndex", "photometry_side_in_index"),
        ("photometrySideOutIndex", "photometry_side_out_index"),
    ]
    # MATLAB is 1-based; Python is 0-based -- subtract 1 from MATLAB's nonzero indices for a fair diff.
    print("\n--- Per-trial index comparison (MATLAB 1-based -> 0-based for diff) ---")
    for matlab_col, python_col in index_cols:
        m = matlab_trial_table[matlab_col].to_numpy()[:n].astype(float)
        p = py_tt[python_col].to_numpy()[:n].astype(float)
        m_valid = m > 0
        m_adj = np.where(m_valid, m - 1, -1)
        both_valid = m_valid & (p >= 0)
        if both_valid.sum() == 0:
            print(f"{python_col}: no trials with both pipelines reporting a valid index")
            continue
        diff = p[both_valid] - m_adj[both_valid]
        print(f"{python_col}: n_both_valid={both_valid.sum()}/{n}  "
              f"mean_diff={diff.mean():+.2f} samples  median={np.median(diff):+.2f}  "
              f"std={diff.std():.2f}  max_abs={np.abs(diff).max():.0f}  "
              f"n_exact_match={int((diff == 0).sum())}")

    m_valid_all = matlab_trial_table["hasAllPhotometryData"].to_numpy()[:n].astype(bool)
    print(f"\nMATLAB hasAllPhotometryData: {m_valid_all.sum()}/{n} trials valid")

    print("\n--- Continuous signal comparison (python zscore vs MATLAB signals/pSignal) ---")
    n_sig = min(len(matlab_signals), len(python_result["zscore"]))
    m_sig = matlab_signals[:n_sig]
    p_sig = python_result["zscore"][:n_sig]
    finite = np.isfinite(m_sig) & np.isfinite(p_sig)
    corr = np.corrcoef(m_sig[finite], p_sig[finite])[0, 1]
    print(f"n_samples={n_sig}  correlation={corr:.6f}  "
          f"mean_abs_diff={np.abs(m_sig[finite] - p_sig[finite]).mean():.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("matlab_processed_mat", type=Path)
    parser.add_argument("raw_session_dir", type=Path)
    parser.add_argument("--matlab-bin", default=DEFAULT_MATLAB_BIN)
    parser.add_argument("--hemisphere", choices=list(HEMISPHERE_CHANNELS), default=DEFAULT_HEMISPHERE)
    parser.add_argument("--export-dir", type=Path, default=None,
                         help="Where to cache the MATLAB export (default: alongside this script, tmp)")
    parser.add_argument("--skip-export", action="store_true",
                         help="Reuse an already-exported reference dir (--export-dir) instead of re-running MATLAB")
    args = parser.parse_args()

    export_dir = args.export_dir or (Path("/tmp") / f"matlab_ref_{args.matlab_processed_mat.stem}")
    if not args.skip_export:
        export_matlab_reference(args.matlab_processed_mat, export_dir, matlab_bin=args.matlab_bin)

    matlab_trial_table, matlab_params, matlab_signals = load_matlab_reference(export_dir)
    python_result = run_python_pipeline(args.raw_session_dir, hemisphere=args.hemisphere)
    compare(matlab_trial_table, matlab_params, matlab_signals, python_result)


if __name__ == "__main__":
    main()
