"""
Same young/old split as run_sm_age_split_comparison.py, on red_l instead of
green_l -- tests whether the LR/RL/RR stronger-in-old word_l2 pattern found
on green_l replicates on a physically distinct channel (same 5 mice, same
sessions modulo per-channel carrier validity, same age cutoff). green_r is
deliberately NOT run here per the user's explicit request.

Usage:
    python run_sm_age_split_comparison_red_l.py
"""

from pathlib import Path

from run_sm_age_split_comparison import main

OUT_ROOT = Path("outputs_fixed/model_series_comparison_sm_age_split_red_l")
FIG_ROOT = Path("figures_fixed_model_series_sm_age_split_red_l")

if __name__ == "__main__":
    main(hemisphere="red_l", out_root=OUT_ROOT, fig_root=FIG_ROOT)
