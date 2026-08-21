"""
Split SM green_l sessions into young/old age groups (same median-age_bin cut
used for the behavioral kappa comparison, see plot_bandit_behavior_comparison.py
and mixedlm_bandit_behavior.py: age_numeric_days < 100 -> young, >= 100 -> old)
and run run_model_series_comparison_sm_red_l.main_red_l on each group
separately, to test for a neural correlate of the behavioral kappa
(stickiness) decline with age found there.

Model 2b_word_l1 (Z ~ C(word_l1)) is the direct neural analogue of kappa:
word_l1 encodes exactly the same thing kappa's "stick" term draws on
behaviorally (this trial's choice as a function of the immediately
preceding trial's own choice+outcome). If neural word_l1 encoding is weaker
in old vs young sessions in parallel with kappa's behavioral decline,
that's the neural correlate; 1_main_effects (Reward + Port + Port:Reward,
no history) is run alongside as the no-history baseline for context.

Excludes the 4 SM sessions flagged fully-forced-choice in
run_bandit_behavior_comparison.py (not real 80/20-bandit trials, would
distort the same way they'd distort a behavioral fit).

truncate_at_side_out=True (mechanism-A fix, min_retained_frac=0.5 default)
is used for both groups; censor_prev_trial is left off here (its own
pre-event masking, when combined with truncate_at_side_out, produces the
gapped/discontinuous pooled-plot visualization flagged earlier this
project -- unwanted for a young-vs-old comparison meant to be visually
clear) -- DECISION_WINDOW_S-based CV numbers are identical either way per
that same earlier finding.

Usage:
    python run_sm_age_split_comparison.py
"""

from pathlib import Path

import pandas as pd

from run_model_series_comparison_sm_red_l import DEFAULT_MICE, main_red_l

CHANNEL_REPORT = Path("outputs_fixed/sm_corrected_channel_report.csv")
HEMISPHERE = "green_l"
MODEL_NAMES = ["1_main_effects", "2b_word_l1"]
AGE_CUTOFF_DAYS = 100  # matches plot_bandit_behavior_comparison.py's median split

EXCLUDE_FORCED_SESSIONS = {
    ("SM1N", "062324"), ("SM1L", "062424"), ("SM2N", "062424"), ("SM3FR", "071824"),
}

OUT_ROOT = Path("outputs_fixed/model_series_comparison_sm_age_split")
FIG_ROOT = Path("figures_fixed_model_series_sm_age_split")


def get_age_split_session_dirs(mice=DEFAULT_MICE, hemisphere=HEMISPHERE):
    report = pd.read_csv(CHANNEL_REPORT, dtype={"date": str})
    valid = report[(report[f"{hemisphere}_carrier_valid"] == True) & (report["mouse"].isin(mice))]  # noqa: E712
    valid = valid[~valid.apply(lambda r: (r["mouse"], r["date"]) in EXCLUDE_FORCED_SESSIONS, axis=1)]

    age_days = valid["age_bin"].str.lstrip("P").astype(float)
    young = sorted(Path(p) for p in valid.loc[age_days < AGE_CUTOFF_DAYS, "session_dir"])
    old = sorted(Path(p) for p in valid.loc[age_days >= AGE_CUTOFF_DAYS, "session_dir"])
    return young, old


def main(hemisphere=HEMISPHERE, out_root=OUT_ROOT, fig_root=FIG_ROOT):
    """hemisphere/out_root/fig_root : override to run this same young/old
    split on a different channel -- e.g. run_sm_age_split_comparison_red_l.py
    calls main(hemisphere="red_l", out_root=.../sm_age_split_red_l, ...) to
    test whether the green_l LR/RL/RR age pattern replicates on a distinct
    physical channel.
    """
    young_dirs, old_dirs = get_age_split_session_dirs(hemisphere=hemisphere)
    print(f"young (age_bin < P{AGE_CUTOFF_DAYS}): {len(young_dirs)} sessions")
    print(f"old (age_bin >= P{AGE_CUTOFF_DAYS}): {len(old_dirs)} sessions")

    for label, dirs in (("young", young_dirs), ("old", old_dirs)):
        print(f"\n{'=' * 70}\nSM {hemisphere} -- {label}\n{'=' * 70}")
        main_red_l(
            mice=DEFAULT_MICE, hemisphere=hemisphere, model_names=MODEL_NAMES,
            out_dir=out_root / label / "results", fig_dir=fig_root / label,
            truncate_at_side_out=True, session_dirs=dirs,
        )


if __name__ == "__main__":
    main()
