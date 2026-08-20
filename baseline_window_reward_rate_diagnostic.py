"""
Follow-up to baseline_reward_rate_diagnostic.py: if the 60s rolling baseline
window is widened, does its correlation with local reward rate shrink or
disappear?

Reuses the SAME session sample (fp_sessions/sm_sessions helpers) and the
SAME local_reward_rate() construction as the original diagnostic, but for
each session loads the raw data/demodulates ONCE (return_dff_intermediates
=True) and then, from the cached pre-baseline `demodulated` trace, cheaply
recomputes final_mean at several window sizes -- no repeated raw loads or
re-demodulation. At each candidate window size W, BOTH the baseline window
and the local-reward-rate window are set to W (matched), since that's what
actually changing BASELINE_WINDOW_SEC in config/params.py would do to the
real pipeline -- this directly answers "if we used a W-second baseline,
would this confound still be there at that timescale?"

Usage:
    python baseline_window_reward_rate_diagnostic.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from baseline_reward_rate_diagnostic import FP_COHORTS, SM_REPORT, fp_sessions, local_reward_rate, sm_sessions
from config.params import FINAL_SAMPLE_FREQ_HZ
from pipeline import run_session

N_PER_GROUP = 3
FIG_OUT_DIR = "/private/tmp/claude-501/-Users-kevinmastro-Documents-Github-Photometry-Analysis/7f5518e6-fdff-4f87-b214-32dae2787477/scratchpad/baseline_diag_figs"
OUT_CSV = Path("outputs_fixed/baseline_window_reward_rate_diagnostic.csv")

WINDOW_SECONDS = [60.0, 120.0, 180.0, 240.0]


def process_one(session_dir, hemisphere, age_bin, group_label):
    try:
        result = run_session(session_dir, hemisphere=hemisphere, output_dir=FIG_OUT_DIR,
                              compute_bandit_state=False, return_dff_intermediates=True)
    except Exception as exc:
        print(f"  FAILED {session_dir.parent.name}/{session_dir.name}: {exc}")
        return []

    demodulated = pd.Series(result["dff_intermediates"]["demodulated"])
    n = len(demodulated)
    trial_table = result["trial_table"]
    mouse, date = session_dir.name, session_dir.parent.name

    max_window_samples = int(round(max(WINDOW_SECONDS) * FINAL_SAMPLE_FREQ_HZ))
    half_max = max_window_samples // 2
    step = max(1, int(round(5.0 * FINAL_SAMPLE_FREQ_HZ)))

    rows = []
    for window_sec in WINDOW_SECONDS:
        window_samples = int(round(window_sec * FINAL_SAMPLE_FREQ_HZ))
        final_mean_w = demodulated.rolling(window_samples, center=True, min_periods=window_samples).mean().to_numpy()
        rate_w = local_reward_rate(trial_table, n, window_samples)

        # trim to the region where even the LARGEST candidate window is fully
        # populated, so every window size is compared over the identical
        # stretch of the session (otherwise a bigger window's extra edge loss
        # could itself shift the correlation).
        fm, rt = final_mean_w[half_max:n - half_max], rate_w[half_max:n - half_max]
        ok = np.isfinite(fm) & np.isfinite(rt)
        fm, rt = fm[ok], rt[ok]
        fm_s, rt_s = fm[::step], rt[::step]

        if len(fm_s) < 10 or np.std(rt_s) == 0 or np.std(fm_s) == 0:
            r, p = np.nan, np.nan
        else:
            r, p = stats.pearsonr(fm_s, rt_s)

        print(f"  {group_label:14s} {mouse:6s} {date}  window={window_sec:5.0f}s  r={r:+.3f}  p={p:.3g}  n_pts={len(fm_s)}")
        rows.append(dict(group=group_label, age_bin=age_bin, mouse=mouse, date=date, hemisphere=hemisphere,
                          window_sec=window_sec, r=r, p=p, n_pts=len(fm_s)))
    return rows


def main():
    jobs = []
    for label, master_csv, qc_csv in FP_COHORTS:
        for session_dir, hemisphere, age_bin in fp_sessions(master_csv, qc_csv, N_PER_GROUP):
            jobs.append((session_dir, hemisphere, age_bin, label))
    for session_dir, hemisphere, age_bin in sm_sessions(SM_REPORT, N_PER_GROUP):
        jobs.append((session_dir, hemisphere, age_bin, f"SM_{age_bin}"))

    print(f"{len(jobs)} sessions queued, {len(WINDOW_SECONDS)} window sizes each\n")
    rows = []
    for job in jobs:
        rows.extend(process_one(*job))

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}\n")

    print("Mean r by window size (all sessions pooled):")
    print(df.groupby("window_sec")["r"].agg(["mean", "std", "count"]).to_string())
    print()

    for window_sec, grp in df.groupby("window_sec"):
        r = grp["r"].dropna().to_numpy()
        if len(r) > 1:
            stat, p = stats.wilcoxon(r)
            print(f"window={window_sec:5.0f}s  Wilcoxon r vs 0: statistic={stat:.3f}, p={p:.3g}, "
                  f"frac r>0={np.mean(r > 0):.2f}, n={len(r)}")


if __name__ == "__main__":
    main()
