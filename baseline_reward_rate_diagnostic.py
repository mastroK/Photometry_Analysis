"""
Diagnostic: does the 60s rolling baseline (`final_mean`, the quantity
subtracted -- and whose paired std is divided by -- to produce `zscore`,
the value every downstream GLM/FIR/RPE fit in this project consumes)
covary with local reward rate? And does that covariation differ across
cohort/condition or SM's age bins?

If the baseline itself tracks recent reward history, every history-length
kernel (RPE, N-back reward terms, switch dynamics) built on `zscore` is
confounded by the exact quantity those kernels are trying to explain --
and if the strength of that confound differs by age, an age comparison
built on those kernels is not interpretable.

Samples a modest, representative set of already-QC'd sessions (a few
distinct mice per FP1/FP2 condition; a few sessions per SM age bin, using
green_l -- the reliable SM channel) and reprocesses each with
pipeline.run_session(..., return_dff_intermediates=True) to recover the
actual continuous final_mean baseline at ~18.52Hz. Builds a local
reward-rate signal (rewards inside the SAME centered 60s window used for
the baseline itself) and computes one Pearson r per session between the
two, subsampled every ~5s to blunt the correlation's autocorrelation-
inflated apparent precision (the sign/magnitude is what matters here, not
a p-value that assumes independent samples).

Usage:
    python baseline_reward_rate_diagnostic.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from config.params import BASELINE_WINDOW_SAMPLES, FINAL_SAMPLE_FREQ_HZ
from pipeline import run_session

N_PER_GROUP = 3
FIG_OUT_DIR = "/private/tmp/claude-501/-Users-kevinmastro-Documents-Github-Photometry-Analysis/7f5518e6-fdff-4f87-b214-32dae2787477/scratchpad/baseline_diag_figs"
OUT_CSV = Path("outputs_fixed/baseline_reward_rate_diagnostic.csv")

FP_COHORTS = [
    ("FP1_none", "outputs_fixed/none_cohort_master.csv", "outputs_fixed/none_cohort_qc_report.csv"),
    ("FP1_dcz", "outputs_fixed/fp1_dcz_cohort_master.csv", "outputs_fixed/fp1_dcz_cohort_qc_report.csv"),
    ("FP2_none", "outputs_fixed/fp2_none_cohort_master.csv", "outputs_fixed/fp2_none_cohort_qc_report.csv"),
    ("FP2_dcz", "outputs_fixed/fp2_dcz_cohort_master.csv", "outputs_fixed/fp2_dcz_cohort_qc_report.csv"),
]
SM_REPORT = "outputs_fixed/sm_corrected_channel_report.csv"


def fp_sessions(master_csv, qc_report_csv, n_per_group):
    master = pd.read_csv(master_csv, dtype={"date": str})
    qc = pd.read_csv(qc_report_csv, dtype={"date": str})
    pairs = master[["mouse", "date", "hemisphere"]].drop_duplicates()
    merged = pairs.merge(qc[["mouse", "date", "session_dir"]], on=["mouse", "date"])
    merged = merged.drop_duplicates(subset=["mouse"]).head(n_per_group)
    return [(Path(r.session_dir), r.hemisphere, None) for r in merged.itertuples()]


def sm_sessions(report_csv, n_per_bin):
    report = pd.read_csv(report_csv, dtype={"date": str})
    valid = report[report["green_l_carrier_valid"] == True]  # noqa: E712
    out = []
    for age_bin, grp in valid.groupby("age_bin"):
        grp = grp.drop_duplicates(subset=["mouse"]).head(n_per_bin)
        for r in grp.itertuples():
            out.append((Path(r.session_dir), "green_l", age_bin))
    return out


def local_reward_rate(trial_table, n_samples, window_samples):
    impulses = np.zeros(n_samples)
    valid = trial_table["photometry_side_in_index"] >= 0
    idx = trial_table.loc[valid, "photometry_side_in_index"].to_numpy().astype(int)
    rewarded = trial_table.loc[valid, "was_rewarded"].to_numpy().astype(bool)
    in_range = (idx >= 0) & (idx < n_samples)
    idx, rewarded = idx[in_range], rewarded[in_range]
    impulses[idx[rewarded]] = 1.0
    series = pd.Series(impulses)
    return series.rolling(window_samples, center=True, min_periods=window_samples).sum().to_numpy()


def process_one(session_dir, hemisphere, age_bin, group_label):
    try:
        result = run_session(session_dir, hemisphere=hemisphere, output_dir=FIG_OUT_DIR,
                              compute_bandit_state=False, return_dff_intermediates=True)
    except Exception as exc:
        print(f"  FAILED {session_dir.parent.name}/{session_dir.name}: {exc}")
        return None

    final_mean = result["dff_intermediates"]["final_mean"]
    n = len(final_mean)
    rate = local_reward_rate(result["trial_table"], n, BASELINE_WINDOW_SAMPLES)

    half = BASELINE_WINDOW_SAMPLES // 2
    fm, rt = final_mean[half:n - half], rate[half:n - half]
    ok = np.isfinite(fm) & np.isfinite(rt)
    fm, rt = fm[ok], rt[ok]

    step = max(1, int(round(5.0 * FINAL_SAMPLE_FREQ_HZ)))
    fm_s, rt_s = fm[::step], rt[::step]
    if len(fm_s) < 10 or np.std(rt_s) == 0 or np.std(fm_s) == 0:
        r, p = np.nan, np.nan
    else:
        r, p = stats.pearsonr(fm_s, rt_s)

    mouse, date = session_dir.name, session_dir.parent.name
    print(f"  {group_label:14s} {mouse:6s} {date} ({hemisphere}): r={r:+.3f}  p={p:.3g}  n_pts={len(fm_s)}")
    return dict(group=group_label, age_bin=age_bin, mouse=mouse, date=date, hemisphere=hemisphere,
                r=r, p=p, n_pts=len(fm_s))


def main():
    jobs = []
    for label, master_csv, qc_csv in FP_COHORTS:
        for session_dir, hemisphere, age_bin in fp_sessions(master_csv, qc_csv, N_PER_GROUP):
            jobs.append((session_dir, hemisphere, age_bin, label))
    for session_dir, hemisphere, age_bin in sm_sessions(SM_REPORT, N_PER_GROUP):
        jobs.append((session_dir, hemisphere, age_bin, f"SM_{age_bin}"))

    print(f"{len(jobs)} sessions queued\n")
    rows = [process_one(*job) for job in jobs]
    rows = [r for r in rows if r is not None]

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}\n")

    print("Per-group mean r (final_mean baseline vs local reward rate):")
    print(df.groupby("group")["r"].agg(["mean", "std", "count"]).to_string())

    overall_r = df["r"].dropna().to_numpy()
    if len(overall_r) > 1:
        stat, p = stats.wilcoxon(overall_r)
        print(f"\nWilcoxon signed-rank, per-session r vs 0 (n={len(overall_r)} sessions): "
              f"statistic={stat:.3f}, p={p:.3g}")
        print(f"Fraction of sessions with r > 0: {np.mean(overall_r > 0):.2f}")


if __name__ == "__main__":
    main()
