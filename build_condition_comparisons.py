"""
Cross-condition comparison figures: overlay two conditions' time-resolved
GLM coefficient trajectories (default + expanded formula) and FIR kernels on
the SAME axes, term-by-term / feature-by-feature, rather than requiring the
reader to flip between two separate per-condition figures.

Consumes the beta_df CSVs (run_expanded_glm_analysis.py) and fir_fold_kernels
.npz files (run_fir_all_conditions.py) already saved per condition under
figures_fixed_expanded_glm/ -- no raw data reloading, this is pure
re-plotting of already-fit results.

Usage:
    python build_condition_comparisons.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path("figures_fixed_expanded_glm")
OUT_DIR = Path("figures_fixed_expanded_glm/comparisons")

# (pair_label, condition_a, condition_b) -- condition_a/b must match the
# lowercase cohort_label used when saving each condition's files.
COMPARISONS = [
    ("fp1_none_vs_fp2_none", "fp1_none", "fp2_none"),
    ("fp1_none_vs_fp1_dcz", "fp1_none", "fp1_dcz"),
    ("fp2_none_vs_fp2_dcz", "fp2_none", "fp2_dcz"),
    ("fp1_none_vs_fp1_retrained_none", "fp1_none", "fp1_retrained_none"),
    ("fp1_dcz_vs_fp2_dcz", "fp1_dcz", "fp2_dcz"),
]

COLOR_A = "#3B5B74"  # slate blue
COLOR_B = "#B8763E"  # warm amber


def _condition_label(cond):
    return cond.replace("_", " ").upper()


def overlay_glm(cond_a, cond_b, formula_label, out_path):
    df_a = pd.read_csv(DATA_DIR / f"{cond_a}_glm_{formula_label}_beta_df.csv", index_col=0)
    df_b = pd.read_csv(DATA_DIR / f"{cond_b}_glm_{formula_label}_beta_df.csv", index_col=0)
    peth_time = df_a.index.to_numpy(dtype=float)

    terms_a = {c[:-5] for c in df_a.columns if c.endswith("_beta")}
    terms_b = {c[:-5] for c in df_b.columns if c.endswith("_beta")}
    terms = sorted(terms_a & terms_b, key=lambda t: (t != "Intercept", t))
    missing = (terms_a | terms_b) - (terms_a & terms_b)
    if missing:
        print(f"  NOTE: terms not common to both conditions, skipped from overlay: {sorted(missing)}")

    ncols = 3
    nrows = int(np.ceil((len(terms) + 1) / ncols))  # +1 for R^2 panel
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for i, term in enumerate(terms):
        ax = axes_flat[i]
        for df, color, label in [(df_a, COLOR_A, _condition_label(cond_a)), (df_b, COLOR_B, _condition_label(cond_b))]:
            beta, se = df[f"{term}_beta"].to_numpy(), df[f"{term}_se"].to_numpy()
            ax.plot(peth_time, beta, color=color, lw=1.8, label=label)
            ax.fill_between(peth_time, beta - 1.96 * se, beta + 1.96 * se, color=color, alpha=0.18, lw=0)
        ax.axhline(0, color="0.6", lw=0.8, ls="--")
        ax.axvline(0, color="0.3", lw=0.8, ls=":")
        ax.set_title(term, fontsize=10)
        ax.set_xlabel("Time from Side Port Entry (s)", fontsize=8)
        ax.set_ylabel(r"$\beta$ (per SD)", fontsize=8)
        ax.tick_params(labelsize=8)

    r2_ax = axes_flat[len(terms)]
    for df, color, label in [(df_a, COLOR_A, _condition_label(cond_a)), (df_b, COLOR_B, _condition_label(cond_b))]:
        r2_ax.plot(peth_time, df["r_squared"].to_numpy(), color=color, lw=1.8, label=label)
    r2_ax.axvline(0, color="0.3", lw=0.8, ls=":")
    r2_ax.set_title("Model fit ($R^2$)", fontsize=10)
    r2_ax.set_xlabel("Time from Side Port Entry (s)", fontsize=8)
    r2_ax.tick_params(labelsize=8)
    r2_ax.legend(fontsize=8, frameon=False)

    for ax in axes_flat[len(terms) + 1:]:
        ax.axis("off")

    fig.suptitle(f"{_condition_label(cond_a)} vs {_condition_label(cond_b)} -- {formula_label} formula", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)
    print(f"  Saved {out_path.with_suffix('.png')}")


def overlay_fir(cond_a, cond_b, out_path):
    fa = np.load(DATA_DIR / f"{cond_a}_fir_fold_kernels.npz")
    fb = np.load(DATA_DIR / f"{cond_b}_fir_fold_kernels.npz")
    lag_time_s = fa["lag_time_s"]

    features_a = {k[len("kernel__"):] for k in fa.files if k.startswith("kernel__")}
    features_b = {k[len("kernel__"):] for k in fb.files if k.startswith("kernel__")}
    features = sorted(features_a & features_b)
    missing = (features_a | features_b) - (features_a & features_b)
    if missing:
        print(f"  NOTE: FIR features not common to both conditions, skipped from overlay: {sorted(missing)}")

    ncols = 3
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes_flat[i]
        for f, color, label in [(fa, COLOR_A, _condition_label(cond_a)), (fb, COLOR_B, _condition_label(cond_b))]:
            folds = f[f"kernel__{feat}"]
            mean, sem = folds.mean(axis=0), folds.std(axis=0, ddof=1) / np.sqrt(folds.shape[0])
            ax.plot(lag_time_s, mean, color=color, lw=1.8, label=label)
            ax.fill_between(lag_time_s, mean - sem, mean + sem, color=color, alpha=0.18, lw=0)
        ax.axhline(0, color="0.6", lw=0.8, ls="--")
        ax.axvline(0, color="0.3", lw=0.8, ls=":")
        ax.set_title(feat, fontsize=10)
        ax.set_xlabel("Lag from event (s)", fontsize=8)
        ax.set_ylabel("Kernel weight", fontsize=8)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=8, frameon=False)

    for ax in axes_flat[len(features):]:
        ax.axis("off")

    fig.suptitle(f"{_condition_label(cond_a)} vs {_condition_label(cond_b)} -- FIR kernels", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)
    print(f"  Saved {out_path.with_suffix('.png')}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for pair_label, cond_a, cond_b in COMPARISONS:
        print(f"\n=== {pair_label} ===")
        overlay_glm(cond_a, cond_b, "default", OUT_DIR / f"{pair_label}_glm_default")
        overlay_glm(cond_a, cond_b, "expanded", OUT_DIR / f"{pair_label}_glm_expanded")
        overlay_fir(cond_a, cond_b, OUT_DIR / f"{pair_label}_fir")


if __name__ == "__main__":
    main()
