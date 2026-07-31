"""
Plots for the continuous time-shifted FIR deconvolution GLM (models/fir_glm.py).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="ticks", context="talk")

# Fixed styling for the two continuous parametric modulators -- everything
# else is a dynamic "side_in_<group>" reward-history/type channel (see
# models.fir_glm.build_event_impulses), whose count/labels/order depend on
# whatever group_col was used, so those get an auto-assigned palette instead.
_FEATURE_COLORS = {
    "side_in_x_Qdiff": "#2c3e50",
    "side_in_x_Choice": "#8e44ad",
}
_FEATURE_LABELS = {
    "side_in_x_Qdiff": r"Side In $\times$ |Q diff|",
    "side_in_x_Choice": "Side In x Choice (Left=1)",
}


def plot_fir_kernels(fold_kernels, lag_time_s, title_prefix="", ncols=4):
    """Deconvolved temporal kernels, one panel per feature: mean +/- SEM
    across CV folds (models.fir_glm.fit_fir_glm's fold_betas, reshaped by
    models.fir_glm.reshape_kernels into fold_kernels).

    fold_kernels : {feature_name: (n_folds, n_lag_bins)} array per feature.
    lag_time_s : (n_lag_bins,) shared seconds-from-event axis (0 = the
        feature's own event onset, i.e. side_in for every channel in this
        model -- see models/fir_glm.py's module docstring).
    """
    names = list(fold_kernels.keys())
    n = len(names)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)

    dynamic_names = [name for name in names if name not in _FEATURE_COLORS]
    palette = sns.color_palette("tab10", n_colors=max(len(dynamic_names), 1))
    auto_colors = dict(zip(dynamic_names, palette))

    for i, name in enumerate(names):
        ax = axes[i // ncols][i % ncols]
        betas = np.asarray(fold_kernels[name])
        mean = betas.mean(axis=0)
        sem = betas.std(axis=0, ddof=1) / np.sqrt(betas.shape[0])
        color = _FEATURE_COLORS.get(name, auto_colors.get(name, "0.3"))
        label = _FEATURE_LABELS.get(name, name.removeprefix("side_in_"))

        ax.plot(lag_time_s, mean, color=color, lw=2)
        ax.fill_between(lag_time_s, mean - sem, mean + sem, color=color, alpha=0.3)
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Lag from side_in (s)")
        ax.set_ylabel(r"$\beta$")
        sns.despine(ax=ax)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"{title_prefix} -- FIR deconvolution kernels".strip(" -"))
    fig.tight_layout()
    return fig
