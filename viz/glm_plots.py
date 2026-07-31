"""
Coefficient-trajectory figure for models.glm_encoding.fit_time_resolved_glm.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from config.params import ALIGN_EVENT_LABELS

sns.set_theme(style="ticks", context="talk")


def plot_glm_coefficients(peth_time, beta_df, formula_str, align_event=None,
                           include_intercept=False, n_cols=3):
    """Small-multiples figure: one subplot per regression term's beta(t)
    trajectory (columns of beta_df from fit_time_resolved_glm), each with a
    shaded 95% CI band (beta +/- 1.96*se), plus a final subplot for the
    model's R^2(t)/adjusted R^2(t). axvline(0) marks the aligning event in
    every panel.

    beta_df : output of models.glm_encoding.fit_time_resolved_glm (indexed
        by peth_time, with {term}_beta/{term}_se/... columns plus
        r_squared/r_squared_adj).
    formula_str : the model formula, shown as the figure's suptitle.
    align_event : optional key into config.params.ALIGN_EVENT_LABELS, used
        only to label the shared x-axis (e.g. "Time from Side Port Entry
        (s)") -- purely cosmetic, same role as plot_session_overview's
        align_event.
    include_intercept : include the Intercept term's own subplot (usually
        not scientifically interesting for an encoding model, so off by
        default).
    """
    x_label = f"Time from {ALIGN_EVENT_LABELS[align_event]} (s)" if align_event else "Time from event (s)"

    terms = [c[: -len("_beta")] for c in beta_df.columns if c.endswith("_beta")]
    if not include_intercept:
        terms = [t for t in terms if t != "Intercept"]

    n_panels = len(terms) + 1  # + R^2 panel
    n_cols = max(1, min(n_cols, n_panels))
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, term in zip(axes, terms):
        beta = beta_df[f"{term}_beta"].to_numpy()
        ci = 1.96 * beta_df[f"{term}_se"].to_numpy()
        ax.plot(peth_time, beta, color="#2c3e50")
        ax.fill_between(peth_time, beta - ci, beta + ci, color="#2c3e50", alpha=0.25)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.set_title(term, fontsize=11)
        ax.set_xlabel(x_label)
        ax.set_ylabel(r"$\beta$ (per SD)")
        sns.despine(ax=ax)

    r2_ax = axes[len(terms)]
    r2_ax.plot(peth_time, beta_df["r_squared"], label="$R^2$", color="#1abc9c")
    r2_ax.plot(peth_time, beta_df["r_squared_adj"], label="adj. $R^2$", color="#e74c3c")
    r2_ax.axvline(0, color="k", ls="--", lw=1)
    r2_ax.set_title("Model fit", fontsize=11)
    r2_ax.set_xlabel(x_label)
    r2_ax.set_ylabel(r"$R^2$")
    r2_ax.legend(frameon=False, fontsize=9)
    sns.despine(ax=r2_ax)

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.suptitle(formula_str, fontsize=10, y=1.02)
    fig.tight_layout()
    return fig
