"""
2-panel verification plot: continuous trace + reward-split PETH.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config.params import ALIGN_EVENT_LABELS, DEFAULT_ALIGN_EVENT

sns.set_theme(style="ticks", context="talk")


def _plot_peth(ax, peth_time, windows, label, color):
    if windows.shape[0] == 0:
        return
    mean = windows.mean(axis=0)
    sem = windows.std(axis=0, ddof=1) / np.sqrt(windows.shape[0])
    ax.plot(peth_time, mean, color=color, label=f"{label} (n={windows.shape[0]})")
    ax.fill_between(peth_time, mean - sem, mean + sem, color=color, alpha=0.25)


def plot_session_overview(
    time_axis,
    dff,
    peth_time,
    rewarded_windows,
    unrewarded_windows,
    channel_label,
    title_prefix,
    rewarded_zscore_windows=None,
    unrewarded_zscore_windows=None,
    baseline_window_s=None,
    align_event=DEFAULT_ALIGN_EVENT,
):
    """3-panel verification figure: continuous dF/F trace, PETH in dF/F
    units, and (if the z-scored window arrays are provided) PETH in
    trial-level event-aligned z-score units.

    rewarded_zscore_windows / unrewarded_zscore_windows are the output of
    alignment.windowing.compute_event_aligned_zscore -- distinct from the
    plain dF/F windows, each trial here is normalized against its own
    pre-event baseline rather than plotted in raw dF/F.
    baseline_window_s, if given, is (pre_s, post_s) and is shaded on the
    z-score panel to show where each trial's baseline was taken from.
    align_event : one of config.params.ALIGN_EVENT_COLUMNS's keys
        ('center_in', 'side_in', 'outcome', 'side_out') -- t=0 in peth_time
        is whichever event the caller aligned windows to (see
        alignment.windowing.get_event_indices); only used here to label the
        x-axis/title, not to change any of the plotted data.
    """
    event_label = ALIGN_EVENT_LABELS[align_event]
    has_zscore = rewarded_zscore_windows is not None or unrewarded_zscore_windows is not None
    n_panels = 3 if has_zscore else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 5))

    ax = axes[0]
    ax.plot(time_axis, dff, lw=0.6, color="0.2")
    ax.set_xlabel("Session time (s)")
    ax.set_ylabel(r"$\Delta F/F$")
    ax.set_title(f"{title_prefix} -- demodulated {channel_label} channel")
    sns.despine(ax=ax)

    ax = axes[1]
    _plot_peth(ax, peth_time, rewarded_windows, "Rewarded", "#1abc9c")
    _plot_peth(ax, peth_time, unrewarded_windows, "Unrewarded", "#e74c3c")
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlabel(f"Time from {event_label} (s)")
    ax.set_ylabel(r"$\Delta F/F$")
    ax.set_title(f"PETH: {event_label}, split by outcome")
    ax.legend(frameon=False)
    sns.despine(ax=ax)

    if has_zscore:
        ax = axes[2]
        if rewarded_zscore_windows is not None:
            _plot_peth(ax, peth_time, rewarded_zscore_windows, "Rewarded", "#1abc9c")
        if unrewarded_zscore_windows is not None:
            _plot_peth(ax, peth_time, unrewarded_zscore_windows, "Unrewarded", "#e74c3c")
        ax.axvline(0, color="k", ls="--", lw=1)
        if baseline_window_s is not None:
            ax.axvspan(baseline_window_s[0], baseline_window_s[1], color="0.6", alpha=0.15,
                       label="baseline window")
        ax.set_xlabel(f"Time from {event_label} (s)")
        ax.set_ylabel("Trial-aligned Z-score")
        ax.set_title(f"PETH: trial-level event-aligned Z-score ({event_label})")
        ax.legend(frameon=False)
        sns.despine(ax=ax)

    fig.tight_layout()
    return fig


def plot_peth_by_group(
    peth_time,
    dff_windows,
    group_labels,
    zscore_windows=None,
    channel_label="",
    title_prefix="",
    group_colors=None,
    group_order=None,
    baseline_window_s=None,
    align_event=DEFAULT_ALIGN_EVENT,
):
    """PETH figure split by an arbitrary per-trial group label (e.g.
    Behavioral_State, or a Q_diff quantile bin), generalizing
    plot_session_overview's rewarded/unrewarded split to any grouping.

    group_labels is a 1D array/Series of length dff_windows.shape[0] (and
    zscore_windows.shape[0], if given) -- same row-alignment convention as
    evaluate_word_outcomes' peth_trial_table/zscore_windows pairing in
    behavior/word_encoding.py. NaN/None labels are dropped from the plot.

    group_colors : optional {label: color} dict (e.g. config.params.STATE_COLORS
        when grouping by Behavioral_State). Labels not covered fall back to a
        "tab10" palette.
    group_order : optional explicit label ordering for consistent legends
        across figures; defaults to sorted(unique labels).
    align_event : one of config.params.ALIGN_EVENT_COLUMNS's keys -- see
        plot_session_overview, same axis/title-labeling-only role.
    """
    event_label = ALIGN_EVENT_LABELS[align_event]
    group_labels = pd.Series(np.asarray(group_labels)).reset_index(drop=True)
    valid = group_labels.notna()
    labels_order = list(group_order) if group_order is not None else sorted(group_labels[valid].unique())

    palette = sns.color_palette("tab10", n_colors=len(labels_order))
    colors = {label: (group_colors or {}).get(label, palette[i]) for i, label in enumerate(labels_order)}

    has_zscore = zscore_windows is not None
    n_panels = 2 if has_zscore else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 5))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    for label in labels_order:
        mask = (group_labels == label).to_numpy()
        _plot_peth(ax, peth_time, dff_windows[mask], str(label), colors[label])
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlabel(f"Time from {event_label} (s)")
    ax.set_ylabel(r"$\Delta F/F$")
    ax.set_title(f"{title_prefix} -- PETH by group ({channel_label})".strip(" -"))
    ax.legend(frameon=False)
    sns.despine(ax=ax)

    if has_zscore:
        ax = axes[1]
        for label in labels_order:
            mask = (group_labels == label).to_numpy()
            _plot_peth(ax, peth_time, zscore_windows[mask], str(label), colors[label])
        ax.axvline(0, color="k", ls="--", lw=1)
        if baseline_window_s is not None:
            ax.axvspan(baseline_window_s[0], baseline_window_s[1], color="0.6", alpha=0.15,
                       label="baseline window")
        ax.set_xlabel(f"Time from {event_label} (s)")
        ax.set_ylabel("Trial-aligned Z-score")
        ax.set_title(f"PETH by group: trial-level event-aligned Z-score ({event_label})")
        ax.legend(frameon=False)
        sns.despine(ax=ax)

    fig.tight_layout()
    return fig


_RASTER_EVENTS = ("center_in", "center_out", "side_in", "side_out")
_RASTER_EVENT_LABELS = {
    "center_in": "Center In", "center_out": "Center Out",
    "side_in": "Side In", "side_out": "Side Out",
}
_RASTER_EVENT_COLORS = {
    "center_in": "#3498db", "center_out": "#9b59b6",
    "side_in": "#1abc9c", "side_out": "#e74c3c",
}


def plot_event_raster(trial_table, align_to="side_in", title_prefix=""):
    """Per-trial event-timing raster: align every trial to `align_to` (one of
    'center_in'/'center_out'/'side_in'/'side_out', matching the *_s columns
    added by behavior.sync.align_behavior_to_photometry) and plot the other
    three events' time relative to it, one marker-row per trial, split by
    was_rewarded (circle = rewarded, x = unrewarded) -- shows the
    center-in/center-out/side-in/side-out relationship per trial directly,
    e.g. rewarded trials' side_out sitting farther from side_in than
    unrewarded trials' (reward consumption time).

    Trials missing the `align_to` event's own timestamp are dropped; the
    remaining trials are plotted in their original (chronological) order,
    top-to-bottom -- no other event needs to be present for a trial to be
    included, since not all trials resolve all 4 events.
    """
    if align_to not in _RASTER_EVENTS:
        raise ValueError(f"align_to must be one of {_RASTER_EVENTS}, got {align_to!r}")
    other_events = [e for e in _RASTER_EVENTS if e != align_to]

    align_col = f"{align_to}_s"
    table = trial_table.loc[trial_table[align_col].notna()].reset_index(drop=True)
    align_time = table[align_col].to_numpy()
    is_rewarded = table["was_rewarded"].to_numpy(dtype=bool)
    y = np.arange(len(table))

    fig, ax = plt.subplots(figsize=(8, max(4, 0.05 * len(table))))
    for event in other_events:
        rel_time = table[f"{event}_s"].to_numpy() - align_time
        finite = np.isfinite(rel_time)
        color = _RASTER_EVENT_COLORS[event]
        label = _RASTER_EVENT_LABELS[event]
        ax.scatter(rel_time[finite & is_rewarded], y[finite & is_rewarded],
                   marker="o", s=14, color=color, label=f"{label} (rewarded)")
        ax.scatter(rel_time[finite & ~is_rewarded], y[finite & ~is_rewarded],
                   marker="x", s=14, color=color, label=f"{label} (unrewarded)")

    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlabel(f"Time from {_RASTER_EVENT_LABELS[align_to]} (s)")
    ax.set_ylabel("Trial")
    ax.set_title(
        f"{title_prefix} -- event-timing raster (aligned to {_RASTER_EVENT_LABELS[align_to]})".strip(" -")
    )
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig
