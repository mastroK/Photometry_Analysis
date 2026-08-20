"""
Compare Bernardo's simplified model series -- Models 1, 1b, 2 (2-bit/3-bit
reward history), 2b (1/2/3-letter "word"), 2c (word + side_out split by
reward), 3 (RPE + binned Q_diff + Port, replacing the word categorical with
the sticky Q-learning model's own value estimates) -- against each other,
per mouse.

Model 3 was added after a from-scratch sanity check (against the real
FP2_none master CSV, not this script) confirmed Q_diff is a trustworthy,
near-continuous proxy for the same left/right history axis word_l2/l3
encode: Q_diff predicts actual choice well (AUC 0.87-0.94 per mouse) and
word_l2 categories separate almost perfectly by Q_diff (categories whose
older trial was left-rewarded cluster at very negative Q_diff, right-
rewarded at very positive, everything else near zero) -- see config/params.
py's MODEL3_QDIFF_FORMULA comment. It's compared against word_l2 (not a
replacement for it) specifically to see whether a 3-parameter,
computational-model-grounded encoding can match a 15-parameter descriptive
one. Layering the rest of RPE/Q-values/behavioral state (per the original
EXPANDED_TIME_RESOLVED_GLM_FORMULA) onto whichever structure wins is still
out of scope here.

Both model families in the pipeline are fit and compared:
  - the time-resolved encoding GLM (models/glm_encoding.py): one OLS per PETH
    timepoint, cross-validated R^2 via cross_validate_window_glm on the
    decision-window summary amplitude (see that function's docstring for why
    a per-timepoint model needs a collapsed scalar target for CV).
  - the FIR/ridge deconvolution model (models/fir_glm.py): continuous-trace
    impulse regression, using its existing grouped (GroupShuffleSplit)
    cross-validated R^2 directly.

Model 1b and 2c add a side_out event on top of side_in. The FIR framework
combines both events in one design matrix natively (build_multi_event_mask_
and_groups). The time-resolved encoding GLM cannot: it is a per-timepoint
fit anchored to one event's PETH window, with no shared time axis for two
events to sit on together. So for the encoding-GLM side of 1b/2c, the same
formula is fit twice -- align_event="side_in" and "side_out" -- and reported
as two separate rows/plots rather than one combined number. This is the
practical ceiling of that framework; only FIR truly combines the events.

Comparison is per-mouse (matching rpe_analysis_stats.py's convention of
never making a claim pooled across animals), on the FP2_none cohort (the
same "FP2_none" entry already defined in run_expanded_glm_analysis.COHORTS).

Sanity check built into the design: Model 1 (Z ~ Reward + Port + Port:Reward)
and Model 2b's 1-letter word (Z ~ C(word_l1)) are the same model in
different codings -- word_l1's 4 levels {R,r,L,l} are exactly the 4 cells of
the Port x Reward design -- so their R^2 should match to floating-point
precision. See run_model_series_comparison.py's own inline check in main().

Usage:
    python run_model_series_comparison.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from behavior.word_encoding import add_reward_seq_2
from config.params import (
    FINAL_SAMPLE_FREQ_HZ,
    MODEL1_FORMULA,
    MODEL2_2BIT_FORMULA,
    MODEL2_3BIT_FORMULA,
    MODEL2B_WORD_FORMULAS,
    MODEL3_QDIFF_FORMULA,
    MODEL3_QDIFF_N_BINS,
    MODEL3B_FORMULA,
    MODEL3C_FORMULA,
    MODEL3D_FORMULA,
    PETH_POST_SEC,
    PETH_PRE_SEC,
)
from io_utils.raw_loader import parse_session_id
from models.fir_glm import (
    DEFAULT_ALPHAS,
    DEFAULT_LAG_SECONDS,
    DEFAULT_N_SPLITS,
    DEFAULT_TEST_SIZE,
    build_event_impulses,
    build_multi_event_mask_and_groups,
    build_shifted_design_matrix,
    build_task_mask_and_groups,
    fit_fir_glm,
    reshape_kernels,
)
from models.glm_encoding import cross_validate_window_glm, fit_time_resolved_glm
from pipeline import extract_event_peth, run_session
from run_expanded_glm_analysis import COHORTS, session_hemisphere_lookup
from run_manifest import write_run_manifest
from viz.fir_plots import plot_fir_kernels
from viz.glm_plots import plot_glm_coefficients

COHORT_LABEL = "FP2_none"
OUT_DIR = Path("outputs_fixed/model_series_comparison/results")
FIG_DIR = Path("figures_fixed_model_series")

# Order used for pairwise Wilcoxon comparisons between successive complexity
# levels -- restricted to the side_in-only models, where "more complex"
# strictly nests the simpler model's information (see MODEL1_FORMULA's
# comment on word_l1 in config/params.py). 1b/2c (multi-event) aren't part of
# a single nested ladder, so they're reported but not chained here.
LADDER_ORDER = ["1_main_effects", "2_2bit_seq", "2_3bit_seq", "2b_word_l1", "2b_word_l2", "2b_word_l3"]


# --- FIR impulse builders for each model (see models/fir_glm.py's extended
# build_event_impulses -- these are all thin, model-specific configurations
# of it, not new impulse-building logic) -----------------------------------

def _model1_fir_impulses(trial_table, n_samples, group_values=None):
    port = np.where(trial_table["chose_right"].to_numpy(), 1.0, -1.0)
    reward = trial_table["was_rewarded"].astype(float).to_numpy()
    return build_event_impulses(
        trial_table, n_samples, group_col=None, include_baseline=True,
        parametric_specs=[("Reward", reward), ("Port", port), ("PortxReward", port * reward)],
    )


def _model1b_fir_impulses(trial_table, n_samples, group_values=None):
    impulses = _model1_fir_impulses(trial_table, n_samples)
    impulses.update(build_event_impulses(
        trial_table, n_samples, group_col=None, include_baseline=True, parametric_specs=[],
        event_col="photometry_side_out_index", event_name="side_out",
    ))
    return impulses


def _make_model2_fir_builder(group_col):
    def _builder(trial_table, n_samples, group_values=None):
        port = np.where(trial_table["chose_right"].to_numpy(), 1.0, -1.0)
        return build_event_impulses(
            trial_table, n_samples, group_col=group_col, group_values=group_values,
            include_baseline=False, parametric_specs=[("Port", port)],
        )
    return _builder


def _make_model2b_fir_builder(word_col):
    def _builder(trial_table, n_samples, group_values=None):
        return build_event_impulses(
            trial_table, n_samples, group_col=word_col, group_values=group_values,
            include_baseline=False, parametric_specs=[],
        )
    return _builder


def _make_model2c_fir_builder(word_col):
    def _builder(trial_table, n_samples, group_values=None):
        impulses = build_event_impulses(
            trial_table, n_samples, group_col=word_col, group_values=group_values,
            include_baseline=False, parametric_specs=[],
        )
        # was_rewarded is always exactly {0.0, 1.0} -- fixed group_values,
        # no cross-session vocabulary pass needed for this one.
        impulses.update(build_event_impulses(
            trial_table, n_samples, group_col="was_rewarded", group_values=[0.0, 1.0],
            include_baseline=False, parametric_specs=[],
            event_col="photometry_side_out_index", event_name="side_out",
        ))
        return impulses
    return _builder


def _model3_fir_impulses(trial_table, n_samples, group_values=None):
    rpe = trial_table["RPE"].astype(float).to_numpy()
    port = np.where(trial_table["chose_right"].to_numpy(), 1.0, -1.0)
    return build_event_impulses(
        trial_table, n_samples, group_col="Q_diff_bin", group_values=group_values,
        include_baseline=False, parametric_specs=[("RPE", rpe), ("Port", port)],
    )


def _model3b_fir_impulses(trial_table, n_samples, group_values=None):
    reward = trial_table["was_rewarded"].astype(float).to_numpy()
    port = np.where(trial_table["chose_right"].to_numpy(), 1.0, -1.0)
    return build_event_impulses(
        trial_table, n_samples, group_col="Q_diff_bin", group_values=group_values,
        include_baseline=False, parametric_specs=[("Reward", reward), ("Port", port)],
    )


def _model3c_fir_impulses(trial_table, n_samples, group_values=None):
    reward = trial_table["was_rewarded"].astype(float).to_numpy()
    q_chosen = trial_table["Q_chosen"].astype(float).to_numpy()
    port = np.where(trial_table["chose_right"].to_numpy(), 1.0, -1.0)
    return build_event_impulses(
        trial_table, n_samples, group_col="Q_diff_bin", group_values=group_values,
        include_baseline=False, parametric_specs=[("Reward", reward), ("Qchosen", q_chosen), ("Port", port)],
    )


def _model3d_fir_impulses(trial_table, n_samples, group_values=None):
    reward = trial_table["was_rewarded"].astype(float).to_numpy()
    q_chosen = trial_table["Q_chosen"].astype(float).to_numpy()
    rpe_abs = np.abs(reward - q_chosen)  # exact identity, see MODEL3D_FORMULA's comment
    port = np.where(trial_table["chose_right"].to_numpy(), 1.0, -1.0)
    return build_event_impulses(
        trial_table, n_samples, group_col="Q_diff_bin", group_values=group_values,
        include_baseline=False,
        parametric_specs=[("Reward", reward), ("Qchosen", q_chosen), ("RPEabs", rpe_abs), ("Port", port)],
    )


def _qdiff_bin_edges(q_diff_values, n_bins=MODEL3_QDIFF_N_BINS):
    """Quantile bin edges for Q_diff, computed once (per mouse in
    run_comparison, pooled across all mice in plot_pooled) and then applied
    identically to every table view for that same scope (side_in, side_out,
    per-session FIR trial tables) via _add_qdiff_bins, so a given trial's bin
    identity doesn't depend on which table it happens to be sliced from.
    Outer edges are opened to +/-inf so a value at the extreme of one view
    still falls in-bin when applied to another view's (slightly different)
    range.
    """
    values = np.asarray(q_diff_values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < n_bins:
        return None  # too few valid Q_diff values (Q-learning fit skipped/failed) to bin at all
    _, edges = pd.qcut(values, q=n_bins, retbins=True, duplicates="drop")
    edges = np.asarray(edges, dtype=float)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _add_qdiff_bins(trial_table, edges):
    """edges=None (see _qdiff_bin_edges) -> Q_diff_bin is all-NaN, so Model 3
    just fails gracefully (like any other missing predictor) for that scope
    rather than raising here.

    Bin ids are cast to string labels ("0".."4", NaN preserved as NaN, not
    the literal string "nan") rather than left as an integer/float column --
    fit_time_resolved_glm's standardize=True step z-scores every NUMERIC
    predictor column before the formula's C(...) wraps it, which (since
    z-scoring is a per-column bijection) leaves the row-grouping and the fit
    itself unchanged but relabels each bin with an unreadable float instead
    of a clean bin number. word_l1/l2/l3 and reward_seq_2/3 are string dtype
    for the exact same reason (see _build_predictor_frame's _get_categorical).
    """
    trial_table = trial_table.copy()
    if edges is None:
        trial_table["Q_diff_bin"] = np.nan
    else:
        bin_idx = pd.cut(trial_table["Q_diff"], bins=edges, labels=False)
        trial_table["Q_diff_bin"] = bin_idx.map(lambda v: str(int(v)) if pd.notna(v) else np.nan)
    return trial_table


SIDE_IN = ("photometry_side_in_index",)
SIDE_IN_OUT = ("photometry_side_in_index", "photometry_side_out_index")

MODEL_SPECS = [
    dict(name="1_main_effects", encoding_formula=MODEL1_FORMULA, encoding_events=("side_in",),
         fir_builder=_model1_fir_impulses, fir_events=SIDE_IN, group_col=None),
    dict(name="1b_plus_side_out", encoding_formula=MODEL1_FORMULA, encoding_events=("side_in", "side_out"),
         fir_builder=_model1b_fir_impulses, fir_events=SIDE_IN_OUT, group_col=None),
    dict(name="2_2bit_seq", encoding_formula=MODEL2_2BIT_FORMULA, encoding_events=("side_in",),
         fir_builder=_make_model2_fir_builder("reward_seq_2"), fir_events=SIDE_IN, group_col="reward_seq_2"),
    dict(name="2_3bit_seq", encoding_formula=MODEL2_3BIT_FORMULA, encoding_events=("side_in",),
         fir_builder=_make_model2_fir_builder("reward_seq_3"), fir_events=SIDE_IN, group_col="reward_seq_3"),
    dict(name="2b_word_l1", encoding_formula=MODEL2B_WORD_FORMULAS[1], encoding_events=("side_in",),
         fir_builder=_make_model2b_fir_builder("word_l1"), fir_events=SIDE_IN, group_col="word_l1"),
    dict(name="2b_word_l2", encoding_formula=MODEL2B_WORD_FORMULAS[2], encoding_events=("side_in",),
         fir_builder=_make_model2b_fir_builder("word_l2"), fir_events=SIDE_IN, group_col="word_l2"),
    dict(name="2b_word_l3", encoding_formula=MODEL2B_WORD_FORMULAS[3], encoding_events=("side_in",),
         fir_builder=_make_model2b_fir_builder("word_l3"), fir_events=SIDE_IN, group_col="word_l3"),
    dict(name="2c_word_l1_side_out", encoding_formula=MODEL2B_WORD_FORMULAS[1], encoding_events=("side_in", "side_out"),
         fir_builder=_make_model2c_fir_builder("word_l1"), fir_events=SIDE_IN_OUT, group_col="word_l1"),
    dict(name="2c_word_l2_side_out", encoding_formula=MODEL2B_WORD_FORMULAS[2], encoding_events=("side_in", "side_out"),
         fir_builder=_make_model2c_fir_builder("word_l2"), fir_events=SIDE_IN_OUT, group_col="word_l2"),
    dict(name="2c_word_l3_side_out", encoding_formula=MODEL2B_WORD_FORMULAS[3], encoding_events=("side_in", "side_out"),
         fir_builder=_make_model2c_fir_builder("word_l3"), fir_events=SIDE_IN_OUT, group_col="word_l3"),
    dict(name="3_qdiff_rpe", encoding_formula=MODEL3_QDIFF_FORMULA, encoding_events=("side_in",),
         fir_builder=_model3_fir_impulses, fir_events=SIDE_IN, group_col="Q_diff_bin"),
    dict(name="3b_reward_qdiff", encoding_formula=MODEL3B_FORMULA, encoding_events=("side_in",),
         fir_builder=_model3b_fir_impulses, fir_events=SIDE_IN, group_col="Q_diff_bin"),
    dict(name="3c_reward_qchosen_qdiff", encoding_formula=MODEL3C_FORMULA, encoding_events=("side_in",),
         fir_builder=_model3c_fir_impulses, fir_events=SIDE_IN, group_col="Q_diff_bin"),
    dict(name="3d_reward_qchosen_rpeabs_qdiff", encoding_formula=MODEL3D_FORMULA, encoding_events=("side_in",),
         fir_builder=_model3d_fir_impulses, fir_events=SIDE_IN, group_col="Q_diff_bin"),
]

# Models 3/3b/3c/3d aren't part of the nested complexity ladder (each is a
# different encoding strategy, not strictly more/less complex than any one
# rung) -- they get explicit head-to-head comparisons against word_l2
# instead of a slot in LADDER_ORDER.
HEAD_TO_HEAD_COMPARISONS = [
    ("2b_word_l2", "3_qdiff_rpe"),
    ("2b_word_l2", "3b_reward_qdiff"),
    ("2b_word_l2", "3c_reward_qchosen_qdiff"),
    ("2b_word_l2", "3d_reward_qchosen_rpeabs_qdiff"),
    ("3c_reward_qchosen_qdiff", "3d_reward_qchosen_rpeabs_qdiff"),
]


def _load_all_sessions(session_dirs, hemisphere_for_session, truncate_at_side_out=False, side_out_margin_s=0.0):
    """Run pipeline.run_session exactly ONCE per session -- not once per
    align_event and once more for FIR -- and derive everything else (the
    side_out-aligned PETH, the FIR continuous trace) from that single call's
    already-computed full_trial_table/dff/zscore. The expensive stages (raw
    load, demod, dff/zscore, trial table enrichment, alignment, bandit-state
    fit) are the dominant per-session cost; only the cheap PETH-extraction
    step (pipeline.extract_event_peth) actually differs by align_event, so
    calling it a second time on the same session's already-computed
    full_trial_table/dff (rather than re-running run_session with
    align_event="side_out") gets the second alignment for nearly free.

    Soft-fails/skips a session that errors at either step, same convention
    as models.fir_glm.build_pooled_fir_dataset.

    truncate_at_side_out/side_out_margin_s : forwarded to run_session's
    side_in-aligned pass only (see pipeline.extract_event_peth's docstring
    and run_model_series_comparison_sm_red_l.py's identically-named params)
    -- opt-in, default False. The side_out-aligned pass below is unaffected
    (truncating relative to side_out isn't meaningful, and run_session's own
    check disallows it).

    Returns a list of per-session dicts (mouse, date, trial_table_in,
    zscore_windows_in, peth_time_in, trial_table_out, zscore_windows_out,
    peth_time_out, continuous_trial_table, continuous_zscore).
    """
    pre_samples = int(round(PETH_PRE_SEC * FINAL_SAMPLE_FREQ_HZ))
    post_samples = int(round(PETH_POST_SEC * FINAL_SAMPLE_FREQ_HZ))

    sessions = []
    n_failed = 0
    for session_dir in session_dirs:
        session_dir = Path(session_dir)
        mouse, date = parse_session_id(session_dir)
        hemisphere = hemisphere_for_session[session_dir]
        try:
            result = run_session(session_dir, hemisphere=hemisphere, align_event="side_in",
                                  truncate_at_side_out=truncate_at_side_out,
                                  side_out_margin_s=side_out_margin_s)
        except Exception as exc:
            print(f"WARNING: skipping session {session_dir} ({mouse} {date}): {exc}")
            n_failed += 1
            continue
        try:
            _, peth_trial_table_out, _, zscore_windows_out, _ = extract_event_peth(
                result["full_trial_table"], result["dff"], "side_out",
                pre_samples, post_samples, result["peth_time"],
            )
        except Exception as exc:
            print(f"WARNING: side_out PETH extraction failed for {session_dir} ({mouse} {date}): {exc}")
            n_failed += 1
            continue

        sessions.append(dict(
            mouse=mouse, date=date,
            trial_table_in=add_reward_seq_2(result["peth_trial_table"]),
            zscore_windows_in=result["all_zscore_windows"],
            peth_time_in=result["peth_time"],
            trial_table_out=add_reward_seq_2(peth_trial_table_out),
            zscore_windows_out=zscore_windows_out,
            peth_time_out=result["peth_time"],
            continuous_trial_table=add_reward_seq_2(result["full_trial_table"]),
            continuous_zscore=np.asarray(result["zscore"], dtype=float),
        ))
    print(f"Loaded {len(sessions)} session(s) via a single run_session() pass each "
          f"(skipped {n_failed})")
    return sessions


def _pool_sessions(sessions, which):
    """In-memory pooling of trial_table_{which}/zscore_windows_{which} across
    sessions, tagged with mouse/date -- the same pooling contract as
    models.glm_data.build_pooled_glm_dataset, just operating on the
    already-loaded per-session dicts from _load_all_sessions instead of
    re-running pipeline.run_session per align_event.
    """
    peth_time = None
    table_frames, window_frames = [], []
    for s in sessions:
        pt = s[f"peth_time_{which}"]
        if peth_time is None:
            peth_time = pt
        elif not np.array_equal(peth_time, pt):
            raise ValueError(
                f"{s['mouse']} {s['date']} has a different peth_time grid than earlier sessions "
                f"in this '{which}' pool -- pooling requires an identical PETH_PRE_SEC/PETH_POST_SEC/"
                "FINAL_SAMPLE_FREQ_HZ config across the whole cohort"
            )
        session_table = s[f"trial_table_{which}"].copy()
        session_table["mouse"] = s["mouse"]
        session_table["date"] = s["date"]
        table_frames.append(session_table)
        window_frames.append(s[f"zscore_windows_{which}"])
    zscore_windows = np.vstack(window_frames)
    trial_table = pd.concat(table_frames, ignore_index=True)
    return peth_time, zscore_windows, trial_table


def _pooled_fir_fit(sessions, fir_builder, event_cols, group_col=None,
                     n_lags_seconds=DEFAULT_LAG_SECONDS, n_splits=DEFAULT_N_SPLITS,
                     test_size=DEFAULT_TEST_SIZE, alphas=DEFAULT_ALPHAS, random_state=0):
    """Pool a list of (trial_table, continuous_signal) sessions and fit the
    FIR model, generalizing models.fir_glm.build_pooled_fir_dataset to an
    arbitrary model-specific impulse builder (rather than a single group_col)
    and, when event_cols has more than one entry, a combined multi-event
    task mask (build_multi_event_mask_and_groups).

    If group_col is given, first collects the union of that column's values
    across all sessions (same two-pass rationale as build_pooled_fir_dataset)
    so every session's design matrix has identical columns even if a rare
    pattern (e.g. an 8-way 3-bit sequence) doesn't occur in every session.
    """
    n_lags = int(round(n_lags_seconds * FINAL_SAMPLE_FREQ_HZ))

    group_values = None
    if group_col is not None:
        all_groups = set()
        for trial_table, _ in sessions:
            all_groups.update(trial_table[group_col].dropna().unique())
        group_values = sorted(all_groups)

    y_parts, phi_parts, group_parts = [], [], []
    column_names = None
    group_offset = 0
    for trial_table, continuous_signal in sessions:
        n_samples = len(continuous_signal)
        impulses = fir_builder(trial_table, n_samples, group_values=group_values)
        Phi, cols = build_shifted_design_matrix(impulses, n_lags)
        if column_names is None:
            column_names = cols
        if len(event_cols) > 1:
            mask, groups = build_multi_event_mask_and_groups(trial_table, n_samples, n_lags, event_cols)
        else:
            mask, groups = build_task_mask_and_groups(trial_table, n_samples, n_lags, event_col=event_cols[0])
        y_parts.append(continuous_signal[mask])
        phi_parts.append(Phi[mask])
        group_parts.append(groups[mask] + group_offset)
        group_offset += len(trial_table)

    y = np.concatenate(y_parts)
    Phi = np.vstack(phi_parts)
    groups = np.concatenate(group_parts)
    mask_all = np.ones(len(y), dtype=bool)

    n_groups = len(np.unique(groups))
    n_splits_eff = min(n_splits, max(2, n_groups // 10))
    model, cv_results = fit_fir_glm(
        y, Phi, mask_all, groups, n_splits=n_splits_eff, test_size=test_size,
        alphas=alphas, random_state=random_state,
    )
    kernels, lag_time_s = reshape_kernels(model.coef_, column_names, n_lags)
    fold_kernels, _ = reshape_kernels(cv_results["fold_betas"], column_names, n_lags)
    return dict(model=model, cv_results=cv_results, kernels=kernels, fold_kernels=fold_kernels,
                lag_time_s=lag_time_s, column_names=column_names, n_lags=n_lags)


def _slice_mouse(trial_table, zscore_windows, mouse):
    is_mouse = (trial_table["mouse"] == mouse).to_numpy()
    sub_table = trial_table.loc[is_mouse].reset_index(drop=True)
    sub_windows = zscore_windows[is_mouse]
    return sub_table, sub_windows


def _wilcoxon_pair(wide, a, b):
    """Wilcoxon signed-rank test on the per-mouse delta R^2 (wide[b] -
    wide[a]) vs 0, same pattern as rpe_analysis_stats.py's
    analysis_2_signed_vs_unsigned. None if either model is missing from
    `wide` or fewer than 2 mice have both values.
    """
    if a not in wide.columns or b not in wide.columns:
        return None
    delta = (wide[b] - wide[a]).dropna()
    if len(delta) < 2:
        return None
    try:
        stat, p = wilcoxon(delta)
    except ValueError:
        return None
    return dict(
        wilcoxon_stat=float(stat), wilcoxon_p=float(p),
        n_positive=int((delta > 0).sum()), n_mice=int(len(delta)),
        mean_delta_r2=float(delta.mean()),
    )


def _pairwise_wilcoxon(encoding_df, fir_df):
    """Wilcoxon comparisons between successive LADDER_ORDER models, plus each
    explicit HEAD_TO_HEAD_COMPARISONS pair (Models 3/3b/3c vs word_l2 -- not
    part of the nested ladder, see that list's comment). fir_df may be
    None/empty (FIR skipped via run_comparison's include_fir=False) -- that
    half of `stats` is just left empty rather than erroring.
    """
    stats = {"encoding_glm": {}, "fir_glm": {}}
    enc_side_in = encoding_df[encoding_df["event"] == "side_in"]
    enc_wide = enc_side_in.pivot(index="mouse", columns="model", values="r2_mean")
    fir_wide = (
        fir_df.pivot(index="mouse", columns="model", values="r2_mean")
        if fir_df is not None and len(fir_df) else pd.DataFrame()
    )

    pairs = list(zip(LADDER_ORDER[:-1], LADDER_ORDER[1:])) + HEAD_TO_HEAD_COMPARISONS
    for a, b in pairs:
        for wide, bucket in [(enc_wide, stats["encoding_glm"]), (fir_wide, stats["fir_glm"])]:
            result = _wilcoxon_pair(wide, a, b)
            if result is not None:
                bucket[f"{a}_vs_{b}"] = result
    return stats


def run_comparison(trial_table_in, zscore_in, peth_time_in,
                    trial_table_out, zscore_out, peth_time_out,
                    fir_sessions_by_mouse, out_dir=OUT_DIR, include_fir=True, model_names=None):
    """include_fir=False skips the FIR/RidgeCV half entirely -- that's the
    expensive part (10-fold GroupShuffleSplit CV, each fold a RidgeCV over
    DEFAULT_ALPHAS, on a mouse's full pooled continuous trace), useful for a
    fast first pass on the (much cheaper) encoding-GLM comparison alone.
    fir_glm_model_comparison.csv is only written when include_fir=True.

    model_names : optional subset of MODEL_SPECS['name'] values to run (e.g.
        just the 2-3 models worth paying FIR's cost for) -- None runs all of
        MODEL_SPECS, matching prior behavior. Pass a distinct out_dir when
        using a subset so it doesn't overwrite a prior full-ladder run's CSVs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mice = sorted(trial_table_in["mouse"].unique())
    encoding_rows, fir_rows = [], []
    specs = MODEL_SPECS if model_names is None else [s for s in MODEL_SPECS if s["name"] in model_names]

    for mouse in mice:
        print(f"\n=== {mouse} ===")
        sub_table_in, sub_windows_in = _slice_mouse(trial_table_in, zscore_in, mouse)
        sub_table_out, sub_windows_out = _slice_mouse(trial_table_out, zscore_out, mouse)
        mouse_fir_sessions = fir_sessions_by_mouse.get(mouse, [])

        # Q_diff_bin edges computed once from this mouse's own side_in trial
        # set, then applied identically everywhere else for this mouse (see
        # _qdiff_bin_edges) -- Model 3 is the only model that references
        # Q_diff_bin, so this is a no-op for every other model.
        qdiff_edges = _qdiff_bin_edges(sub_table_in["Q_diff"])
        sub_table_in = _add_qdiff_bins(sub_table_in, qdiff_edges)
        sub_table_out = _add_qdiff_bins(sub_table_out, qdiff_edges)
        mouse_fir_sessions = [(_add_qdiff_bins(tt, qdiff_edges), z) for tt, z in mouse_fir_sessions]

        for spec in specs:
            for event in spec["encoding_events"]:
                sub_table, sub_windows, peth_time = (
                    (sub_table_in, sub_windows_in, peth_time_in) if event == "side_in"
                    else (sub_table_out, sub_windows_out, peth_time_out)
                )
                rhs = spec["encoding_formula"].split("~", 1)[1].strip()
                try:
                    cv = cross_validate_window_glm(sub_windows, peth_time, sub_table, rhs)
                    encoding_rows.append(dict(mouse=mouse, model=spec["name"], event=event,
                                               r2_mean=cv["r2_mean"], r2_std=cv["r2_std"], n_trials=cv["n_trials"]))
                except Exception as exc:
                    print(f"WARNING: encoding GLM CV failed for {mouse}/{spec['name']}/{event}: {exc}")
                    encoding_rows.append(dict(mouse=mouse, model=spec["name"], event=event,
                                               r2_mean=np.nan, r2_std=np.nan, n_trials=np.nan))

            if not include_fir:
                continue
            if not mouse_fir_sessions:
                fir_rows.append(dict(mouse=mouse, model=spec["name"], r2_mean=np.nan, r2_std=np.nan, n_samples_fit=0))
                continue
            try:
                fit = _pooled_fir_fit(mouse_fir_sessions, spec["fir_builder"], spec["fir_events"],
                                       group_col=spec["group_col"])
                fir_rows.append(dict(mouse=mouse, model=spec["name"],
                                      r2_mean=fit["cv_results"]["r2_mean"],
                                      r2_std=fit["cv_results"]["r2_std"],
                                      n_samples_fit=fit["cv_results"]["n_samples_fit"]))
            except Exception as exc:
                print(f"WARNING: FIR CV failed for {mouse}/{spec['name']}: {exc}")
                fir_rows.append(dict(mouse=mouse, model=spec["name"], r2_mean=np.nan, r2_std=np.nan, n_samples_fit=0))

    encoding_df = pd.DataFrame(encoding_rows)
    encoding_df.to_csv(out_dir / "encoding_glm_model_comparison.csv", index=False)

    fir_df = pd.DataFrame(fir_rows) if include_fir else None
    if include_fir:
        fir_df.to_csv(out_dir / "fir_glm_model_comparison.csv", index=False)

    stats = _pairwise_wilcoxon(encoding_df, fir_df)
    with open(out_dir / "summary_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved encoding_glm_model_comparison.csv"
          + (", fir_glm_model_comparison.csv" if include_fir else " (FIR skipped)")
          + f", summary_stats.json to {out_dir}")
    return encoding_df, fir_df, stats


def plot_pooled(trial_table_in, zscore_in, peth_time_in,
                 trial_table_out, zscore_out, peth_time_out,
                 fir_sessions_by_mouse, fig_dir=FIG_DIR, include_fir=True, model_names=None,
                 min_retained_frac=None, cohort_label="FP2_none"):
    """Coefficient-trajectory / kernel plots from ALL of one cohort's mice
    pooled together (not per-mouse -- purely to visualize response shapes;
    the per-mouse CV R^2 tables from run_comparison are the actual
    comparison). include_fir=False skips the (expensive) FIR kernel
    plots/fits. model_names restricts to a subset of MODEL_SPECS (see
    run_comparison).

    min_retained_frac : forwarded to fit_time_resolved_glm -- opt-in (None
    disables, matching that function's own default), pass e.g. 0.5 for a
    caller pooling truncate_at_side_out windows, so the plotted/reported
    curve NaNs out (stops) once fewer than that fraction of the original
    trial count remains, instead of extending a visually continuous line
    across a shrinking, self-selected long-dwelling subsample.

    cohort_label : only used for the FIR kernel plots' title prefix (defaults
    to this module's original "FP2_none" cohort so existing callers are
    unaffected) -- purely cosmetic, doesn't touch any fitted number, but
    matters for figure provenance once this function runs against other
    cohorts too.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    all_fir_sessions = [s for sessions in fir_sessions_by_mouse.values() for s in sessions]
    specs = MODEL_SPECS if model_names is None else [s for s in MODEL_SPECS if s["name"] in model_names]

    # Q_diff_bin edges computed once across ALL pooled mice here (unlike
    # run_comparison's per-mouse edges) -- this function's whole premise is
    # pooled, cross-mouse visualization, so its own binning should match that.
    qdiff_edges = _qdiff_bin_edges(trial_table_in["Q_diff"])
    trial_table_in = _add_qdiff_bins(trial_table_in, qdiff_edges)
    trial_table_out = _add_qdiff_bins(trial_table_out, qdiff_edges)
    all_fir_sessions = [(_add_qdiff_bins(tt, qdiff_edges), z) for tt, z in all_fir_sessions]

    for spec in specs:
        for event in spec["encoding_events"]:
            sub_table, sub_windows, peth_time = (
                (trial_table_in, zscore_in, peth_time_in) if event == "side_in"
                else (trial_table_out, zscore_out, peth_time_out)
            )
            try:
                beta_df = fit_time_resolved_glm(sub_windows, peth_time, sub_table, formula=spec["encoding_formula"],
                                                 min_retained_frac=min_retained_frac)
                fig = plot_glm_coefficients(peth_time, beta_df, spec["encoding_formula"], align_event=event)
                stem = fig_dir / f"encoding_{spec['name']}_{event}"
                fig.savefig(stem.with_suffix(".png"), dpi=150)
                fig.savefig(stem.with_suffix(".svg"))
                beta_df.to_csv(fig_dir / f"encoding_{spec['name']}_{event}_beta_df.csv")
            except Exception as exc:
                print(f"WARNING: plotting failed for encoding {spec['name']}/{event}: {exc}")

        if not include_fir:
            continue
        try:
            fit = _pooled_fir_fit(all_fir_sessions, spec["fir_builder"], spec["fir_events"], group_col=spec["group_col"])
            fig = plot_fir_kernels(fit["fold_kernels"], fit["lag_time_s"], title_prefix=f"{cohort_label} {spec['name']}")
            stem = fig_dir / f"fir_{spec['name']}_kernels"
            fig.savefig(stem.with_suffix(".png"), dpi=150)
            fig.savefig(stem.with_suffix(".svg"))
        except Exception as exc:
            print(f"WARNING: plotting failed for FIR {spec['name']}: {exc}")
    print(f"Saved pooled coefficient plots{' + FIR kernel plots' if include_fir else ' (FIR skipped)'} to {fig_dir}")


def _save_pooled_arrays(out_dir, peth_time_in, zscore_in, trial_table_in,
                         peth_time_out, zscore_out, trial_table_out):
    """Cache the pooled (post-session-loop) arrays plot_pooled/run_comparison
    consume, so a future fit-only or plot-only fix (e.g. a new
    min_resid_dof/min_retained_frac guard) never again requires repeating a
    full raw-session reload just to re-derive them -- this exact rework
    already had to be paid for twice on the SM side of this project. Shared
    here (rather than duplicated per cohort script) since it's fully generic;
    run_model_series_comparison_sm_red_l.py imports this instead of keeping
    its own copy.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "pooled_zscore_windows.npz",
              zscore_in=zscore_in, zscore_out=zscore_out,
              peth_time_in=peth_time_in, peth_time_out=peth_time_out)
    trial_table_in.to_parquet(out_dir / "pooled_trial_table_in.parquet")
    trial_table_out.to_parquet(out_dir / "pooled_trial_table_out.parquet")
    print(f"Cached pooled arrays to {out_dir} (pooled_zscore_windows.npz, "
          "pooled_trial_table_in/out.parquet)")


def main(cohort_label=COHORT_LABEL, include_fir=True, model_names=None, out_dir=OUT_DIR, fig_dir=FIG_DIR,
         truncate_at_side_out=False, side_out_margin_s=0.0, min_retained_frac=None):
    """include_fir=False runs only the (cheap) time-resolved encoding-GLM
    comparison, skipping FIR/RidgeCV entirely -- FIR's 10-fold GroupShuffleSplit
    CV, each fold a RidgeCV over DEFAULT_ALPHAS, on a mouse's full pooled
    continuous trace, is dramatically more expensive than the encoding-GLM
    side and can be run separately (e.g. `main(include_fir=True)` again
    later, or a dedicated FIR-only pass) once a faster path is worth building.

    cohort_label : which run_expanded_glm_analysis.COHORTS entry to run --
        defaults to this module's original COHORT_LABEL ("FP2_none") so a
        zero-arg main() call is unchanged from prior behavior. Pass e.g.
        "FP1_none"/"FP1_dcz"/"FP1_retrained_none"/"FP2_dcz" to run this same
        ladder against a different cohort (this function used to be
        FP2_none-only via a hardcoded module-level lookup).

    model_names : optional subset of MODEL_SPECS['name'] values to run (e.g.
        just the 2-3 models worth paying FIR's cost for). Pass a distinct
        out_dir/fig_dir alongside a subset so it doesn't overwrite a prior
        full-ladder run's output files.

    truncate_at_side_out/side_out_margin_s : opt-in, default False -- see
        run_model_series_comparison_sm_red_l.py's identically-named params
        and pipeline.extract_event_peth's docstring. NaNs out each trial's
        own post-side_out samples in the side_in-aligned PETH windows before
        fitting.

    min_retained_frac : forwarded to plot_pooled -> fit_time_resolved_glm.
        None (default) leaves this at truncate_at_side_out's own default:
        0.5 when truncate_at_side_out=True, None (disabled) otherwise --
        same auto-default convention as main_red_l. Pass explicitly to
        override either default.
    """
    cohort_entry = next((c for c in COHORTS if c[0] == cohort_label), None)
    if cohort_entry is None:
        raise ValueError(f"Cohort {cohort_label!r} not found in run_expanded_glm_analysis.COHORTS")
    _, master_csv, qc_report_csv, exclude_pairs = cohort_entry
    session_dirs, hemisphere_for_session = session_hemisphere_lookup(master_csv, qc_report_csv, exclude_pairs)
    print(f"{cohort_label}: {len(session_dirs)} sessions")

    print("\nLoading sessions (one pipeline.run_session() pass each, reused for side_in, "
          "side_out, and FIR)...")
    sessions = _load_all_sessions(session_dirs, hemisphere_for_session,
                                   truncate_at_side_out=truncate_at_side_out,
                                   side_out_margin_s=side_out_margin_s)

    peth_time_in, zscore_in, trial_table_in = _pool_sessions(sessions, "in")
    peth_time_out, zscore_out, trial_table_out = _pool_sessions(sessions, "out")
    fir_sessions_by_mouse = {}
    for s in sessions:
        fir_sessions_by_mouse.setdefault(s["mouse"], []).append(
            (s["continuous_trial_table"], s["continuous_zscore"])
        )

    _save_pooled_arrays(out_dir, peth_time_in, zscore_in, trial_table_in,
                        peth_time_out, zscore_out, trial_table_out)

    if min_retained_frac is None and truncate_at_side_out:
        min_retained_frac = 0.5

    write_run_manifest(
        out_dir,
        params=dict(
            cohort_label=cohort_label, include_fir=include_fir, model_names=model_names,
            truncate_at_side_out=truncate_at_side_out, side_out_margin_s=side_out_margin_s,
            min_retained_frac=min_retained_frac, n_sessions=len(session_dirs),
        ),
        script="run_model_series_comparison.main",
    )

    encoding_df, fir_df, stats = run_comparison(
        trial_table_in, zscore_in, peth_time_in,
        trial_table_out, zscore_out, peth_time_out,
        fir_sessions_by_mouse, out_dir=Path(out_dir), include_fir=include_fir, model_names=model_names,
    )

    plot_pooled(
        trial_table_in, zscore_in, peth_time_in,
        trial_table_out, zscore_out, peth_time_out,
        fir_sessions_by_mouse, fig_dir=Path(fig_dir), include_fir=include_fir, model_names=model_names,
        min_retained_frac=min_retained_frac, cohort_label=cohort_label,
    )

    # Sanity check: Model 1 and Model 2b's 1-letter word are the same model
    # in different codings -- their side_in R^2 should match closely.
    enc_side_in = encoding_df[encoding_df["event"] == "side_in"]
    m1 = enc_side_in[enc_side_in["model"] == "1_main_effects"].set_index("mouse")["r2_mean"]
    m2b1 = enc_side_in[enc_side_in["model"] == "2b_word_l1"].set_index("mouse")["r2_mean"]
    max_diff = (m1 - m2b1).abs().max()
    print(f"\nSanity check: max |R^2(Model 1) - R^2(Model 2b word_l1)| across mice = {max_diff:.6f} "
          "(should be ~0 -- same model, different coding)")

    return encoding_df, fir_df, stats


if __name__ == "__main__":
    main()
