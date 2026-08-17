# Is the reward-locked photometry kinetics fit measuring a real per-trial transient?

**Status:** Diagnosis complete. No changes made to `alignment/kinetics.py`, `config/params.py`, or any
pipeline output. This report and the four scripts in `validation/` are local only — not committed or
pushed (by request; push is left to the user).

**Scripts (chronological order of the investigation):**
1. `validation/diagnose_kinetics_shape.py`
2. `validation/diagnose_trial_timing.py`
3. `validation/diagnose_tailrisk.py`
4. `validation/diagnose_censored_kinetics.py`

All four are read-only against `outputs_fixed/rpe_analysis_pooled/{pooled_trial_table.parquet,peth_windows.npz}`
(26,296 trials, 11 mice, `zscore_windows` shape (26296,141), `peth_time` spans -2.16 to 5.4s). Outputs
(PNGs/CSVs) live in scratch, not in this repo — rerun the scripts to regenerate them.

---

## 1. The question

`alignment/kinetics.py::compute_onset_and_decay` fits an onset latency (time-to-half-max) and a
single-exponential decay tau to the **trial-averaged** z-scored PETH, per (mouse, reward/omission), over
`KINETICS_METRIC_WINDOW_S=(0, 3.0)`. Two of eleven mice already produced fits the pipeline's own sanity
guard rejects as implausible (tau far exceeding the fitted segment length). Before trusting any of these
numbers as a target for the SST:PV model's Phase 2 (which is meant to reproduce this kinetics readout),
the concern was: is this describing a real discrete bump-then-decay transient, or could it be:

- **(a)** an artifact of averaging temporally heterogeneous/jittered single trials into a smeared-looking
  "slow" mean,
- **(b)** a real slow signal that tracks recent win/loss history across trials (not a per-trial transient
  at all), or
- **(c)** contamination from the **next trial's own events** falling inside the fixed analysis window —
  this task has no hardware ITI (confirmed: no ITI/timeout parameter anywhere in `config/` or the task
  code), trials are animal-paced and only ~2-5s apart, so a window as long as 3.0-5.4s post-outcome can
  easily run into the next trial.

(c) was further split into two independent sub-mechanisms: contamination **amount** (what fraction of a
trial's window is physically occupied by the next trial's own signal) vs. contamination **content**
(whether the next trial's own outcome type is statistically predictable from the current trial's
win/loss-streak tercile, independent of how much of the window it occupies).

---

## 2. Stage 1 — `diagnose_kinetics_shape.py`: what does the fit actually look like?

Ran spaghetti plots (individual traces + mean + fitted exponential), per-trial peak-time histograms,
a tau-vs-subset-size sweep, and a `Rolling_Accuracy`-tercile PETH overlay, prioritizing WCL31 and WCL28
(the two mice with pipeline-rejected decay fits) plus WCL29 as a clean-fit comparison case.

**Findings:**
- The tercile-split PETH overlay is the decisive result, and **it replicates across all three mice,
  including the clean-fit case (WCL29)**: all three `Rolling_Accuracy` terciles are indistinguishable
  pre-event, share a small common bump in the first ~0-0.5s, then **diverge monotonically from ~0.5s
  onward through the entire recorded window (5.4s)** — recent-loss trials keep moving one direction,
  recent-win trials the other, neither returns to baseline within the window.
- Individual-trial peak time is very jittered (SD ≈ 0.9s, often bimodal at the window edges) in every
  mouse checked.
- The tau-vs-subset-size sweep is flat/converges to a plausible tau for WCL29, but does not converge for
  WCL31/WCL28 — consistent with the pipeline's own rejection of their full-population fits.
- Taken together at this stage: the signal looked like a small, roughly history-independent fast
  component sitting on top of a much larger slow component tracking recent win/loss history — **not** a
  clean per-trial discrete transient. This raised, rather than resolved, the concern — it did not yet
  distinguish "real slow signal" from "next-trial contamination."

## 3. Stage 2 — `diagnose_trial_timing.py`: how much of a risk is next-trial contamination, structurally?

Computed, per trial (excluding each session's last trial, which has no valid "next"), the gap from this
trial's outcome (`side_in_s` — confirmed `photometry_outcome_index == photometry_side_in_index` in
`behavior/sync.py`, i.e. reward delivery has no independently timestamped event in this rig) to (a) the
next trial's `center_in_s` (behavioral re-engagement speed) and (b) the next trial's own `side_in_s` (the
actual contamination source for a side_in-aligned window).

**Findings:**
- Median gap ≈ 2.05-2.38s depending on mouse/gap type.
- **87-89% of trials have the next trial's own outcome event falling inside the fixed 5.4s analysis
  window.** This alone establishes that contamination is not a rare edge case — it is the modal
  situation.
- Step 2 (does the gap itself depend on `Rolling_Accuracy` tercile — i.e., do animals re-engage faster
  after wins?) was run and reported to the user as this stage's terminal deliverable per their explicit
  "stop after Step 2" instruction, before any censoring/re-fit work began.

## 4. Stage 3 — `diagnose_tailrisk.py`: which windows are actually at risk, and how much?

Two separate deliverables, as requested: (Steps 1-2) left-tail risk for the **short**, already-safe
windows; (Step 4, prioritized) contamination **degree** for `REWARD_WINDOW_S`.

**Steps 1-2 (FIR / `DECISION_WINDOW_S`, both bounded at 1.0s — `DEFAULT_LAG_SECONDS=1.0` for FIR,
`DECISION_WINDOW_S=(0,1.0)`):**
- No mouse has a 5th-percentile gap under 1.0s.
- The gap<1.0s subset is under 0.1% of all trials, and is not enriched for either `Rolling_Accuracy`
  tercile (chi-square test, negligible skew).
- **Conclusion: FIR and `DECISION_WINDOW_S` are safe from next-trial contamination.** This directly
  answered the user's question about whether the 1.0s GLM/FIR decision window was at risk — it is not.

**Step 4 (`REWARD_WINDOW_S=(1.0,3.0)`, which `KINETICS_METRIC_WINDOW_S=(0,3.0)` fully contains):**
- Defined a continuous per-trial `contamination_fraction = clip((hi - gap) / span, 0, 1)` — the fraction
  of the 2s window that falls after the next trial's own outcome onset.
- **69% of trials have some non-zero contamination; the mean contamination_fraction across all trials is
  0.304** (i.e., on average about 30% of the window is, physically, the next trial's own signal).
- Pooled across all mice, contamination_fraction does **not** differ significantly by tercile. But
  **per-mouse it is real and mouse-specific, in both directions**: WCL24 shows recent-loss trials **less**
  contaminated (Cliff's delta ≈ -0.25) while WCL29/WCL31 show recent-loss trials **more** contaminated
  (delta ≈ +0.13 to +0.16).
- This was the pivotal puzzle carried into Stage 4: WCL24 shows the *same* tercile-divergence pattern as
  WCL29/WCL31 in the original diagnostic, despite having contamination skewed in the *opposite* direction.
  That single fact already falsifies "contamination amount alone explains the divergence" as a universal
  mechanism — it cannot explain WCL24 on its own. It does not yet tell us what *does* explain it, per
  mouse.
- `REWARD_WINDOW_S`'s only consumer in the codebase (`evaluate_word_outcomes`'s `mean_peak_z_reward`/
  `mean_auc_reward` in `pipeline.py`) is computed inside a per-session demo path, printed to console only,
  and never appears in `run_session`'s returned dict or in anything `batch_processor.py` persists or
  aggregates. No manuscript file was found in this repository to complete the originally-requested
  cross-check of whether any published claim depends on `REWARD_WINDOW_S` — **this remains open**; the
  user was asked to point to a manuscript file and has not yet responded.

## 5. Stage 4 — `diagnose_censored_kinetics.py`: adjudicating real signal vs. contamination, per mouse

This is the decisive stage. Rather than asking "is there contamination" (yes, established above), it asks
directly: **if you remove the contamination, does the tercile divergence survive?** — and separately,
**does the next trial's own outcome correlate with the current trial's tercile even where contamination
amount is minimal?**

### Method

- **Censoring**: for every trial, mask its z-scored window to `NaN` from `peth_time >= next_side_in_gap`
  onward (pre-event baseline is never touched). Trials with `next_side_in_gap < 0.5s` are dropped from the
  censored pool entirely (not just masked) — this is a real censoring exclusion, tracked and reported as
  `frac_excluded_short_window` per (mouse, tercile); in practice this was ≤1.4% everywhere.
- **Aggregation**: `np.nanmean` per (mouse × tercile), with `effective_n(t)` (count of non-NaN
  contributing trials at each timepoint) tracked as a **mandatory companion** to every censored mean —
  never interpreted alone.
- **Reliability breakpoints**: `t_full` (last timepoint before effective_n hits 0), `t_50`/`t_25` (last
  timepoint where effective_n/N ≥ 50%/25%).
- **Refit**: `compute_onset_and_decay` (imported unmodified from `alignment/kinetics.py`) is not NaN-aware,
  so the window is truncated to `(0, min(t_full, KINETICS_METRIC_WINDOW_S[1]))` before fitting.

### Bugs found and fixed during this stage (for transparency)

1. **Sign-orientation bug.** The first version fed raw, non-sign-flipped z-scored windows directly into
   `compute_onset_and_decay`, causing nearly every censored fit to fail with "peak_value ≤ 0 after sign
   orientation" — reward trials in this dataset are confirmed negative-going/suppressive, and the fitter
   expects a positive-going peak. Fixed by adding a per-mouse sign-orientation step (computed once across
   all of that mouse's reward trials, matching the convention already used in
   `diagnose_kinetics_shape.py`) before any fitting.
2. **Fixed-window divergence-averaging bug (caught before reporting to the user).** The initial divergence
   metric averaged high-minus-low tercile difference over a fixed 1.0-3.0s window for both censored and
   uncensored traces. Visual inspection of the WCL24/WCL29/WCL31 comparison figure showed the censored and
   uncensored curves tracking closely while `effective_n` was high, then diverging sharply and unreliably
   exactly where `effective_n` collapsed — a small-sample artifact of a shrinking, potentially
   unrepresentative pool of long-gap trials, not genuine signal. **Fixed** by restricting the divergence
   comparison to a per-mouse dynamic window `[1.0s, min(t_50_low, t_50_high)]` — i.e., only comparing
   censored vs. uncensored within the region where both compared terciles still have effective_n ≥ 50%.
   All divergence numbers below use this corrected, dynamically-bounded window, not the original fixed one.

### Step 1 result: does the tercile divergence survive censoring?

| mouse | divergence (uncensored) | divergence (censored) | shrinkage | shared coverage | well-powered window ends at |
|---|---|---|---|---|---|
| WCL23 | -2.336 | -2.407 | -3.0% (grew) | 86.8% | 2.05s |
| WCL24 | -2.017 | -2.050 | -1.6% (grew) | 86.4% | 1.78s |
| WCL25 | -1.480 | -1.409 | +4.8% | 82.0% | 2.11s |
| WCL26 | -1.753 | -1.742 | +0.6% | 86.3% | 2.27s |
| WCL27 | -1.141 | -1.247 | -9.2% (grew) | 86.5% | 1.94s |
| WCL28 | -1.640 | -1.612 | +1.7% | 86.1% | 2.16s |
| WCL29 | -1.316 | -1.359 | -3.3% (grew) | 88.7% | 2.11s |
| WCL30 | -3.567 | -3.353 | +6.0% | 85.3% | 2.54s |
| WCL31 | -1.969 | -1.893 | +3.9% | 86.9% | 2.65s |
| WCL32 | -2.572 | -2.582 | -0.4% (grew) | 86.9% | 2.38s |
| WCL33 | -3.293 | -3.303 | -0.3% (grew) | 92.0% | 2.75s |

("shrinkage" = fraction by which the censored divergence is smaller than the uncensored divergence within
the shared well-powered window; negative means the censored divergence is actually slightly *larger*.)

**Every single mouse retains ≥91% of its uncensored divergence after censoring**, within a well-powered
window (shared effective-N coverage 82-92%). Removing the next trial's own contaminating signal does
**not** meaningfully shrink the tercile divergence for any of the 11 mice, including WCL24, WCL29, and
WCL31 side by side. This directly answers the user's specific question: **no, the divergence does not
collapse for WCL29/WCL31 while persisting for WCL24 — it persists essentially unchanged for all three,
and for all 11 mice.** By the Step 1 criterion alone, contamination amount does not explain the divergence
anywhere in this dataset.

### Step 2 result: does the next trial's own outcome correlate with the current tercile (content)?

**2a — plain P(next trial is a win | current tercile):**

| mouse | P(win \| low) | P(win \| mid) | P(win \| high) |
|---|---|---|---|
| POOLED | 0.773 | 0.799 | 0.815 |
| WCL23 | 0.756 | 0.784 | 0.804 |
| WCL24 | 0.761 | 0.758 | 0.795 |
| **WCL25** | **0.571** | 0.764 | 0.815 |
| WCL26 | 0.829 | 0.793 | 0.821 |
| WCL27 | 0.780 | 0.791 | 0.803 |
| WCL28 | 0.812 | 0.822 | 0.808 |
| WCL29 | 0.802 | 0.820 | 0.842 |
| **WCL30** | **0.740** | 0.777 | 0.794 |
| WCL31 | 0.772 | 0.821 | 0.825 |
| WCL32 | 0.782 | 0.799 | 0.816 |
| WCL33 | 0.784 | 0.826 | 0.827 |

Most mice show a modest, gradual increase in P(next win) from low→high tercile — expected, since streak
state and P(win) are not independent in a learning animal. WCL25 stands out with a much larger low-vs-high
gap (0.571 vs. 0.815, Δ≈0.24) than any other mouse.

**2b — contamination-fraction-quartile-stratified (per mouse quartiles of `contamination_fraction`,
low/high tercile 2×2, chi-square, no correction — matching this pipeline's established convention):**
the sharper test, because it asks whether the autocorrelation survives even in the **lowest**-contamination
quartile (Q1), where the physical overlap with the next trial is smallest.

- **WCL25, Q1 (lowest contamination quartile): risk difference = +0.449** [95% CI 0.262, 0.636],
  chi²=8.30, **p=0.0040** — highly significant, and this is the quartile where contamination amount is
  smallest. This is genuine content-based autocorrelation, not explained by window overlap.
- **WCL30, Q1: risk difference = +0.160** [95% CI 0.054, 0.267], chi²=7.75, **p=0.0054** — same pattern,
  smaller effect.
- No other mouse shows a significant Q1 effect (all other Q1 p-values ≥ 0.055; WCL23/WCL24 Q1 are
  borderline at p≈0.099/0.055 but do not cross significance and don't show the same pattern in Q2-Q4).
- WCL25's Q2-Q4 cells are flagged `underpowered` (n<30 or contingency cells <5) — reported as such, not
  silently dropped, per the analysis plan's explicit flagging rule.

### Step 3 result: per-mouse verdict

Decision rule (applied mechanically, not by cohort-wide vibe): **real slow signal** if divergence persists
past t_50 AND no significant Q1 autocorrelation; **contamination-content artifact** if divergence
vanishes, OR Q1 autocorrelation is significant (regardless of divergence persistence, since a real
content-driven autocorrelation can inflate an otherwise-real-looking divergence).

| mouse | divergence persists (censored) | amount-asymmetry direction | content sig. in lowest quartile | verdict |
|---|---|---|---|---|
| WCL23 | yes | — | no | **real slow signal** |
| WCL24 | yes | less contaminated on recent-loss | no | **real slow signal** |
| WCL25 | yes | — | **yes (p=0.0040)** | **contamination-content artifact** |
| WCL26 | yes | — | no | **real slow signal** |
| WCL27 | yes | — | no | **real slow signal** |
| WCL28 | yes | — | no | **real slow signal** |
| WCL29 | yes | more contaminated on recent-loss | no | **real slow signal** |
| WCL30 | yes | — | **yes (p=0.0054)** | **contamination-content artifact** |
| WCL31 | yes | more contaminated on recent-loss | no | **real slow signal** |
| WCL32 | yes | — | no | **real slow signal** |
| WCL33 | yes | — | no | **real slow signal** |

**9/11 mice: real slow signal.** The tercile divergence in these mice is not explained by next-trial
window contamination, either by amount (censoring barely moves it) or by content (no significant
autocorrelation even in the least-contaminated quartile). This includes WCL24, WCL29, and WCL31 — the
three mice the amount-based contamination data alone had made ambiguous — and directly resolves the
user's puzzle: none of them differ from each other the way the amount-asymmetry data alone suggested they
might.

**2/11 mice (WCL25, WCL30): contamination-content artifact.** These two show a real, mouse-specific
autocorrelation between the current trial's tercile and the next trial's own outcome that survives even
where contamination amount is minimal — i.e., something about these two animals' behavior (e.g. a longer
or shorter streak of the same outcome type propagating into the very next trial) makes the "current
tercile" and "next trial's own signal bleeding into the window" statistically entangled, independent of
how much of the window physically overlaps.

**No mouse fell into "indeterminate/mixed."**

---

## 6. Scope and caveats — read before using any of this as a Phase 2 target

1. **The "real slow signal" verdict is only validated within each mouse's own well-powered window**
   (per-mouse `t_50`, ranging ≈1.78-2.75s post-outcome across the 11 mice — see the Step 1 table above).
   Whether a genuine signal persists further out, to the full 5.4s window originally observed in Stage 1's
   tercile overlay, is **not resolved** by any analysis performed here — past `t_50` the trial pool
   shrinks to a small, likely-unrepresentative subset of unusually long-gap trials, and the data are too
   sparse to check this honestly. Any claim about the signal's duration should be scoped to ~1.8-2.9s post
   -outcome, not the full window.
2. **`KINETICS_METRIC_WINDOW_S=(0,3.0)` and `REWARD_WINDOW_S=(1.0,3.0)` both extend past every mouse's
   `t_50`.** Even for the 9 "real slow signal" mice, the tail of these windows (roughly `t_50` to 3.0s)
   is progressively contamination-vulnerable and low-effective-N; the fits reported by
   `compute_onset_and_decay` over the full window should not be treated as equally reliable throughout —
   they are best-supported near the peak and progressively less supported toward 3.0s.
3. **This is a diagnosis, not a fix.** No revision has been made to `alignment/kinetics.py`. The two
   pipeline-rejected fits (WCL31, WCL28 in the original diagnostic) remain rejected by the existing sanity
   guard; nothing here should be read as validating those specific numeric tau values.
4. **The `REWARD_WINDOW_S` manuscript cross-check is still open** — no manuscript file has been located
   in this repo, and the user has not yet responded to the request to point to one.
5. **Stage 4 of the earlier tail-risk prompt (the "hM4Di dataset" application) was never executed.** No
   distinct hM4Di cohort exists in this repo (`cohort_metadata_FP1/FP2.csv` have empty Treatment/Viral
   fields for every real mouse); the closest match is the existing DCZ-vs-saline within-animal
   pharmacology data. This should be confirmed with the user before that comparison is attempted.
6. For WCL25 and WCL30 specifically, any downstream analysis that treats the reward-locked tercile
   divergence as a clean per-trial signal (kinetics fit, REWARD_WINDOW_S summaries, etc.) should either be
   scoped to exclude these two animals or explicitly caveat that the effect is confounded with
   next-trial-outcome autocorrelation in their data.

## 7. Bottom line

The original concern (Stage 1) — that the tercile divergence might be a discrete-transient illusion
produced by trial-averaging, a slow win/loss-tracking signal, or next-trial contamination — has been
adjudicated per mouse rather than answered with a single cohort-wide verdict, per the user's explicit
instruction not to force a majority answer. For 9 of 11 mice the divergence survives removal of next-trial
contamination by both mechanisms tested and should be treated as a real, slow (not simply discrete
bump-then-decay), win/loss-history-tracking signal, valid within roughly the first two seconds
post-outcome. For 2 of 11 mice (WCL25, WCL30) the same divergence is at least partly an artifact of a
real content-based autocorrelation between the current trial's history state and the very next trial's own
outcome, and should not be taken at face value without that caveat.
