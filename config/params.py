"""
Pipeline constants, sourced either from `processNew.m` (the canonical MATLAB
entry point, newCode/processNew.m) or from a specific session's saved
`processed.params` struct, as noted per-constant. See vertical_slice
validation against session WCL23/060223 for where these were confirmed.
"""

from dataclasses import dataclass

# --- Acquisition parameters --------------------------------------------------
# Source: processNew.m:150-151 (rawSampleFreq, numChannels). Segment size
# 28000 samples / 2000 Hz = 14 channels x 1 s per Raw_*.mat file
# (processNew.m:169-179 auto-detect table).
RAW_SAMPLE_FREQ_HZ = 2000
N_RAW_CHANNELS = 14

# Channel layout (processNew.m:220-268 channelDefs, :284-293 channelNames).
# 0-based indices into the (14, n_samples) raw array.
CH_GREEN_R, CH_RED_R, CH_G_CARRIER, CH_R_CARRIER, CH_GREEN_L, CH_RED_L = range(6)
(
    CH_CENTERPORT,
    CH_RIGHTPORT,
    CH_LEFTPORT,
    CH_LEFT_LICK,
    CH_RIGHT_LICK,
    CH_CENTER_LED,
    CH_LEFT_LED,
    CH_LASER,
) = range(6, 14)


@dataclass(frozen=True)
class ChannelSelection:
    """Which fluorescence channel is "the" signal, and its carrier reference.

    Most sessions in this cohort are single-color (one active GCaMP channel
    per hemisphere on a different ADC input, not a literal two-fluorophore
    pair -- there is no isosbestic reference channel in this rig). Pick the
    hemisphere with real data for a given session via this selection.
    """

    signal_channel: int
    carrier_channel: int
    nominal_carrier_freq_hz: float
    label: str


HEMISPHERE_CHANNELS = {
    "green_r": ChannelSelection(CH_GREEN_R, CH_G_CARRIER, 167.0, "green r"),
    "red_l": ChannelSelection(CH_RED_L, CH_R_CARRIER, 223.0, "red l"),
    # Some mice in this cohort (e.g. WCL28) are implanted/recorded on the
    # opposite side from the green_r norm -- same green carrier frequency,
    # just read off CH_GREEN_L instead of CH_GREEN_R. Confirmed against that
    # mouse's RA-processed params (channelNames/measuredCarrierFreq): green r
    # reads 0 Hz (no signal) while green l reads the real ~167 Hz lock.
    "green_l": ChannelSelection(CH_GREEN_L, CH_G_CARRIER, 167.0, "green l"),
}
DEFAULT_HEMISPHERE = "green_r"

# --- Demodulation (spectBS.m via processNew.m:356-527) -----------------------
FFT_ESTIMATE_N_POINTS = 2 ** 14             # processNew.m:362 -- window used only to *estimate* the true carrier freq
SPECTRAL_WINDOW_SAMPLES = 216               # processNew.m:318 (=2*9*12)
SPECTRAL_WINDOW_OVERLAP_SAMPLES = 108       # processNew.m:319 (=window/2) -> hop = 216-108 = 108 samples
FREQ_STEP_HZ = 1.0                          # processNew.m:311 -- spacing between evaluated frequency bins
FREQ_STEP_WIDTH_BINS = 7                    # processNew.m:312 -- +/- this many 1 Hz bins around the carrier are evaluated
INCL_FREQ_WIN_BINS = 4                      # processNew.m:310 / spectBS.m:42 -- +/- this many bins around the PEAK bin get averaged into the envelope
LOW_PASS_CORNER_HZ = 100.0                  # processNew.m:315 / spectBS.m:45 -- only applied if the demod hop rate exceeds this (does not trigger at the hop rate below)

# hop = window - overlap = 108 raw samples -> this IS the final sample grid.
# finalSampleFreq = rawSampleFreq / (12*9) = 2000/108 ~= 18.5185 Hz (processNew.m:308)
HOP_SAMPLES = SPECTRAL_WINDOW_SAMPLES - SPECTRAL_WINDOW_OVERLAP_SAMPLES  # 108
FINAL_SAMPLE_FREQ_HZ = RAW_SAMPLE_FREQ_HZ / HOP_SAMPLES                  # ~18.5185 Hz
FINAL_TIME_STEP_SEC = 1.0 / FINAL_SAMPLE_FREQ_HZ                         # ~0.054 s ("18 bins ~= 1 s" in the stats scripts)

# --- Baseline / normalization -------------------------------------------------
# processNew_fast_kevin.m:328 (params.detrendWindowTime=60) -- the reference
# pipeline for this cohort (confirmed: its ptsKeep_before/after/finalSampleFreq/
# detrendWindowTime match this cohort's saved processed_*.mat params exactly,
# unlike vanilla processNew.m). Used for BOTH the raw-carrier rolling z-score
# (pre-demodulation, at rawSampleFreq) and the final rolling z-score
# (post-demodulation, at finalSampleFreq) in MATLAB's real double-pass
# scheme -- see preprocessing/demodulate.py's compute_dff_and_zscore.
BASELINE_WINDOW_SEC = 60.0
BASELINE_WINDOW_SAMPLES = int(round(BASELINE_WINDOW_SEC * FINAL_SAMPLE_FREQ_HZ))  # ~1111 samples == MATLAB's signalDetrendWindow
LCM_CARRIERS = 36  # processNew_fast_kevin.m: lcm(carrierPtsPerCycle) for the 167/223 Hz carrier pair, confirmed via processed_WCL23_060223.mat's params.lcmCarriers
RAW_BASELINE_WINDOW_SAMPLES = LCM_CARRIERS * int(
    (BASELINE_WINDOW_SEC * RAW_SAMPLE_FREQ_HZ) // LCM_CARRIERS
)  # MATLAB's rawDetrendWindow ~= 119988 samples @ 2000 Hz raw rate (confirmed exact match against a real session's saved params)

# --- Behavior clock alignment (processBehavior.m) ----------------------------
XCORR_MAX_LAG_POKES = 100                   # processBehavior.m:95
XCORR_ACCEPT_THRESHOLD = 0.5                # processBehavior.m:102

# --- Trial photometry-validity gates (processBehavior.m:284-308, processCeliaWord.m:101-108) ---
# MATLAB's dropFirstDetrendWindow=1 is hardcoded true in every processNew*
# variant, so minPtsOffset == signalDetrendWindow (full window) for the
# primary hasAllPhotometryData gate that feeds the main reward-split PETH;
# a SEPARATE, more lenient half-window gate (minPtsOffset=signalDetrendWindow/2)
# is used by processCeliaWord.m/processByQuantiles.m/etc for word/sequence/
# quantile-conditioned analyses -- these are genuinely two different MATLAB
# gates, not a single convention, confirmed by direct source read.
MIN_PTS_OFFSET_FULL = BASELINE_WINDOW_SAMPLES        # hasAllPhotometryData (main PETH)
MIN_PTS_OFFSET_HALF = BASELINE_WINDOW_SAMPLES / 2.0  # hasP (word/sequence outcome analyses)

# --- PETH window --------------------------------------------------------------
# processNew_fast_kevin.m's ptsKeep_before=40/ptsKeep_after=100 (SAMPLES),
# confirmed a fixed script-wide constant, and confirmed uniform (40, 100)
# across all 46 sessions in the FP1 "none" cohort's saved processed_*.mat
# params -- not session-dependent, so hardcoded directly rather than read
# per-session. (Vanilla processNew.m's generic default is 40/60 -- NOT what
# this cohort was actually processed with.)
PTS_KEEP_BEFORE_SAMPLES = 40
PTS_KEEP_AFTER_SAMPLES = 100
PETH_PRE_SEC = PTS_KEEP_BEFORE_SAMPLES / FINAL_SAMPLE_FREQ_HZ   # 2.16 s
PETH_POST_SEC = PTS_KEEP_AFTER_SAMPLES / FINAL_SAMPLE_FREQ_HZ   # 5.4 s

# --- PETH alignment event selection --------------------------------------------
# Which trial_table column (photometry-clock sample index, from
# behavior.sync.align_behavior_to_photometry) each --align-to choice pulls, and
# its human-readable axis/title label. 'outcome' has no separately-timestamped
# raw signal in this rig -- reward is delivered essentially instantaneously on
# the rewarded side poke (rewardDurationLeft/Right solenoid-open times are
# ~55-60 ms, sub-final-sample -- confirmed against WCL23/060223's parameters
# CSV), and this codebase already models "reward/consummatory response" as a
# delayed sub-window OFF side_in (see REWARD_WINDOW_S below) rather than a
# separately timestamped event. So 'outcome' reuses the side_in photometry
# index -- see behavior/sync.py for where photometry_outcome_index is set.
ALIGN_EVENT_COLUMNS = {
    "center_in": "photometry_center_in_index",
    "side_in": "photometry_side_in_index",
    "outcome": "photometry_outcome_index",
    "side_out": "photometry_side_out_index",
}
ALIGN_EVENT_LABELS = {
    "center_in": "Center Port Entry",
    "side_in": "Side Port Entry",
    "outcome": "Reward Delivery / Feedback",
    "side_out": "Side Port Exit",
}
DEFAULT_ALIGN_EVENT = "side_in"

# --- Trial-level event-aligned z-scoring --------------------------------------
# Per-trial baseline window, in seconds relative to the aligning event
# (e.g. choice/side-port entry at t=0). Distinct from BASELINE_WINDOW_SEC
# above, which drives the CONTINUOUS rolling z-score/dF/F used for GLMs,
# DREADD comparisons, and state classifiers. This trial baseline is used
# only to normalize individual PETH/event-aligned traces, so each trial is
# scored relative to its own immediate pre-event activity rather than a
# session-wide rolling window.
PETH_BASELINE_PRE_EVENT_S = -2.0
PETH_BASELINE_POST_EVENT_S = -0.5

# --- Trial action/outcome word labels (processCeliaWord.m, KM_processCeliaWord.m) ---
# Trailing-window word length(s), in trials, to compute per trial (see
# behavior/word_encoding.py). newCode/KM_processCeliaWord.m:22-23 default is a
# single "levels" value (commonly 3 or 4); the user wants 1, 2, and 3 all
# available at once rather than picking one.
WORD_ENCODING_LEVELS = (1, 2, 3)

# --- Explicit lag columns / sequence strings (behavior/word_encoding.py) ------
# How many trials back (n-1, n-2, ..., n-LAG_N) to expose as explicit
# 1_Reward/1_Choice/1_Switch-style columns and as concatenated sequence
# strings (reward_seq_3, choice_seq_3, switch_seq_3). Not a MATLAB port --
# this is the "classic" lag/sequence convention requested directly by the lab.
LAG_N = 3

# --- Word/sequence outcome evaluation (behavior/word_encoding.py) ------------
# Sub-windows (seconds relative to choice/side-port entry, t=0) used by
# evaluate_word_outcomes() to summarize photometry per word/sequence group.
# Decision window: immediate post-choice-entry response. Reward window:
# later consummatory/licking-related response. Both fall inside the PETH's
# [-PETH_PRE_SEC, +PETH_POST_SEC] span.
DECISION_WINDOW_S = (0.0, 1.0)
REWARD_WINDOW_S = (1.0, 3.0)

# Default word/sequence columns (and how many top-N most frequent patterns
# per column) shown by pipeline.py's demo outcome summary table.
SUMMARY_GROUP_COLUMNS = (
    "word_l2_generic", "word_l3_generic", "reward_seq_3", "choice_seq_3", "switch_seq_3",
    "Behavioral_State",
)
SUMMARY_TOP_N = 5

# --- Raw-photometry loading ----------------------------------------------------
RAW_LOAD_N_WORKERS = 32  # thread pool size for loading Raw_*.mat segments over the network share (I/O-latency-bound, not CPU-bound)

# --- Sticky Q-learning fit / behavioral-state classifier ----------------------
# Ported from mastro_mouse_bandit_analysis (qlearning.py, states.py; commit
# 3554989) -- see external/bandit_state_model.py. Fit fresh per session
# against this pipeline's own trial_table (per lab decision: the trial_table
# is the single source of truth, not a join against that package's external
# nosepoke-aging dataset).
BANDIT_MIN_TRIALS = 30   # qlearning.py fit_session_level's min_trials default
BANDIT_N_STARTS = 20     # qlearning.py fit_session_level's n_starts default

ALPHA_BOUNDS = (1e-4, 1 - 1e-4)
BETA_BOUNDS = (1e-3, 50.0)
KAPPA_BOUNDS = (-10.0, 10.0)

BETA_BOUND_THRESH = 49.9
KAPPA_BOUND_THRESH = 9.9
ALPHA_BOUND_LO = 0.001
ALPHA_BOUND_HI = 0.999
NLL_CHANCE_THRESH = 0.70

ROLLING_WINDOW = 5
ROLLING_MIN_PERIODS = 3

# Data-driven classifier v2 thresholds (states.py) -- simplified from an
# original 6-state scheme to 4 states after K-Means validation showed a
# silhouette peak at K=4 rather than 6.
BIAS_DEV_THRESH = 0.45
BIAS_LOW_PRIGHT = 0.4
BIAS_HIGH_PRIGHT = 0.6
EXPLOITATION_ACC_THRESH = 0.65
EXPLOITATION_SWITCH_THRESH = 0.15

STATES_ORDERED = ("Exploitation", "Exploration", "Left Bias", "Right Bias")
STATE_COLORS = {
    "Exploitation": "#1abc9c",
    "Exploration": "#f39c12",
    "Left Bias": "#2980b9",
    "Right Bias": "#e74c3c",
}

# Number of |Q_diff| quantile bins used by viz.traces.plot_peth_by_group when
# grouping PETHs by Q-value-difference magnitude instead of Behavioral_State.
QDIFF_N_BINS = 3

# --- Time-resolved GLM encoding model (models/glm_encoding.py) ----------------
# Default predictor formula for fit_time_resolved_glm, referencing the clean
# names models.glm_encoding._build_predictor_frame builds from trial_table's
# raw columns: Choice<-chose_right, Reward<-was_rewarded,
# Reward_lag{1,2,3}<-{k}_Reward (behavior/word_encoding.py's explicit lag
# columns), Q_diff_abs<-Q_diff.abs() (external/bandit_state_adapter.py),
# Behavioral_State (external/bandit_state_adapter.py).
DEFAULT_TIME_RESOLVED_GLM_FORMULA = (
    "Z ~ Choice * Reward + Reward_lag1 + Reward_lag2 + Reward_lag3 + Q_diff_abs + C(Behavioral_State)"
)
