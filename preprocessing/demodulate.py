"""
Carrier-frequency estimation, lock-in demodulation, and dF/F / rolling
z-score computation -- the Python replacement for processNew.m's
demodulation block (processNew.m:356-527) and spectBS.m.
"""

import numpy as np
import pandas as pd

from config.params import (
    BASELINE_WINDOW_SAMPLES,
    FFT_ESTIMATE_N_POINTS,
    FREQ_STEP_HZ,
    FREQ_STEP_WIDTH_BINS,
    HOP_SAMPLES,
    INCL_FREQ_WIN_BINS,
    RAW_BASELINE_WINDOW_SAMPLES,
    RAW_SAMPLE_FREQ_HZ,
    SPECTRAL_WINDOW_SAMPLES,
)


def estimate_carrier_freq(raw_channel, fs=RAW_SAMPLE_FREQ_HZ, n_points=FFT_ESTIMATE_N_POINTS):
    """FFT peak-finding on a single chunk to get the true (as-recorded)
    carrier frequency -- processNew.m normalizes (z-scores) the chunk before
    the FFT (processNew.m:365 `normalize(...)`) so that amplitude doesn't
    affect peak-finding; we do the same here.
    """
    chunk = raw_channel[:n_points]
    chunk = (chunk - chunk.mean()) / chunk.std()
    spectrum = np.abs(np.fft.rfft(chunk)) / n_points
    freqs = np.fft.rfftfreq(n_points, d=1.0 / fs)
    peak_idx = np.argmax(spectrum)
    candidate_freq = freqs[peak_idx]
    pts_per_cycle = int(np.floor(fs / candidate_freq))          # processNew.m:376
    measured_freq = fs / pts_per_cycle                          # processNew.m:377
    return measured_freq, pts_per_cycle


def demodulate_envelope(
    raw_channel,
    carrier_freq,
    fs=RAW_SAMPLE_FREQ_HZ,
    window_samples=SPECTRAL_WINDOW_SAMPLES,
    hop_samples=HOP_SAMPLES,
    freq_step=FREQ_STEP_HZ,
    freq_step_width=FREQ_STEP_WIDTH_BINS,
    incl_freq_win=INCL_FREQ_WIN_BINS,
):
    """Lock-in style demodulation matching spectBS.m.

    MATLAB calls `spectrogram(rawData, window, noverlap, freqRange, fs)`
    with `freqRange` a literal vector of specific frequencies (not just FFT
    bin spacing) -- with window=216 samples that call uses the Goertzel
    algorithm internally to evaluate the short-time DFT at exactly those
    frequencies. We replicate that directly: for each overlapping window we
    take the dot product with complex exponentials at each of the 15 target
    frequencies (mathematically identical to what MATLAB's spectrogram/
    Goertzel path computes; only an overall normalization constant differs,
    which cancels out once we take dF/F or a z-score downstream).

    Window is Hamming-windowed: MATLAB's `spectrogram(x, N, ...)` defaults to
    `hamming(N)` when given a scalar window length (not scipy's default
    Tukey/boxcar), so we build the Hamming window explicitly.

    hop_samples=108 at fs=2000 Hz means each output sample is 0.054 s apart
    -- i.e. the STFT hop rate ITSELF lands exactly on the pipeline's final
    ~18.52 Hz sample grid (processNew.m:308-309); no separate decimation step
    is needed. This also means the conditional low-pass filter in spectBS.m
    (`if filtFreq < params.lowPassCorner: filtSig = sig`, spectBS.m:44-46)
    never actually triggers for this window/overlap configuration, since
    18.52 Hz < 100 Hz -- so we skip it too.
    """
    freqs = carrier_freq + freq_step * np.arange(-freq_step_width, freq_step_width + 1)  # processNew.m:423-425, 15 freqs

    n = len(raw_channel)
    n_frames = 1 + (n - window_samples) // hop_samples
    frame_starts = hop_samples * np.arange(n_frames)
    frame_idx = frame_starts[:, None] + np.arange(window_samples)[None, :]

    window = np.hamming(window_samples)
    frames = raw_channel[frame_idx] * window  # (n_frames, window_samples)

    basis = np.exp(-2j * np.pi * np.outer(freqs, np.arange(window_samples)) / fs)  # (n_freqs, window_samples)
    stft_at_freqs = basis @ frames.T  # (n_freqs, n_frames), complex -- Goertzel-equivalent
    amplitude = np.abs(stft_at_freqs)

    peak_bin = int(np.argmax(amplitude.mean(axis=1)))  # spectBS.m:27-30
    lo = max(0, peak_bin - incl_freq_win)
    hi = min(len(freqs), peak_bin + incl_freq_win + 1)
    envelope = amplitude[lo:hi, :].mean(axis=0)  # spectBS.m:42

    return envelope, freqs[peak_bin]


def compute_dff_and_zscore(raw_channel, carrier_freq, envelope, window_samples=BASELINE_WINDOW_SAMPLES,
                            raw_window_samples=RAW_BASELINE_WINDOW_SAMPLES):
    """MATLAB's real double-pass baseline/z-score scheme
    (processNew_fast_kevin.m:447-541, rollingZ.m) -- confirmed the reference
    implementation for this cohort (see config.params module docstring/
    comments). This is NOT a single pass on one demodulated envelope; it's
    two independent rolling-baseline passes at two different sample rates:

      1. Rolling z-score of the RAW (undemodulated), carrier-modulated
         signal, at the raw ~2kHz rate, BEFORE demodulation:
         flSignal = (flRaw - movmean(flRaw, rawDetrendWindow)) / movstd(flRaw, rawDetrendWindow).
      2. Demodulate that pre-z-scored trace -- a SEPARATE demodulation pass
         from `envelope` (the plain single-pass demodulation of the raw
         channel -- MATLAB's processed.signals_raw, informational only, not
         used downstream by anything).
      3. A SECOND, independent rolling z-score of THAT demodulated trace, at
         the final ~18.52Hz rate (rollingZ.m) -- this is MATLAB's
         processed.signals (pSignal), the value every PETH/GLM in the
         reference pipeline actually uses.

    `dff` has no MATLAB equivalent -- processNew_fast_kevin.m's rawf0/
    signals_raw are never combined into a literal dF/F% anywhere in the
    reference pipeline. Kept here as a Python-only convenience metric
    (single-pass dF/F from the plain `envelope`, for visualization only) --
    not expected to match any MATLAB output. `zscore` is the value that must
    match, and does (see validation/compare_to_matlab.py).
    """
    raw_channel = np.asarray(raw_channel, dtype=float)
    raw_s = pd.Series(raw_channel)
    raw_mean = raw_s.rolling(raw_window_samples, center=True, min_periods=1).mean()
    raw_std = raw_s.rolling(raw_window_samples, center=True, min_periods=1).std(ddof=1)

    fl_signal = ((raw_s - raw_mean) / raw_std).to_numpy()
    zero_std = (raw_std == 0).to_numpy()
    fl_signal[zero_std] = 0.0  # processNew_fast_kevin.m's stdZeros guard

    demodulated, _ = demodulate_envelope(fl_signal, carrier_freq)

    final_s = pd.Series(demodulated)
    final_mean = final_s.rolling(window_samples, center=True, min_periods=1).mean()
    final_std = final_s.rolling(window_samples, center=True, min_periods=1).std(ddof=1)
    # rollingZ.m has no zero-std guard on this second pass -- replicate as-is (can produce inf/nan).
    zscore = ((final_s - final_mean) / final_std).to_numpy()

    env_s = pd.Series(envelope)
    env_mean = env_s.rolling(window_samples, center=True, min_periods=1).mean()
    dff = ((env_s - env_mean) / env_mean).to_numpy()

    return dff, zscore, env_mean.to_numpy()
