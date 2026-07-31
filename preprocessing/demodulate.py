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


def compute_dff_and_zscore(envelope, window_samples=BASELINE_WINDOW_SAMPLES):
    """From the single demodulated envelope, compute a rolling-baseline dF/F
    and a rolling z-score. This is the lab's chosen standard going forward:
    ONE demodulation pass, then both quantities derived from the same
    rolling mean/std of that envelope -- simpler and physically cleaner than
    MATLAB's two-baseline scheme (which rolling-z-scores the raw carrier
    signal before demodulating, then demodulates twice, then rolling-z-scores
    the demodulated trace again -- processNew.m:427-527).
    """
    s = pd.Series(envelope)
    rolling_mean = s.rolling(window_samples, center=True, min_periods=1).mean()
    rolling_std = s.rolling(window_samples, center=True, min_periods=1).std(ddof=1)

    dff = (s - rolling_mean) / rolling_mean
    zscore = (s - rolling_mean) / rolling_std

    return dff.to_numpy(), zscore.to_numpy(), rolling_mean.to_numpy()
