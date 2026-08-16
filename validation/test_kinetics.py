"""
Standalone recovery test for alignment.kinetics.compute_onset_and_decay: build
a synthetic trace with KNOWN onset latency and KNOWN decay tau directly (no
production code involved in constructing it), and assert the fitted values
recover the truth within a reasonable tolerance.

No pytest in this environment -- plain asserts, run directly:
    python validation/test_kinetics.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from alignment.kinetics import compute_onset_and_decay  # noqa: E402


def make_synthetic_trace(dt=0.02, t_start=-1.0, t_end=3.0, t_peak=0.4, peak_value=3.0, tau=0.6):
    """Linear rise from 0 at t=0 to peak_value at t=t_peak (known
    time-to-half-max = t_peak * onset_fraction, exactly, for onset_fraction
    fitted on a perfectly linear ramp), flat 0 baseline before t=0, then a
    noiseless single-exponential decay z(t) = peak_value * exp(-(t-t_peak)/tau)
    after the peak. No production code (compute_event_aligned_zscore,
    compute_onset_and_decay itself, etc.) is used to build this -- it's
    assembled directly from the definitions the pipeline is supposed to
    recover.
    """
    peth_time = np.arange(t_start, t_end + dt / 2, dt)
    trace = np.zeros_like(peth_time)

    pre_peak = (peth_time >= 0) & (peth_time <= t_peak)
    trace[pre_peak] = peak_value * (peth_time[pre_peak] / t_peak)

    post_peak = peth_time > t_peak
    trace[post_peak] = peak_value * np.exp(-(peth_time[post_peak] - t_peak) / tau)

    return trace, peth_time


def test_recovers_known_onset_and_tau():
    t_peak_true, peak_value, tau_true = 0.4, 3.0, 0.6
    onset_fraction = 0.5
    expected_onset = t_peak_true * onset_fraction  # exact, on a linear ramp

    trace, peth_time = make_synthetic_trace(t_peak=t_peak_true, peak_value=peak_value, tau=tau_true)
    onset_latency_s, decay_tau_s, r_squared, diagnostics = compute_onset_and_decay(
        trace, peth_time, metric_window_s=(0.0, 2.0), onset_fraction=onset_fraction,
    )

    print(f"expected onset={expected_onset:.4f}s, got {onset_latency_s:.4f}s")
    print(f"expected tau={tau_true:.4f}s, got {decay_tau_s:.4f}s, r_squared={r_squared:.6f}")
    print(f"diagnostics: {diagnostics}")

    assert diagnostics["skip_reason"] is None, f"decay fit was skipped: {diagnostics['skip_reason']}"
    assert abs(onset_latency_s - expected_onset) < 0.01, (
        f"onset latency off by {abs(onset_latency_s - expected_onset):.4f}s (tolerance 0.01s)"
    )
    assert abs(decay_tau_s - tau_true) / tau_true < 0.02, (
        f"decay tau off by {abs(decay_tau_s - tau_true) / tau_true:.2%} (tolerance 2%)"
    )
    assert r_squared > 0.999, f"expected a near-perfect fit on noiseless data, got r_squared={r_squared:.4f}"
    print("PASSED: known-onset/known-tau recovery within tolerance\n")


def test_negative_peak_is_skipped_not_forced():
    # A trace that never goes positive in the metric window -- compute_onset_and_decay
    # itself does no sign correction (that's compute_group_kinetics's job), so
    # this must come back as an explicit skip, not a fabricated fit.
    peth_time = np.arange(-1.0, 2.0, 0.02)
    trace = -3.0 * np.exp(-np.clip(peth_time, 0, None) / 0.5)
    onset_latency_s, decay_tau_s, r_squared, diagnostics = compute_onset_and_decay(
        trace, peth_time, metric_window_s=(0.0, 2.0),
    )
    assert np.isnan(decay_tau_s) and np.isnan(r_squared), "expected the fit to be skipped for a non-positive peak"
    assert diagnostics["skip_reason"] is not None
    print(f"PASSED: negative-peak trace correctly skipped ({diagnostics['skip_reason']})\n")


def test_too_short_post_peak_segment_is_skipped():
    # Peak sits right at the end of metric_window_s -- too few post-peak
    # samples to fit a decay, must be skipped (NaN), not forced.
    peth_time = np.arange(-1.0, 1.05, 0.02)
    trace = np.where(peth_time < 0, 0.0, 2.0 * (peth_time / 1.0))  # still rising at window end (t=1.0)
    onset_latency_s, decay_tau_s, r_squared, diagnostics = compute_onset_and_decay(
        trace, peth_time, metric_window_s=(0.0, 1.0),
    )
    assert np.isnan(decay_tau_s), "expected the decay fit to be skipped (too few post-peak samples)"
    assert diagnostics["skip_reason"] is not None and "post-peak" in diagnostics["skip_reason"]
    assert not np.isnan(onset_latency_s), "onset latency should still be computed even when decay is skipped"
    print(f"PASSED: too-short post-peak segment correctly skipped ({diagnostics['skip_reason']})\n")


if __name__ == "__main__":
    test_recovers_known_onset_and_tau()
    test_negative_peak_is_skipped_not_forced()
    test_too_short_post_peak_segment_is_skipped()
    print("All kinetics tests passed.")
