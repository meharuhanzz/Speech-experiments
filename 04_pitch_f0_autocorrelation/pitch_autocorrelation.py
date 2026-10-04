"""Experiment 04: Pitch (F0) estimation via autocorrelation.

The classic time-domain pitch detector: autocorrelate a frame with
itself at increasing lags, and the lag with the strongest correlation
(after the zero-lag peak) corresponds to the signal's period -- so
f0 = sample_rate / best_lag. Tested across a range of known synthetic
f0 values and under added noise, so "how accurate is this" has an
exact numeric answer rather than a vibe.
"""
import numpy as np

SAMPLE_RATE = 16000


def synthesize_voiced_frame(f0, duration=0.04, sr=SAMPLE_RATE, n_harmonics=5, noise_std=0.0, seed=0):
    """A harmonic series at f0 -- closer to real voiced speech than a
    pure sine, since real voiced excitation is a buzzy pulse train with
    many harmonics, not a single tone."""
    rng = np.random.RandomState(seed)
    t = np.arange(0, duration, 1.0 / sr)
    signal = np.zeros_like(t)
    for h in range(1, n_harmonics + 1):
        signal += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t)
    signal += rng.normal(0, noise_std, size=t.shape)
    return signal


def autocorrelation_pitch(frame, sr, fmin=60, fmax=400):
    frame = frame - frame.mean()
    corr = np.correlate(frame, frame, mode="full")
    corr = corr[len(corr) // 2:]  # keep only non-negative lags

    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)
    search_region = corr[min_lag:max_lag]
    if len(search_region) == 0 or search_region.max() <= 0:
        return None

    best_lag = min_lag + np.argmax(search_region)
    return sr / best_lag


if __name__ == "__main__":
    true_f0s = [100, 150, 200, 250, 300]
    print(f"{'true f0':>10} | {'estimated f0':>14} | {'error':>8} | {'error %':>8}")
    errors = []
    for f0 in true_f0s:
        frame = synthesize_voiced_frame(f0, noise_std=0.0)
        estimated = autocorrelation_pitch(frame, SAMPLE_RATE)
        error = estimated - f0
        errors.append(abs(error))
        print(f"{f0:>10} | {estimated:>14.2f} | {error:>8.2f} | {abs(error)/f0:>7.2%}")

    print(f"\nmean absolute error across {len(true_f0s)} test frequencies (clean signal): "
          f"{np.mean(errors):.2f} Hz")

    print(f"\n--- robustness to noise, at a fixed true f0=150 Hz ---")
    print(f"{'noise_std':>10} | {'estimated f0':>14} | {'error %':>8}")
    for noise_std in [0.0, 0.5, 1.0, 2.0, 4.0]:
        frame = synthesize_voiced_frame(150, noise_std=noise_std, seed=1)
        estimated = autocorrelation_pitch(frame, SAMPLE_RATE)
        if estimated is None:
            print(f"{noise_std:>10} | {'no pitch found':>14} | {'--':>8}")
        else:
            error_pct = abs(estimated - 150) / 150
            print(f"{noise_std:>10} | {estimated:>14.2f} | {error_pct:>7.2%}")

    print("\nSignal RMS for reference (harmonic series, 5 harmonics, amplitude "
          f"1/h each): {np.sqrt(np.mean(synthesize_voiced_frame(150, noise_std=0.0)**2)):.3f} "
          "-- noise_std=4.0 is a genuinely harsh, low-SNR case relative to that.")
