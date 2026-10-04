"""Experiment 01: The Discrete Fourier Transform, from scratch.

Implements the DFT as the textbook formula actually says it is --
X[k] = sum_n x[n] * exp(-2j*pi*k*n/N) -- as a direct O(N^2)
matrix multiplication, no FFT trick. Run on a synthetic signal built
from known frequencies, so "did it work" has an exact answer: the
magnitude spectrum should peak exactly at the frequencies we put in,
and the result should match numpy's FFT to numerical precision.
"""
import numpy as np

SAMPLE_RATE = 8000  # Hz
DURATION = 0.5      # seconds


def make_multitone_signal(freqs_hz, amplitudes, sr=SAMPLE_RATE, duration=DURATION):
    t = np.arange(0, duration, 1.0 / sr)
    signal = np.zeros_like(t)
    for f, a in zip(freqs_hz, amplitudes):
        signal += a * np.sin(2 * np.pi * f * t)
    return signal, t


def dft_from_scratch(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    W = np.exp(-2j * np.pi * k * n / N)
    return W @ x


def magnitude_spectrum(X, sr, N):
    freqs = np.fft.fftfreq(N, d=1.0 / sr)
    mag = np.abs(X) / N * 2  # scale to approximate sine amplitude
    # only the positive-frequency half is meaningful for a real signal
    half = N // 2
    return freqs[:half], mag[:half]


if __name__ == "__main__":
    true_freqs = [220, 440, 1000]   # Hz: A3, A4, and a higher tone
    true_amps = [1.0, 0.5, 0.25]
    signal, t = make_multitone_signal(true_freqs, true_amps)
    N = len(signal)
    print(f"Signal: {N} samples at {SAMPLE_RATE} Hz ({DURATION}s), "
          f"built from tones at {true_freqs} Hz with amplitudes {true_amps}")

    import time
    t0 = time.time()
    X_scratch = dft_from_scratch(signal)
    t_scratch = time.time() - t0

    t0 = time.time()
    X_numpy = np.fft.fft(signal)
    t_numpy = time.time() - t0

    max_diff = np.max(np.abs(X_scratch - X_numpy))
    print(f"\nfrom-scratch DFT vs numpy.fft.fft: max abs difference = {max_diff:.2e}")
    print(f"timing: from-scratch O(N^2) = {t_scratch*1000:.1f}ms, "
          f"numpy FFT (O(N log N)) = {t_numpy*1000:.1f}ms "
          f"({t_scratch/t_numpy:.0f}x slower -- this is exactly why nobody "
          f"actually uses the direct formula on real audio)")

    freqs, mag = magnitude_spectrum(X_scratch, SAMPLE_RATE, N)
    print(f"\n--- recovering the true frequencies from the magnitude spectrum ---")
    peak_bins = []
    for true_f in true_freqs:
        nearest_idx = np.argmin(np.abs(freqs - true_f))
        peak_bins.append(nearest_idx)
        print(f"  true freq {true_f:5d} Hz -> nearest DFT bin at {freqs[nearest_idx]:.1f} Hz, "
              f"magnitude {mag[nearest_idx]:.3f} (true amplitude was {true_amps[true_freqs.index(true_f)]})")

    # is the energy actually concentrated at exactly these bins, or spread out?
    total_energy = np.sum(mag ** 2)
    peak_energy = np.sum(mag[peak_bins] ** 2)
    print(f"\nFraction of total spectral energy sitting in just these 3 bins "
          f"(out of {len(mag)} total): {peak_energy/total_energy:.1%}")
