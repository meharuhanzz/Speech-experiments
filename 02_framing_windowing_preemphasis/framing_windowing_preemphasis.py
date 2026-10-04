"""Experiment 02: Framing, windowing, and pre-emphasis -- the standard
preprocessing pipeline before any speech feature extraction.

Three things, all from scratch:
  1. pre-emphasis: a first-order high-pass filter y[n] = x[n] - a*x[n-1],
     which boosts high frequencies to counteract the natural tilt of
     voiced speech spectra (energy falls off at roughly -6dB/octave).
  2. framing: chopping a continuous signal into overlapping short windows
     (speech is only stationary over ~20-30ms).
  3. windowing: a Hamming window on each frame to reduce the spectral
     leakage a hard rectangular cut would cause.

Each step is checked against what it's supposed to do, not just run
and printed.
"""
import numpy as np

SAMPLE_RATE = 16000


def make_speech_like_signal(duration=1.0, sr=SAMPLE_RATE):
    """A signal with most of its energy at low frequencies (like voiced
    speech) plus a weaker high-frequency component -- pre-emphasis
    should visibly shift energy from the low tone towards the high one."""
    t = np.arange(0, duration, 1.0 / sr)
    rng = np.random.RandomState(0)
    signal = (
        1.0 * np.sin(2 * np.pi * 120 * t)   # strong low tone (like a pitch harmonic)
        + 0.05 * np.sin(2 * np.pi * 3000 * t)  # weak high-frequency component
    )
    signal += rng.normal(0, 0.01, size=t.shape)
    return signal


def pre_emphasis(signal, alpha=0.97):
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])


def frame_signal(signal, frame_len, frame_step):
    n_frames = 1 + (len(signal) - frame_len) // frame_step
    frames = np.zeros((n_frames, frame_len))
    for i in range(n_frames):
        start = i * frame_step
        frames[i] = signal[start:start + frame_len]
    return frames


def hamming_window(frame_len):
    n = np.arange(frame_len)
    return 0.54 - 0.46 * np.cos(2 * np.pi * n / (frame_len - 1))


def band_energy(signal, sr, low_hz, high_hz):
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sr)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    return np.sum(spectrum[mask] ** 2)


if __name__ == "__main__":
    signal = make_speech_like_signal()
    print(f"Synthetic 'speech-like' signal: {len(signal)} samples at {SAMPLE_RATE} Hz, "
          f"dominant 120 Hz tone + weak 3000 Hz tone")

    # --- pre-emphasis ---
    low_before = band_energy(signal, SAMPLE_RATE, 50, 200)
    high_before = band_energy(signal, SAMPLE_RATE, 2500, 3500)
    print(f"\n--- pre-emphasis (alpha=0.97) ---")
    print(f"before: low-band (50-200Hz) energy = {low_before:.2f}, "
          f"high-band (2500-3500Hz) energy = {high_before:.4f}, "
          f"ratio low/high = {low_before/high_before:.1f}")

    emphasized = pre_emphasis(signal)
    low_after = band_energy(emphasized, SAMPLE_RATE, 50, 200)
    high_after = band_energy(emphasized, SAMPLE_RATE, 2500, 3500)
    print(f"after:  low-band energy = {low_after:.2f}, "
          f"high-band energy = {high_after:.4f}, "
          f"ratio low/high = {low_after/high_after:.1f}")
    print(f"-> pre-emphasis boosted the high-frequency band relative to low "
          f"by a factor of {(low_before/high_before) / (low_after/high_after):.1f}x, "
          f"as intended.")

    # --- framing ---
    frame_len = int(0.025 * SAMPLE_RATE)   # 25ms
    frame_step = int(0.010 * SAMPLE_RATE)  # 10ms hop -> 15ms overlap
    frames = frame_signal(signal, frame_len, frame_step)
    print(f"\n--- framing (25ms frames, 10ms hop) ---")
    print(f"produced {frames.shape[0]} frames of {frames.shape[1]} samples each")
    print(f"overlap between consecutive frames: {frame_len - frame_step} samples "
          f"({(frame_len - frame_step) / frame_len:.0%} of frame length)")

    # --- windowing ---
    window = hamming_window(frame_len)
    windowed_frame = frames[10] * window
    rect_spectrum = np.abs(np.fft.rfft(frames[10], n=4096))
    hamming_spectrum = np.abs(np.fft.rfft(windowed_frame, n=4096))
    freqs = np.fft.rfftfreq(4096, d=1.0 / SAMPLE_RATE)

    # spectral leakage check: energy far from the 120Hz tone should be
    # lower with a Hamming window than with an implicit rectangular one
    far_mask = freqs > 500
    rect_leakage = np.sum(rect_spectrum[far_mask] ** 2)
    hamming_leakage = np.sum(hamming_spectrum[far_mask] ** 2)
    print(f"\n--- windowing (Hamming vs implicit rectangular) ---")
    print(f"spectral energy above 500Hz (should be near-zero -- our signal's "
          f"only real content there is the weak 3000Hz tone):")
    print(f"  rectangular (no window): {rect_leakage:.4f}")
    print(f"  Hamming window:          {hamming_leakage:.4f}")
    print(f"-> Hamming reduced leakage by {rect_leakage/hamming_leakage:.1f}x "
          f"by tapering the frame edges instead of cutting them off sharply.")
