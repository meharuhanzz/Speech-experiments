"""Experiment 06: Dynamic Time Warping for isolated-word matching.

The pre-neural approach to isolated word recognition: represent each
word as a sequence of feature vectors (here, frame energies in a few
frequency bands -- a crude stand-in for MFCC frames), then use DTW to
find the lowest-cost alignment between two sequences of possibly
different lengths. The same word spoken at a different speaking rate
should have a low DTW cost against its own template; a different word
should have a much higher cost, even after accounting for length
differences that a plain Euclidean distance couldn't handle at all.
"""
import numpy as np

SAMPLE_RATE = 16000


def synthesize_word(formant_sequence, duration, sr=SAMPLE_RATE, f0=150):
    """A toy 'word' as a sequence of vowel-like segments, each with its
    own formant pair -- a crude stand-in for a sequence of phonemes."""
    n_segments = len(formant_sequence)
    seg_dur = duration / n_segments
    signal = []
    for formants in formant_sequence:
        n = int(seg_dur * sr)
        t = np.arange(n) / sr
        period = int(sr / f0)
        excitation = np.zeros(n)
        excitation[::period] = 1.0
        sig = excitation.copy()
        for f_formant in formants:
            r = np.exp(-np.pi * 80 / sr)
            theta = 2 * np.pi * f_formant / sr
            a1, a2 = 2 * r * np.cos(theta), -r * r
            filtered = np.zeros(n)
            for i in range(2, n):
                filtered[i] = sig[i] + a1 * filtered[i - 1] + a2 * filtered[i - 2]
            sig = filtered
        signal.append(sig / (np.max(np.abs(sig)) + 1e-9))
    return np.concatenate(signal)


def extract_frame_features(signal, sr=SAMPLE_RATE, frame_len=400, frame_step=200, n_bands=4):
    """Crude band-energy features per frame -- stands in for a proper
    MFCC pipeline (built from scratch in experiment 03) so this
    experiment can focus purely on the DTW algorithm itself."""
    n_frames = max(1, 1 + (len(signal) - frame_len) // frame_step)
    band_edges = np.linspace(0, sr / 2, n_bands + 1)
    features = np.zeros((n_frames, n_bands))
    for i in range(n_frames):
        start = i * frame_step
        frame = signal[start:start + frame_len]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))
        spectrum = np.abs(np.fft.rfft(frame))
        freqs = np.fft.rfftfreq(frame_len, d=1.0 / sr)
        for b in range(n_bands):
            mask = (freqs >= band_edges[b]) & (freqs < band_edges[b + 1])
            features[i, b] = np.log(np.sum(spectrum[mask] ** 2) + 1e-10)
    return features


def dtw_distance(seq_a, seq_b):
    """Standard DTW with Euclidean local cost, from scratch."""
    n, m = len(seq_a), len(seq_b)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            local_cost = np.linalg.norm(seq_a[i - 1] - seq_b[j - 1])
            cost[i, j] = local_cost + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return cost[n, m] / (n + m)  # normalize by path length so longer sequences aren't unfairly penalized


if __name__ == "__main__":
    WORD_A = [[300, 2300], [700, 1200], [400, 900]]   # "word A"'s 3 phoneme-like segments
    WORD_B = [[800, 1200], [300, 2300], [600, 1700]]  # "word B": different segments

    print("Synthesizing: word A spoken at normal speed, word A spoken slowly "
          "(1.6x duration), and word B (a different word) at normal speed.")

    signal_a_normal = synthesize_word(WORD_A, duration=0.6)
    signal_a_slow = synthesize_word(WORD_A, duration=0.6 * 1.6)
    signal_b_normal = synthesize_word(WORD_B, duration=0.6)

    feat_a_normal = extract_frame_features(signal_a_normal)
    feat_a_slow = extract_frame_features(signal_a_slow)
    feat_b_normal = extract_frame_features(signal_b_normal)

    print(f"\nfeature sequence lengths: A-normal={len(feat_a_normal)} frames, "
          f"A-slow={len(feat_a_slow)} frames, B-normal={len(feat_b_normal)} frames")

    dtw_same_word_diff_rate = dtw_distance(feat_a_normal, feat_a_slow)
    dtw_diff_word_same_rate = dtw_distance(feat_a_normal, feat_b_normal)

    print(f"\n--- DTW distance (normalized by alignment path length) ---")
    print(f"word A (normal) vs word A (1.6x slower):  {dtw_same_word_diff_rate:.3f}")
    print(f"word A (normal) vs word B (different word): {dtw_diff_word_same_rate:.3f}")

    naive_euclidean_possible = len(feat_a_normal) == len(feat_b_normal)
    print(f"\nA plain frame-by-frame Euclidean distance would need equal-length "
          f"sequences; A-normal vs A-slow have different lengths "
          f"({len(feat_a_normal)} vs {len(feat_a_slow)} frames) so that "
          f"comparison isn't even possible without DTW's warping.")

    if dtw_diff_word_same_rate > dtw_same_word_diff_rate:
        ratio = dtw_diff_word_same_rate / dtw_same_word_diff_rate
        print(f"\n-> DTW correctly rates the same word at a different speaking "
              f"rate as {ratio:.1f}x closer than a genuinely different word "
              f"spoken at the same rate -- it's warping through the speed "
              f"difference while still being sensitive to actual phonetic "
              f"content.")
    else:
        print("\n-> DTW did NOT separate same-word-different-rate from "
              "different-word as expected here.")
