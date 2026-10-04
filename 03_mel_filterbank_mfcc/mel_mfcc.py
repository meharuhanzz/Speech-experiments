"""Experiment 03: Mel filterbank + MFCC, from scratch.

The whole pipeline by hand: Hz<->mel conversion, triangular mel
filterbank construction, framing+FFT power spectrum, log mel energies,
and the DCT-II used to decorrelate them into cepstral coefficients.

The real test isn't "does this run" -- it's whether the result has the
property MFCCs are actually used for in ASR: being mostly invariant to
pitch (F0) while being sensitive to the vocal-tract resonances (formants)
that distinguish one vowel from another. Two synthetic vowels (different
formant structure) are each synthesized at two different pitches, and we
check that same-vowel-different-pitch MFCC distance is smaller than
different-vowel-same-pitch distance.
"""
import numpy as np

SAMPLE_RATE = 16000


def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


def mel_filterbank(n_filters, n_fft, sr):
    low_mel, high_mel = hz_to_mel(0), hz_to_mel(sr / 2)
    mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filters = np.zeros((n_filters, n_fft // 2 + 1))
    for m in range(1, n_filters + 1):
        left, center, right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        for k in range(left, center):
            if center > left:
                filters[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            if right > center:
                filters[m - 1, k] = (right - k) / (right - center)
    return filters, hz_points[1:-1]


def dct2(x, n_coeffs):
    """Manual DCT-II, the same transform used to decorrelate log-mel
    energies into cepstral coefficients."""
    N = len(x)
    result = np.zeros(n_coeffs)
    n = np.arange(N)
    for k in range(n_coeffs):
        result[k] = np.sum(x * np.cos(np.pi * k * (2 * n + 1) / (2 * N)))
    return result


def synthesize_vowel(formants, f0, duration=0.3, sr=SAMPLE_RATE, bandwidth=80):
    """Source-filter vowel synthesis: an impulse train at f0 (the voiced
    excitation) run through resonant filters at each formant frequency."""
    n_samples = int(duration * sr)
    t = np.arange(n_samples) / sr

    period_samples = int(sr / f0)
    excitation = np.zeros(n_samples)
    excitation[::period_samples] = 1.0

    signal = excitation.copy()
    for f_formant in formants:
        r = np.exp(-np.pi * bandwidth / sr)
        theta = 2 * np.pi * f_formant / sr
        a1 = 2 * r * np.cos(theta)
        a2 = -r * r
        filtered = np.zeros(n_samples)
        for n in range(2, n_samples):
            filtered[n] = signal[n] + a1 * filtered[n - 1] + a2 * filtered[n - 2]
        signal = filtered
    return signal / (np.max(np.abs(signal)) + 1e-9)


def extract_mfcc(signal, sr=SAMPLE_RATE, n_fft=512, n_filters=26, n_ceps=13):
    window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n_fft) / (n_fft - 1))
    frame = signal[:n_fft] * window if len(signal) >= n_fft else np.pad(signal, (0, n_fft - len(signal))) * window

    spectrum = np.fft.rfft(frame, n=n_fft)
    power = (np.abs(spectrum) ** 2) / n_fft

    fbank, _ = mel_filterbank(n_filters, n_fft, sr)
    mel_energies = fbank @ power
    log_mel = np.log(mel_energies + 1e-10)

    return dct2(log_mel, n_ceps)


if __name__ == "__main__":
    fbank, centers = mel_filterbank(n_filters=26, n_fft=512, sr=SAMPLE_RATE)
    print(f"Mel filterbank: {fbank.shape[0]} filters over {fbank.shape[1]} FFT bins")
    print(f"filter center frequencies (Hz), first 5: {np.round(centers[:5])}")
    print(f"filter center frequencies (Hz), last 5:  {np.round(centers[-5:])}")
    print("-> spacing should grow wider at higher frequencies (that's what "
          "'mel-scale' means): "
          f"gap between filters 1-2 = {centers[1]-centers[0]:.0f}Hz, "
          f"gap between filters 24-25 = {centers[24]-centers[23]:.0f}Hz")

    # vowel-like formant structures (rough /i/ and /a/ ballpark values)
    VOWEL_I_FORMANTS = [300, 2300]   # close front vowel: low F1, high F2
    VOWEL_A_FORMANTS = [800, 1200]   # open back vowel: high F1, low F2

    print("\n--- synthesizing two vowels, each at two different pitches ---")
    mfccs = {}
    for vowel_name, formants in [("i", VOWEL_I_FORMANTS), ("a", VOWEL_A_FORMANTS)]:
        for f0 in [110, 220]:  # a "low" and a "high" pitch, same vowel
            sig = synthesize_vowel(formants, f0)
            mfccs[(vowel_name, f0)] = extract_mfcc(sig)
            print(f"  vowel '{vowel_name}' at f0={f0}Hz: formants={formants}")

    def dist(a, b):
        return np.linalg.norm(mfccs[a][1:] - mfccs[b][1:])  # skip c0 (overall energy)

    same_vowel_diff_pitch = dist(("i", 110), ("i", 220))
    same_vowel_diff_pitch_a = dist(("a", 110), ("a", 220))
    diff_vowel_same_pitch_110 = dist(("i", 110), ("a", 110))
    diff_vowel_same_pitch_220 = dist(("i", 220), ("a", 220))

    print(f"\n--- MFCC distance (coefficients 1-12, skipping c0) ---")
    print(f"same vowel 'i', pitch 110 vs 220 Hz:  {same_vowel_diff_pitch:.2f}")
    print(f"same vowel 'a', pitch 110 vs 220 Hz:  {same_vowel_diff_pitch_a:.2f}")
    print(f"different vowel, same pitch 110 Hz:   {diff_vowel_same_pitch_110:.2f}")
    print(f"different vowel, same pitch 220 Hz:   {diff_vowel_same_pitch_220:.2f}")

    avg_same_vowel = (same_vowel_diff_pitch + same_vowel_diff_pitch_a) / 2
    avg_diff_vowel = (diff_vowel_same_pitch_110 + diff_vowel_same_pitch_220) / 2
    print(f"\naverage same-vowel-different-pitch distance: {avg_same_vowel:.2f}")
    print(f"average different-vowel-same-pitch distance: {avg_diff_vowel:.2f}")
    if avg_diff_vowel > avg_same_vowel:
        print(f"-> MFCC distance is {avg_diff_vowel/avg_same_vowel:.1f}x larger across "
              f"vowels than across pitches on the same vowel: it's doing its job, "
              f"separating phonetic identity from pitch.")
    else:
        print("-> MFCC did NOT separate vowel identity from pitch as expected here.")
