"""Experiment 07: Formant estimation via Linear Predictive Coding (LPC).

LPC models each sample as a linear combination of the previous p
samples -- a reasonable model for the vocal tract acting as an
all-pole resonant filter on the glottal excitation. The LPC
coefficients are found by solving the normal equations from the
signal's autocorrelation (Levinson-Durbin recursion, from scratch),
and the filter's resonant peaks -- found as the angles of its complex
roots -- are the estimated formants. Checked against the known
formant frequencies used to synthesize the test vowels in the first
place, and cross-checked against scipy's Toeplitz solver for the
Levinson-Durbin step itself.
"""
import numpy as np
from scipy.linalg import solve_toeplitz

SAMPLE_RATE = 16000


def synthesize_vowel(formants, f0=130, duration=0.3, sr=SAMPLE_RATE, bandwidth=60):
    n_samples = int(duration * sr)
    period = int(sr / f0)
    excitation = np.zeros(n_samples)
    excitation[::period] = 1.0
    signal = excitation.copy()
    for f_formant in formants:
        r = np.exp(-np.pi * bandwidth / sr)
        theta = 2 * np.pi * f_formant / sr
        a1, a2 = 2 * r * np.cos(theta), -r * r
        filtered = np.zeros(n_samples)
        for n in range(2, n_samples):
            filtered[n] = signal[n] + a1 * filtered[n - 1] + a2 * filtered[n - 2]
        signal = filtered
    return signal / (np.max(np.abs(signal)) + 1e-9)


def autocorrelate(signal, max_lag):
    n = len(signal)
    return np.array([np.sum(signal[:n - lag] * signal[lag:]) for lag in range(max_lag + 1)])


def levinson_durbin(r, order):
    """From-scratch Levinson-Durbin recursion: solves the LPC normal
    equations in O(p^2) instead of the O(p^3) a generic solver would
    need, by exploiting the Toeplitz structure of the autocorrelation
    matrix."""
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]
    for i in range(1, order + 1):
        acc = r[i] + np.sum(a[1:i] * r[i - 1:0:-1])
        k = -acc / e
        a_new = a.copy()
        for j in range(1, i):
            a_new[j] = a[j] + k * a[i - j]
        a_new[i] = k
        a = a_new
        e *= (1 - k ** 2)
    return a, e


def lpc_coefficients_scratch(signal, order):
    r = autocorrelate(signal, order)
    a, _ = levinson_durbin(r, order)
    return a


def lpc_coefficients_scipy_crosscheck(signal, order):
    """Same normal equations, solved with scipy's generic Toeplitz
    solver instead of hand-written Levinson-Durbin, as a correctness
    cross-check."""
    r = autocorrelate(signal, order)
    R = r[:order]
    rhs = -r[1:order + 1]
    a_rest = solve_toeplitz(R, rhs)
    return np.concatenate([[1.0], a_rest])


def formants_from_lpc(a, sr):
    roots = np.roots(a)
    roots = roots[np.imag(roots) >= 0]  # complex roots come in conjugate pairs
    angles = np.angle(roots)
    freqs = angles * sr / (2 * np.pi)
    mags = np.abs(roots)
    # keep only roots close to the unit circle (real resonances, not noise)
    order = np.argsort(freqs)
    freqs, mags = freqs[order], mags[order]
    resonant = (freqs > 0) & (mags > 0.85)
    return np.sort(freqs[resonant])


if __name__ == "__main__":
    test_vowels = {
        "i (close front)": [300, 2300],
        "a (open back)": [800, 1200],
        "u (close back)": [350, 800],
    }

    for name, true_formants in test_vowels.items():
        signal = synthesize_vowel(true_formants)
        order = 8  # needs >= 2 poles per formant
        a_scratch = lpc_coefficients_scratch(signal, order)
        a_scipy = lpc_coefficients_scipy_crosscheck(signal, order)
        coeff_diff = np.max(np.abs(a_scratch - a_scipy))

        estimated = formants_from_lpc(a_scratch, SAMPLE_RATE)
        print(f"\nvowel '{name}': true formants = {true_formants} Hz")
        print(f"  from-scratch Levinson-Durbin vs scipy Toeplitz solver, "
              f"max coefficient difference: {coeff_diff:.2e}")
        print(f"  estimated resonances from LPC roots: {np.round(estimated).tolist()} Hz")

        for true_f in true_formants:
            nearest = estimated[np.argmin(np.abs(estimated - true_f))]
            error_pct = abs(nearest - true_f) / true_f
            print(f"    true {true_f} Hz -> nearest estimated {nearest:.0f} Hz "
                  f"(error {error_pct:.1%})")
