# Speech Experiments

Eight small, self-contained speech/audio-processing experiments,
simple to medium difficulty — a companion to
[NLP-experiments](https://github.com/meharuhanzz/NLP-experiments) and
[CV-experiments](https://github.com/meharuhanzz/CV-experiments). Same
rules: every test signal is synthesized inside the script (sine waves,
harmonic series, a simple source-filter vowel model), never a real
recording, so there's always an exact known ground truth — a true
frequency, a true formant, a true speech/silence boundary — to check
the result against. Classic DSP/speech techniques are implemented from
scratch where seeing the actual math is the point (DFT, mel filterbank
+ MFCC, autocorrelation pitch, DTW, Levinson-Durbin LPC), then
cross-checked against numpy/scipy where a trusted reference exists.

Every number below came from actually running the script. Several
experiments turned up a result that contradicted my first guess, and
the write-up was corrected to match what the code showed.

```bash
pip install -r requirements.txt
cd 01_dft_from_scratch && python dft_from_scratch.py
```

## Contents

| # | Experiment | Level | Idea |
|---|---|---|---|
| 01 | [DFT from scratch](01_dft_from_scratch/) | Simple | Direct O(N²) DFT, exact match to `numpy.fft`, recovers known tone frequencies exactly |
| 02 | [Framing, windowing, pre-emphasis](02_framing_windowing_preemphasis/) | Simple | The standard preprocessing trio, each effect measured, not just asserted |
| 03 | [Mel filterbank + MFCC](03_mel_filterbank_mfcc/) | Medium | Full MFCC pipeline from scratch; tests whether it actually separates vowel identity from pitch |
| 04 | [Pitch (F0) via autocorrelation](04_pitch_f0_autocorrelation/) | Simple–medium | Classic time-domain pitch tracker, accuracy measured against known synthetic F0s under increasing noise |
| 05 | [Voice Activity Detection](05_voice_activity_detection/) | Medium | Energy+ZCR VAD scored with real precision/recall; finds exactly why it breaks at high noise |
| 06 | [DTW word alignment](06_dtw_word_alignment/) | Medium | Dynamic Time Warping distinguishes the same word at a different speaking rate from a genuinely different word |
| 07 | [LPC formant estimation](07_lpc_formant_estimation/) | Medium | Levinson-Durbin recursion from scratch, recovers known formants to within 0.2-5% |
| 08 | [Spectral subtraction denoising](08_snr_spectral_subtraction/) | Medium | Noise estimation + denoising measured in real SNR dB, including when it backfires |

## 01 — DFT from scratch

A direct O(N²) implementation of the DFT formula matched
`numpy.fft.fft` to **1.94e-09 max absolute difference** — and was
**433x slower**, which is exactly why nobody uses the direct formula
on real audio. All 3 known tones (220, 440, 1000 Hz) were recovered
at their exact amplitudes, with 100% of spectral energy landing in
just those 3 bins out of 2000.

## 02 — Framing, windowing, pre-emphasis

Pre-emphasis boosted a weak 3000 Hz component relative to a dominant
120 Hz tone by **392.5x** (first-order high-pass doing exactly what
it's supposed to). Framing at 25ms/10ms produced the expected 60%
frame overlap. A Hamming window cut spectral leakage into an
unrelated frequency band by **2.9x** compared to an implicit
rectangular window.

## 03 — Mel filterbank + MFCC

Filter spacing confirmed mel-scale behavior directly: 75 Hz between
the first two filters, 643 Hz between two filters near the top of the
range. The real test: two synthetic vowels (formants [300, 2300] Hz
vs [800, 1200] Hz), each synthesized at two different pitches (110 Hz
and 220 Hz). MFCC distance across *vowels* came out **3.0x larger**
than MFCC distance across *pitches* of the same vowel — confirming
MFCCs do what they're actually used for in ASR: representing phonetic
content largely independent of pitch.

## 04 — Pitch via autocorrelation

Near-perfect on a clean signal (0.00-0.63% error across 100-300 Hz).
Degraded gracefully as noise increased (0.31% → 4.76% error), then
broke down completely at high noise — the estimate jumped to ~104 Hz
against a true 150 Hz, a classic octave/sub-harmonic tracking error.

## 05 — Voice Activity Detection

```
noise_std=0.6: precision=1.000 recall=0.858
noise_std=1.0: precision=0.000 recall=0.000  (zero predicted frames, not wrong ones)
```

Not the precision collapse I expected — a **recall** collapse, and a
specific mechanism: noise raises a frame's zero-crossing rate, so the
ZCR gate (originally there to *reject* noisy/unvoiced frames) starts
rejecting real speech frames too once the noise is loud enough, until
at noise_std=1.0 it rejects everything. Separately: disabling the ZCR
gate entirely dropped F1 from 0.982 to 0.707 at moderate noise — the
gate is doing real work, right up until it isn't.

## 06 — DTW word alignment

The same synthetic word at normal speed (46 frames) vs. 1.6x slower
(75 frames) — different lengths, so plain Euclidean distance isn't
even applicable. DTW distance for same-word-different-rate came out
**2.0x lower** than for a genuinely different word at the same rate,
which is the entire point of using DTW over fixed-length comparison
for isolated word recognition.

## 07 — LPC formant estimation

Levinson-Durbin recursion, hand-derived, matched scipy's generic
Toeplitz solver to **~1e-12** on the LPC coefficients themselves.
Formants recovered from the LPC roots landed within **0.2% to 5.2%**
of the true synthesis parameters across three different vowels.

## 08 — Spectral subtraction denoising

```
input SNR  -0.4 dB -> output  4.5 dB   (+4.9 dB improvement)
input SNR 19.6 dB -> output  9.0 dB   (-10.6 dB -- actively worse)
```

Output SNR plateaus around 7-9 dB regardless of input: spectral
subtraction helps a lot when the input is bad, but the spectral floor
(needed to avoid musical-noise artifacts) imposes its own distortion
once the input was already clean — it's a fix for bad SNR, not a free
quality upgrade. Separately: deliberately over-estimating the noise
by 3x made the output 2.5 dB *worse*, not better — more aggressive
subtraction isn't a safe default; getting the noise estimate right
matters more than how hard you subtract.
