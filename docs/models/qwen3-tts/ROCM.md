# ROCm 10.1 migration

October 24, 2026, Strix Halo `gfx1151`. CustomVoice, VoiceDesign and Base use
the original BF16 checkpoints and F32 waveform decoder. ROCm 7.2.3 uses pinned
Nix dependencies and Clang 22. ROCm 10.1.0 was first checked with Clang 24/GCC
15.2 in `rocm/dev-ubuntu-26.04:10.1.0-full`, then built through
`nix build` with cached Clang 24/GCC 16.2 and the repository's matching
ROCr patch. Source `65bf6798` includes main `67d1f6ac`; this model's kernels are unchanged
from the independently checked `6d43e786`.
Builds and inference run separately.

## Required correction

ROCm 10.1 redirects single-frame in-place decoder GEMMs to a GEMV with a
different reduction tree. Splitting a stream at frame 301 then changes its
audio, breaking the exact cached-reference contract.

Single-frame projections now use a separate zero-beta input, retaining the
matrix reduction. Its storage is prepared with the weights to avoid allocation
in a streamed tail. Longer chunks keep their existing dispatch. No precision,
sampling rule or quality limit changes.

| Migration check | Result |
| --- | --- |
| Official talker/predictor and Base encoder controls | Original limits pass on container and Nix 10.1 |
| Fixed-code waveform versus official | MAE 1.33e-7, maximum error 1.28e-6 on both; limits 1e-5/1e-4 |
| Incremental decoder across 300-frame reset | Exact after the correction on 7.2.3, container 10.1 and Nix 10.1 |
| Cached-reference suffix, replacement and unrelated-request isolation | Exact after the correction, both compilers |
| Seed replay, PCM/WAV agreement, C2 requests, cancellation/reuse | All three variants pass the bounded HTTP checks |
| Default-sampled English intelligibility | Same WER on both: CustomVoice 1/37 words; VoiceDesign and Base 0/37 |

Intelligibility uses the paragraph in [QUALITY.md](QUALITY.md), seed 42 and
natural EOS, recognized with the same pinned 7.2.3 ASR control. Same-seed audio
can differ between compilers; exact replay is required within each runtime.
This small sample does not establish multilingual or long-form equivalence.

## Final Nix timing control

Production binaries, resident HTTP, 64 codec frames, same paragraph, seed 42,
talker temperature/top-k/top-p 0.7/20/0.85 and predictor 0.8/30/0.9:

| Warm request | ROCm 7.2.3 | ROCm 10.1 |
| --- | ---: | ---: |
| CustomVoice | 2,019 ms | 2,037 ms |
| VoiceDesign | 2,026 ms | 2,050 ms |
| Base, cached reference | 2,096 ms | 2,122 ms |

Final buffered, streamed and post-cancellation timings pass the 5% and 3 ms
gate. Concurrent clients must be compared by admission order, since the
device admits them serially. All six natural-completion WAVs match their
previously scored container/control output bytes within the same compiler
version, preserving the independent intelligibility results above. Base emits
158 codec frames on 7.2.3 and 157 on 10.1; that natural-completion wall time is
not a matched-work timing control.

The reference revisions and existing upstream parity limits remain in
[QUALITY.md](QUALITY.md). Raw profiles, requests and WAVs are retained under
`gufo-models-rocm101-20261023/port-models` on the test host.
[BENCHMARKS.md](BENCHMARKS.md) keeps its published figures.
