# Qwen3-TTS benchmarks

Measured on gfx1151 with the release build, warm, for the canonical sentence at
seed 42.

| | Native | Official ROCm | Ratio |
|---|---|---|---|
| Canonical request | `9.05`–`9.16` s for 23.12 s of audio | `98.00` s for 21.84 s | `10.9x` throughput |
| Real-time factor | `2.55x` | `0.22x` | — |
| Per codec frame | `31.0` ms | — | — |

Per-variant endpoint latency:

| Variant | Audio | Request |
|---|---|---|
| CustomVoice | 23.12 s | `9.09` s |
| VoiceDesign | 6.96 s | `2.90` s |
| Base ICL | 5.20 s | `2.69` s, or `2.45` s reusing the reference clip |

## Streaming

2026-09-20, Nix production binary, CustomVoice, resident model, 64 codec frames
(5.12 seconds of audio), seed 42; talker top-k 20 / top-p 0.85 / T 0.7,
predictor top-k 30 / top-p 0.9 / T 0.8. One warmup, one check per transport:

| Transport | First audio | Complete request |
| --- | ---: | ---: |
| Buffered WAV | 2.036 s | 2.036 s |
| PCM HTTP | 0.324 s | 2.068 s |
| SSE | 0.322 s | 2.059 s |
| WebSocket | 0.361 s | 2.099 s |

All four return identical PCM bytes. Two simultaneous requests also reproduce
those bytes, finishing in 2.037 / 4.073 s. These bounded controls measure early
delivery; they do not claim increased TTS model throughput or natural EOS.

A default-sampling CustomVoice paragraph reaches EOS at 208 frames / 16.64 s
of audio in 6.71 / 6.59 s on two requests, with exact full-WAV replay and 0%
word error in the native ASR check.

## Concurrent HTTP

Current sampled 64-frame CustomVoice control (same input as above):

| C | Request completion times |
| ---: | --- |
| 1 | 2.036 s |
| 2 | 2.037 / 4.073 s |
| 4 | TODO |
| 6 | TODO |
| 8 | TODO |

TTS model execution is serialized; output streaming improves first-audio latency.
Queued requests are cancellable. Single-request variant numbers above use
separate texts and cannot be compared as equivalent workloads.

## Reproduce

Use the [model server examples](README.md), a fixed text/reference voice,
checkpoint identity and seed. Measure resident requests after warmup; report
output audio duration, wall time and RTF. Use sampled requests for natural EOS;
limit codec frames for greedy operator controls. Profiler runs are separate.

Retained profile: talker/predictor GEMV 22.7 ms per frame, waveform decode
2.2 ms, other kernels 3.1 ms, dispatch gaps 3.8 ms. Projection traffic is near
the measured DRAM ceiling; these bounded-profile components are not a new
end-to-end latency measurement.
