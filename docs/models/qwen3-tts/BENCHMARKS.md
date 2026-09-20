# Qwen3-TTS benchmarks

2026-09-20, Linux gfx1151, BF16, Nix production build
`8e4ba46f9dd0` (binary SHA-256 prefix). Resident models; ASR 1.7B loaded but idle.
The same [37-word paragraph](EVALUATION.md) is used for all three variants:
seed 42, automatic language, talker T 0.7 / top-k 20 / top-p 0.85,
predictor T 0.8 / top-k 30 / top-p 0.9.

## Single request

64 codec frames = 5.12 seconds of audio. Buffered latency is the median of three
requests after one warmup; PCM streaming is one separate check.
RTF = request time / generated audio duration, lower is better.

| Variant | Buffered WAV | RTF | First PCM audio | Complete PCM |
| --- | ---: | ---: | ---: | ---: |
| CustomVoice | 2.018 s | 0.394 | 0.201 s | 2.059 s |
| VoiceDesign | 2.024 s | 0.395 | 0.201 s | 2.065 s |
| Base ICL | 2.234 s | 0.436 | 0.345 s | 2.247 s |

All three variants return identical buffered/streamed PCM and repeat their
seeded waveforms exactly. Base uses a cached 15.05-second reference clip with
its complete transcript. Encoder features and waveform-reference state are
cached; reference-code talker prefill remains part of each request.
Its latency is not directly comparable to text-only CustomVoice/VoiceDesign.

The first streamed chunk contains four codec frames (320 ms of audio);
subsequent chunks contain 16. This changes delivery latency, not generated audio.
Natural-EOS duration and intelligibility are reported in
[evaluation](EVALUATION.md), separately from fixed-work speed checks.

## Transports and concurrency

One warmed loopback check per transport; WebSocket uses TCP_NODELAY:

| Transport | First audio | Complete request |
| --- | ---: | ---: |
| PCM HTTP | 0.199 s | 2.053 s |
| SSE | 0.201 s | 2.051 s |
| WebSocket | 0.199 s | 2.048 s |

WAV, PCM, SSE and WebSocket return identical PCM bytes.

| C | Request completion times |
| ---: | --- |
| 1 | 2.018 s |
| 2 | 2.025 / 4.046 s |
| 4 / 6 / 8 | TODO |

TTS execution remains serialized; streaming delivers audio earlier.
Queued requests are cancellable. Concurrency does not imply batched model work.

## Profile and reproduction

Isolated warm 64-frame profiles exclude loading, warmup and idle server time:

| Variant | Dispatches | GPU work | GPU busy / kernel span | Projection share |
| --- | ---: | ---: | ---: | ---: |
| CustomVoice | 68,241 | 1.899 s | 90.5% | 76.6% |
| VoiceDesign | 68,054 | 1.915 s | 90.5% | 76.1% |
| Base ICL | 69,737 | 2.107 s | 90.9% | 69.3% |

Projection traffic remains the main bottleneck. Base additionally spends 16.7%
in library prefill/waveform operations and 5.7% in talker/predictor attention.
Profile overhead is excluded from the timing tables.

Use the [server examples](README.md) with the settings above. Record model and
binary identities, exact text/reference, output duration, wall time and RTF.
Warm the resident model once; run the profiler separately with
`tools/prof/prof.py` and `--stages qwen-tts`. Raw profiles and audio stay outside
Git. Retained and rejected optimizations are listed in
[experiments](EXPERIMENTS.md).
