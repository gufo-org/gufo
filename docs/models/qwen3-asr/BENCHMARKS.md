# Qwen3-ASR benchmarks

2026-09-21, Linux gfx1151, BF16, Nix production binary `ce73ab914d06`
(SHA-256 prefix). Resident model; one warmup and three timed requests.
RTF = request time / input duration, lower is better.

Warm-file-cache CLI model setup: **0.37 s**. Cold HTTP readiness after dependency
removal: **TODO**. The decoder excludes the audio tower's weights, saving
**606 MiB** of device storage; the complete 49-token transcript still passes.

## Single request

15.05125-second English recording, 49 greedy output tokens matching the official
reference. Timings include audio decoding, features, encoder and text generation;
model loading is excluded.

| Metric | Result |
| --- | ---: |
| Median request | 985.67 ms |
| RTF | 0.0655 |
| Audio seconds per wall second | 15.27 |
| Audio encoder | 33.25 ms |
| Text prefill and generation | 944.25 ms |

Stage medians are measured separately; do not sum them into another headline.
Native convolutions remove MIOpen and improve the encoder by 8.2% in the matched
control. Complete-request latency remains about 0.99 s because text generation
dominates; no substantial whole-model gain is claimed.
See [quality](QUALITY.md) and the
[measurement record](artifacts/native-convolution.json).

## Concurrent HTTP

| C | Current latency / throughput |
| ---: | --- |
| 1 | See the resident control above; HTTP adds transport overhead. |
| 2 / 4 / 6 / 8 | TODO: refresh after chunk-level admission changes. |

GPU work remains serialized between chunks. CPU frontend preparation can overlap
for two requests, and long recordings yield device admission between chunks.
File SSE and committed Realtime utterances preserve buffered transcript output.

## Profile and reproduction

The final warm request in the separate profile has 13,253 dispatches,
1,275.19 ms of GPU work and a 1,308.67 ms kernel span: **97.4% GPU-busy**.
It contains three native convolution launches; no im2col tensor or MIOpen
launch loop remains. Text and library projections dominate. Profiling overhead
is excluded from the timing table; the measured unprofiled request is 985.67 ms.

```sh
nix build
nix develop -c python3 src/models/qwen3_asr/tools/benchmark.py \
  --model "$MODEL" --audio speech.wav --warmup 1 --repeat 3
```

The benchmark uses the CLI's decoded sample count and supports float WAV.
Record fixture/model/binary identities. Profile separately with
`tools/prof/prof.py --stages qwen-asr`; raw traces stay outside Git.
Long-form corpus accuracy and current concurrent throughput remain **TODO**.
