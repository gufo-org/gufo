# Qwen3-ASR benchmarks

2026-09-20, Linux gfx1151, BF16, Nix production binary `8e4ba46f9dd0`
(SHA-256 prefix). Resident model; one warmup and three timed requests.
RTF = request time / input duration, lower is better.

## Single request

15.05125-second English recording, 49 greedy output tokens matching the official
reference. Timings include audio decoding, features, encoder and text generation;
model loading is excluded.

| Metric | Result |
| --- | ---: |
| Median request | 988.80 ms |
| RTF | 0.0657 |
| Audio seconds per wall second | 15.22 |
| Audio encoder | 35.28 ms |
| Text prefill and generation | 943.98 ms |

Stage medians are measured separately; do not sum them into another headline.
The matched unchanged control is 986.68 ms: this pass preserves speed, without
an established ASR throughput gain. Four complete-recording transcripts remain
unchanged. See [evaluation](EVALUATION.md) for exactness and reference scope.

## Concurrent HTTP

| C | Current latency / throughput |
| ---: | --- |
| 1 | See the resident control above; HTTP adds transport overhead. |
| 2 / 4 / 6 / 8 | TODO: refresh after chunk-level admission changes. |

GPU work remains serialized between chunks. CPU frontend preparation can overlap
for two requests, and long recordings yield device admission between chunks.
File SSE and committed Realtime utterances preserve buffered transcript output.

## Profile and reproduction

The isolated warm request has 13,346 dispatches, 963.05 ms of GPU work and a
996.35 ms kernel span: **96.7% GPU-busy**. Text projections account for **77.6%**
of GPU time, library audio/prefill projections 10.8%, text attention 6.1% and
text normalization 2.8%. Projection bandwidth remains the main bottleneck.
Profiling overhead is excluded from the timing table.

```sh
nix build
nix develop -c python3 src/models/qwen3_asr/tools/benchmark.py \
  --model "$MODEL" --audio speech.wav --warmup 1 --repeat 3
```

The benchmark uses the CLI's decoded sample count and supports float WAV.
Record fixture/model/binary identities. Profile separately with
`tools/prof/prof.py --stages qwen-asr`; raw traces stay outside Git.
Long-form corpus accuracy and current concurrent throughput remain **TODO**.
