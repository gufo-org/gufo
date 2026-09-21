# Qwen3.8 27B on Strix Halo

Linux x86-64, gfx1151, 128 GB unified memory; Nix release binaries.
Production targets: **UD-Q4_K_XL / UD-Q8_K_XL**. Recommended DFlash2 draft:
**Q4_K_M**, with **adaptive** as the default controller. Q8_0 and BF16 drafts
remain supported; a full comparison across context depths is **TODO**.

PNG/JPEG image input uses the matching BF16 projector with AR or DFlash2.
Native MTP is CLI-only. [Image usage and quality checks](README.md#images).

Each table identifies its workload and measurement date. Unmeasured points
are **TODO**; the concurrency sweep still needs a performance refresh.

## Loading and continuation

2026-09-21, one cold-file-cache launch to HTTP readiness, including Q4_K_M
DFlash2 and two sessions at context capacity 262144:

| Target | Ready |
| --- | ---: |
| Q4 | 6.38 s |
| Q8 | 8.60 s |

A 626-token Q4+DFlash2 prompt snapshot occupies **216 MiB**, independent of
unused context capacity. Cancel/continue reused 661 tokens and prefilled
61 new tokens; restoring the prompt snapshot took **3.13 ms**.
These are bounded controls, not a long-conversation latency distribution.

## Single user, autoregressive

Standard sweep: **pp2048 / tg128**, C1. Cells are prefill / generation tok/s.
`TODO` means the current implementation needs a qualified measurement.

| Context depth | Q4 | Q8 |
| ---: | ---: | ---: |
| 0 | **622.00 / 11.74** | **445.86 / 7.11** |
| 4,096 | TODO | TODO |
| 8,192 | TODO | TODO |
| 12,288 | TODO | TODO |
| 16,384 | TODO | TODO |
| 32,768 | **467.63 / 10.30** | TODO |

Measured **2026-09-19**, one warmed release sample per point; Q4 d0 is the
mean of two controls (PP range 620.38–623.61 tok/s).
## Single user, DFlash2

**pp2048 / tg128**, C1, greedy, Q4_K_M draft and adaptive controller.
Cells are prefill / generation tok/s. Depth zero refreshed **2026-09-21**;
32K measured **2026-09-19**. One warmed sample per point.

| Context depth | Q4 target | Q8 target |
| ---: | ---: | ---: |
| 0 | **635.75 / 27.86** | **464.32 / 25.03** |
| 4,096 | TODO | TODO |
| 8,192 | TODO | TODO |
| 12,288 | TODO | TODO |
| 16,384 | TODO | TODO |
| 32,768 | **442.86 / 31.01** | TODO |

The repetitive Q4 CLI input accepts 96.5% of proposals at d32768.
Greedy outputs match AR. Natural prompts have different acceptance and speed.

Draft-precision control, **2026-09-16**: cached `prose_tides`, **tg64**,
adaptive, greedy C1, one warmed release sample per draft. Generation tok/s:

| Draft | Q4 target | Q8 target |
| --- | ---: | ---: |
| Q4_K_M | **23.51** | **15.61** |
| Q8_0 | 22.66 | 15.33 |
| BF16 | 21.96 | 14.11 |

Full depth comparison across all three drafts and MTP performance: **TODO**.
[Prompts, artifact identities and quality checks](EVALUATION.md).

## Multiple users, autoregressive

Short generation controls, **2026-09-15/16**: `prose_tides`, context capacity 4096,
cached prompt, **tg64**, one warmup and one measured round. All concurrency
tables report **aggregate delivered tok/s**. C1 is the regression control.

| Concurrency | Q4 | Q8 |
| ---: | ---: | ---: |
| 1 | 11.82 | 7.10 |
| 2 | 22.98 | 14.38 |
| 4 | 41.87 | 27.13 |
| 6 | 57.01 | 38.72 |
| 8 | 68.08 | 49.03 |

pp2048/tg128 across all five context depths at C2/4/6/8: **TODO**.

## Multiple users, DFlash2

Q4_K_M draft, adaptive controller, greedy **tg64**, cached prompts, context
capacity 4096. **2026-09-16**, three warmed rounds for C1/C4 repetition and Q8
C6 repetition; one warmed pass for the remaining cells.
All three mixed-corpus prompts are warmed before measurement.
Aggregate delivered tok/s:

| Concurrency | Q4 repetition | Q4 mixed corpus | Q8 repetition | Q8 mixed corpus |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 62.08 | 36.46 | 48.85 | 24.77 |
| 2 | 84.07 | 41.85 | 78.64 | 33.00 |
| 4 | 93.62 | 46.48 | 89.60 | 35.66 |
| 6 | 96.06 | 55.02 | 93.48 | 42.34 |
| 8 | 100.37 | 56.73 | 96.81 | 43.92 |

Repetition accepts **100%** of proposals. The mixed corpus contains 24 requests:
eight each of code, JSON and prose; acceptance is **61.21% for Q4 / 49.13% for
Q8**. Throughput is total output tokens divided by the sum of measured
request-group spans. C8 mixed latency (median / p95): **Q4 6.75 / 9.33 s;
Q8 8.36 / 12.37 s**. Physical widths, output hashes and acceptance counts are
checked at every concurrency; both C1 controls retain their performance.
The quality guide records each control's release.

Target quantized projections are the main scaling limit. C1 DFlash2 already
verifies seven or eight positions together; C2/C4 increase that work to
14–16/28–32 positions. Larger batches currently bring limited additional
throughput. The **100 tok/s at C2 / 150 tok/s at C4** targets remain unmet.

Controller and verification details: [evaluation](EVALUATION.md).

## Reproduce and maintain quality

The shared image encoder, measured separately on 2026-09-20 with
`mmproj-BF16.gguf`, takes **1252 ms** for a 1024×1024 RGB image (1024 merged
tokens). This is a warm encode, excluding image preprocessing, first weight
upload and language-model prefill. Q4 and Q8 use the same projector. Embeddings
are byte-identical to the prior encoder; see [experiments](EXPERIMENTS.md).

```sh
nix build
MODEL=/path/to/Qwen3.8-27B-UD-Q4_K_XL.gguf
DRAFT=/path/to/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
./result/bin/gufo bench --model "$MODEL" \
  -p 2048 -n 128 -d 0,4096,8192,12288,16384 -c 1 -r 2 -v
./result/bin/gufo bench --model "$MODEL" \
  --speculative dflash2 --dflash-model "$DRAFT" \
  -p 2048 -n 128 -d 0,4096,8192,12288,16384 -c 1 -r 2 -v
```

Use `-p 4096` for pp4096. Qwen `bench` is greedy and C1; use
[the HTTP benchmark](../../../tools/README.md) for sampled or concurrent requests.
DFlash2 prefill includes feature capture and draft context injection.

Start with `nix develop -c python3 tools/qwen27b/check.py fast`, then run the
[affected quality checks](EVALUATION.md) before publishing speed. Greedy
speculation must match AR IDs; sampled verification must preserve the target
distribution and reproduce seeded runs within the same configuration.
Independent original-target/MTP qualification and the full context/concurrency
sweep: **TODO**.

Optimization targets: **prefill scaling with context, and Q4/Q8 generation
at C2, C4, C6 and C8 with and without DFlash2**. C1 must retain its performance. Track aggregate
throughput, per-user latency, physical batch width, output correctness and
seeded sampling at every concurrency level. Screen changes with short runs;
check retained changes across C1/2/4/6/8 before publishing speed.
