# Qwen3.8 27B on Strix Halo

Linux x86-64, gfx1151, 128 GB unified memory; Nix release binaries.
Production targets: **UD-Q4_K_XL / UD-Q8_K_XL**. Recommended DFlash2 draft:
**Q4_K_M**, with **adaptive** as the default controller. Q8_0 and BF16 drafts
remain supported; a full comparison across context depths is **TODO**.

## Single user, autoregressive

Standard sweep: **pp2048 / tg128**, C1. Cells are prefill / generation tok/s.
`TODO` means the current implementation needs a qualified measurement.

| Context depth | Q4 | Q8 |
| ---: | ---: | ---: |
| 0 | **610.56** / TODO | **503.17** / TODO |
| 4,096 | TODO | TODO |
| 8,192 | TODO | TODO |
| 12,288 | TODO | TODO |
| 16,384 | TODO | TODO |

Prefill measured with 12 warmed release samples for Q4 (2026-09-14) and
two for Q8 (2026-09-13).
Q4 uses FP16 activations with packed quantized weights; Q8 retains native wave64.

## Single user, DFlash2

Short control, **2026-09-15**: cached `prose_tides`, **tg64**, adaptive,
greedy C1, one warmed release sample per draft. Generation tok/s:

| Draft | Q4 target | Q8 target |
| --- | ---: | ---: |
| Q4_K_M | **23.63** | **15.72** |
| Q8_0 | 22.79 | 15.49 |
| BF16 | 22.15 | 14.26 |

pp2048/tg128 at depths **0 / 4,096 / 8,192 / 12,288 / 16,384**,
for Q4 and Q8 targets: **TODO**. MTP performance: **TODO**.
[Prompts, artifact identities and quality checks](eval/README.md).

## Multiple users, autoregressive

Short generation control, **2026-09-15**: `prose_tides`, context capacity 4096,
cached prompt, **tg64**, one warmup and one measured round. All concurrency
tables report **aggregate delivered tok/s**. C1 is the regression control.

| Concurrency | Q4 | Q8 |
| ---: | ---: | ---: |
| 1 | 11.82 | 7.10 |
| 2 | 22.98 | 14.38 |
| 4 | 41.87 | 27.13 |
| 6 | 57.01 | 38.72 |
| 8 | 67.77 | 49.65 |

pp2048/tg128 across all five context depths at C2/4/6/8: **TODO**.

## Multiple users, DFlash2

Q4_K_M draft, adaptive controller, greedy **tg64**, cached prompts, context
capacity 4096. **2026-09-15–16**, one warmed pass; Q4 C1/C4 repetition uses
three warmed rounds. C1 warms all three mixed-corpus prompts.
Aggregate delivered tok/s:

| Concurrency | Q4 repetition | Q4 mixed corpus | Q8 repetition | Q8 mixed corpus |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 61.69 | 36.70 | 48.79 | 25.10 |
| 2 | 81.33 | 41.47 | 73.99 | 31.64 |
| 4 | 90.84 | 45.95 | 85.21 | 35.12 |
| 6 | 93.07 | 53.98 | 90.14 | 41.80 |
| 8 | 97.20 | 55.93 | 93.32 | 43.11 |

Repetition accepts **100%** of proposals. The mixed corpus contains 24 requests:
eight each of code, JSON and prose; acceptance is **61.21% for Q4 / 49.13% for
Q8**. Throughput is total output tokens divided by the sum of measured
request-group spans. C8 mixed latency (median / p95): **Q4 6.91 / 9.44 s;
Q8 8.53 / 12.58 s**. Physical widths, output hashes and acceptance counts are
checked at every concurrency; both C1 controls retain their performance.
The quality guide identifies the separate C1 mixed-corpus control.

Target quantized projections are the main scaling limit. C1 DFlash2 already
verifies seven or eight positions together; C2/C4 increase that work to
14–16/28–32 positions. Larger batches currently bring limited additional
throughput. The **100 tok/s at C2 / 150 tok/s at C4** targets remain unmet.

Drafting and context injection share projections across requests. Exact
verification uses wider row groups and skips rejected suffixes while retaining
private proposals, sampling draws and controller feedback. **Adaptive remains
the default**; controller comparisons and pp2048/tg128 across the five context
depths remain **TODO**.
[Prompts, methodology and quality checks](eval/README.md).

## Reproduce and maintain quality

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
[the HTTP benchmark](../../tools/README.md) for sampled or concurrent requests.
DFlash2 prefill includes feature capture and draft context injection.

Start with `nix develop -c python3 tools/qwen27b/check.py fast`, then run the
[affected quality checks](eval/README.md) before publishing speed. Greedy
speculation must match AR IDs; sampled verification must preserve the target
distribution and reproduce seeded runs within the same configuration.
Independent original-target/MTP qualification and the full context/concurrency
sweep: **TODO**.

Current optimization target: **Q4 and Q8 generation at C2, C4, C6 and C8,
with and without DFlash2**. C1 must retain its performance. Track aggregate
throughput, per-user latency, physical batch width, output correctness and
seeded sampling at every concurrency level. Screen changes with short runs;
check retained changes across C1/2/4/6/8 before publishing speed.
