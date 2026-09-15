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
| Q4_K_M | **23.66** | **15.79** |
| Q8_0 | 22.90 | 15.49 |
| BF16 | 22.13 | 14.26 |

pp2048/tg128 at depths **0 / 4,096 / 8,192 / 12,288 / 16,384**,
for Q4 and Q8 targets: **TODO**. MTP performance: **TODO**.
[Prompts, artifact identities and quality checks](eval/README.md).

## Multiple users, autoregressive

Short generation control, **2026-09-15**: `prose_tides`, context capacity 4096,
cached prompt, **tg64**, one warmup and one measured round. All concurrency
tables report **aggregate delivered tok/s**. C1 is the regression control.

| Concurrency | Q4 | Q8 |
| ---: | ---: | ---: |
| 1 | 11.79 | 7.10 |
| 2 | 22.98 | 14.38 |
| 4 | 41.84 | 27.17 |
| 6 | 57.08 | 38.74 |
| 8 | 67.74 | 49.66 |

pp2048/tg128 across all five context depths at C2/4/6/8: **TODO**.

## Multiple users, DFlash2

Q4_K_M draft, same workload and units. **Adaptive remains the default.**
Fixed 1 uses `--draft-policy fixed --draft-tokens 1`. Each request keeps its
own sampling, caches and recurrent state.

| Concurrency | Q4 adaptive | Q4 fixed 1 | Q8 adaptive | Q8 fixed 1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | **23.66** | 18.55 | **15.79** | 12.03 |
| 2 | 29.17 | **30.72** | **24.73** | 21.09 |
| 4 | 31.63 | **44.60** | 25.45 | **34.43** |
| 6 | 32.79 | **46.77** | 25.61 | **43.14** |
| 8 | 34.20 | **49.18** | 26.79 | **46.38** |

Greedy output matches AR throughout. Temperature 0.8 / seed 42 reproduces C1
output within each target/controller configuration at C2/4/6/8. Higher
concurrency favors shorter blocks on this prompt. Separate JSON and repetition
controls favor adaptive at C8 on both targets; see the [quality guide](eval/README.md).
Controller tuning across workloads and the pp2048/tg128 depth sweep remain **TODO**.

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
