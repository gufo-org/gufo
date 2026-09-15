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

Current short control, **2026-09-14**, Q4 target, adaptive, greedy C1.
One warmed release sample per draft; all 32 generated IDs match AR and
acceptance is 22.2%. Prefill includes feature capture and draft injection.
These **tg32 controls are not the full tg128 sweep**.

| Draft | pp2048 tok/s | tg32 tok/s |
| --- | ---: | ---: |
| Q4_K_M | **573.56** | **16.42** |
| Q8_0 | 569.45 | 16.05 |
| BF16 | 574.41 | 15.10 |

pp2048/tg128 at depths **0 / 4,096 / 8,192 / 12,288 / 16,384**,
for Q4 and Q8 targets: **TODO**. MTP performance: **TODO**.
[Prompts, artifact identities and quality checks](eval/README.md).

## Multiple users, autoregressive

Short generation control, **2026-09-15**: `prose_tides`, context capacity 4096,
cached prompt, **tg64**, one warmup and one measured round. Cells are
**aggregate / per-user whole-request tok/s**. C1 is the regression control.

| Concurrency | Q4 | Q8 |
| ---: | ---: | ---: |
| 1 | 11.79 / 11.79 | 7.10 / 7.10 |
| 2 | 22.89 / 11.45 | 14.36 / 7.18 |
| 4 | 41.38 / 10.35 | 27.00 / 6.75 |
| 6 | 55.99 / 9.33 | 38.02 / 6.34 |
| 8 | 65.77 / 8.22 | 48.70 / 6.09 |

pp2048/tg128 across all five context depths at C2/4/6/8: **TODO**.

## Multiple users, DFlash2

Q4_K_M/adaptive, same short workload and units. Target verification shares
projections across requests; draft generation and recurrent state remain
independent for each user.

| Concurrency | Q4 target | Q8 target |
| ---: | ---: | ---: |
| 1 | 23.66 / 23.66 | 15.68 / 15.68 |
| 2 | 24.81 / 12.40 | 16.18 / 8.09 |
| 4 | 28.92 / 7.23 | 17.70 / 4.42 |
| 6 | 29.71 / 4.95 | 17.46 / 2.91 |
| 8 | 30.94 / 3.87 | 18.14 / 2.27 |

Greedy output matches AR at every width. Temperature 0.8 / seed 42 reproduces
C1 output within each target/mode configuration at C2/4/6/8. These controls
do not replace the pp2048/tg128 depth sweep, which remains **TODO**.

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
