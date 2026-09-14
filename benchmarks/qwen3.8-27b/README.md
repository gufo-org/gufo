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
| 0 | **613.65** / TODO | **503.17** / TODO |
| 4,096 | TODO | TODO |
| 8,192 | TODO | TODO |
| 12,288 | TODO | TODO |
| 16,384 | TODO | TODO |

Prefill measured with six warmed release samples for Q4 (2026-09-14) and
two for Q8 (2026-09-13).
Q4 uses FP16 activations with packed quantized weights; Q8 retains native wave64.

## Single user, DFlash2

Current short control, **2026-09-14**, Q4 target, adaptive, greedy C1.
One warmed release sample per draft; all 32 generated IDs match AR and
acceptance is 22.2%. Prefill includes feature capture and draft injection.
These **tg32 controls are not the full tg128 sweep**.

| Draft | pp2048 tok/s | tg32 tok/s |
| --- | ---: | ---: |
| Q4_K_M | **572.84** | **16.48** |
| Q8_0 | 569.07 | 16.09 |
| BF16 | 571.57 | 15.09 |

pp2048/tg128 at depths **0 / 4,096 / 8,192 / 12,288 / 16,384**,
for Q4 and Q8 targets: **TODO**. MTP performance: **TODO**.
[Prompts, artifact identities and quality checks](eval/README.md).

## Multiple users, autoregressive

Q4 and Q8, pp2048/tg128 at the same five depths:

| Concurrency | Aggregate prefill / per-user generation |
| ---: | --- |
| 2 | TODO |
| 4 | TODO |

## Multiple users, DFlash2

Q4 and Q8 targets with Q4_K_M/adaptive, same workload and reporting units:

| Concurrency | Aggregate prefill / per-user generation |
| ---: | --- |
| 2 | TODO |
| 4 | TODO |

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
Independent original-target/MTP qualification and complete C>1 parity: **TODO**.
