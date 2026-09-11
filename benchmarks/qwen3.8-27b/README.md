# Qwen3.8 27B on Strix Halo

Production targets are **UD-Q4_K_XL and UD-Q8_K_XL**. Optional speculative
routes are MTP and DFlash2. BF16 targets are quality references; DFlash2
drafts in Q4_K_M, Q8_0 and BF16 are being compared. Linux x86-64, gfx1151, 128 GB unified memory; Nix release builds.

Qualification is in progress. `TODO` means no current qualified measurement.
Prefill defaults to **pp2048** (`-p 4096` remains available), generation uses **tg128**, and `C` means concurrent
requests. Generation rates below are per user; aggregate rate is `C` times
that number. Draft precision is selected separately from target precision.

## Single user, autoregressive

| Context depth | Q4 pp / tg (tok/s) | Q8 pp / tg (tok/s) |
| ---: | ---: | ---: |
| 0 | 427.8 / 11.50 | 491.1 / 7.07 |
| 4,096 | 406.8 / 11.33 | 470.9 / 7.01 |
| 8,192 | 390.8 / 11.15 | 451.2 / 6.95 |
| 12,288 | 375.1 / 10.96 | 434.2 / 6.88 |
| 16,384 | 362.0 / 10.77 | 414.4 / 6.81 |

## Single user, speculative

Prefill / generation in tok/s; fixed blocks of seven proposed tokens. Each
pairing matches all 128 AR token IDs at every depth. MTP performance: **TODO**.

| Target | Depth | DFlash2 Q4_K_M | DFlash2 Q8_0 | DFlash2 BF16 |
| --- | ---: | ---: | ---: | ---: |
| Q4 | 0 | 386.3 / 19.24 | 386.6 / 16.22 | 386.7 / 15.53 |
| Q4 | 4,096 | 372.8 / 12.94 | 372.5 / 12.63 | 374.0 / 12.11 |
| Q4 | 8,192 | 361.1 / 35.81 | 360.6 / 35.65 | 360.6 / 34.35 |
| Q4 | 12,288 | 348.0 / 33.09 | 347.3 / 32.11 | 348.2 / 30.95 |
| Q4 | 16,384 | 334.7 / 13.53 | 335.1 / 13.79 | 332.6 / 13.31 |
| Q8 | 0 | 450.1 / 21.57 | 448.4 / 21.38 | 451.3 / 20.46 |
| Q8 | 4,096 | 432.6 / 14.87 | 432.0 / 14.46 | 431.9 / 13.90 |
| Q8 | 8,192 | 415.0 / 33.63 | 416.1 / 33.49 | 414.5 / 32.29 |
| Q8 | 12,288 | 398.2 / 34.91 | 396.9 / 34.81 | 396.9 / 33.64 |
| Q8 | 16,384 | 373.9 / 31.38 | 381.5 / 31.26 | 379.5 / 30.24 |

These are single timed repetitions after warmup at revision `023a13a`, not
estimates of measurement variance. Acceptance varies with the synthetic
continuation, so generation speed need not decrease monotonically with depth.
[Raw measurements and token hashes](eval/dflash2-depths.json).

## Multiple users, autoregressive

Cells are aggregate prefill / per-user generation, in tok/s.

| Context depth | Q4 C2 | Q8 C2 | Q4 C4 | Q8 C4 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | TODO | TODO | TODO | TODO |
| 4,096 | TODO | TODO | TODO | TODO |
| 8,192 | TODO | TODO | TODO | TODO |
| 12,288 | TODO | TODO | TODO | TODO |
| 16,384 | TODO | TODO | TODO | TODO |

## Multiple users, speculative

Cells are aggregate prefill / per-user generation, in tok/s. Batched
speculative serving is being qualified before these numbers are published.

| Context depth | Q4 C2 | Q8 C2 | Q4 C4 | Q8 C4 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | TODO | TODO | TODO | TODO |
| 4,096 | TODO | TODO | TODO | TODO |
| 8,192 | TODO | TODO | TODO | TODO |
| 12,288 | TODO | TODO | TODO | TODO |
| 16,384 | TODO | TODO | TODO | TODO |

## Draft choice and latest optimization

All three drafts pass the pinned upstream operator comparison. Final precision
selection remains open: compare companions against the same target and prompts,
since acceptance depends on the target's continuation. Controller comparisons
follow kernel optimization.

Latest short release A/B against `85b9994`: **C1, pp2048, tg128, depth 0**,
one timed repetition after warmup. Values are baseline → candidate, in tok/s.
This is separate from the earlier full depth sweep above.

| Target | Draft | Prefill | Generation |
| --- | --- | ---: | ---: |
| Q4 | Q4_K_M | 388.6 → 397.9 | 19.40 → 19.92 |
| Q4 | Q8_0 | 387.8 → 397.4 | 16.33 → 16.79 |
| Q4 | BF16 | 388.2 → 396.7 | 15.63 → 16.05 |
| Q8 | Q4_K_M | 451.8 → 464.4 | 21.61 → 22.30 |
| Q8 | Q8_0 | 448.2 → 464.0 | 21.42 → 22.17 |
| Q8 | BF16 | 452.1 → 466.2 | 20.49 → 21.22 |

Recurrence state stays in registers across verification rows; committed replay
batches rows per layer and skips unused output work. Feature injection shares
BF16 weights across sixteen FP32 rows. Generation improves **2.7–3.6%** and
prefill **2.2–3.5%** in this probe; all twelve speculative traces match AR.
Full logits, recurrent state and all 270 upstream-qualified draft trace files
remain byte-identical. [Measurements, profiles and quality evidence](eval/dflash2-recurrence.json).

## Reproduce

```sh
nix build
MODEL=/path/to/Qwen3.8-27B-UD-Q4_K_XL.gguf
DRAFT=/path/to/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
MTP=/path/to/mtp-Qwen3.8-27B-Q4_0.gguf

./result/bin/gufo bench --model "$MODEL" \
  -p 2048 -n 128 -d 0,4096,8192,12288,16384 -c 1 -r 2 -v
./result/bin/gufo bench --model "$MODEL" \
  --speculative dflash2 --dflash-model "$DRAFT" \
  -p 2048 -n 128 -d 0,4096,8192,12288,16384 -c 1 -r 2 -v
./result/bin/gufo bench --model "$MODEL" \
  --speculative mtp --mtp-model "$MTP" \
  -p 2048 -n 128 -d 0,4096,8192,12288,16384 -c 1 -r 2 -v
```

Only DFlash2 is implemented; the legacy `dflash` and `dflash-2` CLI spellings
select the same backend. The GGUF architecture name remains `dflash`.

DFlash2 prefill includes target feature capture and draft context injection.
The verbose depth traces must match AR token IDs for the same target.

Use short probes while developing. Run hardware jobs sequentially, profile in
a separate pass, and alternate baseline/candidate release measurements.
Synthetic depth sweeps measure the engine; chat corpus comparisons measure
acceptance on useful workloads. Report both without combining their rates.

## Quality and feature parity

The [quality report](eval/README.md) records the maintained checks, pinned
DFlash2 formulas, evidence and remaining gaps. Start with
`nix develop -c python3 tools/qwen27b/check.py fast`, then run only the affected
kernel/model suites. Publish speed only after the affected quality checks pass; label short probes explicitly.

Production execution has one implementation per supported shape/weight type;
Qwen kernel, precision and verification environment switches are removed,
along with inactive fusion/prefetch policies and their kernels.
DFlash2 retains FP32 values in a bounded history ring. MTP replay uses the
committed target features. The optimized correctness build keeps assertions
and symbols; benchmark only the Nix release binaries.

Remaining: independent original-target/MTP qualification, MTP and C>1 speed
measurements, and speculative serving parity for C>1.
