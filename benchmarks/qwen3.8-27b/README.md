# Qwen3.8 27B on Strix Halo

Production targets are **UD-Q4_K_XL and UD-Q8_K_XL**. Optional speculative
routes are MTP and DFlash2. BF16 targets are quality references; DFlash2
drafts in Q4_K_M, Q8_0 and BF16 are being compared. Linux x86-64, gfx1151, 128 GB unified memory; Nix release builds.

Qualification is in progress. `TODO` means no current qualified measurement.
Prefill defaults to **pp2048** (`-p 4096` remains available), generation uses **tg128**, and `C` means concurrent
requests. Generation rates below are per user; aggregate rate is `C` times
that number. Draft precision is selected separately from target precision.

## Single user, autoregressive

Reference sweep at `023a13a`; full refresh: TODO.

| Context depth | Q4 pp / tg (tok/s) | Q8 pp / tg (tok/s) |
| ---: | ---: | ---: |
| 0 | 427.8 / 11.50 | 491.1 / 7.07 |
| 4,096 | 406.8 / 11.33 | 470.9 / 7.01 |
| 8,192 | 390.8 / 11.15 | 451.2 / 6.95 |
| 12,288 | 375.1 / 10.96 | 434.2 / 6.88 |
| 16,384 | 362.0 / 10.77 | 414.4 / 6.81 |

## Single user, speculative

Current qualified prefill / generation in tok/s; fixed blocks of seven
proposed tokens. One timed repetition after warmup. Every measured pairing
matches all 128 AR token IDs. MTP performance: **TODO**.

| Target | Depth | DFlash2 Q4_K_M | DFlash2 Q8_0 | DFlash2 BF16 |
| --- | ---: | ---: | ---: | ---: |
| Q4 | 0 | 395.1 / 20.91 | 396.5 / 17.67 | 397.5 / 16.90 |
| Q4 | 4,096 | 383.7 / 13.99 | TODO | TODO |
| Q4 | 8,192 | TODO | TODO | TODO |
| Q4 | 12,288 | TODO | TODO | TODO |
| Q4 | 16,384 | TODO | TODO | TODO |
| Q8 | 0 | 466.2 / 22.99 | 464.6 / 22.82 | 466.8 / 21.90 |
| Q8 | 4,096 | 445.9 / 15.84 | TODO | TODO |
| Q8 | 8,192 | TODO | TODO | TODO |
| Q8 | 12,288 | TODO | TODO | TODO |
| Q8 | 16,384 | TODO | TODO | TODO |

[Measurements and quality checks](eval/dflash2-rollback.json).
Unmeasured cells await refresh; the [previous full depth sweep](eval/dflash2-depths.json)
remains available as a historical reference. Acceptance varies with the
continuation, so generation speed need not decrease monotonically with depth.

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

All three drafts pass the pinned upstream operator comparison. Precision
selection remains open because acceptance depends on the target and prompt.
Controller comparisons follow kernel optimization.

The latest pass saves **50.5 MiB per speculative session** with compact
rollback, fuses draft normalization/RoPE and speeds up Q4 convolution
projections. Final profiles reduce copy time by 19%, draft QK/RoPE by
17–20%, and those Q4 projections by 7%; total generation changes remain small.
Q5/IQ4 prefill fusion and Q8 token grouping were slower or inconsistent in
model tests and are removed. Cache addressing now preserves exact logits across
mixed session capacities and at a 262,144-token logical context.
[Measurements, quality checks and rejected experiments](eval/dflash2-rollback.json).

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
