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

## Draft choice

Compare companions against the **same target and chat-framed prompts**.
Q4 and Q8 targets can generate different continuations, so their acceptance
rates do not directly rank the companions. Each pairing must reproduce its
own target's greedy token IDs before its speed qualifies.

Three chat prompts × 128 generated tokens × two interleaved repetitions.
All **72/72 baseline/candidate cases reproduce every target token ID**. These are chat
generation rates including prompt processing, separate from the depth sweep.

| Target | DFlash2 Q4_K_M (tok/s) | DFlash2 Q8_0 (tok/s) | DFlash2 BF16 (tok/s) |
| --- | --- | --- | --- |
| Q4_K_XL | 25.53 | 25.64 | 24.53 |
| Q8_K_XL | 24.05 | 23.77 | 23.06 |

All three drafts pass the pinned upstream operator comparison. The retained
kernel and launch changes reduce draft GPU time by about **4%** across all
three formats; measured end-to-end improvement is **0.2–0.7%** because target
verification dominates. Q4's artifact is 0.85 GiB smaller than Q8's. Final
selection remains open while draft and verification kernels are optimized;
controller comparisons follow that work.
The [optimization record](eval/dflash2-optimization.json) contains interleaved
samples, artifact/token hashes, kernel ablations and before/after profiles.

Further verification-head and embedding changes retain exact outputs. Their
additional end-to-end effect is below 0.2% in a short interleaved probe; see
the [verification optimization record](eval/dflash2-verification.json).

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
kernel/model suites. Release speed qualifies only after complete quality runs.

Production execution has one implementation per supported shape/weight type;
Qwen kernel, precision and verification environment switches are removed,
along with inactive fusion/prefetch policies and their kernels.
DFlash2 retains FP32 values in a bounded history ring. MTP replay uses the
committed target features. The optimized correctness build keeps assertions
and symbols; benchmark only the Nix release binaries.

Remaining: independent original-target/MTP qualification, MTP and C>1 speed
measurements, and speculative serving parity for C>1.
