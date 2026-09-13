# Qwen3.8 27B on Strix Halo

Production targets are **UD-Q4_K_XL and UD-Q8_K_XL**. Optional speculative
routes are MTP and DFlash2. BF16 targets are quality references; DFlash2
drafts in Q4_K_M, Q8_0 and BF16 are being compared. Linux x86-64, gfx1151, 128 GB unified memory; Nix release builds.

Qualification is in progress. `TODO` means no current qualified measurement.
Prefill defaults to **pp2048** (`-p 4096` remains available), generation uses **tg128**, and `C` means concurrent
requests. Generation rates below are per user; aggregate rate is `C` times
that number. Draft precision is selected separately from target precision.

## Single user, autoregressive

Q4 depth 0 has a [paired AR refresh](eval/dflash2-mixed-formats.json);
depth 4096 uses the [preceding refresh](eval/dflash2-scalar-row-reuse.json).
Other rows remain the reference sweep at `023a13a`. Full refresh: TODO.

| Context depth | Q4 pp / tg (tok/s) | Q8 pp / tg (tok/s) |
| ---: | ---: | ---: |
| 0 | 420.2 / 11.74 | 491.1 / 7.07 |
| 4,096 | 405.0 / 11.51 | 470.9 / 7.01 |
| 8,192 | 390.8 / 11.15 | 451.2 / 6.95 |
| 12,288 | 375.1 / 10.96 | 434.2 / 6.88 |
| 16,384 | 362.0 / 10.77 | 414.4 / 6.81 |

## Single user, speculative

Recommended **Q4_K_M draft, adaptive controller**, pp2048/tg128, in tok/s.
One timed repetition after warmup; every measured row matches all 128 AR IDs.
These depth rows precede the latest kernel changes. Refresh: **TODO**.

| Context depth | Q4 pp / tg | Q8 pp / tg |
| ---: | ---: | ---: |
| 0 | 398.1 / 24.18 | 466.2 / 24.14 |
| 4,096 | 384.9 / 17.00 | 447.3 / 16.30 |
| 8,192 | TODO | TODO |
| 12,288 | TODO | TODO |
| 16,384 | TODO | TODO |

[Q4 depth measurements](eval/dflash2-head-iq4.json);
[preceding Q8 measurements](eval/dflash2-batch-widths.json).
The [controller comparison](eval/dflash2-controllers.json) remains separate.
Earlier [fixed-block precision results](eval/dflash2-rollback.json) and the
[full depth sweep](eval/dflash2-depths.json) remain historical references.
Other depths with the new default and MTP performance: **TODO**.

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

**Q4_K_M is the recommended draft; adaptive is the default controller.**
All three drafts pass the pinned upstream operator comparison. A short C1
comparison on three chat prompts (explanation, code, reasoning), tg128,
one repetition per pairing, includes prefill:

| Target | Draft | Fixed tok/s | Adaptive tok/s |
| --- | --- | ---: | ---: |
| Q4 | Q4_K_M | 28.85 | **29.44** |
| Q4 | Q8_0 | 28.98 | 28.39 |
| Q4 | BF16 | 27.77 | 27.09 |
| Q8 | Q4_K_M | **26.36** | **26.36** |
| Q8 | Q8_0 | 26.16 | 26.12 |
| Q8 | BF16 | 25.48 | 25.50 |

Every continuation matches all 128 AR token IDs. Q8 still wins the coding
prompt; fixed can be faster with Q8/BF16 drafts. Draft files occupy
1.06 / 1.92 / 3.60 GiB respectively; BF16 has no measured production speed
advantage here. These short chat results do not replace the synthetic depth
table. [Controller and precision comparison](eval/dflash2-controllers.json).

**Latest Q4 target/Q4 draft:** greedy C1, including prefill and excluding
model loading. Two timed samples per binary in ABBA order; JSON has four
across two comparisons. No separate warmup. Changes compare binaries within
this probe, not against earlier tables measured at different times.

| Workload | Current tok/s | Paired change |
| --- | ---: | ---: |
| Repetition, tg128, fixed-7 | **54.15** | +0.3% |
| JSON, tg300, adaptive | **48.50** | +0.4% |
| Prose, tg300, adaptive | **22.12** | +0.1%, effectively flat |

Every token ID and acceptance statistic is unchanged. Alternating FMA
multiplication operands in Q5 batch-eight projections reduces their profiled
GPU time by 0.93%, with no spills or changed accumulation order.
[Measurements and quality checks](eval/dflash2-fma-order.json).

[Mixed-format scalar specialization](eval/dflash2-mixed-formats.json) raises
AR generation to **11.74 tok/s** at depth zero. Earlier work covers
[two-token verification](eval/dflash2-width2.json) and
[compact Q4 staging at widths 3–8](eval/dflash2-q4-staging.json).
The [quality report](eval/README.md) indexes the remaining evidence.

Compiler scheduling flags did not change the emitted kernels; scheduling
boundaries and expanded Q5 codes were slower. Rounded EMA and a smaller
initial controller prior regressed prose. Broader FMA scheduling changes
showed no meaningful release gain. None is retained.

The upstream headline uses different artifacts/power and excludes prefill;
it is not a matched engine comparison. [Source audit and comparison limits](eval/llama-comparison.json).
The [rollback pass](eval/dflash2-rollback.json) saves **50.5 MiB per session**
and qualifies mixed cache capacities and logical context 262,144.

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

Select DFlash2 with `--speculative dflash2`. The GGUF architecture identifier
is `dflash`; it is not a CLI backend name.

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

All supported sampling controls pass the bounded AR/DFlash2 matrix across
both targets, three draft quants and both controllers. The
[sampling record](eval/sampling-strategies.json) includes probability controls,
seeded replay and all six HTTP adapters; it is not a capability evaluation.

Production execution has one implementation per supported shape/weight type;
Qwen kernel, precision and verification environment switches are removed,
along with inactive fusion/prefetch policies and their kernels.
DFlash2 retains FP32 values in a bounded history ring. MTP replay uses the
committed target features. The optimized correctness build keeps assertions
and symbols; benchmark only the Nix release binaries.

Remaining: independent original-target/MTP qualification, MTP and C>1 speed
measurements, and speculative serving parity for C>1.
