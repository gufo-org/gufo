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
depth 4096 has a [fresh AR control](eval/dflash2-controller-cost.json).
Other rows remain the reference sweep at `023a13a`. Full refresh: TODO.

| Context depth | Q4 pp / tg (tok/s) | Q8 pp / tg (tok/s) |
| ---: | ---: | ---: |
| 0 | 420.2 / 11.74 | 491.1 / 7.07 |
| 4,096 | 408.5 / 11.64 | 470.9 / 7.01 |
| 8,192 | 390.8 / 11.15 | 451.2 / 6.95 |
| 12,288 | 375.1 / 10.96 | 434.2 / 6.88 |
| 16,384 | 362.0 / 10.77 | 414.4 / 6.81 |

## Single user, speculative

Recommended **Q4_K_M draft, adaptive controller**, pp2048/tg128, in tok/s.
One timed repetition after warmup; every measured row matches all 128 AR IDs.
Q4 depth 4096 uses the measured-cost controller below; the other entries
precede the latest changes. Full refresh: **TODO**.

| Context depth | Q4 pp / tg | Q8 pp / tg |
| ---: | ---: | ---: |
| 0 | 398.1 / 24.18 | 466.2 / 24.14 |
| 4,096 | 386.4 / 17.06 | 447.3 / 16.30 |
| 8,192 | TODO | TODO |
| 12,288 | TODO | TODO |
| 16,384 | TODO | TODO |

[Current Q4 depth-4096 control](eval/dflash2-controller-cost.json);
[earlier Q4 depth measurements](eval/dflash2-head-iq4.json);
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
All three drafts pass the pinned upstream operator comparison. An earlier C1
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
model loading. Each workload uses two samples per binary in ABBA order
after warmup on the quiet host; prose also has a BAAB confirmation, pooled
with the initial samples. These are short controls.

| Workload | Before tok/s | Current tok/s | Change |
| --- | ---: | ---: | ---: |
| Repetition, tg128, fixed-7 | 56.49 | **56.61** | +0.21% |
| JSON, tg300, adaptive | 53.38 | **53.66** | +0.53% |
| Prose, tg300, adaptive | 23.58 | **23.58** | flat |

The latest [mixed-format projection change](eval/dflash2-remaining-formats.json)
shares activations across four output rows for selected Q6/IQ projections.
The affected 1,341 calls take 11.53% less GPU time; overall gains remain small
because these calls account for only 5% of the profile. All 16 continuations,
102 verification logit rows, 102 feature rows, C3 cache controls and 90 Q4
draft traces remain exact. Arithmetic and allocations are unchanged.
Earlier [contiguous FFNs](eval/dflash2-contiguous-ffn.json) remove 1,554
projection launches; [feature transfers](eval/dflash2-feature-transfer.json)
improved JSON 1.93% and repetition 1.45%.

The controller uses [measured Q4 verification costs](eval/dflash2-controller-cost.json);
Q8 keeps its previous formula. Its earlier chat comparison was flat in
aggregate, with code/reasoning losing 1.0–1.6%; the depth-4096 control lost 1.0%.

[Mixed-format scalar specialization](eval/dflash2-mixed-formats.json) raises
AR generation to **11.74 tok/s** at depth zero. Earlier work covers
[two-token verification](eval/dflash2-width2.json) and
[compact Q4 staging at widths 3–8](eval/dflash2-q4-staging.json).
The [quality report](eval/README.md) indexes the remaining evidence.

Lossless row packing, residual-precision WMMA/INT8, precomputed activation
sums and removing full-tile bounds checks were slower. Alignment hints
changed no instructions. Further [integer-matrix and final-barrier probes](eval/dflash2-matrix-probes.json)
also lost. [Lossless Q6 head expansion](eval/dflash2-feature-transfer.json)
increased bandwidth demand and lost 14–26%; rejected experiments remain excluded.

**Same host and identical Q4 target/draft files**, at `b55a120`, raw tg300,
including prefill:

| Workload / policy | Gufo tok/s | Laurent Vulkan tok/s |
| --- | ---: | ---: |
| JSON / adaptive | **51.29** | 36.49–37.81 |
| Prose / adaptive | **22.91** | 19.08–19.20 |
| JSON / fixed-7 | **49.74** | 37.22–38.77 |

Two samples per engine; exploratory sequential controls. Fixed-7 has identical
tokens, acceptance and round counts. All six Gufo continuations match AR.
Upstream prose differs from its AR and across repeats; this does not establish
a semantic quality regression or a controller bug. Its FP4 headline uses
different artifacts and decode-only timing. The author describes 65.6 tok/s
as a short 115 W burst and reports 55.0 tok/s on the everyday power profile;
the headline remains unmatched. [Power caveat provenance](eval/dflash2-remaining-formats.json);
[matched-artifact measurements and checks](eval/llama-comparison.json).
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
