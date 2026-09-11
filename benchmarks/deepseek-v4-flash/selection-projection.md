# C1 projection, normalization and selection

2026-09-11; DS4 Flash 0731 on gfx1151. Baseline `a760f81`, using the same
pinned target/support artifacts as the main README. All model timings use Nix
release binaries; kernel experiments use the production arithmetic flags.

## Retained implementation

- **Ordered F16 projection:** reuse the existing vector-load kernel for C1
  HC, router and output-head projections with aligned chunks. The 32 chunks,
  FMA sequence and final reduction order are unchanged. Isolated HC/router
  kernels improve about 2.2×; the vector-only release improves tg128 from
  16.75 to 17.30 tok/s at depth zero and 14.64 to 15.01 at 16K.
- **HC RMS normalization:** reuse the existing register-cached kernel for
  scalar and verifier calls at width 16,384. It reads each input once and
  preserves the reduction tree. The scalar API now delegates to the row API.
  Cached norm plus vector projection is about 1.7× faster than generic norm
  plus vector projection in the isolated experiment.
- **Exact partial top-k:** retain input keys in registers, skip common leading
  bits, partition the boundary bucket until at most 1,024 candidates remain,
  then sort full score/index keys. This preserves rank order, including the
  lower-index tie rule. The route covers 1,025–16,384 scores, top-512 and up to
  six query rows; wider batches and larger contexts keep their existing sorts.
- **Zero ties:** both zero signs now map to one radix key. This fixes the
  existing radix selector's disagreement with the numeric comparator on
  `+0`/`-0` ties. Neither score precision nor model/cache precision changes.

No runtime selectors or new command-line options are introduced.

## Release results

Two repetitions per depth, baseline `a760f81` against the combined candidate:

| Depth | Baseline pp2048 | Current pp2048 | Baseline tg128 | Current tg128 | TG gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 458.40 | 458.84 | 16.74 | 17.78 | 6.2% |
| 4,096 | 444.03 | 443.65 | 15.08 | 15.90 | 5.4% |
| 8,192 | 435.26 | 436.20 | 14.84 | 15.75 | 6.1% |
| 12,288 | 431.68 | 431.58 | 14.73 | 15.61 | 6.0% |
| 16,384 | 423.62 | 423.78 | 14.64 | 15.47 | 5.7% |

Throughput is tok/s. All **1,280 paired token positions match exactly** across
both repetitions and all five depths. Prefill stays within measured variation.
Both processes retained over 28.7 GiB available under a 12 GiB memory guard.
The vector-only controls and combined results are retained in
[selection-projection.json](selection-projection.json), including source and
binary identities, individual process summaries, repeated hashes and guards.

## Quality and profiling

The existing `ds4.projections` check covers C1 HC/router/output-head projection
against scalar arithmetic and an independent CPU formula. Sixteen cached RMS
cases compare every output bit and check a double-precision CPU formula,
including zeros, very small values, large dynamic range and 1/2/6/8 rows.

The existing `ds4.attention` executable now includes `indexer_test.hip`:
72 top-k cases repeat twice against independent CPU ordering and the production
fallback sorts. Coverage includes quantized ties, both zero signs, infinities,
tightly clustered scores and per-query causal masks with fewer than 512 visible
rows. These checks remain under the two existing kernel test names and
`tools/ds4/check.py kernels`.

Target quality passes with the unchanged pinned trajectory (116/128 top-1,
rank sum 142, worst rank 3), wide-prefill fingerprint and 29/29 concurrent
session choices. The full maintained DSpark check passes: all 736 scalar replay
choices and frontier logits through C8/16K are exact, as are repeated outputs,
changing membership and snapshot continuations. Budget and state bounds pass.
The run retained at least 15.80 GiB available under the 12 GiB memory guard.
The complete serving/CLI and external-reference checks were not repeated for
this kernel change; their previous qualification remains separately pinned.
The 75-question capability comparison remains pending separate approval.

The production profile at 4K, covering 16 generated tokens, explains the gain:

| Kernel family | Baseline GPU ms | Current GPU ms |
| --- | ---: | ---: |
| Ordered F16 projections | 55.83 | 32.39 |
| Plain HC RMS | 28.46 | 3.99 |
| Indexer selection, including small-set fallback | 5.74 | 5.34 |
| All decode kernels | 1,016.54 | 965.51 |

Both profiles contain 29,109 launches, trimmed from the first decode HC
projection to exclude prefix preparation. The production partial selector
uses 120 VGPRs and 272 bytes of private storage at the profiled shape; the
standalone register figures below do not describe this build. Q8 projections
remain about 51% of decode GPU time, followed by MoE gate/up at 13%.
These profiles diagnose kernel cost; the headline sweep runs without profiling.

```sh
nix develop -c tools/ds4/check.py kernels
nix develop -c build/gpu-test/tests/models/deepseek_v4_flash/ds4_attention_test --benchmark
# Run once per immutable baseline/candidate Nix release:
./result/bin/gufo bench --model "$MODEL" -c 1 -p 2048 -n 128 \
  -d 0,4096,8192,12288,16384 -r 2 -v
```

## Experiment decisions

- Rejected full HC norm/projection fusion: bit-exact, but slower than
  separate generic norm plus vector projection. Swizzling LDS did not help.
- Retained existing cached RMS instead of adding another normalization kernel.
- Rejected the first partial selector on large C1 sets; retain the parallel
  tree beyond 16,384 scores.
- Added common-prefix compression after measuring slowdowns on large ties and
  tightly clustered scores; the retained kernel addresses both slowdowns
  within its selected range, with zero spills.
- New selector resources: 117/181 VGPRs, 12/8 waves per SIMD and 17,096 bytes LDS
  for the 8K/16K specializations in the standalone build. Production resources
  are recorded separately in the final profile.
- Deferred additional 2K/4K specializations: all 72 cases remain exact and
  isolated selection saves roughly 1–3 μs per call, but the profiled selector
  is only 0.5% of decode GPU time. No end-to-end win is qualified for these
  extra variants; the retained implementation uses two capacities.
