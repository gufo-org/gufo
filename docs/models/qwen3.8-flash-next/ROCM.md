# ROCm 10.1 migration status

Updated October 23, 2026. Scope: Flash-Next UD-Q4_K_XL with its shared-Q8_0
MTP sidecar on Strix Halo `gfx1151`.

**Focused compatibility and numerical checks pass on both runtimes.**
Sampled generation improves; some cold-start and disk timing observations
remain inconclusive. At the time of these measurements, the default Nix
toolchain was ROCm 7.2.3. Main
`ab122e44` adds the optional ROCm 10.1 build and its ROCr queue-memory patch.
The working branch is rebased onto main `ade4f4d7`, including the expanded
greedy draft-head eligibility from #739 and the Cachix configuration from
#743/#747. Those cache updates do not change inference code or numerical flags.

## Tested configurations

| Component | Default toolchain | 10.1 container |
| --- | --- | --- |
| ROCm | 7.2.3, pinned Nix dependencies | 10.1.0, Ubuntu 26.04 container |
| HIP compiler | Clang 22 | Clang 24, `d6f6cb691863e9258034ab48c1dfce21689d5606` |
| Host C++ compiler | GCC 15.3 | GCC 15.2 |
| ROCr | Repository queue-memory patch | Matching 10.1 queue-memory patch |
| Production source | Candidate rebased onto `ade4f4d7` | Same candidate |

The 10.1 image is `rocm/dev-ubuntu-26.04:10.1.0-full`; its digest and loaded
runtime verification are recorded in [QUALITY.md](QUALITY.md).
Both controls use the same hardware, weights, settings and ordered requests.
Inference and builds run separately.
Controller comparisons use frozen controls based on main `41d93463` plus
the selector change; the candidate adds the scoped sampled-head cost curve.

The repository now uses ROCm 10.1 through `nix build` and `nix develop`;
the former optional `.#rocm10` outputs have been removed. The measurements here use the container configuration
above; they do not qualify every possible ROCm installation.

## Validation

| Area | State |
| --- | --- |
| Build, loading and serving | Pass on both tested runtimes. |
| Selector scheduling | Bounded unrolling reduces newer-compiler register spills without changing scores or selected blocks. Clang 22 retains its original path. |
| Numerical regression within 10.1 | 18.6 million logits remain byte-identical with the selector change; independent selector and MTP checks pass. |
| Request behavior | Greedy/default sampling, C2, cache replay, stops, cancellation and image cases pass. |
| Completed tasks | Extraction, executable generated tests and 32K file-edit tool calls pass on both runtimes. |
| AR performance | Matched d0/d32K measurements are comparable; 10.1 d32K prefill improves from 1,622 to 1,648 tok/s with the selector change. |
| Sampled MTP performance | The scoped sampled C1 curve improves code, prose and thinking on both compilers. Short replies and full target-row verification retain their established costs. |

These are focused checks, not a complete model or context sweep. Initial
timing flags, targeted controls and qualification limits remain in
[QUALITY.md](QUALITY.md#sampled-draft-head-costs-october-23-2026).
The standard [benchmark tables](BENCHMARKS.md) retain their original toolchain
and measurement provenance; they are not a ROCm 10.1 refresh.

## Sampled generation

Default sampling, C1, 128 output tokens, warm replay; rates are tok/s.
Both controls include main `41d93463` and the selector change.

| Prompt | 7.2.3 before | 7.2.3 candidate | 10.1 before | 10.1 candidate |
| --- | ---: | ---: | ---: | ---: |
| Code | 62.72 | 70.75 | 70.47 | 73.50 |
| Prose | 48.72 | 51.58 | 47.86 | 55.75 |
| Thinking | 54.04 | 57.40 | 45.71 | 55.20 |

These bounded screens improve on both compilers. Completed extraction,
generated tests and 32K file edits also pass their task checks. Sampled
answers can differ in length: total completion time does not always improve.
Individual timing flags remain visible in the quality notes; these numbers
are not a claim that every request is faster. In an alternating disk control,
the unchanged binary takes 99.8 and 128.0 ms to restore the same branch,
versus 97.4 ms for the candidate. Its storage/copy timing remains inconclusive.

## Numerical differences

Compiler versions produce different logits and seeded answers. Reproducing
the older output is not an accuracy criterion: in a first-layer capture,
10.1 agrees more often with independent FP32 and FP64 quantization references.
An experiment forcing the older rounding was rejected. Existing production
rounding, target sampling and rejection correction remain unchanged.

The cost update changes draft length, preserving target arithmetic and
rejection correction. It uses committed acceptance history, never runtime
timings. Seeded requests retain their configured concurrency; full-head,
full target-row, greedy and C>1 cost curves stay unchanged. Cache restoration
retains existing scratch usable by either head without allocating additional rows.

The first cycle keeps the conservative policy until acceptance is observed.
Short requests and tails of at most two blocks also keep their established
costs. The serving adapter passes the remaining output budget to the engine,
which still bounds every physical block. This fixes the initial candidate's
8–12-token continuation slowdown; its focused replay passes all 84 timing
phases with identical output and draft counts.

Full target-row verification also keeps the original costs: pricing it as
compact verification widened constrained tool output and increased a
thinking/file-edit pause from 111 to 120 ms. The focused final replay passes
all 63 phases, with that pause at 113 ms and identical output/cache work.

See [the quantization check](QUALITY.md#cross-compiler-quantization-check) and
[experiment decisions](EXPERIMENTS.md).
