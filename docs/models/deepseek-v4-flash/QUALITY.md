# DeepSeek V4 Flash quality

**DSpark matches Gufo AR on the retained greedy checks, including C>1.
Official target-model parity is not yet established.** The Flash 0731 mixed IQ2/Q2/Q8
weights and DSpark sidecar are [pinned](artifacts/model-identities.json).

| Check | Retained result |
| --- | --- |
| DSpark greedy replay | 736/736 token choices match AR; maximum full-logit error 0, through C8 / 16K |
| HTTP AR/DSpark, C1/2/4/6/8 | All 63 tg128 responses match fresh AR; mixed/repetitive prompts, October 30, 108 GiB GTT |
| Sparse-indexer boundary | DSpark C1 and AR C2: 0/128 token differences, bit-identical full logits |
| Reproducibility | 1280/1280 choices and 20 full-logit vectors repeat exactly |
| Sampling / sessions | p/q acceptance, residual correction, seeded replay, EOS/cancellation, three-turn continuation and disk restore pass |
| Optimized versus Debug | **33/2327** greedy choices differ |
| Historical capability set | **53/75** correct; 22 failures, no execution errors, nine length finishes |

Evidence: [DSpark sampling/replay](artifacts/dspark-sampling.json),
[boundary checks](artifacts/indexer-boundary.json), the HTTP cohorts in
`artifacts/multi-*-gufo-*.json`,
[qualification](artifacts/quality-qualification.json) and
[formula audit](artifacts/prefill-formula-audit.json).
Gufo-path agreement cannot detect shared target errors. Sampled DSpark need not
match AR's same-seed sequence; replay requires the same execution configuration
and schedule. The build-sensitive target differences remain under investigation.
Ask before rerunning the 75-question capability set.

## Reproduce

Use the smallest affected [maintained suite](../../../tools/ds4/README.md):

```sh
nix develop -c tools/ds4/check.py fast
nix develop -c tools/ds4/check.py kernels
nix develop -c tools/ds4/check.py model --model "$MODEL" --dspark-model "$DSPARK"
```

Independent FP64 operators and exact sparse top-k checks supplement model/state
replay. Keep full-logit captures outside Git; do not relax quality gates.

## Benchmark method

Measured October 30, 2026 on one 128 GB Strix Halo `gfx1151` node, power
profile Balanced, GTT limit 108 GiB. Gufo `07206ea4` is the production Nix
package and passed `tools/ds4/check.py fast` (5/5). The reference is
antirez/ds4 `0aaea5a2` built by `.#ds4-reference` with ROCm 10.1. Both run over HTTP through
`tools/bench/model-bench.py run --fresh`, with the same target/sidecar, greedy,
thinking off, pp2048/tg128 and one warmed sample per point. Depth is cached
prefix length. C1/2/4/6/8 prefill all sessions from a C1 prompt checkpoint
before timed decoding, preserving prefill arithmetic; rates sum individual
decode rates. Memory is the C1/AR peak device-global HIP memory in use
(`hipMemGetInfo` total minus free) at capacity 262144.

Loading is C1/DSpark at capacity 262144. Before each launch,
`POSIX_FADV_DONTNEED` evicts the target and DSpark files and `cachestat`
confirms 0 resident pages. Runtime libraries are not evicted. The September
reference loading time (150.87 s) used a different eviction method, storage and
ROCm version, so it is not comparable.

Gufo DSpark completions equal Gufo AR at depth 0. At deeper depths, the DSpark
request caches one more prefix token than the AR request (`cache_n` one higher,
`prompt_n` one lower), so the AR/DSpark text check applies only at depth 0.
The September artifacts show the same offset.

Gufo prefill arithmetic depends on free device memory. The DeepSeek backend
fills a lazy Q8-to-F16 weight cache until free memory reaches its reserve (5% of
device memory, at least 4 GiB), then logs `q8 fp16 cache budget exhausted; using
q8 kernels`. At the node's 98 GiB GTT limit, the cache stopped filling during
the C1 checkpoint prefill for AR at C8 and DSpark at C6. Those outputs differed
from isolated AR, on `main` at `1fc829ea` as well. At 98 GiB, the reference's
depth-0 prefill measured 153 tok/s instead of 237 tok/s. Neither change occurs
at 108 GiB, the limit the September tables also used.

Performance baselines and unavailable comparisons are described in
[benchmarks](BENCHMARKS.md). Commands/counts remain in [artifacts](artifacts/bench.json) and the
[benchmark workflow](../../../.agents/skills/benchmark-model/SKILL.md).

## ROCm 10.1 migration

Focused cross-compiler checks, matched controls and remaining limits are in
[ROCM.md](ROCM.md). The original reference fixtures and tolerances above are
unchanged; migration measurements use separately recorded controls.
