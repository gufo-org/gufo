# DeepSeek V4 Flash quality

**DSpark matches Gufo AR on the retained greedy checks, including C>1.
Official target-model parity is not yet established.** The Flash 0731 mixed IQ2/Q2/Q8
weights and DSpark sidecar are [pinned](artifacts/model-identities.json).

| Check | Retained result |
| --- | --- |
| DSpark greedy replay | 736/736 token choices match AR; maximum full-logit error 0, through C8 / 16K |
| HTTP AR/DSpark, C1/2/4/6/8 | All 63 tg128 responses match fresh AR; mixed/repetitive prompts, September 24 |
| Sparse-indexer boundary | DSpark C1 and AR C2: 0/128 token differences, bit-identical full logits |
| Reproducibility | 1280/1280 choices and 20 full-logit vectors repeat exactly |
| Sampling / sessions | p/q acceptance, residual correction, seeded replay, EOS/cancellation, three-turn continuation and disk restore pass |
| Optimized versus Debug | **33/2327** greedy choices differ |
| Historical capability set | **53/75** correct; 22 failures, no execution errors, nine length finishes |

Evidence: [DSpark sampling/replay](artifacts/dspark-sampling.json),
[boundary/HTTP checks](artifacts/indexer-boundary.json),
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

September 24, 2026; same target/sidecar, greedy, thinking off, pp2048/tg128,
one warmed sample per point. Depth is cached prefix length. C1/2/4/6/8 prefill
all sessions from a C1 prompt checkpoint before timed decoding, preserving
prefill arithmetic; rates sum individual decode rates. Loading uses cold files,
C1/DSpark/capacity 262144; memory uses C1/AR peak global HIP allocation.

Performance baselines and unavailable comparisons are described in
[benchmarks](BENCHMARKS.md). Commands/counts remain in [artifacts](artifacts/bench.json) and the
[benchmark workflow](../../../.agents/skills/benchmark-model/SKILL.md).
