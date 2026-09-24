# DeepSeek V4 Flash evaluation

**Target parity remains unresolved.** Antirez is an independent comparison,
not official ground truth. AR/DSpark agreement cannot detect shared target errors.

## Quality

| Check | Retained result |
| --- | --- |
| Optimized trajectory | **115/128** top-1 (required ≥116); rank sum 145 (≤142), worst rank 4 (≤3) |
| Optimized versus Debug | **33/2327** greedy choices differ |
| Post-prefill logits | **4/20** comparisons fail; post-decode vectors pass |
| Reproducibility | **1280/1280** choices and 20 full-logit vectors repeat exactly |
| Operators / state | FP64 formulas, 156 exact top-k cases and 736 DSpark replay choices pass |
| [Sparse-indexer boundary](artifacts/indexer-boundary.json) | DSpark C1 and AR C2: 0/128 token differences, bit-identical full logits |
| [Reference DSpark control](artifacts/reference-qualification.json) | Antirez C1 differs from its own AR after fresh and prepared prefill; performance comparison excluded |
| Serving | Sampling, EOS isolation, cancellation, three-turn continuation and disk restore pass |
| Capability | **53/75**, 22 failures, zero execution errors, nine length finishes; full refresh pending |

Evidence: [qualification](artifacts/quality-qualification.json),
[prefill comparisons](artifacts/antirez-ds4-ar-comparison.json),
[formulas](artifacts/prefill-formula-audit.json),
[operators](artifacts/captured-operator-audit.json),
[DSpark sampling](artifacts/dspark-sampling.json).
DSpark uses p/q acceptance and residual correction; seeded replay requires the
same execution configuration and schedule. The boundary fix does not resolve
the separate target-parity alerts. Ask before rerunning the 75-question evaluation.

## Checks

[Maintained tools](../../../tools/ds4/README.md): `fast` checks wiring/sampling,
`kernels` checks independent formulas, `model` checks logits and serving/state.
Run affected checks while iterating; do not relax quality gates.

```sh
nix develop -c tools/ds4/check.py fast
nix develop -c tools/ds4/check.py kernels
nix develop -c tools/ds4/check.py model --model "$MODEL" --dspark-model "$DSPARK"
```

## Benchmark method

Same Flash 0731 target/support GGUF, greedy, thinking off, pp2048/tg128;
one warmed sample per point. Depth is cached prefix length. C1/2/4/6/8 use
the same d0 prompts and prefill every session before timing decoding; throughput
is the sum of individual request rates. Gufo uses a C1-built prompt checkpoint
to hold prefill arithmetic fixed. These are decode-focused concurrency tests.
At 64K, the mixed task requests a story because the summary stopped before tg128.
Loading: C1/DSpark, capacity 262144, cold model files. Memory: C1/AR, same capacity,
peak HIP-used memory. [Build/model identities](artifacts/model-identities.json).

Reference: [antirez/ds4 `0aaea5a2`](https://github.com/antirez/ds4/tree/0aaea5a238fb41a35106a551e73c8409dfb751ac),
ROCm 7.2.3 / gfx1151, no inference patches. Gufo reports HTTP stage timings;
Antirez's native log timers are checked against HTTP token counts. Unsupported
comparisons are **N/A**, with reasons beside the tables. Antirez disables ROCm
DSpark under native batching, including `--batched-session 1`; its C1 DSpark
also fails the retained AR-equivalence control.

```sh
nix build .#ds4-reference --out-link result-ds4-reference
nix develop -c python3 tools/bench/model-bench.py --model deepseek-v4-flash \
  --gguf "$MODEL" --dspark "$DSPARK" run --target gufo --todo
# Reference: --target reference --reference-binary result-ds4-reference/bin/ds4-server
```

The [benchmark driver](../../BENCHMARKS.md) retains counts, acceptance and commands
in artifacts. The [optional reference package](../../../.devops/nix/ds4-reference.nix)
is excluded from Gufo and the default dev shell; [build smoke evidence](artifacts/reference-smoke.json).
