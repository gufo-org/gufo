# DS4 development tools

Run from the repository root inside `nix develop`. Build, correctness, and benchmark requirements
are in [the DS4 benchmark README](../../benchmarks/deepseek-v4-flash/README.md).

| Tool | Purpose |
| --- | --- |
| `tools/ds4/check.py fast` | CLI, chat framing, pinned dataset audit, and answer grading |
| `tools/ds4/check.py kernels` | Q8/IQ2/F16 and attention oracles, plus complete official HC formulas; no model required |
| `tools/ds4/check.py reference --model "$MODEL" --upstream "$UPSTREAM" --output /tmp/ds4-reference` | Repeated AR scoring against 105 official continuations; independent full logits and 128 forced tokens at 0/4K/8K/12K/16K, with matched 2K/4K prefill calls |
| `tools/ds4/check.py reference --prefill-only --model "$MODEL" --upstream "$UPSTREAM" --output /tmp/ds4-prefill` | Same matched prefill grid and exact repeat check; skip continuation scoring and decode replay |
| `tools/ds4/check.py all --model "$MODEL" --dspark-model "$DSPARK"` | All nine DS4 CTest checks, including changing concurrency, snapshots through 16K, mixed sampling, and prefix/disk reuse |
| `tools/ds4/import-eval.py` | Rebuild pinned fixtures from an upstream checkout; default capability subset, `--suite official` for the 0731 continuations |
| `result/bin/gufo eval --questions 75 --greedy --output /tmp/ds4-quality.json` | Pinned capability evaluation through the real HTTP server (add `--base-url`) |
| `result/bin/gufo bench -c 1,2,4,6,8 -p 2048 -n 128 -d 0,4096,8192,12288,16384 -r 2 -v` | Release model sweep; per-request output hashes and draft counters (add model paths) |
| `tools/ds4/check.py benchmark --ar-log /tmp/ar.log --dspark-log /tmp/dspark.log --output /tmp/bench.json` | Check all 25 points and every member of both repeats; require repeated hashes/counters and matching AR/DSpark tokens |
| `tools/serving/gufo-serving-bench.py` | Shared HTTP concurrency, scheduling, and acceptance measurement |
| `tools/quant/speculative-corpus.py` | Shared AR/speculative text comparison on the fixed corpus |
| `tools/prof/prof.py` | Shared rocprofv3 capture, rollup, and A/B diff |

The attention oracle also supports isolated timing with the production
kernel. Build it as a standalone microbenchmark through Nix:

```sh
nix develop -c tools/bench/build.sh tests/models/deepseek_v4_flash/attention_test.hip
nix develop -c /tmp/attention_test --benchmark
```

The reference suite requires a clean `antirez/ds4` checkout at
`6289c516273979173abbc062209a81dd3706b804`. It builds upstream through the
test-only Nix/CMake adapter, leaving upstream model code unchanged. Each engine
runs separately. The new output directory retains manifests, scores, full
logits, actual prefill capacities, build/model identities, and `comparison.json`.
The full suite requires exact Gufo
repeats, the existing full-logit bounds, and no detected increase in 100-case
NLL (paired case bootstrap, 95% interval, seed 731; allowance of `5e-10` for
upstream's nine-decimal TSV rounding). The five smoke scores are
reported separately. Passing this finite sample cannot establish the absence
of every quality issue.
Antirez is an independent implementation, not an authoritative oracle. The
command reports threshold violations for investigation; retain the measurements
and resolve discrepancies against official formulas and task scores rather
than changing arithmetic merely to match Antirez.
For prefill investigations, `--prefill-only` keeps the same ten prompt/call-size
combinations and two Gufo repetitions. It omits the continuation/NLL gates and
after-decode vectors explicitly. Reports retain the existing raw-logit bounds
and add centered RMSE, Jensen–Shannon divergence (nats), maximum probability
difference, and top-token agreement; these diagnostics do not relax a gate.

Model inference timings always use the release `result/bin/gufo`. Do not use
CTest/debug model executables for performance claims. Temporary profiler traces
and experiment artifacts stay outside Git; retained experiment decisions belong
in the benchmark README. There are no production reference-route switches.
