# DS4 development tools

Run from the repository root inside `nix develop`. Build, correctness, and benchmark requirements
are in [the DS4 benchmark README](../../benchmarks/deepseek-v4-flash/README.md).

| Tool | Purpose |
| --- | --- |
| `tools/ds4/check.py fast` | CLI, chat framing, pinned dataset audit, and answer grading |
| `tools/ds4/check.py kernels` | Bitwise Q2 kernel reference comparison; no model required |
| `tools/ds4/check.py all --model "$MODEL" --dspark-model "$DSPARK"` | Complete DS4 quality suite, including actual model inference |
| `tools/ds4/import-eval.py` | Rebuild the pinned capability fixture from the audited upstream revision |
| `tools/serving/gufo-serving-bench.py` | Shared HTTP concurrency and acceptance measurement |
| `tools/quant/speculative-corpus.py` | Shared AR/speculative text comparison on the fixed corpus |
| `tools/prof/prof.py` | Shared rocprofv3 capture, rollup, and A/B diff |

The Q2 numerical oracle also supports isolated timing with the production
kernel. Build it as a standalone microbenchmark through Nix:

```sh
nix develop -c tools/bench/build.sh tests/models/deepseek_v4_flash/q2_down_test.hip
nix develop -c /tmp/q2_down_test --benchmark
```

Model inference timings always use the release `result/bin/gufo`. Do not use
CTest/debug model executables for performance claims. Temporary profiler traces
and experiment artifacts stay outside Git; retained experiment decisions belong
in the benchmark README. There are no production reference-route switches.
