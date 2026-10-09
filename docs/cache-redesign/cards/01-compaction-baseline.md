# Card 01 · Compaction mechanics baseline

This records the compaction scenario family in card 01's combined test-only PR. It adds
`cache-compaction` to the functional runner, selected explicitly and excluded
from `all`. The cache implementation is unchanged.

## Contract

The fixture preserves one system prompt and a declared `read_archive` tool
while replacing the rest of a near-limit conversation with a summary. A small
tool result calibrates the append length using the loaded model's reported
tokens. The large prompt must reach 75–95% of the configured context; the
summary must drop at least half that capacity.

A short warm branch before the large append measures a reusable coherent
system/tools checkpoint. Compaction must reuse at least that boundary, and
must not reuse beyond the seed prompt into abandoned history. The summary
changes the answer from ALPHA to BETA. Both unchanged retries must avoid
prefill, except for the existing documented assistant-opening allowance when
the server log proves a retry-copy refusal under pressure. The next turn must
advance reuse to the compacted conversation.

All eight warm requests precede the eight uncached controls. Every warm answer,
finish reason and generated-token count must exactly match its control.
Reported cache/prefill accounting and per-request metrics are mandatory. The fast harness
tests reject misses, excessive reuse, frozen continuation boundaries, stale
answers and unexpected retry work.

## Reproduce

Build clean current main with `nix build`. Run once per mode, using the same
production binary and separate output directories:

```sh
nix develop -c python3 tests/functional/run.py \
  --record-baseline --output /tmp/compaction-ar --sampling-preset qwen38 \
  --suite cache-compaction -- \
  /path/to/main/result/bin/gufo serve llm \
  --model /path/to/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --mtp-model /path/to/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --speculative off --sessions 1 --context 32768 --think off
```

Use `--speculative mtp` and a fresh output directory for MTP. Keep the sidecar
in both commands so the AR run verifies the explicit override. No disk cache
is enabled by this suite.

## Scope

This baseline uses deterministic client-authored tool calls/results and a
summary, with actual seed/continued assistant replies. It does not exercise a
real client's compaction policy, summary generation, streaming progress, a >64K drop, abandonment
priority under pressure, restart/disk behavior, or other model families.
Other scenario families are recorded in the [combined baseline](01-functional-baseline.md). These measurements
are a current-main step baseline, not new-cache qualification or a timing
comparison; there is one sample per request and no speedup claim.

Assistant outputs, usage, request/output hashes, reports and server logs
stay in the listed local artifact directories. Raw HTTP bodies are not retained.
Per-request work and TTFT are recorded below; full raw measurements stay outside Git.

## Recorded results · 2026-10-09

Both runs passed on clean main `d221a01c`, built with `nix build`, on AMD
Strix Halo gfx1151 (125 GiB unified memory), Linux 7.2.9 and ROCm 7.2.3.
The harness used pinned Python 3.14.6 and OpenAI SDK 2.41.1. The target was
Flash-Next UD-Q4_K_XL (four shards), with the shared Q8_0 MTP sidecar.
The binary SHA-256 is `de0b42cbd49d8dacf853513983760c39c50e5931c432926ab7293db0d0230047`.
The harness SHA-256 is `4820f6294d0fcae05a406e78c5fae1473dd886cf136a639260d5da42d3798cfc`;
the flake.lock SHA-256 is `faabad820ec09caab7351bf48910cec895a38cf1766e098e3065aea86f405ed6`.
Model sources and documented identities are in
[model-identities.json](../../models/qwen3.8-flash-next/artifacts/model-identities.json).
No generated test output is committed.

The prompt dropped from **27,542 to 2,302 tokens** (25,240 removed). Both
modes restored the measured 2,048-token shared checkpoint and prefilled only
254 summary tokens. Both unchanged retries prefilled zero tokens. Continuation
reused 2,304 of 2,327 tokens. All eight warm outputs matched their uncached
controls exactly. AR executed zero drafts; MTP proposed 48 and accepted 18.

All request results below passed. TTFT is the server measurement in ms;
reuse is the API’s `cached_tokens`, including live-state hits. Prefill, restore,
decode, queue and wall timings are retained separately in JSON.

| Mode | Request | Prompt | Reused | Prefilled | TTFT (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| AR | `seed` | 2277 | 0 | 2277 | 1685.971 |
| AR | `shared_probe` | 2277 | 2048 | 229 | 327.133 |
| AR | `sample` | 5912 | 2277 | 3635 | 2289.571 |
| AR | `near_limit` | 27542 | 2277 | 25265 | 16004.001 |
| AR | `large_retry` | 27542 | 27542 | 0 | 2.863 |
| AR | `summary` | 2302 | 2048 | 254 | 400.804 |
| AR | `summary_retry` | 2302 | 2302 | 0 | 2.178 |
| AR | `continue` | 2327 | 2304 | 23 | 158.519 |
| AR | `seed_cold` | 2277 | 0 | 2277 | 1598.554 |
| AR | `shared_probe_cold` | 2277 | 0 | 2277 | 1438.540 |
| AR | `sample_cold` | 5912 | 0 | 5912 | 3714.870 |
| AR | `near_limit_cold` | 27542 | 0 | 27542 | 17038.756 |
| AR | `large_retry_cold` | 27542 | 0 | 27542 | 17167.037 |
| AR | `summary_cold` | 2302 | 0 | 2302 | 1487.616 |
| AR | `summary_retry_cold` | 2302 | 0 | 2302 | 1467.575 |
| AR | `continue_cold` | 2327 | 0 | 2327 | 1620.112 |
| MTP | `seed` | 2277 | 0 | 2277 | 1751.093 |
| MTP | `shared_probe` | 2277 | 2048 | 229 | 335.480 |
| MTP | `sample` | 5912 | 2277 | 3635 | 2398.382 |
| MTP | `near_limit` | 27542 | 2277 | 25265 | 16784.072 |
| MTP | `large_retry` | 27542 | 27542 | 0 | 3.415 |
| MTP | `summary` | 2302 | 2048 | 254 | 402.933 |
| MTP | `summary_retry` | 2302 | 2302 | 0 | 2.037 |
| MTP | `continue` | 2327 | 2304 | 23 | 156.848 |
| MTP | `seed_cold` | 2277 | 0 | 2277 | 1505.902 |
| MTP | `shared_probe_cold` | 2277 | 0 | 2277 | 1489.474 |
| MTP | `sample_cold` | 5912 | 0 | 5912 | 3913.012 |
| MTP | `near_limit_cold` | 27542 | 0 | 27542 | 18216.332 |
| MTP | `large_retry_cold` | 27542 | 0 | 27542 | 18053.213 |
| MTP | `summary_cold` | 2302 | 0 | 2302 | 1532.032 |
| MTP | `summary_retry_cold` | 2302 | 0 | 2302 | 1527.583 |
| MTP | `continue_cold` | 2327 | 0 | 2327 | 1555.754 |

Raw artifacts: `/tmp/gufo-card01-compaction-fn-ar-03` and
`/tmp/gufo-card01-compaction-fn-mtp`. Earlier attempts remain retained: the
first lacked the SDK; the second exposed an overly strict harness expectation
that required almost the full system prefix despite a coherent earlier
checkpoint. Their reports and distinct harness hashes remain outside Git as
superseded evidence, not cache failures under the final contract.

Validation: all 63 functional-runner unit tests passed with the pinned
Python environment; `python3 tools/ci/check-docs.py` and `git diff --check`
passed. Production paths and C++ sources were unchanged.
