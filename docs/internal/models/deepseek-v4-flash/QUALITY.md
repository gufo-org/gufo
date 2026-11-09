# DeepSeek V4 Flash quality

**DSpark matches Gufo AR on the retained greedy checks, including C>1.
Official target-model parity is not yet established.** The Flash 0731 mixed IQ2/Q2/Q8
weights and DSpark sidecar are [pinned](artifacts/model-identities.json).

| Check | Retained result |
| --- | --- |
| DSpark greedy replay | 736/736 token choices match AR; maximum full-logit error 0, through C8 / 16K |
| HTTP AR/DSpark, C1/2/4/6/8 | All 63 tg128 responses match fresh AR; mixed/repetitive prompts, October 30, 108 GiB GTT |
| Sparse-indexer boundary | DSpark C1 and AR C2: 0/128 token differences, bit-identical full logits |
| DSpark target state | `ds4.dspark_verify`: after every cycle through 32 tokens, snapshot target state equals AR |
| Reproducibility | 1280/1280 choices and 20 full-logit vectors repeat exactly |
| Sampling / sessions | p/q acceptance, residual correction, seeded replay, EOS/cancellation, three-turn continuation and disk restore pass |
| Official continuations | `ds4.target` limit: NLL ≤ 0.407 and ≥ 53/100 first tokens; ROCm 10.1 scores 0.4026 and 59 |
| Optimized versus Debug | **33/2327** greedy choices differ |
| Historical capability set | **53/75** correct; 22 failures, no execution errors, nine length finishes |

Evidence: [DSpark sampling/replay](artifacts/dspark-sampling.json),
[boundary checks](artifacts/indexer-boundary.json), the HTTP cohorts in
`artifacts/multi-*-gufo-*.json`,
[qualification](artifacts/quality-qualification.json) and
[formula audit](artifacts/prefill-formula-audit.json).
Gufo-path agreement cannot detect shared target errors. Sampled DSpark need not
match AR's same-seed sequence; replay requires the same execution configuration
and schedule. Ask before rerunning the 75-question capability set.

## Target quality limit

`ds4.target` first compares the SHA-256 of `GUFO_DEEPSEEK_V4_FLASH_MODEL` with
the pinned target file. It then scores the 100 official continuations, 2,313
teacher-forced tokens from the hosted model, with ordinary decoding. The
token-weighted NLL must be at most 0.407, and at least 53 of the 100 first
tokens must match. Scoring takes about 3.5 minutes on gfx1151.

| Build | NLL | First tokens |
| --- | ---: | ---: |
| Gufo, September 11, ROCm 7.2.3 | 0.3994 | 56 |
| Gufo `f86cd0d9`, ROCm 7.2.3 | 0.4006 | 56 |
| Gufo `211dba49`, ROCm 10.1 | 0.4026 | 59 |
| antirez/ds4 `6289c516` | 0.4034 | 54 |
| Rejected compressor and router prototype, September 11 | 0.4073 | 53 |
| Rejected compressor prototype, September 11 | 0.4085 | 56 |

The limits admit every accepted build and reject both prototypes that the
[operator audit](artifacts/captured-operator-audit.json) rejected. The scorer is
deterministic for one build, so the margin of 0.0044 over ROCm 10.1 covers only
arithmetic changes. Accepted Gufo builds since September span 0.0032. Token
choices move more than these scores: at unchanged source, ROCm 10.1 changed
greedy agreement with a 128-token antirez/ds4 trajectory from 115 to 106 while
NLL moved from 0.4006 to 0.4026.

## DSpark target state

A DSpark cycle verifies its anchor and drafts in one pass. That pass projects
the attention and indexer compressor pairs with decode's one-row kernel, so
each verified row stores the compressor state that a one-token decode stores.
Until November 2026, verification projected the 4,096×256 indexer compressor
as two single matrices. That shape is also the router's, so it took the ordered
batch kernel, and the stored rows differed from AR in their low bits. Logits
cannot show these rows below about 2,048 tokens: while a layer holds at most
512 compressed rows, the indexer selects all of them. Past that depth the rows
change indexer scores, and a greedy chat turn at a depth of 2,048 tokens
diverged from AR at token 78 of 128.

`ds4.dspark_verify` syncs an AR session and a DSpark session on one short chat
prompt and generates 32 greedy tokens. After every DSpark cycle, the test
requires the AR token choices and the AR target sections of the session
snapshot: tokens, logits, raw rows, compressed rows, and compressor state. It
names each section that differs. The snapshot header, the DSpark request
state, and the drafter rows differ by mode, so the test does not compare them.

## Prefill agreement limits

`ds4.target` also compares batched prefill with token-by-token evaluation of
the same tokens: a 273-token prompt and conversational prompts of 32, 64 and
96 tokens. Every comparison requires finite logits and needs the sequential
choice to rank at most third in the batched logits. The full-logit limits are:

| Prompt | RMSE | Cosine | Maximum error | Worst measured RMSE / cosine / maximum error |
| --- | ---: | ---: | ---: | --- |
| 273 tokens | ≤ 1.12 | ≥ 0.979 | ≤ 5.0 | 0.56 / 0.99 / 2.79 |
| 32 tokens | ≤ 1.5 | ≥ 0.97 | ≤ 7.5 | 1.23 / 0.975 to 0.979 / 6.30 |
| 64 and 96 tokens | ≤ 1.12 | ≥ 0.979 | ≤ 5.0 | 0.70 / 0.99 / 3.89 |

The 1.12, 0.979 and 5.0 limits were set for the 273-token prompt in August and
copied to the conversational prompts in September, before any ROCm 10.1 run.
The worst values come from six builds measured on gfx1151 in November:
`f86cd0d9` and `270b5333` with ROCm 7.2.3, `f86cd0d9` with ROCm 7.2.3 and
assertions enabled, `f86cd0d9` and `f07fd2fe` with ROCm 10.1, and `f86cd0d9`
with ROCm 10.1 and the Clang 24 router and Sinkhorn order turned off. Every
ROCm 7.2.3 build passes the original limits at 32 tokens, with at most
0.51 / 1.00 / 2.52. Every ROCm 10.1 build exceeds them only at 32 tokens, and
the official continuation NLL of those builds is at most 0.4026. The test
prints cosine with two decimals. ROCm 10.1 prints 0.98 and fails 0.979, so its
value is between 0.975 and 0.979.

The 32-token limits leave about 20% over the worst build in each measure:
RMSE 1.23 to 1.5, maximum error 6.30 to 7.5, and 1 − cosine at most 0.025 to
0.03. At 64 and 96 tokens the original limits keep at least 28% over the worst
build in each measure, so they stay unchanged.

Full-logit RMSE and cosine weigh every vocabulary entry equally, so tokens that
sampling never selects dominate them. A better long-term check is the KL
divergence `KL(p_seq || p_batched)` from the sequential to the batched token
distribution, together with the maximum error over the 64 highest sequential
logits. That check is future work.

## Reproduce

Use the smallest affected [maintained suite](../../../../tools/ds4/README.md):

```sh
nix develop -c tools/ds4/check.py fast
nix develop -c tools/ds4/check.py kernels
nix develop -c tools/ds4/check.py model --model "$MODEL" --dspark-model "$DSPARK"
```

Independent FP64 operators and exact sparse top-k checks supplement model/state
replay. Keep full-logit captures outside Git; do not relax quality gates.

`ds4.dspark` replays C8 at depth 16384 only when HIP reports at least 100 GiB
of device memory. With the target and DSpark files loaded, that scenario holds
91.5 GiB after its base snapshot and adds 0.97 GiB per session, so it runs out
of memory at a 98 GiB GTT limit and passed at 108 GiB. On a smaller device the
test runs every other check, prints `SKIP` with both sizes and exits with 77,
which CTest reports as skipped, not passed.

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
[benchmark workflow](../../../../.agents/skills/benchmark-model/SKILL.md).

## ROCm 10.1 migration

Focused cross-compiler checks, matched controls and remaining limits are in
[ROCM.md](ROCM.md). The original reference fixtures are unchanged, and the
32-token prefill limit above is the only tolerance widened for ROCm 10.1.
Migration measurements use separately recorded controls.
