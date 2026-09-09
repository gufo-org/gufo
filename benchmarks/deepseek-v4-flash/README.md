# DeepSeek V4 Flash on Strix Halo

Status: 2026-09-09. Linux x86-64, AMD Strix Halo `gfx1151`, 128 GB unified
memory. Use release binaries from `nix build` for every model performance run.
`C` is simultaneous users. Decode tok/s is **per user**, unless labelled aggregate;
prompt throughput and whole-request throughput are different measurements.

| Artifact | Pin |
| --- | --- |
| Target | `antirez/deepseek-v4-gguf`, revision `1cd7b564460821938add0475a60b942c409295e0` |
| Target file | `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf` (80.76 GiB) |
| DSpark support | `DeepSeek-V4-Flash-DSpark-support-0731.gguf`, revision `e7f04037032990db0346398d249baf9fb9df1ccc` |
| Quality reference | `antirez/ds4` revision `84cc882352757baf628a1776badf7cc54d584e28` |

## Single user, autoregressive

| Prompt tokens | Prefill tok/s | Decode tok/s |
| ---: | ---: | ---: |
| 32 | 66.32 | 16.82 |
| 64 | 117.28 | 16.73 |
| 128 | 197.13 | 16.58 |
| 256 | 270.10 | 16.55 |
| 512 | 364.38 | 16.47 |
| 1,024 | 441.59 | 16.22 |
| 2,048 | 501.68 | 15.80 |

Each row measures prefill from an empty context and 64 generated tokens after
a prefix of the stated length. These are separate benchmark runs; context
preparation is excluded from decode timing. Values are means of three runs.
The CLI uses a fixed repeating token sequence; natural prompt acceptance varies.

## Single user, DSpark

| Prompt tokens | Prefill tok/s | Decode tok/s |
| ---: | ---: | ---: |
| 32 | 66.55 | 41.38 |
| 64 | 117.49 | 40.60 |
| 128 | 196.78 | 32.37 |
| 256 | 270.45 | 26.07 |
| 512 | 363.87 | 25.50 |
| 1,024 | 439.68 | 24.10 |
| 2,048 | 498.68 | 17.38 |

DSpark adapts its draft tail from three tokens up to the requested/artifact
limit. Low acceptance temporarily returns C1 to autoregressive decoding.
Sampled requests use autoregressive decoding. Attaching the support artifact
selects DSpark in `prompt`, `chat`, `bench`, and `serve`; explicit
`--speculative off` takes precedence.

## Multiple users, autoregressive

| Users | Per-user decode tok/s | Aggregate request output tok/s |
| ---: | ---: | ---: |
| 2 | 14.88 | 29.59 |
| 4 | 11.98 | 47.48 |
| 6 | 9.70 | 57.54 |
| 8 | 9.73 | 76.66 |

HTTP `repetition_sequence`, 64 output tokens, homogeneous synchronized rounds,
default server caching, one warmup and three measured repetitions. AR restores
the warmed 31-token prefix; DSpark rebuilds it because snapshots lack support
state. Decode is the median per request. Aggregate output includes that prefill
and measures the complete round; it is not a cache-matched prefill comparison.
Request-local attention and state are retained while target projections share
weight reads across active users.

## Multiple users, DSpark

| Users | Per-user decode tok/s | Aggregate request output tok/s | Draft acceptance |
| ---: | ---: | ---: | ---: |
| 2 | 23.59 | 32.82 | 88.0% |
| 4 | 15.04 | 38.61 | 91.7% |
| 6 | 7.66 | 32.23 | 73.9% |
| 8 | 6.86 | 36.39 | 78.8% |

Concurrent requests share support computation and verify independently sized
blocks. The measured draft ceilings are three at C2/C4/C8 and two at C6.
DSpark speed depends on acceptance; it does not beat autoregressive batching
on every workload: AR wins decode at C6/C8 and aggregate request throughput from
C4 because its warmed prefix can be restored. On this same HTTP case, C1 reaches **31.36 tok/s** with DSpark and **17.09 tok/s** without it. C1
therefore retains the highest per-user performance in the DSpark sweep.

## Reproduce

```sh
git add <changed-paths>
nix build
MODEL=/path/to/target.gguf
DSPARK=/path/to/DSpark-support.gguf

# Omit --dspark-model for autoregressive runs.
./result/bin/gufo bench --model "$MODEL" --dspark-model "$DSPARK" \
  -p 2048,32,64,128,256,512,1024,2048 -n 0 -r 3
./result/bin/gufo bench --model "$MODEL" --dspark-model "$DSPARK" \
  -p 0 -n 64 -d 0,32,64,128,256,512,1024,2048 -r 3
./result/bin/gufo bench --model "$MODEL" --dspark-model "$DSPARK" \
  -p 128 -n 16 --validate-prefill 128
./result/bin/gufo serve --host 127.0.0.1 --port 8080 --sessions 8 llm \
  --model "$MODEL" --dspark-model "$DSPARK"

# In another terminal, while the server is running:
nix develop -c tools/serving/gufo-serving-bench.py \
  --base-url http://127.0.0.1:8080 --gufo ./result/bin/gufo \
  --suite benchmarks/qwen3.8-27b/speculative-corpus.json \
  --case repetition_sequence --corpus-layout homogeneous \
  --concurrency 1,2,4,6,8 --max-tokens 64 --warmup 1 --repetitions 3 \
  --output /tmp/ds4-serving.json
```

Discard the initial 2K prefill and depth-zero decode rows as warmups. Benchmark
work is fixed token count, including stop-token IDs, with no EOS resampling.
DSpark depth prefixes are rebuilt outside timing because target snapshots do
not contain the support cache. For HTTP sweeps use the shared
`tools/serving/gufo-serving-bench.py`; keep corpus, output length, cache setting,
concurrency, warmup, and repetitions identical across comparisons.

## Quality contract

All DS4-owned checks are registered in
[`tests/models/deepseek_v4_flash/CMakeLists.txt`](../../tests/models/deepseek_v4_flash/CMakeLists.txt).
Shared sampler, scheduler, and HTTP tests stay with those shared components.
Formatting and static analysis run once through Nix, outside the runtime suites. The [tools index](../../tools/ds4/README.md) lists maintained runners.

| Check | Why we maintain it |
| --- | --- |
| `ds4.template`, `ds4.cli`, `ds4.dataset`, `ds4.eval` | Official framing, CLI option wiring, pinned dataset integrity, and answer grading |
| `ds4.q2-down` | Actual production Q2 kernel versus independent reference, bitwise finite output at C2/C4/C6/C8, uniform and ragged groups |
| `ds4.target` | Official token-ID goldens, pinned 128-token trajectory, full-logit prefill/decode comparison, exact 2K prefill repeats, concurrent state and snapshot restoration |
| `ds4.dspark` | Scalar replay quality; exact repeated tokens/logits/counters/positions at C1/C2/C4/C6/C8; varied budgets, compression boundaries, prompt seeding, and safe snapshot fallback |
| `ds4.serving` | Sampling, incremental prefill, scheduler state, prefix reuse, cancellation, and server adapter behavior |

```sh
nix develop -c tools/ds4/check.py fast
nix develop -c tools/ds4/check.py kernels
nix develop -c tools/ds4/check.py all --model "$MODEL" --dspark-model "$DSPARK"
nix build --no-link .#checks.x86_64-linux.pr
# With a C1 DSpark server running, repeat and compare response/reasoning/grades:
./result/bin/gufo eval --base-url http://127.0.0.1:8080/v1 \
  --questions 4 --greedy --output /tmp/ds4-quality.json
```

The pinned target trajectory must retain at least 116/128 top-1 choices, rank
sum at most 142, and worst rank at most 3. State comparisons require finite
logits, RMSE at most 1.12, cosine at least 0.979, and max error at most 5.
DSpark C1 scalar replay additionally requires at least 125/128 top-1 choices
and worst rank at most 2 on four diverse continuations. These bounds qualify
numerical quality; they do not promise identical AR and DSpark text.

Repeatability requires identical inputs, seeds, and execution configuration.
Batch-size/composition invariance is not established: the fixed-width baseline
matches C1/C2 text on 7/10 corpus cases. The current homogeneous DSpark HTTP
sweep repeats exactly at every tested C; asynchronous AR scheduling can change
batch shapes and text within the same nominal C. Do not hide changed outputs behind a
throughput average. Every optimization must pass the relevant kernel oracle,
the retained model gates, and repeated release A/B runs. Compare output hashes
and draft counters alongside speed; widen coverage when a change affects new
shapes. Do not weaken quality thresholds to accept a faster candidate.

The current greedy DSpark capability smoke passed the first four pinned cases
twice: answers `B`, `C`, `70`, `C`, identical visible text and reasoning, 879
completion tokens per run, no errors or length finishes. This and the
[retained AR evaluation](eval/README.md) are regression samples, not official
benchmark scores. Full capability sweeps remain TODO.

## Experiments

- Retained: grouped Q2 activation reuse; matched C2-C8 decode improvement of about 3%, with bitwise kernel agreement.
- Retained: three Q2 tiles per pass at C6; about 0.5% additional decode improvement.
- Retained: shared adaptive C1 policy; repetition64 improved from 18.81 to 31.45 tok/s on the preceding release.
- Rejected: MMQ for speculative verification; numerical agreement fails.
- Rejected: learned-confidence draft trimming and alternate grouped gate variants; no qualified speed win.
- Simplified: specialized grouped gate/up arguments; 136→128 VGPRs, no scratch, exact fingerprints, throughput within ±0.1% in three interleaved A/B pairs.
- Rejected: hardcoding the gate input width; unchanged registers and kernel time (942.6→942.5 ms), no meaningful speed gain.
- Rejected: two prefill Q2 column fragments instead of four; exact logits, but pp2048 fell 502.09→490.74 tok/s and kernel time rose 1,337→1,535 ms.
- Profile: 2K prefill spends 37.8% of GPU time in quantized matrix multiplication and 17.8% in MoE down projection. Further TG/PP speedups remain TODO; neutral cleanups are not counted as gains.
