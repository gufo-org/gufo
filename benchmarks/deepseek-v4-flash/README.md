# DeepSeek V4 Flash on Strix Halo

Linux x86-64, AMD Strix Halo `gfx1151`, 128 GB unified memory. All model timings
use Nix release binaries. `C` is simultaneous requests. The sweep adds **2,048
prompt tokens** or generates **128 tokens** at each listed context depth.

Results are means of two repetitions. The [full report](speed-matrix.json)
retains deviations, token hashes, draft counters, and binary/model identities.
C1 retains the highest per-user generation rate at every depth.

| Artifact | Pin |
| --- | --- |
| Target | `antirez/deepseek-v4-gguf`, revision `1cd7b564460821938add0475a60b942c409295e0` |
| Target file | `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf` (80.76 GiB) |
| DSpark support | `DeepSeek-V4-Flash-DSpark-support-0731.gguf`, revision `e7f04037032990db0346398d249baf9fb9df1ccc` |
| Short-trajectory comparison | `antirez/ds4` revision `84cc882352757baf628a1776badf7cc54d584e28` |
| Official 0731 continuations | `antirez/ds4` revision `6289c516273979173abbc062209a81dd3706b804` |
| Official computation | `deepseek-ai/DeepSeek-V4-Flash-0731` revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062` |

## Single user, autoregressive

| Context depth | pp2048 tok/s | tg128 tok/s |
| ---: | ---: | ---: |
| 0 | 455.16 | 16.72 |
| 4,096 | 443.34 | 15.06 |
| 8,192 | 435.44 | 14.80 |
| 12,288 | 431.90 | 14.62 |
| 16,384 | 425.92 | 14.52 |

## Single user, DSpark

| Context depth | pp2048 tok/s | tg128 tok/s |
| ---: | ---: | ---: |
| 0 | 452.23 | 16.50 |
| 4,096 | 440.47 | 39.53 |
| 8,192 | 432.81 | 38.50 |
| 12,288 | 427.24 | 37.74 |
| 16,384 | 420.87 | 36.03 |

Warm C1 prefill reaches **468.5 / 464.8 tok/s at pp2048** and
**492.9 / 487.7 tok/s at pp4096** (AR / DSpark; four warm repetitions).
The first C1/depth-zero prefill row above includes first-use work, which lowers
its mean and increases its deviation. The full report retains both measurements.

## Multiple users, autoregressive

Cells are **aggregate pp2048 / per-user tg128**, in tok/s.

| Context depth | C2 | C4 | C6 | C8 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 469.66 / 16.24 | 467.92 / 13.89 | 467.03 / 11.46 | 466.11 / 9.93 |
| 4,096 | 443.33 / 13.64 | 441.27 / 10.64 | 441.46 / 8.36 | 440.98 / 7.12 |
| 8,192 | 434.47 / 13.22 | 434.07 / 10.15 | 433.70 / 7.92 | 433.93 / 6.70 |
| 12,288 | 430.31 / 12.92 | 429.46 / 9.81 | 429.64 / 7.62 | 430.04 / 6.41 |
| 16,384 | 424.25 / 12.74 | 424.18 / 9.63 | 423.73 / 7.45 | 424.06 / 6.25 |

## Multiple users, DSpark

Cells are **aggregate pp2048 / per-user tg128**, in tok/s.

| Context depth | C2 | C4 | C6 | C8 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 463.78 / 16.11 | 463.05 / 13.00 | 460.27 / 10.82 | 461.10 / 9.63 |
| 4,096 | 438.88 / 24.78 | 438.76 / 14.21 | 438.71 / 9.18 | 438.59 / 6.98 |
| 8,192 | 431.19 / 23.88 | 431.52 / 13.69 | 431.51 / 8.84 | 431.49 / 6.57 |
| 12,288 | 426.60 / 23.14 | 427.06 / 13.25 | 427.04 / 8.59 | 427.18 / 6.65 |
| 16,384 | 420.83 / 22.50 | 421.07 / 12.86 | 420.97 / 8.34 | 421.23 / 6.44 |

At 4K–16K, DSpark speeds up generation by **2.48–2.62× at C1,
1.77–1.82× at C2, 1.34–1.35× at C4, and 1.10–1.13× at C6**.
C8 is about 2% slower at 4K/8K and 3–4% faster at 12K/16K.
Depth-zero generation is 0.8–6.4% slower with DSpark on this workload.

DSpark chooses draft lengths from [measured cycle costs](cost-calibration.json)
and backs off when acceptance is too low. It skips cycles whose verification
cost cannot be repaid even at full acceptance. Decisions use token history and
offline calibration; wall-clock timing never changes token decisions. Sampled requests
use target decoding while eligible greedy peers retain DSpark batching. Changes
in concurrency reset the decision history. Complete support state and controller
state survive snapshots; a new request reusing a prefix starts fresh statistics.

## C1 memory with a 262,144-token capacity

The 80.76 GiB target weights are only part of the footprint: DSpark adds
5.58 GiB of support weights, and inference also needs projection caches,
KV state, and work buffers. The runtime reuses scratch across consecutive
attention and FFN stages, omits the dense attention mask, and grows compressed
KV storage and indexer score scratch with actual context use. Cache precision
is unchanged; prefill reserves 128 tokens of decode headroom.

| C1 DSpark workload | Previous GPU allocation (GiB) | Current (GiB) | Saved (GiB) |
| --- | ---: | ---: | ---: |
| pp2048/4096 + tg128 | 98.79 | 90.74 | 8.06 |
| 16K prefix, pp4096 + tg128 | 98.79 | 91.46 | 7.33 |

These are driver GTT measurements, not whole-system RAM usage. The
[memory report](memory-c1-262k.json) retains matched release runs, exact output
hashes and draft counts, timings, and memory observations. Scratch reuse and
growing KV save another **6.07 GiB** at pp2048/4096 beyond the earlier indexer
scratch reduction. The controls found no material speed regression; a separate
256-token continuation checks growth during generation. Only up to 20,608
tokens are used; capacity does not imply filling the window. KV storage grows
with longer inputs.

## Reproduce

```sh
git add <changed-paths>
nix build
MODEL=/path/to/target.gguf
DSPARK=/path/to/DSpark-support.gguf

# Omit --dspark-model for autoregressive runs; use -c 1 for single user.
./result/bin/gufo bench --model "$MODEL" --dspark-model "$DSPARK" \
  -c 1,2,4,6,8 -p 2048 -n 128 -d 0,4096,8192,12288,16384 -r 2 -v
# Capture stdout and stderr for each run, then validate the pair:
nix develop -c tools/ds4/check.py benchmark \
  --ar-log /tmp/ar.log --dspark-log /tmp/dspark.log --output /tmp/bench.json

# Large capacity, single-user server; ordinary prompts suffice for memory checks.
./result/bin/gufo serve --host 127.0.0.1 --port 19231 --sessions 1 llm \
  --model "$MODEL" --dspark-model "$DSPARK" --context 262144
```

Prefill and generation are separate measurements. The prefix uses a fixed
repeating token sequence, identical across concurrent requests; generation
follows the model’s greedy choices. Depth-zero generation uses a 16-token seed. Context preparation and
snapshot restoration occur outside timing. Prefill runs each request in
sequence and reports aggregate prompt throughput; decode reports 128 divided
by the whole batch's elapsed seconds.
Stop-token IDs count toward the fixed workload. Verbose output records generated
token hashes and draft counters for every request and repetition. Natural
prompts can have different acceptance and speed. Run model jobs and builds
sequentially, including correctness runs; the 128 GB memory is shared by CPU
and GPU. Keep other CPU/GPU work out of timed runs. HTTP scheduling measurements
use the shared `tools/serving/gufo-serving-bench.py` with matched prompts, caching,
output budgets, concurrency, and repetitions.

## Quality contract

**Current status:** [all nine checks pass](quality-qualification.json), including
736 exact DSpark scalar replay choices through C8/16K.
The repeated speed matrix passes exact output/hash and draft-counter checks:
53,760 generated tokens across both modes and repetitions. Its DSpark run kept
at least 13.89 GiB available under a 12 GiB memory guard.
Full capability qualification remains pending.
Antirez comparisons flag four post-prefill differences; they do not establish
which implementation is more accurate.
See the [scores and limits](eval/README.md).

DS4 checks live in [one directory](../../tests/models/deepseek_v4_flash/CMakeLists.txt).
Shared sampler, scheduler, and HTTP tests remain with their shared components.
The [tools index](../../tools/ds4/README.md) lists the maintained entry points.

| Check | Required coverage |
| --- | --- |
| `ds4.template`, `ds4.cli`, `ds4.dataset`, `ds4.eval` | Official framing, option wiring, pinned fixture integrity, answer grading |
| `ds4.projections` | 54 Q8/IQ2/F16 shape cases against scalar kernels and independent formulas; two HC cases against the official FP32 projection/RMS formula |
| `ds4.attention` | 28 target/support arithmetic cases plus 16 official DSpark window cases; double-precision references, poisoned stale rows, ring wrap, masks and sparse causal indices |
| `ds4.target` | Official token goldens, pinned trajectory, full-logit prefill/decode comparisons, exact 2K logits at 4K/262K capacities, concurrent state isolation, bounds |
| `ds4.dspark` | Scalar quality; exact tokens/logits/counters at fixed and changing C; short budgets; complete snapshot continuation through 16K; policy backoff and fork isolation |
| `ds4.serving` | Mixed sampling and actual batch widths, bounded prefill under arrivals, C1 warm-prefix equality at 262K capacity, disk identity, cancellation, context exhaustion |

```sh
nix develop -c tools/ds4/check.py fast
nix develop -c tools/ds4/check.py kernels
nix develop -c tools/ds4/check.py all --model "$MODEL" --dspark-model "$DSPARK"
nix develop -c tools/ds4/check.py reference --model "$MODEL" \
  --upstream /path/to/pinned-antirez-checkout --output /tmp/ds4-reference
nix build --no-link .#checks.x86_64-linux.pr
# Against a C1 DSpark server; repeat and compare response, reasoning and grades:
./result/bin/gufo eval --base-url http://127.0.0.1:8080/v1 \
  --questions 16 --greedy --output /tmp/ds4-quality.json
```

The pinned target trajectory requires at least 116/128 top-1 choices, rank sum
at most 142, and worst rank at most 3. State comparisons require finite logits,
RMSE ≤1.12, cosine ≥0.979, and max error ≤5. DSpark scalar replay requires exact greedy tokens (including tie-breaking)
and frontier logits. The eight-scenario C1/C2/C4/C6/C8 replay through 16K
covers **736 token choices** across four diverse continuations. Never weaken these bounds to accept a faster implementation.
The replay records speculative frontiers, frees those sessions, then checks
one scalar session at a time, keeping at most C model sessions resident.

Repeatability means identical inputs, seeds, and execution schedule, including
prefill boundaries and batch membership. AR/DSpark output equality passes this
speed workload; general text equality and batch-composition invariance are not
established. Check repeated output hashes and draft decisions alongside
speed; throughput alone cannot qualify a change. Retain an optimization only
after the relevant numerical oracle, model checks, and repeated release A/B
measurements pass. During iteration, run the affected fast/kernel checks; run
the complete model/serving suite for retained changes. Extend coverage for newly affected shapes and shared runners.
AR qualification also requires the external 100-case continuation comparison:
agreement between DSpark and AR cannot detect a bug shared by both paths.
The scorer records every reference token; compare likelihood and greedy
agreement against a matched independent upstream control. The same command
compares full logits before and after 128 forced tokens at all five depths,
using matched 2K and 4K prefill calls. It checks exact repeated Gufo output,
identical input token IDs, and actual prefill capacities.
It retains the full-logit bounds above and rejects a detectable 100-case NLL
increase using a paired case bootstrap 95% interval (seed 731).
Antirez is an independent implementation, not ground truth. Cross-engine logit
differences require investigation against the official computation and task
scores; do not add approximations solely to match another implementation.
Separate GGUF quantization error from additional kernel approximations. The HC
oracle uses GGUF weight precision with the official FP32 activation formula.
Repeated-word prompts probe numerical behavior, not long-context task accuracy.
Capability subset results: TODO. Full capability scores remain TODO; see the
[retained AR evaluation](eval/README.md) for earlier regression samples.

## Experiments

- Retained: scalar-equivalent verifier projections and attention; exact 736-token replay across the maintained concurrency/depth matrix.
- Retained: attention ring indexing; removes integer division without changing arithmetic.
- Retained: reusable request scratch and complete DSpark snapshots; exact continuation across changing batch membership and session capacities.
- Retained: exact-size weight allocations remove 6.74 GiB of arena padding; the complete guarded DSpark suite kept at least 13.84 GiB available.
- Retained: official DSpark feature injection and 128-row attention window, checked independently across ring boundaries.
- Retained: direct IQ2 activation reads; C8 verifier gate/up GPU time 140.94→62.29 ms, total verifier 531.08→452.74 ms, exact scalar outputs.
- Rejected: F16 rounding inside fused HC; closer to Antirez's logits, but adds an approximation absent from the official formula.
- Retained: concurrent cost policy calibrated at 0/4K/16K; the repeated full sweep passes output equality and reports the gains and losses above.
- Rejected: speculative MMQ (numerical mismatch), confidence trimming and alternate grouped gate variants (no qualified speed win).
- Rejected: two prefill Q2 column fragments; exact logits, pp2048 fell 502.09→490.74 tok/s.
- Prefill profile, pp2048 at 4K: quantized matrix multiplication takes 35.8% of kernel time, MoE down projection 17.1%, and attention 14.1%. Further kernel gains remain TODO.
