# DeepSeek V4 Flash on Strix Halo

Linux x86-64, AMD `gfx1151`, 128 GB unified memory. Nix release binaries.
`C` is simultaneous requests. Single-user measurements use **pp2048 / tg128**;
depth precedes the measured operation. Unknown current measurements are **TODO**.

**Target parity remains open:** the optimized build misses the historical
trajectory gate, and four post-prefill differential alerts remain unresolved.
DSpark replay does not establish target correctness. [Evidence and limits](EVALUATION.md).

Target: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`
(80.76 GiB), `antirez/deepseek-v4-gguf` revision
`1cd7b564460821938add0475a60b942c409295e0`. DSpark support:
`DeepSeek-V4-Flash-DSpark-support-0731.gguf`, revision
`e7f04037032990db0346398d249baf9fb9df1ccc`.

Cold-file-cache launch to HTTP readiness: **36.82 s** on 2026-09-21,
including DSpark and two sessions at context capacity 4096. The integer
expert-routing tensor loads without conversion; serving replay remains exact
under the [state checks](EVALUATION.md).

## Single user, autoregressive

Latest depth control: **2026-09-19**, two release measurements at d0/d32K.
Values are mean ± standard deviation; initial kernel setup contributes to
depth-zero prefill variation. The final packing layout needs a d64K speed refresh.

| Context depth | pp2048 tok/s | tg128 tok/s |
| ---: | ---: | ---: |
| 0 | 450.26 ± 33.18 | 17.74 ± 0.00 |
| 4,096 | TODO | TODO |
| 8,192 | TODO | TODO |
| 12,288 | TODO | TODO |
| 16,384 | TODO | TODO |
| 32,768 | 422.18 ± 3.61 | 14.86 ± 0.00 |
| 65,536 | TODO | TODO |

## Single user, DSpark

Same workload and measurement date: two measurements at d0, one at d32K.
DSpark prefill includes support-state work.

| Context depth | pp2048 tok/s | tg128 tok/s |
| ---: | ---: | ---: |
| 0 | 453.49 ± 25.96 | 17.45 ± 0.00 |
| 4,096 | TODO | TODO |
| 8,192 | TODO | TODO |
| 12,288 | TODO | TODO |
| 16,384 | TODO | TODO |
| 32,768 | 417.80 | 35.30 |
| 65,536 | TODO | TODO |

The CLI uses a repeating token sequence. Natural prompts can have substantially
different acceptance and speed. Depth-zero generation starts from 16 tokens;
it does not follow the measured 2048-token prefill.

## Multiple users, autoregressive

Current pp2048/tg128 depth sweep: **TODO**. Report aggregate prompt throughput
and per-user generation throughput at each depth.

| Users | d0 / d4K / d8K / d12K / d16K / d32K |
| ---: | --- |
| 2 | TODO |
| 4 | TODO |
| 6 | TODO |
| 8 | TODO |

## Multiple users, DSpark

Current pp2048/tg128 depth sweep: **TODO**, including acceptance and output
checks at every concurrency.

| Users | d0 / d4K / d8K / d12K / d16K / d32K |
| ---: | --- |
| 2 | TODO |
| 4 | TODO |
| 6 | TODO |
| 8 | TODO |

Latest fixed-prompt sampled HTTP control (2026-09-19, C2, temperature 1,
top-p 0.95, seed 7): **17.40 tok/s per user** for an autumn explanation,
**21.01** for a repeating pattern, and **15.27** for a short naming task.
Two fresh-server measurements; details and limits are in the
[sampling evidence](EVALUATION.md#dspark).

DSpark combines offline cycle costs and acceptance history with checkpoint
confidence for filtered sampled C>1 requests. These use exact hybrid top-8
proposals; C1 and unfiltered sampling retain point-mass proposals. Stochastic
verification preserves the target distribution, with its own seeded trace.
Greedy behavior is unchanged. Requests keep private RNG, proposal, acceptance
and controller state; prefix reuse starts fresh request statistics. Live timing
never changes token decisions.

## Memory

C1 with context capacity **262,144**. GPU-visible unified allocations exclude
separate CPU memory; capacity does not imply filling the window.

| DSpark workload | GPU allocation |
| --- | ---: |
| pp2048/4096 + tg128 | 90.74 GiB |
| 16K prefix, pp4096 + tg128 | 91.46 GiB |

Measured in the earlier C1 capacity control; a current memory refresh is TODO. Target and
support weights, KV and working buffers all contribute. Compressed KV and score
scratch grow with actual use. Prefill reuses scratch across ordered stages;
indexer scoring borrows the idle attention-output range for temporary F16 keys.
Queries share that range and feed WMMA directly. Persistent KV precision is
unchanged.

## Reproduce

```sh
nix build
MODEL=/path/to/target.gguf
DSPARK=/path/to/DSpark-support.gguf
./result/bin/gufo bench --model "$MODEL" \
  -c 1 -p 2048 -n 128 -d 0,32768 -r 2 -v
./result/bin/gufo bench --model "$MODEL" --dspark-model "$DSPARK" \
  -c 1 -p 2048 -n 128 -d 0,32768 -r 2 -v
```

Use `-c 1,2,4,6,8` for concurrency and
`-d 0,4096,8192,12288,16384,32768` for the full depth sweep. Use `-p 4096`
for pp4096. Sampled controls use the same `--temperature` and `--seed` in both
modes. Context preparation and restoration are outside timing. Verbose output
records hashes and draft counts; compare these alongside speed. Run model
jobs, builds and profiles sequentially. The HTTP benchmark is
`tools/serving/gufo-serving-bench.py`; distinguish cold requests from cache hits.
