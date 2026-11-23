# Inkling 2 Mini benchmarks

Strix Halo (Ryzen AI MAX+ 395, gfx1151, 128 GB), ROCm 10.1, November 23,
2026. Measured ceilings on this host (`tools/bench/gfx1151_peak.hip`): 240.5
GB/s DRAM read, 54.8 TFLOPS F16 WMMA.

## End to end

`gufo bench -m Inkling-2-Mini-UD-Q4_K_XL.gguf -p N -n 128 -r 2` against
llama.cpp `94f3af3d0` `llama-bench -fa 1` (HIP, the same file).

| Workload | Gufo | llama.cpp HIP |
| --- | --- | --- |
| pp512 | 1,626 tokens/s | not measured |
| pp2048 | 2,268 tokens/s | 383.5 tokens/s |
| pp8192 | 2,209 tokens/s | not measured |
| tg128 after 64 tokens | 47.0 tokens/s | 24.3 tokens/s (tg128) |
| tg128 after 2,048 tokens | 43.9 tokens/s | |
| tg128 after 8,192 tokens | 42.0 tokens/s | |

Batched decode (`inkling2_bench --ctx 8192 --batch 2,4`, aggregate):

| Sessions | 64-token context | 4,096-token context |
| --- | --- | --- |
| 1 | 46.1 | 43.5 |
| 2 | 68.3 | 61.7 |
| 4 | 93.4 | 80.2 |

The imported extension measured 44.7 / 86.2 tokens/s (1 / 4 sessions, short
context) and 1,366 tokens/s at a 4K prompt on Gufo 0.35.0; on this build its
kernels gave 1,395 tokens/s at pp2048 and 45.7 tokens/s decode.

### Windows

Same machine, Windows 11, TheRock 10.0.0, 96 GB Variable Graphics Memory,
`gufo bench -r 2`:

| Workload | Gufo (Windows) |
| --- | --- |
| pp512 / pp2048 / pp8192 | 1,648 / 2,271 / 2,199 tokens/s |
| tg128 after 512 / 2,048 / 8,192 tokens | 46.0 / 44.8 / 42.8 tokens/s |

The decode weight prefetch costs more than it saves on Windows: its
side-stream waits in the replayed graph (ABBA, C = 1: 39.4 tokens/s with
it, 46.0 without; `inkling2_bench` at a 2,048-token context, 2 / 4 / 8
sessions: 56.4 / 81.3 / 104.3 with it, 64.5 / 88.0 / 109.7 without). `Engine::Create` therefore times replayed one-session decode steps of a
scratch session with and without it and keeps it only when it is faster. The
prefetch only warms the cache, so the logits are unchanged. `inkling2_spec_test`,
`inkling2_cache_test` and `inkling2_bench --verify` (2, 4 and 8 sessions) pass
bit for bit, and six HTTP requests (greedy, seeded, a tool call and an image)
return identical responses with and without DSpark.

## Prefill changes (all byte-identical)

pp2048 progression, each step measured with the previous one in place:

| Change | pp2048 |
| --- | --- |
| Imported extension, ported | 1,395 |
| Loop-controller gate and apply re-tiled to 64 tokens x 64 rows per block (k-order kept) | 1,566 |
| Controller apply grid with a token tile's channel blocks adjacent | 1,658 |
| Residual and message-bank copies replaced by buffer swaps | ~1,665 |
| 128-token MoE gate/up kernel: 8 waves of 32 tokens x (32 gate + 32 up) | 1,697 |
| Activations straight from global memory into the WMMA A operand | 1,859 |
| Same 128 x 128 kernel for MoE down and the dense projections | 1,903 |
| MoE tail fused (slot sum, post conv, residual add) | 1,912 |
| WMMA skipped on 16-token tiles past an item's end | 2,037 |
| 128-token tiles from 256-token ubatches (was 1,024; pp512 1,378 → 1,483) | 2,041 |

The generic attention-residual mix (`mix_k`) dropped from 72 to 46 ms per
pp2048 by unrolling its loops, which issues loads ahead without reordering the
additions.

Second round (November 23), `inkling2_bench` pp2048 after a warm-up prefill
(`gufo bench` measured 2,036 before and 2,210 after; pp512 1,483 → 1,618,
pp8192 1,960 → 2,156). Outside the GEMMs nearly every prefill kernel streams
19 MB F32 activation buffers at the DRAM ceiling, so the gains come from
removing passes and from overlapping bandwidth-bound work with the
compute-bound GEMMs:

| Change | pp2048 |
| --- | --- |
| Before (PR head) | 1,990 |
| Down and K = 2304 dense GEMMs: one straight-line WMMA body per live-tile count (no per-tile branches between the LDS loads and the WMMAs); skinny sink / conv-tap projections with float4 LDS reads and a prefetched k-chunk | 2,029 |
| F16 GEMM operands written by the norm, mix and attention kernels that produce them (no separate F32 → F16 passes); 64 x 64 tiled gate for the single-job FFN controller | 2,077 |
| Attention post-conv and residual add in one pass (as the MoE tail) | 2,085 |
| Relative-bias table rows a key tile meets staged in LDS | 2,095 |
| K/V short convolutions evaluated inside the KV-cache write; unread F32 window-layer norm outputs skipped | 2,108 |
| k, v, rel and the skinny projections on a side stream beside q and gate | 2,126 |
| The next layer's messages mixed on the side stream beside the MoE gate/up GEMM | 2,143 |
| That mix also writes the messages' inverse RMS, so the controller gates skip half their prologue (ubatches of 1,024+ tokens) | 2,162 |

Decode is unchanged (45.9 tokens/s after 64 tokens, 44.0 after 2,048) and
batched decode still matches single-session decode bit for bit.

Rejected variants are in [Experiments](EXPERIMENTS.md).

## Decode changes (all byte-identical)

tg128 after 64 tokens, alternating A/B on one build each:

| Change | tg128 |
| --- | --- |
| Imported extension kernels | 45.7 |
| One session: expert GEMV grid with an expert's row blocks adjacent | 46.0 |
| 8 waves per block in every weight GEMV (rows per block only) | 46.8 |
| A pass's 12 message mixes in one launch (they share the pass's bank) | 47.0 |

At 47 tokens/s a step takes 21.3 ms: about 16.8 ms of weight GEMVs at 87–98%
of the DRAM ceiling (the LM head reads 492 MB in 2.1 ms), about 3.4 ms of small
kernels and under 1 ms of gaps. Rejected variants are in [Experiments](EXPERIMENTS.md).

## DSpark

Greedy speculative decoding against plain decoding of the same prompt
(`inkling2_bench --dspark ... --depth D`, WikiText-2, 256 tokens; output
identical in every run). Fixed depths, before the adaptive width:

| Context | Depth 1 | Depth 2 | Depth 3 | AR |
| --- | --- | --- | --- | --- |
| 64 | 57.6 (87% accepted) | 64.0 (75%) | 68.2 (70%) | 45.0 |
| 2,000 | 44.8 (48%) | 43.6 (35%) | 39.2 (26%) | 43.1 |
| 6,000 | 53.4 (80%) | 59.0 (67%) | 57.9 (57%) | 41.9 |

A depth-3 cycle at short context: propose 4.7 ms, verify 38.2 ms, commit
2.2 ms (`inkling2_bench` host timings). Verifying 2/3/4 rows of one session
(`--spec`) takes 27.8/32.9/37.8 ms against 22.1 ms for one decode step; the
extra rows mostly cost the distinct experts they route to (a 4-row step
streams about 22 experts per visit instead of 8).

HTTP, one session, default settings (`gufo serve llm --model ...
--dspark-model ...`), 400 tokens, thinking on; the text equals the AR
server's for all five requests:

| Request | AR | DSpark (adaptive) |
| --- | --- | --- |
| Python function, greedy | 45.9 | 66.7 |
| Why the sky is blue, greedy | 46.1 | 56.0 |
| First 15 primes, greedy | 46.1 | 83.8 |
| Haiku, temperature 0.8, seed | 46.0 | 59.6 |
| Translation, temperature 0.7, seed | 43.3 | 50.8 |

Sampled with the model's defaults (temperature 1, no truncation; coupled
proposals), seeded, 400 tokens; identical text again:

| Request | AR | DSpark |
| --- | --- | --- |
| Python function | 43.2 | 51.8 |
| Why the sky is blue | 43.1 | 50.7 |
| Short story, temperature 0.6 | 43.3 | 46.8 |
| Ten fruits, min-p 0.05 | 44.4 | 49.8 |
| Haiku, temperature 0.8, top-p 0.95 (greedy proposals) | 46.0 | 66.7 |

At temperature 1 about 60% of the coupled proposals are accepted; the host
sampler's pass over 200K logits (about 1.5 ms per selected token, AR too) is
the larger cost of those cycles.

### Verification at depth (multi-row global attention)

A verification step's rows are consecutive positions of one session, so on
global visits one block per (KV head, split) now reads each K/V tile once for
all rows (their 3 heads each fill the WMMA query slots one row used to leave
empty). Keys past a row's position are masked, so every row's result is the
one-row kernel's: `inkling2_spec_test` and `--spec` report 0 mismatched rows.

| `inkling2_bench --spec` | before | after |
| --- | --- | --- |
| R = 2, ctx 64 / 4,000 / 32,000 | 27.7 / 29.5 / 38.2 ms | 27.8 / 29.4 / 34.6 ms |
| R = 4, ctx 64 / 4,000 / 32,000 | 37.5 / 39.8 / 57.7 ms | 37.5 / 39.4 / 43.6 ms |
| DSpark depth 3, ctx 32,000 (AR 36.1 tok/s) | 34.8 tok/s | 44.2 tok/s |

### Wider verification (up to seven drafts)

The drafter's block holds seven proposals; the width policy now chooses among
all of them (the same expected-tokens-per-cost rule, verify costs below as its
prior). Every row is still exact (`inkling2_spec_test`: 0 mismatched rows for
2 to 8 rows).

| `inkling2_bench --spec` rows | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ctx 64 (ms) | 27.7 | 32.9 | 37.6 | 43.1 | 47.8 | 53.6 | 58.9 |
| ctx 32,000 (ms) | 34.6 | | 43.8 | 50.2 | 60.6 | 66.7 | 77.0 (before the 4 + 4 split) |

Over HTTP (one session, all responses byte-identical to plain decoding):

| Workload | AR | up to 3 drafts | up to 7 drafts |
| --- | --- | --- | --- |
| Six coding tasks, 1,024 tokens, greedy | 46.0 tok/s | 70.7 | 74.3 |
| The same, default sampling (T = 1, seeded) | 43.4 tok/s | 61.2 | 62.8 |
| Primes list (greedy) | 46.1 | 83.8 | 87.5 |
| Explanation (greedy) | 46.1 | 56.9 | 55.4 |
| Translation (T = 0.7) | 43.3 | 51.6 | 50.2 |

A replay of the coding runs drafting seven tokens every cycle accepted all
seven in 15% of the cycles and none in 24% (mean 2.6).

### Verification and drafting, round 3

Exact changes (every row still byte-identical; oracle dumps unchanged):
multi-row projections no longer spill (the tiny-matrix rows' token loop was
unrolled and hoisted every token's loads: 232 to 3,672 bytes of scratch per
lane), the three-input projections read their weights once for up to five
tokens, the attention row groups of 6 to 8 rows run interleaved, and the
multi-row LM head takes two rows per wave. Drafting uses the low-rank
proposal logits (the drafter's rounding only; acceptance unchanged on the
replays).

| `inkling2_bench --spec` rows | 2 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- |
| ctx 64, before (ms) | 27.7 | 37.6 | 43.1 | 47.8 | 53.6 | 58.9 |
| ctx 64, after (ms) | 27.4 | 36.1 | 41.3 | 45.8 | 51.2 | 57.2 |
| ctx 32,000, after (ms) | 33.4 | 43.2 | 49.1 | 55.4 | 60.9 | 68.2 |

Plain decode: 21.9 ms per token (22.0). DSpark propose at depth 7: 4.66 ms
per cycle (7.62).

| HTTP, all outputs byte-identical to AR | AR | before | after |
| --- | --- | --- | --- |
| Six coding tasks, greedy | 46.0 tok/s | 74.3 | 78.0 |
| Six coding tasks, default sampling | 43.4 tok/s | 62.8 | 65.7 |
| Primes (greedy) | 46.1 | 87.5 | 91.9 |
| Explanation (greedy) | 46.1 | 55.4 | 57.7 |
| Translation (T = 0.7) | 43.3 | 50.2 | 52.2 |


### Concurrent requests (November 23, round 4)

Requests decoding at the same time draft together (one drafter pass reads
each weight row once for all of them: `Engine::ProposeBatch`) and are checked
in one `Engine::VerifyBatch` step of up to eight rows from several sessions.
Rows go to the drafts most likely to survive while they pay for their share
of the step (`ChooseDsparkBatchWidths`); the costs by total rows are learnt
online, and a back-off per batch width stops drafting that buys nothing.
Greedy and sampled requests share a step: greedy rows take the device
argmax, sampled rows the host logits (temperature-only requests with coupled
proposals), and the requests' selections run on parallel threads.

Global-attention decode rows now take the split length of their own
position's bucket instead of the batch's largest. Before, a request batched
with one past 4K tokens got a different split, so its logits depended on
what ran beside it (`inkling2_spec_test`: 24 of 36 batched decode rows and 37
of 54 batched verification rows differed from lone decoding; 0 now).

HTTP, `--sessions 4`, eight requests of 400 tokens (four greedy, four with
the default sampling and a seed; coding and prose), aggregate tokens/s. All
48 outputs are byte-identical across AR and DSpark and across C:

| C | AR | DSpark, sessions drafting one after another | DSpark, one drafter pass and parallel selection |
| --- | --- | --- | --- |
| 1 | 44.8 | 60.7 | 60.8 |
| 2 | 65.5 | 75.8 | 77.7 |
| 4 | 88.0 | 86.9 | 90.1 |

Before this round, concurrent requests decoded AR (the AR column).

With four requests the eight-row step leaves four draft rows, and each must
beat a plain step's 10.8 ms per token; the policy drafts mostly for the
greedy coding requests there.

### Prefill, round 3

`inkling2_bench` pp2048 2,170 → 2,210 tokens/s (pp512 1,500 → 1,515, pp8192
2,075 → 2,115); `gufo bench` pp2048 2,210 → 2,268, pp512 1,618 → 1,626,
pp8192 2,156 → 2,209. Oracle dumps byte-identical:

| Change | Kernel time per pp2048 (serialized) |
| --- | --- |
| Controller apply: float4 LDS reads, epilogue inputs loaded before the products | 43.8 → 35.0 ms |
| Controller inputs' inverse RMS from the attention-residual mix (summed in the gate prologue's order), so the gates no longer re-read them | gates 37.9 → 32.8 ms |

Under the counter profiler (kernels serialized) the pp2048 kernels sum to
about the parallel wall time: MoE gate/up 5.0 ms and down 3.7 ms per visit
(about 70% and 47% of the WMMA ceiling for the live 16-token tiles), dense
projections 161 ms, attention 62 ms, attention-residual mixes 64 ms, MoE
tails 54 ms. The gate/up kernel's loop alone (dequantisation, weight and
activation loads removed) runs at 65% of the WMMA ceiling.

The exact changes left are small: every remaining kernel is either at its
DRAM floor (mixes, MoE tails, norms) or a WMMA loop already near its own
ceiling, and none of the tried restructurings beat it (see
[Experiments](EXPERIMENTS.md)). The ideas that would reach about 2,400 tokens/s
change the arithmetic and need their own quality qualification first: F16 MoE
slot rows instead of F32 (about 45 ms per pp2048), the relative-position bias
as a WMMA product, a one-pass prefill mix, and the controller inputs' inverse
RMS in the mixes' own reduction order (listed under "Not exact, not taken").

## Round 5: up to eight concurrent requests, decode overheads (November 23)

All changes exact: oracle dumps of both paths byte-identical, `inkling2_bench
--verify` 0 differences for batches of 4 and 8 (192 of 192 rows),
`inkling2_spec_test` all pass, and every HTTP response below byte-identical
to the baseline's.

### Eight sessions in one step

Batched decode and the drafter now take up to eight sessions (they took four;
the decode kernels were already eight rows wide). Steps of 5 to 8 sessions
share one weight pass.

| Change | 1 row | 4 rows | 8 rows |
| --- | --- | --- | --- |
| Before (decode step, ctx 64) | 21.85 ms | 41.88 ms | (two steps of 4) |
| K/V preparation and the split reduction inside the decode attention launch; the controllers' norm inside their apply launch (649 → 505 launches per token) | 21.76 | 41.73 | 66.1 |
| A norm's nine loads per thread issued together (explicit FMAs keep each sum) | 21.65 | | |
| 16-wave blocks for GEMVs of 4+ rows (their quantized inputs fill a CU's LDS, so 8-wave blocks left one block per CU) | | 41.78 | 64.6 |
| A visit's q (7.5 MB) read on a side stream while its inputs are mixed, its output projection and router while it attends (20 blocks; the DRAM is mostly idle during those latency-bound kernels, and the GEMVs then find the weights in the last-level cache) | 20.68 | 40.83 | 63.9 |

Verification of 2 / 4 / 6 / 8 rows of one session: 27.3 / 35.9 / 46.1 /
57.5 → 26.2 / 35.1 / 45.1 / 54.5 ms. Decode after 4,096 / 16,384 / 32,000
tokens: 23.18 / 25.11 / 27.49 → 22.10 / 24.11 / 26.58 ms per token (at 32K
the eight global visits read about 1 GB of K/V per token, near the DRAM
ceiling).

Prefetch variants (decode step at ctx 64): more of the projections (q and
part of gate) 20.79 → 20.82–21.20 ms; 40 / 80 / 160 blocks instead of 20
21.29 / 21.53 / 21.66; the controllers' weights before q 21.16; the next
visit's q beside this visit's experts 21.25 (competes with, or is evicted
by, the expert stream).

### Drafting

| Change | per cycle |
| --- | --- |
| The draft context of every session in a step injected in one pass (one GEMV per weight for all rows instead of a WMMA GEMM per session) | commit 2.09 → 1.96 ms |
| Drafter GEMVs of one input in one launch (q/k/v, gate/up, context k/v) | propose 3.47 → 3.37 ms (depth 3) |
| Low-rank proposal table in int8 with one scale per token (16.8 MB per chain step instead of 33.5) | propose at depth 7 4.31 → 3.32 ms |

Acceptance is unchanged on the replays (79 of 153 at depth 3, 191 of 469 and
114 of 962 at depth 7, contexts 64 and 2,000).

`--draft-tokens` is now honoured (it was ignored) and defaults to the
drafter's seven proposals (the provider declared 3, which the runner never
applied).

### HTTP

`gufo serve --sessions 8 --context 32768`, eight prompts (coding and prose),
256 tokens each, even requests greedy and odd ones seeded with the default
sampling (T = 1), aggregate tokens/s. Every response is byte-identical across
builds, AR and DSpark, and C:

| C | AR before | AR after | DSpark before | DSpark after |
| --- | --- | --- | --- | --- |
| 1 | 45.6 | 48.1 | 67.9 | 71.8 |
| 2 | 62.0 | 66.6 (65.5–68.3, three runs) | 67.0 | 71.0 |
| 4 | 86.1 | 87.9 | 84.3 | 89.1 |
| 8 | 87.2 | 112.4 | 90.3 | 111.1 |

At C = 8 the step is bound by expert weights: the MoE kernels take 70% of it
and stream about 250 GB/s over the 50 or so distinct experts its 64
(token, slot) pairs route to. Drafting cannot pay there: each extra
verification row costs about 4.7 ms (the experts it adds) and survives about
60% of the time, roughly 8 ms per drafted token against 8.3 ms per token for
the plain eight-row step, so the policy does not draft at C = 8 and DSpark
trails AR by the context injection (about 1.4%).

Where a C = 1 step went before the prefetch (21.5 ms span): weight GEMVs about 16.8 ms at
86–98% of the DRAM ceiling, 505 launches of which the latency-bound small
kernels (mixes, attention, router, controllers) take about 3.6 ms and the
launch gaps about 1 ms. A C = 1 DSpark cycle (36.5 ms at three rows) keeps
the GPU 96% busy: verification 31.9 ms, drafter 3.1 ms. The host sampler
(temperature 1: a double-precision exp per vocabulary entry, in token order)
costs about 1.5 ms per sampled token outside the GPU.

## Round 6: host-side selection, copy drafts, batched policy (November 23)

Engine and kernels are unchanged, so every logit is the round-5 one. Every
HTTP response below is byte-identical to the round-5 build's, AR and DSpark,
at every C (8 prompts x 2 modes x 4 widths, 14 more single requests and 8
copy requests), and the functional suites record the same outputs (see
[Quality](QUALITY.md)).

### Where a decode step's time goes

`rocprofv3` kernel trace of plain decode at a 64-token context, weight prefetch
off (the profiler hangs on the prefetch side stream): 21.7 ms per step, of
which 17.3 ms are weight GEMVs at 85 to 95% of the DRAM ceiling, 3.3 ms the
latency-bound kernels and 1.0 ms the gaps between 505 launches (2.0 us each,
whatever the kernels).

| Kernels (per step) | Time | Rate |
| --- | --- | --- |
| MoE gate/up, 48 visits x 8 experts | 5.74 ms | 215 GB/s (NVFP4 visits) |
| Attention projections (q, gate, rel, k, v) | 4.66 ms | 216 GB/s |
| MoE down | 3.03 ms | 202 GB/s |
| LM head | 2.15 ms | 228 GB/s |
| Attention output projection | 1.77 ms | 204 GB/s |
| Attention-residual mixes (98) | 1.02 ms | 10.4 us each |
| Attention (48) | 0.93 ms | 19.3 us each |
| Loop controllers (gate and apply, 48 each) | 0.88 ms | 7.3 and 11.0 us |
| Router (48) | 0.47 ms | 9.7 us |

A four-row verification step (35.9 ms) spends 21.3 ms in the MoE kernels
against 8.8 ms for one row: each extra row costs about 4.2 ms of expert
weights (about half of its 8 experts per visit are new to the step) and
0.5 ms elsewhere. Both steps are bound by the bytes of the distinct experts
their rows route to.

### Sampled requests

A request that samples (the model's default: temperature 1, no truncation)
paid 1.3 ms per token on the host that a greedy request does not: the logits
row came back through a pageable buffer, and the sampler built a 3.2 MB list
of candidates per token and passed over it four times.

| Change | Sampled plain decode |
| --- | --- |
| Before | 21.35 ms per token (greedy 20.03) |
| The whole-row temperature draw without the candidate list: the same weights from the same operands, summed in token order, the same quotients subtracted from the same uniform (`SamplerState::Sample`, all models); the exps shared with a few threads that take chunks as they wake | 20.78 |
| The row's logits through the pinned buffer; the maximum taken on the floats and the widening folded into the threaded pass | 20.57 |

`sampling_test` draws 1,200 rows both ways (penalties, min-p, masked entries,
single-survivor rows, 200,058-entry rows): same token and RNG state every
time. In the microbenchmark a 200,058-entry draw takes 0.36 ms against
1.16 ms.

Several sampled rows (a verification step's, or several requests') ran each
on a new thread; a thread woken for one row takes about 1 ms where the
calling thread takes 0.4 ms. Up to three rows now run one after another on
the calling thread, more on persistent selection threads: four requests,
two of them sampled, select in 0.84 ms instead of 1.6 ms.

### Copy drafts

A sampled request whose last 24 generated tokens repeat an earlier stretch of
its context (5 where the last accepted copy left off; Gufo's
`speculative::PromptLookup` rules) takes its proposals from the tokens that
followed that stretch instead of from the drafter. The drafter's coupled draw
misses about a fifth of the tokens of a copied stretch; the copied token
misses almost none. Proposals only: the request's sampler still selects
every token.

| HTTP, one request, 500 tokens, default sampling, seeded | Before | After | Copied proposals kept |
| --- | --- | --- | --- |
| Rewrite a 60-line file with one command added | 75.7 tok/s | 90.1 | 267 of 268 |
| Repeat a paragraph, then translate a sentence | 54.3 | 63.5 | 99 of 102 |
| Rename a class throughout a file | 80.3 | 84.8 | 176 of 176 |
| Write a function (nothing to copy) | 72.9 | 74.8 | none proposed |

Greedy requests do not copy: the drafter already copies as well (rewrite
84.0 tok/s, 357 of 433 proposals kept), and where a match's continuation
differs the copied proposals displace better ones (a function written from
scratch: 81.4 to 77.8 tok/s with copies).

### HTTP, before and after

`gufo serve --sessions 8 --context 32768`, the round-5 workload (eight
prompts, 256 tokens, even requests greedy and odd ones seeded with the
default sampling), aggregate tokens/s:

| C | AR before | AR after | DSpark before | DSpark after |
| --- | --- | --- | --- | --- |
| 1 (greedy) | 48.3 | 48.0 | 71.7 | 72.7 |
| 2 | 59.7 | 62.3 | 71.1 | 73.9 |
| 4 | 88.0 | 91.1 | 88.5 | 91.1 |
| 8 | 113.0 | 114.1 | 111.7 | 109.9 to 112.9 (four runs) |

Six seeded requests with the default sampling, one at a time, 400 tokens:
AR 46.6 to 48.4 tok/s, DSpark 59.3 to 61.1 (1,050 of 1,624 proposals kept,
as before).

With four requests drafting lost about 2% to plain batched steps (five of
the eight rows are anchors, and the drafter's pass for four requests takes
5.8 ms); the back-off for batched cycles now compares against measured plain
steps of the same request count and tolerates no loss, so DSpark matches AR
there. At eight requests no row is left to draft in and DSpark trails AR by
the draft-context injection (about 1.5%).

## Round 7: wide copy verification (November 23)

A request decoding alone verifies up to 64 rows in one step when the rows
after the drafter's proposals are copied from its context (prompt lookup).
Every row keeps the arithmetic of a one-token decode, so the output is still
byte-identical to plain decoding (`inkling2_spec_test` at 2 to 64 rows, the
oracle on both paths, `inkling2_cache_test`, HTTP hashes).

Steps of 9 to 64 rows read each weight once for all rows:

- The dense Q8_0 projections and the LM head use int8 WMMA. Each 16-element
  integer dot product of the GEMV is one WMMA entry; the float scaling and the
  lanes' butterfly are replayed in the GEMV's order (the leaves are spread
  over a block's eight waves and combined from LDS).
- The MoE groups a step's (row, slot) pairs by expert. A block reads each
  expert row once for up to four pairs; the per-pair dot products are
  unchanged.
- Attention runs in groups of eight rows.

`inkling2_bench --spec`, context 64, ms per verification step:

| Rows | 8 | 9 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- | --- |
| Before (8-row steps one after another) | 55.6 | n/a | n/a | n/a | n/a |
| Grouped MoE + 8-row GEMV launches | 55.6 | 74.8 | 96.4 | 177.9 | 344.9 |
| Final | 55.6 | 62.5 | 80.9 | 133.6 | 245.4 |
| Per row, final | 6.95 | 6.94 | 5.06 | 4.18 | 3.83 |

At context 20,000: 16 rows 93.0 ms, 64 rows 292.7 ms.

The width starts at eight rows and doubles after two whole copies of that
width, up to 64; any rejected copy returns it to eight. Greedy requests append
a copy to the drafter's proposals when the context they extend continues an
earlier stretch; while a copy keeps being kept, the next cycle copies first
without drafting.

HTTP, one request, DSpark, outputs identical to before:

| Request | Before | After |
| --- | --- | --- |
| Rewrite a file, greedy, 1,500 tokens | 103.2 tok/s | 118.1 tok/s |
| Rewrite a file, sampled, 1,500 tokens | 100.1 tok/s | 117.9 tok/s |
| Rename in a file, greedy (801 tokens) | 98.5 tok/s | 116.4 tok/s |
| Rewrite, sampled, 400 tokens | 82.7 tok/s | 94.2 tok/s |
| Code from scratch, greedy, 400 tokens | 77.8 tok/s | 76.4 tok/s |

AR and batched requests are unchanged (C = 1, 4, 8 within noise, same hashes).

Rejected (all exact):

| Variant | Result |
| --- | --- |
| Expert pairs grouped 8 per block (LDS for 8 inputs) | 64 rows 324 ms vs 272 with 4 (one block per CU) |
| Wide dp4a GEMVs looping 8-token groups in one launch | spilled (8 rows of LPR 16 need > 256 VGPRs); with 4-token groups 64 rows 264 ms vs 245 with WMMA |
| MoE by int8 WMMA (pairs of an expert as A rows, NVFP4 decoded to int8) | 64 rows 241 ms vs 245, 16 rows 117 vs 81: a 16-row tile has about four live pairs |
| Dense WMMA from 4 rows (DSpark and batched steps) | 8 rows 61.4 vs 55.6 ms: half-empty tiles |

