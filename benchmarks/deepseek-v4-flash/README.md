# DeepSeek V4 Flash Q2-imatrix on Strix Halo

Status: 2026-09-03. This page is the current functional and performance
snapshot, not an optimization history.

## Model

| Field | Value |
| --- | --- |
| Repository | `antirez/deepseek-v4-gguf` |
| Snapshot | `1cd7b564460821938add0475a60b942c409295e0` |
| Artifact | `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf` |
| Size | 80.76 GiB |
| Engine source | `antirez/ds4` at `84cc882352757baf628a1776badf7cc54d584e28` |
| Backend | Model-private ROCm/HIP implementation for `gfx1151` |

The DeepSeek graph, quantized layouts, session state, dispatch, and numerical
kernels live under `src/models/deepseek_v4_flash`. They do not call Qwen
kernels or share mutable Qwen state.

## Run

Build through the repository flake and define the artifact path:

```sh
git add .
nix build

MODEL=/path/to/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
```

Run a terminal prompt:

```sh
./result/bin/gufo prompt \
  --model "$MODEL" \
  --raw \
  --prompt 'The capital of France is' \
  -n 16 \
  -t 0
```

Run the sparse long-context benchmark. Each prompt row adds a 2K suffix to the
prepared depth, and each generation row produces 128 autoregressive tokens.
Frontiers are extended incrementally and restored from in-memory snapshots.

```sh
./result/bin/gufo bench \
  --model "$MODEL" \
  --n-prompt 2048 \
  --n-gen 128 \
  --n-depth 2048,8192,16384,32768,65536 \
  --repetitions 1 \
  --verbose
```

The same artifact can be served through the existing OpenAI-compatible
completion and chat endpoints:

```sh
./result/bin/gufo serve \
  --host 127.0.0.1 \
  --port 8080 \
  llm \
  --model "$MODEL"
```

The retained first-four Antirez DS4 HTTP capability run and independent repeat
are documented in [eval/README.md](eval/README.md). They are Gufo regression
baselines, not official dataset scores.

## Current Results

The Gufo rows use the release package, one repetition, a 2K prompt suffix, and
128 generated tokens. The DS4 reference is the supplied same-machine result
for the same artifact. Prompt rows compare the final context after adding the
2K suffix; generation rows compare the prepared depth.

| Prepared depth | Gufo `pp2048` | DS4 `pp2048` | Delta | Gufo `tg128` | DS4 `tg128` | Delta | Snapshot bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K | 206.33 | 205.49 | +0.4% | 15.63 | 14.76 | +5.9% | 52,184,460 |
| 8K | 204.22 | 197.13 | +3.6% | 14.65 | 13.87 | +5.6% | 136,750,476 |
| 16K | 197.79 | 190.09 | +4.1% | 14.39 | 13.63 | +5.6% | 249,505,164 |
| 32K | 179.18 | 171.83 | +4.3% | 13.66 | 12.93 | +5.6% | 475,014,540 |
| 64K | not remeasured | not supplied | - | 12.42 | 11.91 | +4.3% | 926,033,292 |

The model loads 80.76 GiB of tensor spans in about 21 seconds. The 64K run
plans 82.07 GiB total, including model, KV state, and working buffers.
The 2K decode value was confirmed by two paired candidate samples; the 8K-32K
rows come from one sparse-depth sweep. The 64K
prompt row was not rerun because the retained prompt-kernel gain was already
stable through 32K.

### Weight placement

DS4 keeps the copied device-arena policy. A source-selected candidate instead
registered the full read-only GGUF mapping with HIP and addressed weights
through its device alias. Separate release packages used the same artifact,
kernel routes, 2K prepared depth, 2K prompt, and 128 generated tokens.

| Policy | Cache state | Load | `pp2048` | `tg128` | Max RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| HIP-mapped GGUF | cold | 23.19 s | 64.70 tok/s | 14.15 tok/s | 81.20 GiB |
| HIP-mapped GGUF | warm | 2.49 s | 153.63 tok/s | 8.93 tok/s | 81.23 GiB |
| Copied device arenas | interleaved repeat | 23.11 s | 205.64 tok/s | 15.47 tok/s | 0.72 GiB |

Even with the file cache warm, direct mapping regressed prompt throughput by
25.3% and generation throughput by 42.3%. It also added about 80.5 GiB of
process RSS and roughly 21.25 million minor faults because registration
first-touched the complete mapping. The mapped candidate passed the unchanged
quality envelope exactly: pinned trajectory 116/128 top-1, rank sum 142,
worst rank 3; batched-prefill RMSE 0.478146, cosine 0.99617, maximum error
2.39995, and sequential choice rank 1. The candidate was therefore rejected
for placement performance and memory behavior, not numerical quality.

A separate full-prompt comparison isolates the retained prompt kernels:

| Prompt | Baseline | Current | Delta |
| ---: | ---: | ---: | ---: |
| 4K | 194.31 | 202.15 | +4.0% |
| 8K | 203.54 | 209.01 | +2.7% |
| 16K | 203.75 | 210.02 | +3.1% |

The 4K values are means from an interleaved baseline/candidate replay. The 8K
and 16K values are matched release-package runs. The current paired attention
route keeps FP32 compressed KV authoritative and maintains a derived FP16 mirror
for ratio-4 attention layers. Indexed-attention time falls from 82.07 ms to
55.72 ms per layer (-32.1%); fused mirror production adds 0.22 ms total. The
wave32 kernel uses 80 VGPRs, 57,992 bytes LDS, zero scratch, and 32 waves per
workgroup. The mirror adds about 21, 42, 84, 168, and 336 MiB at 4K, 8K, 16K,
32K, and 64K respectively, while snapshot payloads remain unchanged.

### Resident server scheduling

The OpenAI-compatible server keeps independent DeepSeek sessions resident in
the common text scheduler. DeepSeek currently advertises physical width one,
so C=2 and C=4 requests make fair round-robin progress through an exact serial
fallback rather than a native batched decode kernel.

Release-package qualification used distinct raw prompts, 16 greedy output
tokens per request, a 512-token context, and isolated replays of every
concurrent request:

| Workload | Decode rate after TTFT | Whole-request aggregate | Result |
| --- | ---: | ---: | --- |
| C=1 | 16.47 tok/s | 11.50-11.53 tok/s | +1.3% decode rate against the prior 16.25 tok/s baseline |
| C=2 | Width-one serialized | 11.18-11.20 tok/s | Both outputs exactly match isolated execution |
| C=4 | Width-one serialized | 11.21-11.22 tok/s | All outputs exactly match isolated execution |

The C=1 decode rate is the reciprocal of the 60.73 ms median inter-token
latency across four steady samples after graph warmup. It is the number
comparable to `tg` throughput and is consistent with the 15-16 tok/s
longer-context results above. The previous direct-server baseline had a
61.54 ms steady inter-token latency, or 16.25 tok/s.

The 11.x tok/s values are a different metric: generated tokens divided by
whole HTTP wall time, including roughly 477-532 ms of prompt prefill per short
request. The C=2 and C=4 rows are two steady concurrent samples after graph
warmup. They are useful end-to-end workload measurements, but must not be
reported as DeepSeek decode or `tg` throughput.

Whole-request aggregate throughput does not yet scale with concurrency because
every physical model advance remains width one; the current benefit is
resident state, overlap, fair scheduling, cancellation, and prefix reuse
without reloading the model.

Clean server startup measurements reported about 89.4, 90.0, and 90.2 GiB of
consumed system-available memory at C=1, C=2, and C=4 respectively. Thus the
incremental resident-session cost was about 0.60 GiB at C=2 and 0.82 GiB at
C=4 relative to C=1, while the 80.76 GiB model tensor cache remained shared.
A short 20-token state snapshot contained 14.1 MiB after prefill and 15.2 MiB
after generation.

## Quality and Integration

- The pinned upstream DS4 CLI and packaged Gufo CLI produce the exact same
  four-token greedy continuation, ` Paris. It is`, for the same raw prompt and
  artifact.
- A pinned 128-token teacher-forced trajectory keeps at least 116/128 reference
  tokens at top-1, every reference token within top-3, and aggregate rank at
  most 142. This catches sustained numerical drift without treating
  free-running near-tie flips as state corruption.
- A 273-token batched-prefill versus sequential-state comparison matches the
  immutable pre-optimization envelope: RMSE at most 1.12, cosine at least
  0.979, maximum logit error at most 5.0, and the sequential winner remains
  within the batched top-3.
- Full logits are finite after real GGUF prefill.
- Snapshot restore reproduces the exact position, greedy token, and logit
  vector.
- Direct model sessions and the HTTP backend produce identical raw and chat
  token sequences.
- Cancellation returns the request session cleanly for exact subsequent reuse.
- Terminal prompt, benchmark, raw completion, and chat completion use the same
  model-owned engine and tokenizer.

## Successful

- Adapted the required DS4 graph and ROCm kernels into repository-owned C++20
  `runtime` and `kernels/rocm` packages.
- Removed standalone distributed, tensor-parallel, SSD weight streaming,
  multi-GPU placement, CPU reference, MTP, steering, and embedded hotlist
  implementations from the production closure.
- Converted the private fork to native ROCm/HIP naming and APIs.
- Kept DeepSeek code, kernels, state, and dispatch isolated from Qwen.
- Reused the existing Gufo CLI, benchmark, and OpenAI-compatible server.
- Matched the DS4 state payload sizes and stayed close to or ahead of the
  supplied throughput curve through 64K.
- Reduced the six-expert Q2-down kernel by 7.8% with a two-way compiler unroll;
  `tg128` improved 1.2% at 2K and 0.6-0.7% at 8K-32K without changing VGPR,
  LDS, or scratch allocation.
- Reused each routed-MoE activation tile across four Q2-down output fragments
  and aliased the epilogue over dead staging LDS. Full-prompt throughput improves
  1.1-2.7%, while the canonical 2K-suffix prompt rows improve about 4-5%
  through 32K.
- Added a derived FP16 compressed-KV mirror and 32-head wave32 indexed-attention
  route for large prompt batches. Full-prompt throughput improves 2.7-4.0% and
  canonical 2K-suffix prompt rows improve 2.4-3.3% through 32K, with neutral
  autoregressive throughput and unchanged serialized state.

## gfx1151 prefill kernels (September 3, 2026)

Measured on the artifact above with `-p 4096 -n 16 -r 2`, alternating
configurations within one script so APU throttling cannot favour an order.
The quality column is the pinned 128-token trajectory from
`deepseek_v4_flash_engine_test`; its floor is 116/128 top-1, rank sum 142,
worst rank 3.

| Stack | `pp4096` | `tg16` | Pinned trajectory | Prefill `rmse` | Gate |
| --- | ---: | ---: | --- | ---: | --- |
| `16d5e30` baseline | 188.5 | 16.5 | 116/128, 142, 3 | 0.478146 | pass |
| Retained default | 427.3 | 16.6 | 116/128, 142, 3 | 0.41 | pass |
| Retained default, MMQ disabled | 270.1 | 16.6 | 116/128, 142, 3 | — | pass |
| Plus the attention output-B hipBLASLt fallback | 410.5 | 16.6 | 114/128, 150, 5 | — | fail |

Four interleaved rounds of the two binaries: baseline 188.24 / 188.22 / 187.67 /
190.04, retained 399.22 / 426.69 / 428.83 / 445.80 (the first round is cold);
four warm rounds of the final binary read 417.33 / 427.61 / 437.34 / 426.80. The
same pairing at 8,192 tokens, which chunks: 198.83 / 199.60 against
431.84 / 434.53, **+117%**.

Two different noise regimes, and telling them apart matters for every A/B below:

- **Alternating two binaries is ±5%.** Each arm is a fresh process that re-reads
  an 80 GiB artifact, so the thermal and page-cache state differs per arm; the
  retained stack swung 399.22 to 445.80 across four rounds this way.
- **Alternating configurations inside one binary is under 1%.** Six warm runs of
  one binary under two `DS4_CUDA_MMQ_X_MAX` settings read 431.39 / 427.19 /
  430.81 and 430.73 / 426.37 / 428.94.

So sweep with an env switch in one process wherever a knob allows it, and treat
any cross-binary delta under 5% as unmeasured. `rocm-smi` during a 4,096-token
prefill reads 138 to 140 W at 2.78 to 2.83 GHz against the 2.90 GHz the
register-resident peak harness reaches, so the prefill does sit at the package
power cap, but the cap is not what produces the cross-binary spread.

The retained default is **+126% prompt** and flat decode with the pinned
trajectory and the greedy continuation unchanged from the baseline, and with the
273-token batched-versus-sequential prefill comparison not merely inside its
envelope but *tighter than the baseline's*: `rmse` 0.478146 to 0.38, `cosine`
0.99617 to 1.00, `max_error` 2.39995 to 2.04, sequential winner still rank 1.
This stack also reproduces the retained first-four capability baseline greedily
through `gufo serve`, extracting `B`, `C`, `70`, `C` — the same answers as
[eval/README.md](eval/README.md) — 4/4 with no execution errors.

### Width scoping is what made the accelerated routes retainable

Every accelerated prefill route is gated on `DS4_ROCM_WIDE_PREFILL_ROWS`, 128
rows, rather than on `n_tokens > 1`. Below that width a batch is a decode step,
a DSpark verification block, or a short resumed batch, and those are the routes
the pinned trajectory's teacher-forced 15- and 17-row batches measure, where the
envelope demands bit-identical output. Above it the governing envelope is the
273-token prefill comparison, which has real tolerances, so a reordered
reduction can be justified there on its own evidence.

That distinction is the whole difference between this table and the first
attempt at the same work: applying the routes at `n_tokens > 1` put the
trajectory at 103/128, and scoping them to prompt-chunk width alone recovers
116/128 for all of them except hipBLASLt routing.

### Retained routes

Added this round, all width-scoped and all gate-clean:

| Route | Evidence |
| --- | --- |
| Routed MMQ column-tile grid bounded by the real largest bucket | `mul_mat_q` 3,143 to 2,792 ms at kernel level; pure launch geometry |
| Wide Q2-down grid compacted to the populated `(hot expert, row group)` pairs | 1,363 to 1,190 ms at kernel level; pure launch geometry |
| Inverse rotated tail folded into the F16 attention group pack | `rope_tail_kernel` 254.73 to 28.08 ms, the pack unchanged at 159.5 ms: one read-modify-write pass over 512 MiB per layer removed |
| Resumed mixed-window chunks routed to the same rocWMMA producer as the first chunk | `pp8192` 406.4 / 410.8 to 425.1 / 429.8 tok/s; the ratio-128 layers of every chunk after the first were reaching the scalar F32 kernel |
| Attention score pass with both operands row-major | -36.8% indexed and -28.3% mixed window in the ablation harness, output identical to 1.7e-06 |
| Routed expert sum folded into the hyper-connection expansion | +2.1% (442.9 against 433.9 tok/s mean, best round 452.58). Bit-identical: both forms accumulate the same F16 slots in ascending expert order in F32, and the buffer the separate form stored was F32, so the round trip was lossless. Estimated at 0.2% from traffic alone and worth ten times that -- removing a whole kernel launch beats the byte accounting |
| Deterministic assignment scatter parallelized | ran one thread per block walking all 24,576 pairs, 539 us per call. Two passes over contiguous per-thread ranges with one block-wide prefix sum keeps the output byte-identical. A first attempt that scanned in chunks of the block width was slower -- 18 barriers per chunk, 1,728 per block |
| One published F16 mirror of the normalized attention rows, shared by the layer's projections | `f32_to_f16_vec4_kernel` 423 to 321 launches and 97.13 to 57.32 ms at 1,024 tokens, so about 160 ms at 4,096. Bit-identical: the conversion is row-local, so every consumer sees the bytes it would have produced. The graph clears the mirror at each layer boundary, before any buffer it names can be rewritten |

Bit-identical, so they hold at every width:

- A `256x128x32` rocWMMA F16 GEMM replaces Tensile's `MT64x32x8` choice for the
  `4096 x chunk x 8192` attention output-B projection: 2,753.69 ms to 558.03 ms
  per 4,096-token chunk. Accumulators stay in registers for the whole K loop.
- The Q8_0-to-F16 transposed weight cache is built by an LDS-tiled kernel
  instead of one output element per lane. The scalar form put a wave's 32 stores
  8,192 B apart, one cache line each and all on one memory channel: 2,306.89 ms
  to 44.10 ms of first-prefill latency, and it is what collapsed the run-to-run
  spread from `±32` to `±6` tok/s.
- The 32-head wave32 rocWMMA attention producer is single-pass. Folding the
  running maximum into the accumulator removes the second traversal of every KV
  row block, its staging and its barriers: 1,206.87 ms to 902.52 ms. The 1 GiB
  score cache the old second pass read is gone with it.
- The routed Q2-down kernel stages each expert row's raw Q2_K block once per
  256-value K slab instead of re-reading it on all sixteen `BK` steps. Each
  re-read was six scalar sub-word loads from a two-byte-aligned 84-byte block.
  1,879.68 ms to 1,597.32 ms, `+4.4%` to `+5.0%` end to end, and the kernel then
  prefetches the next K step's mid tile into registers for a further `+3.6%`.
- The F32-to-F16 stages and the attention group-head pack move four elements per
  thread. At one element per thread a wave issued 128 B of loads against 64 B of
  stores, so every store was a partial line: 420.63 ms to 271.74 ms and
  255.46 ms to 161.80 ms.
- Plain RMS norm holds its row in registers, dropping the second full-row read
  (263.40 ms to 233.92 ms), and the routed Q2-down epilogue pages one
  accumulator fragment at a time instead of two, halving that kernel's dynamic
  LDS.
- The scalar cold-expert Q2-down launch is skipped when every populated expert
  reached the WMMA hot list, which is the common case at prompt-chunk width.

Reordering, and therefore scoped to prompt-chunk width:

- The vendored llama.cpp MMQ tier (`kernels/rocm/mmq`, see its `VENDOR.md`) owns
  routed IQ2 gate/up and dense Q8 prefill. Worth 270.1 to 363.9 tok/s. Its
  column-tile cap is already at its optimum: sweeping `DS4_CUDA_MMQ_X_MAX`
  through 80 / 64 / 48 / 32 / 24 gives 420.18 / 416.04 / 407.17 / 368.03 /
  361.59 tok/s, so the 80 cap wins despite leaving about 40% of each tile as
  padding at 96 columns per expert.
- Mixed-window prefill attention uses the same rocWMMA producer as the indexed
  layers: 915.82 ms to 316.35 ms. It is the one route that lowers precision
  rather than reordering, converting Q and KV to F16 where the scalar kernel
  stayed in F32, and it leaves the trajectory at 116/128 while the prefill
  comparison moves `rmse` 0.478146 to 0.479653 and `max_error` 2.39995 to
  2.16629.
- The query head norm and rotated tail run as one pass over the row, about
  250 ms per chunk. Staging the row in LDS is not bit-identical even with the
  expression written identically and the tail still stored and reloaded, because
  this backend builds with `-ffast-math`.
- hipBLASLt algorithm pinning uses the profiled gfx1151 candidate for
  prompt-chunk shapes and the heuristic's first choice below that width.
- Projections that ship on hipBLAS move to hipBLASLt for a measured subset of
  sites, `DS4_ROCM_LT_ROUTE_DEFAULT_MASK` = 23. Each site was gated on its own:
  the dense Q8 F32 and F16-result projections, the F16 projection and the paired
  F16 projection each hold the trajectory at 116/128 rank sum 142, while the
  attention output-B fallback alone drops it to 114/128 rank sum 150 and is
  excluded. That fallback only runs when the `256x128` rocWMMA tile rejects the
  shape, which at prompt-chunk width means an `n` that is not a multiple of 128,
  so excluding it costs nothing at a 4,096-token prompt.

`GUFO_DEEPSEEK_ROCM_MMQ=0`, `GUFO_DEEPSEEK_ROCM_MIXED_WINDOW_WMMA=0`,
`GUFO_DEEPSEEK_ROCM_FUSED_QNORM_ROPE=0`, `GUFO_DEEPSEEK_ROCM_HIPBLASLT_CANDIDATE=<n>`
and `GUFO_DEEPSEEK_ROCM_HIPBLASLT_ROUTING=<mask>` reach each route individually.

### Comparison against the upstream ds4 optimization

[antirez/ds4#887](https://github.com/antirez/ds4/pull/887) is the same class of
work on the same GPU and artifact, reporting 187.26 to 292.86 tok/s at a
4,096-token prefill on ROCm 10.0 with rocBLAS `5.6.0.8d1ae90e`. This tree is on
ROCm 7.2.3, so library versions differ, but the retained default here is 418.8
tok/s, `+43%` on that figure.

Upstream's own quality evidence, scored with `score_official` on the official
100-case Flash manifest at context 4096, is worth recording because it is a
different instrument from the gate used here:

| Revision | Average NLL | First-token matches | Average greedy prefix |
| --- | ---: | ---: | ---: |
| ROCm 10.0 pre-optimization | 0.402107528 | 55/100 | 4.650 |
| ROCm 10.0 with the PR | 0.403662338 | 55/100 | 4.690 |
| CUDA 13.0, either revision | 0.404811251 | 55/100 | 5.150 |

So upstream's ROCm arithmetic moved too — average NLL by `+0.387%`, greedy
prefix 4.650 to 4.690 — and their API top-1 agreement moved 85.863% to 85.949%.
Their CUDA rows are byte-identical only because the PR does not touch CUDA. They
did not preserve numerics; they measured that the change was benign across 100
cases and shipped it.

The gate here is a stricter instrument on a narrower sample: a pinned 128-token
teacher-forced trajectory with a hard 116/128 floor that the baseline hits
exactly. It cannot distinguish a benign reordering from a real regression, and
it fails on any reordering at all — including, if it were applied here,
upstream's own. That is why the routes above are scoped by width rather than
loosened, and why the numbers in the first table were reachable without touching
the envelope.

### Nothing in this prefill is bandwidth bound

A `rocprofv3 --pmc FETCH_SIZE` pass over a 4,096-token chunk settles what the
five hot kernels are actually waiting on. `FETCH_SIZE` is bytes that reach
memory, so it prices each kernel against the 240 GB/s DRAM read ceiling the peak
harness measures:

| Kernel | Calls | Fetched | Time | Achieved |
| --- | ---: | ---: | ---: | ---: |
| `mul_mat_q` dense Q8 | 215 | 69.7 GB | 910 ms | 76.6 GB/s |
| `moe_down_q2K_hotlist_wmma_wide` | 43 | 51.0 GB | 1,192 ms | 42.8 GB/s |
| `mul_mat_q` routed IQ2 | 86 | 48.4 GB | 1,876 ms | 25.8 GB/s |
| `attention_mixed_heads32_wmma<true,true>` | 21 | 7.7 GB | 875 ms | 8.8 GB/s |
| `attention_mixed_heads32_wmma<false,false>` | 20 | 5.5 GB | 697 ms | 7.9 GB/s |

The routed gate/up moves 48 GB at 11% of the DRAM ceiling. That killed two
plausible theories at once. Its activation tile is re-read once per row tile, so
the logical traffic per call is 7.5 GB against 562 MB fetched -- the MALL absorbs
93% of it -- and doubling `DS4_ROCM_WMMA_MMQ_Y` to 128, which halves those
re-reads at unchanged wave occupancy, measured 399.7 / 398.7 against 400.8 /
413.5 tok/s. Not retained.

The same counter run also rules out the workgroup geometry. Both routed kernels
launched a rectangular grid sized by the largest expert bucket, and the router
skew at this width puts that near 3,500 rows against a mean bucket of 96, so
about 90 to 96% of their workgroups exited immediately after reserving their full
shared-memory tile. Compacting both -- the MMQ column-tile bound and the wide
Q2-down `(hot expert, row group)` work list, both retained because they are pure
launch geometry and cannot be worse -- is worth `mul_mat_q` 3,143 to 2,792 ms and
the Q2-down kernel 1,363 to 1,190 ms at kernel level, and nothing measurable
end-to-end. A grid-stride variant over the column tiles
(`DS4_MMQ_COL_SLOTS`, default off) was neutral at every slot count.

### The attention score pass reads one operand the expensive way

`tools/bench/dsv4_attn_mixed_bench.hip` removes one phase of the mixed producer
at a time, keeping the barriers, at 1,024 tokens:

| Phase removed | Indexed (ratio 4) | Mixed window (ratio 128) |
| --- | ---: | ---: |
| — (full) | 6.34 ms | 4.36 ms |
| KV staging | -17.7% | -28.9% |
| QK matrix ops | **-60.8%** | **-64.5%** |
| Online softmax | -4.8% | -8.3% |
| PV matrix ops | -7.6% | -11.9% |
| Barriers | -17.7% | -24.2% |

The score pass and the value pass issue the same number of matrix ops per row
block, yet the score pass costs about eight times as much. It was not the
dependency chain: four independent accumulators over the same K sum measured
-0.2% to +2.4%. It was not parallelism: splitting K across four wave groups
instead of running on two of the block's 32 waves recovered only 4.5 to 8.8%.
It was not LDS traffic: hoisting the loop-invariant Q fragments into registers,
which halves the pass's LDS reads, recovered 5.5 to 8.8% and at one wave group
cost 55% because 32 held fragments spill.

It was the operand layout. K was a `col_major` matrix_b, which rocwmma gathers.
Staging Q transposed instead turns the product into `scoresT = KV . Q^T` with
both operands row-major:

| Variant | Indexed | Mixed window |
| --- | ---: | ---: |
| Transposed Q | -31.0% | -21.4% |
| Transposed Q, K split over 2 wave groups | **-36.8%** | **-28.3%** |
| Transposed Q, K split over 4 wave groups | -38.1% | -28.0% |

Every variant's `heads` output agrees with the baseline to 1.7e-06 absolute.
Two wave groups is what ships: four needs 6 KiB of reduction scratch, which does
not fit beside the production 1,024-entry top-k row table.

### Where the remaining time goes

Both routed-MoE kernels are latency-bound, not throughput-bound. A
`rocprofv3 --pmc` pass on a 2,048-token prefill:

| Kernel | Waves | VALU/wave | LDS/wave | VGPR | Scratch |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mul_mat_q` | 74.9 M | 888 | 166 | 192 | 0 |
| `moe_down_q2K_hotlist_wmma_wide` | 15.4 M | 3,018 | 836 | 96 | 0 |

The Q2-down kernel's whole instruction stream is 46.4 G wave-VALU plus 12.9 G
wave-LDS, which the GPU can issue in about 330 ms scaled to a 4,096-token chunk,
against 1,597 ms measured: **9 to 18% issue utilisation**, no scratch, and VGPRs
that allow sixteen waves per SIMD. `mul_mat_q` sits at about 23% by the same
accounting. Both are waiting, and what they are short of is resident waves,
because their tile layouts need enough LDS to cap residency at two (MMQ) to six
(Q2 down) workgroups per CU.

Everything that trades LDS for work per barrier therefore loses, and every width
or depth knob is neutral:

| Change | Result |
| --- | --- |
| Emitting the F16 hyper-connection row from the norm that produces it, removing a separate conversion pass | 416.92 / 423.71 versus 418.76 / 418.88 tok/s, so within noise for 134 MiB of resident buffer; not retained |
| Routing the 32,768 x chunk x 1,024 query-B projection to the rocWMMA tile | the 2.9 GiB transposed weight cache is refused by the shared F16 budget guard, so the route never fires |
| `NFRAG` 8 versus 4 | 417.69 / 415.46 versus 406.53 / 406.54 tok/s |
| Mid-tile register prefetch | 421.0 tok/s, and does not stack with `NFRAG` 8 |
| `MTILES` 8 versus 4 | 377.14 versus 377.41 tok/s |
| One barrier per K step by double-buffering both tiles | 410.1 versus 417.9 tok/s: 4 KiB more LDS drops residency six to four workgroups per CU |
| Attention score pass across eight wave groups instead of two | 374.47 to 363.14 tok/s; the ablation harness later showed why -- the score pass is bound by its operand layout, not by how many waves run it |
| Attention score pass with four independent accumulators | -0.2% to +2.4% in the ablation harness: the K reduction is not dependency-latency bound |
| Hoisting the loop-invariant Q fragments into registers | -5.5% at four wave groups, +55% at one (32 fragments spill); the transposed-Q staging subsumes it |
| `DS4_ROCM_WMMA_MMQ_Y` 128 instead of 64, halving the routed activation re-reads | 399.7 / 398.7 versus 400.8 / 413.5 tok/s |
| Every macro tile, wave split and swizzle for the attention output projections | the shipped 256x128x32 W4x2 SWZ2 tile is the best of 17 geometries in `tools/bench/dsv4_attn_out_gemm_bench.hip` at 23.9 TFLOP/s; larger tiles, `BK` 64, and register double-buffering all lose (see the table there) |
| Taking the output GEMM's fragments off `col_major` the way the attention score pass did, by transposing at LDS write time | 11.49 to 26.58 ms for A and 12.95 ms for B. The attention win does **not** generalize: there Q is staged once per block and read 40 x 32 times, so a staging transpose is amortized away; here both tiles are re-staged every K step and read about twice, so the eight scalar LDS writes per 16-byte global read dominate |
| Routed IQ2 column width above the old 80 cap, which the tile-fill arithmetic predicted would be the largest remaining lever | a dead tie. The mean expert bucket is 96 rows, so 80-column tiles leave about 45% of the tile work empty and 128-column tiles about 32%, yet warm interleaved runs read 429.8 versus 428.7 tok/s mean. The specialization was generalized off `mmq_x == 80` so every width the selector can reach stays on the raw-IQ2 path, and the default stays 80 |

A 500 tok/s target needs 4,096 tokens in 8.19 s and 600 needs 6.83 s. This
prefill is about 105 TFLOP of arithmetic, so 600 is 15.4 TFLOP/s sustained
across everything, including roughly 1.7 s of memory-bound norm, convert and
pack work that does almost no arithmetic; the retained default sustains 10.9.
Even removing every one of those 1.7 s -- which is not achievable -- lands at
about 530.

The counter evidence above says why the gap does not close with kernel work.
Nothing is near the DRAM ceiling, so there is no bandwidth to reclaim. The two
routed-MoE kernels hold 3.4 s between them and have now resisted grid
compaction, tile height, column width, `NFRAG`, `MTILES`, barrier count and
residency; MMQ's IQ2 tile cannot shrink because `iu8` WMMA consumes
byte-expanded codes, and the Q2-down layout resisted every knob. The two
attention producers hold 1.1 s and the transposed-Q rewrite takes 30 to 37% of
that in isolation, which is the single largest remaining win found and is worth
about 4% of the chunk. The two output projections hold 1.1 s and are already on
the best of 17 measured geometries, at 2x hipBLASLt.

What makes 500 and 600 both out of reach is not one missing optimization but
that the routed gate/up, which holds 2.8 s, has now been shown inert to every
schedule-shaped lever available:

| Lever | Predicted | Measured |
| --- | --- | --- |
| Workgroup count (90-96% of the grid was empty) | large | kernel-level -351 ms, end-to-end nil |
| Activation re-reads (`MMQ_Y` 64 -> 128, halves them) | -850 ms | nil |
| Tile fill (column width 80 -> 96/112/128) | -1.0 s at 88% fill | nil |
| Fragment operand layout | -30% by analogy with attention | 2x worse |
| DRAM traffic | — | 25.8 GB/s of a 240 GB/s ceiling: no headroom to reclaim |
| Resident waves per CU, 8 -> 16, by splitting warps along the column axis (`DS4_ROCM_WMMA_MMQ_NCW`) | -0.9 to -1.2 s if wave-starved | **18% slower**: 354.9 against 419.6 tok/s mean |
| Deriving the IQ2 sign masks arithmetically instead of loading `ksigns64`, removing one of the two dependent indexed loads per eight weights from the inner loop | the loader is stalled, not computing, and VALU is 89% idle | bit-identical (rmse 0.41, max_error 1.95) and **neutral**: 435.1 against 439.6 tok/s mean. The sign lookup is not the cost either |
| Staging the Q2-down weight tile as `[k][n]` so its value fragments load row-major (`GUFO_DEEPSEEK_ROCM_WIDE_DOWN_B_ROWMAJOR`) | -25% by analogy with attention | **5% slower**: 401.4 against 423.3 tok/s mean. The attention win needs a staging-to-read ratio of about 1,280; this tile is read 16 times per dequantize and the strided stores cost more |

#### What the cost actually is: instruction count, at a shared issue ceiling

A `rocprofv3` pass over a 2,048-token chunk, one counter group at a time (this
hardware refuses more than a couple at once -- `FETCH_SIZE WRITE_SIZE` together
already returns error code 38), comparing the routed IQ2 dispatches against the
dense Q8 dispatches of the *same* `mul_mat_q` kernel:

| | IQ2_XXS routed | Q8_0 dense |
| --- | ---: | ---: |
| dispatches | 86 | 215 |
| time | 1,281.9 ms | 488.5 ms |
| VALU wave-instructions | **45.5 G** | **15.2 G** |
| VALU issue rate | 35.5 G/s (**15.3%** of capacity) | 31.1 G/s (**13.4%**) |
| waves resident per SQ | **15.6** | **15.7** |
| L2 hit | 83.5% | 75.4% |
| MemUnitBusy | 55.2% | 41.5% |

The two paths sit at the **same residency and the same issue rate**, and the time
ratio (2.62x) tracks the instruction ratio (3.00x). So the routed path is not
starved of waves relative to the dense one, and it is not stalled differently: it
simply issues three times the instructions, for less useful work. Per output
element it costs about 20 VALU wave-instructions against the dense path's one --
that is the IQ2 expansion, counted rather than assumed.

This explains every null result above at once. Grid compaction, tile height, tile
width, warp decomposition and operand layout all change *how* the instructions are
scheduled; none of them change how many there are, and the time is proportional to
how many there are. It also explains why replacing the `ksigns64` load with four
VALU operations was slightly negative rather than positive: it traded a cached
load for more of the resource that actually gates the kernel.

Two further consequences. Both kernels leave 85% of VALU capacity unused, so the
15% ceiling is itself a dependency-stall property of the inner loop rather than a
throughput limit -- shortening the dependency chain per weight would help as much
as removing instructions. And the requantization direction is now supported rather
than assumed: a routed format with fewer operations per weight is the one change
that moves the term the time is proportional to.

Its cost tracks neither bytes, nor tiles, nor workgroups, nor operand layout,
and the counters say it is not issuing either. 74.9 M waves at 888 VALU and 166
LDS instructions each is 66.5 G VALU and 12.4 G LDS operations, which this GPU
retires in about 297 ms and 111 ms respectively against 2,792 ms measured: **11%
VALU and 5% LDS utilisation**, a 31x stall factor. It is waiting on memory it
cannot hide, and what it is short of is resident waves.

Upstream's decomposition pins that at 8 waves per CU of the 32 the hardware
offers. Waves per CU is `(65,536 / LDS_per_workgroup) * nwarps`, and `nwarps` is
`mmq_y / 16` because a warp owns `mmq_y / nwarps` rows and that must be at least
the 16-row accumulator tile, so LDS and `nwarps` both scale with `mmq_y` and the
product is invariant -- which is why `mmq_y` 128, doubling both, changed nothing.

**That was the one structural lever left, so it was built and measured.**
`DS4_ROCM_WMMA_MMQ_NCW` adds a second decomposition axis: `NCW` warps share a
row group and take every `NCW`-th column tile, giving `nwarps = (mmq_y/16)*NCW`
at unchanged shared memory and therefore 16 resident waves per CU at `NCW = 2`.
It is correct -- the pinned trajectory holds at 116/128, 142, 3 and the prefill
envelope improves to `rmse` 0.38, `max_error` 1.76 -- and it is **18% slower**:
367.93 / 350.56 / 346.34 tok/s against 406.94 / 429.48 / 422.49, and a forced
96-column width does not recover it (348.26 / 354.55). Halving the columns per
warp also halves the matrix ops that each A-fragment load feeds, and that costs
more than the extra waves recover.

So wave starvation is falsified as well. The default stays `NCW = 1`, which
reproduces upstream exactly (verified: the same gate output, `rmse` 0.41 and
`max_error` 2.15, as before the refactor), and the knob is kept because it
records the attempt.

With bytes, tiles, workgroups, operand layout, issue rate and resident waves all
eliminated, what remains is the per-warp serial dependency inside the IQ2 tile:
each A-fragment load feeds a fixed number of matrix ops and nothing available
changes that ratio in the favourable direction. Moving it needs a routed-expert
format whose tile does not need byte-expanded codes for `iu8` WMMA -- a
quantization change, not a kernel change.

The two attention producers are the one place a rewrite still pays: 1.1 s at 8
to 15% of the WMMA ceiling, of which the transposed-Q staging takes 30 to 37% in
isolation. Going further there means a FlashAttention-2 tiling with several query
tokens per block, which the ratio-4 layers block because each token carries its
own top-k row set, so it would have to be built for the ratio-128 layers alone.

**Resolved.** Normalizing the hyper-connection row straight to F16 is retained
unconditionally and carries no switch, because it is byte-identical: the gate
reads 116/128, 142, 3 with `rmse` 0.41 and `max_error` 2.15 either way.

The earlier drift to `rmse` 0.53 was not the projection and not the rounding
primitive. A same-input comparison settled the projection question outright --
running the F32 entry and `hip_matmul_f16_f16_input_tensor` back to back on
byte-identical halves differs in **0 of 98,304 outputs**. What was left was the
norm: a *separate* F16 kernel with source identical to `rms_norm_plain_regs_kernel`
computed a slightly different `scale`, because under `-ffast-math` `rsqrtf` need
not lower to the same instruction sequence in two different functions. Giving the
one kernel two nullable outputs -- `float *` and `__half *`, one compilation
shared by both callers -- makes the halves exactly what converting its floats
would produce. The F32 store is skipped entirely when only halves are wanted, so
the conversion pass and the store both go.

It stays scoped to `n_tokens >= 128`: below that the F32 entry deliberately
replays decode's per-row reduction so a verified row's greedy choice matches
autoregressive decode, and the F16 route bypasses that. Removing the scoping fails
the trajectory outright at 110/128, 154, 4.

### Off-the-shelf flash attention does not fit this model

Checked rather than assumed, because head_dim 512 is far outside what the tuned
libraries ship:

- **AOTriton** (already linked for MiniMax H3) rejects it: the shipped
  `libaotriton_v2` carries the string `head_dim > 192  Input unsupported`.
- **Composable Kernel** at this ROCm version ships only the `ck/` v1 tree, whose
  matrix path is XDLOPS, and no `ck_tile` FMHA; nothing under `include/ck_tile`
  and no gfx11 FMHA kernels exist to call.
- Upstream FlashAttention-2 caps head_dim at 256 and FlashAttention-3 is
  Hopper-only.

A hand-written Triton kernel would lower through the same LLVM and WMMA path as
the current rocWMMA producer with less control over the LDS layout -- and layout
is exactly where this kernel's largest win came from, the 31% the transposed-Q
score pass returned. DS4 attention also needs attention sinks, a 128-token raw
window, ratio-4 compressed KV and per-token top-k 512 indexing in one kernel,
which no stock interface expresses.

### Writing a faster attention producer: what worked and what did not

Retained, and arithmetically free: **the online-softmax wave already evaluates
`exp(score - m_new)` for its own lane while forming the block sum**, which is
exactly what the separate probability pass recomputed. Writing it there deletes
that pass and the barrier in front of it. Worth 1.6% on the window kernel in
isolation and about 1.5% end-to-end, and it *improves* the envelope: `max_error`
2.15 to 1.95 at unchanged `rmse` 0.41, because the fused form keeps one product
where the separate pass let `-ffast-math` contract a different way.

Rejected, both measured in `tools/bench/dsv4_attn_mixed_bench.hip`:

| Attempt | Reasoning | Measured |
| --- | --- | --- |
| Q^T fragments held in registers, the freed 34 KiB of staging arena recycled as a multi-buffered KV pipeline | the ablation prices exposed KV staging at 29% and barriers at 24%, and this is the only change that attacks both | **+15% to +135%**. The held fragments spill: 16 per wave at two K groups, 8 at four, against a 96-VGPR budget at full occupancy. Four groups with three KV buffers is the least bad at +15% |
| Score pass split across four wave groups instead of two | -4.4% on the kernel in isolation | nothing end-to-end (433.1 against 433.2 tok/s) and the extra reassociation of the score sum moved `rmse` 0.41 to 0.48, no longer clearly better than the baseline's 0.478146. Not worth the margin |

The wave32 producer's row table was also cut to a 512-entry local cap, since the
launcher only dispatches it at `top_k == 512`; that is what freed the shared
memory the four-group split needed, and it stays as headroom.

### A FlashAttention-2 tiling cannot help at head_dim 512

This was the last candidate and it closes by counting, not by measurement.

An attention block stages a KV tile once and reuses it for whatever queries the
block holds. Q costs 1 KiB of shared memory per (token, head) at head_dim 512,
the 16-row KV tile costs 16 KiB, and the score and probability tiles about 3 KiB,
so the Q budget is roughly 45 KiB -- about 45 (token, head) pairs, and `H` has to
divide the 64 heads. For a `T` tokens by `H` heads block on the window-only
layers, the KV rows staged over a 4,096-token chunk are

    (n_tokens / T) * (n_head / H) * (window + T - 1 + n_comp)

The leading product is `n_tokens * n_head / (T * H)`, constant at a fixed Q
budget, so the only free term is `window + T - 1 + n_comp`, which is **minimized
at T = 1**. Tokens in a block each add a staged row; heads add none. At `T * H`
of 32:

| block | KV rows staged |
| --- | ---: |
| **1 token x 32 heads** (shipped) | **1,310,720** |
| 2 x 16 | 1,318,912 |
| 4 x 8 | 1,335,296 |
| 8 x 4 | 1,368,064 |
| 32 x 1 (textbook FA2) | 1,564,672 |

So the shipped geometry is already the optimum, and a textbook FA2 tile is 19%
worse. FlashAttention's usual win comes from head_dim 64 or 128, where many
tokens' Q fits beside the KV tile; at 512 that inverts and heads are the operand
worth batching. Going past 32 heads needs 64 KiB of Q alone, which leaves no room
for the KV tile, and splitting K to make room re-stages Q per row block -- ten
times the Q traffic.

One lesson from the two wins in this round: **the byte-traffic estimate
underprices a fusion that removes a kernel outright.** The routed-sum fold was
predicted at 22 ms from its 128 MiB round trip and delivered ten times that.

## Failed

- Automatic CMake discovery was not reliable for the header-only ROCm
  dependencies in a clean Nix sandbox. Explicit flake-provided include roots
  are retained.
- The upstream DS4 frontend build is not imported. Gufo owns CLI, HTTP,
  benchmarking, cancellation, and session pooling.
- Q8 projection row grouping (`1/2/4/8`) and high-compression row grouping
  (`8/16/32`) produced no repeatable end-to-end improvement.
- Q2-down LDS aliasing alone was noise (`192.01` versus `192.13 tok/s` at
  4K); it is retained only because it enables the faster four-fragment kernel.
- An 8K prefill capacity regressed the 8K prompt (`201.80` versus `210.21
  tok/s`) and was noise in a 16K replay (`204.14` versus `203.62 tok/s`), so
  the existing 4K chunk policy remains.
- Wave32 indexed attention against the FP32 compressed cache was slower on its
  own (`193.49` versus `195.73 tok/s` at 4K); the route is retained only with
  fused production of the derived FP16 mirror.
- Native MMQ, device expert queues, cached hipBLASLt projection routing, and
  dense-Q8 32/128-token tile variants either failed the quality envelope or
  regressed the 4K prompt and were removed. MMQ and hipBLASLt routing were
  revisited on September 2, 2026 and are now large wins rather than regressions,
  but they still fail the envelope; see "gfx1151 prefill kernels" above.
- Widening the batched attention output-A rocWMMA tile to any multiple of 128
  rows, which lets `rank=1024` through, cost 855.99 ms against rocBLAS's
  563.72 ms for the same call: each group's A panel stops being MALL-resident
  across the n sweep, which is the whole premise of that tile.
- Direct HIP registration of the full GGUF mapping was numerically valid but,
  even warm, regressed 2K prompt/decode throughput by 25.3%/42.3% and added
  about 80.5 GiB of process RSS, so copied device arenas remain authoritative.

## To Do

- Add model-owned thinking/reasoning mode and effort controls; the current chat
  template intentionally uses the no-thinking path.
- Retain and compare the first four-case `gufo eval` HTTP regression baseline;
  Pi/coding-agent evaluation remains deferred under #153.
- Profile and optimize the model-owned gfx1151 kernels under #155.
- Continue improving general-workload DSpark acceptance and verifier cost under
  #156 and #222. The retained fast route exceeds the 24--25 tok/s target on
  sustained high-acceptance decoding; see "DSpark speculative decoding" below.
- Add model-owned offline calibration/imatrix tooling only when a new
  quantization recipe requires it.
- Add native concurrent decode batching and restart-safe SSD state reuse
  through the serving milestones.

## DSpark speculative decoding (September 1, 2026)

Status: retained as an opt-in DS4-only backend under #156. Qwen continues to use
DFlash2; DSpark is not a replacement for the Qwen backend.

The optimized route seeds each five-token support-model block with the target's
already-known first token and verifies six rows. Four verifier-prefix snapshots
allow partial accepts to commit without replay. If five rows are accepted before
the sixth rejects, the runtime restores prefix four and evaluates only row five.
If a prefix snapshot cannot be committed, the runtime automatically restores the
frontier and replays the accepted tokens through one-token decode.

The retained gfx1151 implementation adds:

- lazy prompt-to-support KV seeding with contiguous cache-coverage tracking;
- verifier-prefix snapshots for accepted lengths one through four;
- grouped tile-4 execution for the six selected IQ2 gate/up experts;
- compact exact Q2 down-pair accumulation and narrow exact Q8 shapes for 3, 5,
  and 6 rows;
- a support-only 16384-to-24 FP16 kernel;
- managed support-weight residency;
- an acceptance scheduler that rejects clearly weak text after four probes,
  evaluates marginal text after eight, and backs off for 64 tokens below the
  measured 48% break-even point.

All production choices are enabled by default once
`--speculative dspark --dspark-model` is selected. There are no user-facing
policy knobs to tune.

The implementation follows the upstream DSpark quality contract: every
committed token is target-verified, but retaining batched verifier state is not
byte-identical to sequential decode. Sequential replay remains an internal
safety fallback and self-test, not a separate user policy.

### Quality and throughput by task type

Greedy generation, 128 tokens per prompt, chat-v2 framing, using the shared
ten-category corpus:

| Category | Exact vs AR | AR tok/s | DSpark tok/s | Speedup | Support accept | Attempts/skipped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| expository | no | 16.50 | 16.27 | 0.99x | 43.8% | 16/77 |
| code | no | 16.63 | 17.68 | 1.06x | 53.1% | 35/0 |
| reasoning | yes | 16.27 | 23.60 | 1.45x | 78.5% | 26/0 |
| summarization | no | 15.67 | 16.01 | 1.02x | 47.5% | 8/15 |
| Italian | no | 16.43 | 16.05 | 0.98x | 40.0% | 9/101 |
| Chinese | no | 16.42 | 15.57 | 0.95x | 23.3% | 6/115 |
| structured | yes | 15.95 | 21.30 | 1.33x | 69.0% | 29/0 |
| creative | no | 16.32 | 15.73 | 0.96x | 20.0% | 5/118 |
| repetitive | yes | 16.26 | **27.56** | **1.69x** | 94.8% | 23/0 |
| instruction | no | 16.44 | 15.65 | 0.95x | 23.3% | 6/115 |

Aggregate: AR 16.32 tok/s, DSpark 18.00 tok/s, 1.10x mean and 1.00x
median speedup. The scheduler attempted 163 blocks and skipped 541. Its observed
60.7% acceptance is selection-biased because it stops sampling weak requests.

A pre-retention characterization with backoff suppressed attempted 379 blocks
and measured 43.4% true support-token acceptance, 60.9% independent positional
agreement, and 21.4% full blocks. The earlier 39.9% report was not comparable:
it used raw prompts, included the target-known anchor in acceptance, and sampled
only a handful of blocks. The corresponding anchor-free number was 27.9%.

The raw repetitive corpus case is the stable throughput target:

| Workload | Exact vs AR | AR tok/s | DSpark tok/s | Speedup | Support accept |
| --- | ---: | ---: | ---: | ---: | ---: |
| `red, blue, blue` sequence | yes | 15.66 | **29.05** | **1.85x** | 100.0% |

This exceeds the requested 24--25 tok/s regime and is over 3x the original
8.78 tok/s DSpark route.

The internal sequential-replay check is byte-identical on all ten categories at
64 generated tokens. It is retained as a correctness test, not exposed as a
production policy.

### Prompt processing and verifier checks

Three interleaved 252-token prompt runs measured median prefill of 5,358.8 ms
autoregressive and 4,967.6 ms with DSpark attached, a 7.3% improvement within
run-to-run variance. Prompt processing does not regress.

The exact verifier and rollback checks passed at every production width:

| Rows | Exact | AR replay | Verify best |
| ---: | ---: | ---: | ---: |
| 2 | 2/2 | 119.19 ms | 96.71 ms |
| 5 | 5/5 | 296.59 ms | 159.74 ms |
| 6 | 6/6 | 358.04 ms | 173.44 ms |

Issue #222's 75 ms verifier target is not met. The six-row route is retained
because seed-plus-five changes the amount of target work avoided, not because
the verifier itself reached that issue's projection target.

### Profile and retained/rejected decisions

Matched 64-token `rocprofv3` runs show total GPU kernel time falling from
7,471.87 ms autoregressive to 5,478.87 ms with DSpark, a 26.7% reduction. One
malformed ROCm dispatch with `end < start` was ignored; `tools/prof/prof.py`
now validates this and reports the count.

The one-time target Q8 transpose accounts for 2,454.51 ms and is not a
steady-state DSpark bottleneck. The principal repeated DSpark kernels were:

| Kernel family | Aggregate GPU time |
| --- | ---: |
| selected-expert IQ2 gate/up, tile 4 | 493.67 ms |
| compact Q2 down-pair accumulation | 381.71 ms |
| six-row prequantized Q8 projections | 375.18 ms |
| grouped verifier activation projection | 254.86 ms |

Routed MoE remains the repeated verifier bottleneck for #222. Prefix snapshots
were retained because they remove partial-accept replay and raise the production
corpus from 15.37 to 18.00 tok/s once paired with acceptance scheduling.

### Reproduction

```sh
TARGET=/var/llms/huggingface/hub/models--antirez--deepseek-v4-gguf/snapshots/1cd7b564460821938add0475a60b942c409295e0/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
DSPARK=/var/llms/huggingface/hub/models--antirez--deepseek-v4-gguf/snapshots/e7f04037032990db0346398d249baf9fb9df1ccc/DeepSeek-V4-Flash-DSpark-support-0731.gguf

tools/quant/speculative-corpus.py \
  --binary ./result/bin/gufo \
  --model "$TARGET" \
  --backend dspark \
  --draft-model "$DSPARK" \
  --max-tokens 128 \
  --allow-mismatch \
  --allow-sparse

tools/quant/speculative-corpus.py \
  --binary ./result/bin/gufo \
  --model "$TARGET" \
  --backend dspark \
  --draft-model "$DSPARK" \
  --case repetition_sequence \
  --prompt-mode raw \
  --max-tokens 128
```

## Historical DSpark baseline (superseded)

Status: in progress under #156. The path is opt-in and is not wired into
generation, so nothing here affects the numbers above. DSpark is DeepSeek's own
drafter; it is unrelated to the Qwen DFlash and DFlash-2 paths.

| Field | Value |
| --- | --- |
| Support artifact | `DeepSeek-V4-Flash-DSpark-support-0731.gguf` (5.58 GiB resident) |
| Repository | `antirez/deepseek-v4-gguf` |
| Architecture | `deepseek4-dspark`, 3 stages, `block_size` 5, `markov_rank` 256 |
| Target features | layers 40, 41, 42, fused by `main_proj` (12288 -> 4096) |

### Measured

Greedy, 4K context, `--speculative dspark --dspark-model`.

| Stage | Cost |
| --- | ---: |
| Draft block (3 stages, 5 + 1 encoder rows) | 21 ms |
| Verification block, 5 rows | 152 ms |
| Autoregressive token | 60-65 ms |

Verification costs `59 ms + 18 ms/row` and is exact against autoregressive decode
at 2, 5, and 6 rows. Rollback restores the compressor frontier and reproduces the
pre-block continuation.

### Per task type

Ten prompts from the shared speculative corpus
(`benchmarks/qwen3.8-27b/speculative-corpus.json`), greedy, 32 tokens each,
through `tools/quant/speculative-corpus.py --backend dspark`:

| Prompt | Category | Exact | AR tok/s | DSpark tok/s | Speedup | Acceptance |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| expository_pangram | expository | yes | 13.42 | 11.30 | 0.84x | 24.0% |
| cpp_ring_buffer | code | yes | 15.45 | 11.31 | 0.73x | 24.0% |
| reasoning_train | reasoning | yes | 16.10 | 11.42 | 0.71x | 32.0% |
| summary_gpu | summarization | yes | 14.51 | 11.46 | 0.79x | 24.0% |
| italian_explanation | multilingual | yes | 15.55 | 11.55 | 0.74x | 20.0% |
| chinese_explanation | multilingual | yes | 16.71 | 11.31 | 0.68x | 28.0% |
| json_schedule | structured | yes | 16.71 | 11.32 | 0.68x | 24.0% |
| creative_station | creative | yes | 14.68 | 11.43 | 0.78x | 28.0% |
| repetition_sequence | repetitive | yes | 16.14 | 8.82 | 0.55x | 46.0% |
| instruction_debug | instruction | yes | 16.28 | 11.35 | 0.70x | 20.0% |

Aggregate: exact 10/10, AR 15.49 tok/s, DSpark 11.06 tok/s, speedup 0.71x,
median 0.72x, acceptance 28.7%.

**Quality is preserved exactly**: every category reproduces autoregressive
decode's completion byte for byte. **Throughput is not.** The path is opt-in
behind `--speculative dspark`, off by default, and on this hardware with this
drafter it should stay off.

### Why: exactness and speed are mutually exclusive here

The only saving speculative decoding can offer is *not running* a decode step for
an accepted token. Taking that saving means keeping the key/value rows the batched
verification pass wrote. Those rows are not the bytes one-token decode would have
written, and the difference compounds: a 48-token generation with fully accepted
blocks diverged from autoregressive decode at the tail even though every accepted
token was individually correct.

So there are two regimes, both measured:

| Commit policy | Byte-exact vs decode | Aggregate |
| --- | --- | ---: |
| Replay accepted prefix through decode (historical default) | yes | 0.71x |
| Keep verifier cache state | no | 0.54x |

Direct commit does not currently help because it only applies when *every* drafted
row is accepted, and at a mean accepted length of 1.35 out of 5 that is rare. The
replay path therefore dominates either way, at one full decode step per accepted
token.

**In the exact regime the theoretical maximum is 1.0x**, reached by never drafting:
every emitted token still costs a decode step, plus the draft and verification on
top. The break-even controller exists to approach that bound rather than to win;
the residual 0.71x is the cost of its periodic probes on short requests.

### What would actually make this faster

A cycle costs `21 ms draft + 152 ms verify`. Measured per-position draft accuracy
is 0.62, so prefix matching yields a mean accepted length of `sum p^k` = 1.47 over
five positions, against a break-even of about 1.9 accepted.

| Change | Expected |
| --- | --- |
| Per-row prefix snapshots (`DS4_SPEC_PREFIX_SLOTS` upstream) so partial accepts restore rather than replay | ~1.3x, and only in the non-byte-exact regime |
| Narrower verification cost (#222) | another 20-30 ms per block |
| Tree or multi-candidate verification | the only route to ~2x |

Lengthening the block does not help: at `p = 0.62`, five positions give 1.47 and
eight give 1.56. The binding constraint is the prefix-matching topology combined
with this checkpoint's per-position accuracy, not kernel cost or dispatch.

### The bound, as an inequality

For a block of `r` verified rows with mean accepted length `a`:

```
speedup <= decode_ms * (1 + a) / (draft_ms + fixed_ms + marginal_ms * r)
```

Floors for the cost terms on this checkpoint, from weight traffic at the 242 GB/s
DRAM ceiling rather than from the current kernels:

Routed-expert traffic is the term that decides this, and it depends on how much
the rows of a block share experts. Measured with
`GUFO_DEEPSEEK_ROCM_MOE_EXPERT_SPREAD=1` over 45 five-row blocks: a block issues
30 `(row, expert)` pairs but touches a **median of 19 distinct experts** (mean
17.7, range 14-26). About 40% of expert traffic is therefore the same weights
reloaded for a different row, and the small-batch route currently pays all of it
because it is indexed per `(row, expert)` pair.

| Term | Floor | Why |
| --- | ---: | --- |
| `fixed_ms` | ~33 ms | 6 GiB dense Q8 weights plus the first row's 6 experts, once per block |
| `marginal_ms` | ~6 ms | about 3 newly touched experts per additional row, plus its LM head row |
| `draft_ms` | ~20 ms | measured; the three DSpark stages are small |
| `decode_ms` | ~65 ms | measured autoregressive step |

At `r = 5` and the measured `a = 1.44` that gives
`65 * 2.44 / (20 + 33 + 30) = 1.91x` as the ceiling **with perfect kernels and
expert deduplication**. So "almost 2x" is reachable with this drafter; the gap is
kernel efficiency, not drafter accuracy.

An earlier revision of this section put the ceiling at 1.67x by charging every
`(row, expert)` pair a full expert load. That was wrong: it ignored the overlap
measured above.

### The ladder to get there

Each step is measured or derived from the terms above; verification is 152 ms today
against a 63 ms floor.

| Step | Verify | Aggregate |
| --- | ---: | ---: |
| Today, replaying accepted prefixes through decode | 152 ms | 0.71x |
| Per-row prefix snapshots, so no accept ever replays | 152 ms | ~0.92x |
| Expert deduplication for narrow batches | ~132 ms | ~1.03x |
| Dense projections at the bandwidth floor (#222) | ~90 ms | ~1.4x |
| All terms at their floors | ~63 ms | ~1.9x |

Two constraints on that ladder. First, every step past the first requires the
non-byte-exact commit regime, because the saving *is* not re-running decode for an
accepted token. Second, prefix snapshots are the prerequisite: without them a
partial accept replays about 86 ms per cycle, which is more than the entire
verification budget at the floor.

### Ablations

Conventions a support checkpoint does not record, each settled by measured
acceptance rather than assumption (12 cycles, first prompt above):

| Variant | First-token hits | Acceptance |
| --- | ---: | ---: |
| **Block at L, Markov on, forward slots, non-causal, direct KV** | **5/12** | **11.7%** |
| Hyper-connection-form KV injection | 4/12 | 10.0% |
| Causal block attention | 5/12 | 8.3% |
| Markov off (per-position argmax) | 3/12 | 5.0% |
| Reversed slot order | 3/12 | 5.0% |
| Block at L-1 | 2/12 | 5.0% |

Every structural choice in the retained configuration wins its comparison, which
is why the remaining gap is being treated as numerical. The losing A/B paths
were removed rather than exposed as user configuration.

Acceptance is one scalar at the end of a ten-stage chain and cannot localize the
fault; the next step is a numerical oracle that compares the fused feature, each
stage's hidden state, and the base logits against the upstream DSpark path for a
fixed input.
