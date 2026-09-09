# DeepSeek V4 Flash Q2-imatrix on Strix Halo

Status: 2026-09-09. Current concurrent DSpark qualification appears under
"Concurrent DSpark qualification by text type"; dated measurements below retain
their original workload and policy.

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

The model loads 80.76 GiB of tensor spans in about 21 seconds. The startup
copier issues sixteen aligned direct reads in parallel, sizes each chunk to the
current tensor span, copies through pinned buffers, and releases the roughly
1 GiB staging pool as soon as the model cache is ready. The same release path
took 141.79 seconds with one synchronous direct read in flight; retained cold
runs take 21.05-21.48 seconds. Final runs after hours of builds and profiling
measured 22.71 and 23.97 seconds. A queue-depth-16 storage check still sustains
6.21-6.44 GiB/s, so this spread and an earlier isolated 29.75-second run are
system variance rather than a return to serialized I/O.

The 64K run plans 82.07 GiB total, including model, KV state, and working buffers.
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

### Prompt throughput against prompt length

One warm process, a 4,096-token point first so the one-time weight mirrors and
plan selection land outside the measured rows, arms alternated within each pair:

| prompt | before | now | delta |
| ---: | ---: | ---: | ---: |
| 128 | 143.04 | 180.38 | +26.1% |
| 256 | 212.58 | 260.91 | +22.7% |
| 512 | 293.84 | 354.77 | +20.7% |
| 768 | 343.15 | 391.80 | +14.2% |
| 1,024 | 374.01 | 421.63 | +12.7% |
| 1,536 | 405.24 | 467.96 | +15.5% |
| 2,048 | 494.35 | 493.66 | -0.1% |
| 4,096 | 529.49 | 529.54 | +0.0% |

The follow-up short-width routes were measured against commit `cb663c8` with
the same release binary harness, five repetitions, and a 16-token warm point:

| prompt | `cb663c8` | current | delta |
| ---: | ---: | ---: | ---: |
| 32 | 53.63 | 61.43 | **+14.5%** |
| 64 | 72.64 | 103.41 | **+42.4%** |
| 96 | 86.06 | 147.86 | **+71.8%** |
| 128 | 169.01 | 183.08 | **+8.3%** |
| 256 | 261.61 | 262.26 | +0.2% |

Routed MMQ now starts at 32 rows. Dense MMQ and hipBLASLt start at 64, while
the accuracy-sensitive mixed-attention, fused norm/rope, output-B padding, and
inverse-rope routes stay at 96. At 32 rows, a two-pair Q2 down tile matches the
sparse expert buckets and a compact live-expert list avoids launching against
all 256 experts. The 32/64/96 full-logit comparisons remain top-1/rank-1
against sequential evaluation.

A chat turn prefills only what the prefix cache does not already hold, so these
narrow widths and not the 4,096-token chunk are what a conversation pays. See
"The conversational prompt widths, which were at half speed" below for the two
mechanisms and their measurements.

The follow-up routed-MMQ tile sweep also found that the vendored 80-column
selector was too wide from 1,024 through 2,048 prompt tokens. Holding that
interval at 48 columns improved pp1024 from 420.05 to 436.19 tok/s (+3.8%),
pp1536 from 465.67 to 475.08 (+2.0%), and pp2048 from 492.84 to 497.86 (+1.0%).
The measured 4,096-token route remains on the default selector.

The remaining narrow-prompt profile put the scalar Q2 down projection at
145.4 ms for pp512. Moving experts with at least four rows to the existing
WMMA route reduced it to 85.3 ms; total Q2 down time fell by 47.6 ms. A
2/3/4/6/8/10-row threshold sweep measured
357.88/359.70/361.05/358.64/355.71/352.63 tok/s, respectively. The retained
four-row threshold applies only from 64 through 2,048 prompt rows:

| prompt | eight-row threshold | final release | delta |
| ---: | ---: | ---: | ---: |
| 128 | unchanged | 189.62 | control |
| 256 | 262.79 | 266.63 | +1.5% |
| 512 | 355.71 | 361.54 | +1.6% |
| 1,024 | 435.69 | 438.70 | +0.7% |
| 1,536 | 474.29 | 477.81 | +0.7% |
| 2,048 | 499.48 | 500.27 | +0.2% |

Applying the threshold below 64 rows changed the pinned trajectory to 81/128
top-1 tokens, so that broader policy was rejected. The scoped policy restores
116/128 top-1 tokens, aggregate rank 142, and worst rank 3. Custom four- through
seven-row WMMA tiles were neutral at pp512 and were also removed.

The September 6 closeout separates singleton cold experts from multi-row
buckets, keeps the two-pair scalar tile through 128 prompt rows, and uses a
32-row WMMA tile for hot experts at those same widths. At pp128, scalar Q2-down
time fell from 250.93 to 240.46 ms and the hot WMMA kernel from 361.27 to
335.27 ms; total profiled GPU span fell from 2,050.16 to 2,025.54 ms. A final
five-repetition release sweep after a 4K warm point measured:

| prompt | final tok/s |
| ---: | ---: |
| 128 | 191.17 |
| 256 | 267.54 |
| 512 | 360.74 |
| 1,024 | 435.92 |
| 1,536 | 477.56 |
| 2,048 | 499.33 |
| 4,096 | 525.76 |

The 512-2,048 rows remain inside the established cross-run spread. The narrow
routes preserve each dot-product and routed-slot reduction order.

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
the common text scheduler. Without DSpark it advertises native physical widths
2 through 8 in addition to the unchanged width-one decode path. Dense HC/Q/KV
projections, attention output, FFN/MoE, and the LM head run layer-synchronously
over the concurrent rows. Each row still updates its own raw, compressed, and
indexer attention state at its own absolute position.

The narrow attention-output kernel quantizes every row exactly as C=1 does,
then loads each Q8 weight block once and carries one accumulator per concurrent
session. This preserves the per-row block and warp reduction order while
reusing the weight stream across W2-W8. With DSpark, C1 keeps speculative
decoding. A C2-C8 wave switches to the exact target W2-W8 plan before prefill,
so it does not spend time capturing support features or drafting tokens that
the concurrent target path cannot consume. `GUFO_DEEPSEEK_SESSION_BATCH=0`
provides an operational fallback.

Exact physical plans remove the old C5/C6 padding to W8. The W6 attention
output-A specialization improves the measured C6 decode rate without changing
C1. At C7/C8, the existing sorted-expert gate/up route now groups all rows that
selected an IQ2 expert and loads that expert's weights once for the group. Q2
down keeps the qualified decode route and reduction order. This supersedes and
removes the narrower first-owner kernel. The compact C2-C8 gate routes also no
longer copy 256 device expert counters to the host on every layer when no later
route consumes the host copy.

Release-package qualification on September 5, 2026 used an 80-token prompt,
32 greedy output tokens, a 512-token context, one warmup, and three measured
rounds:

| Concurrent requests | Plan | Request p50 | TTFT p50 | ITL p50 | Aggregate tok/s | vs C=1 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | serial fallback | 3,006.9 ms | 1,153.2 ms | 59.8 ms | 37.25 | control |
| 2 | W2 | 4,956.9 ms | 1,761.0 ms | 103.1 ms | 45.05 | **+20.9%** |
| 4 | W4 | 8,554.3 ms | 2,984.2 ms | 179.7 ms | 51.73 | **+38.9%** |
| 8 | W8 | 15,441.9 ms | 5,481.2 ms | 321.3 ms | 56.91 | **+52.8%** |

Per-request inter-token latency grows because one device pass advances every
active row, but total delivered throughput now scales with concurrency. A
separate ten-round C=1 control measured 37.73 aggregate tok/s, 17.26 decode
tok/s, and 59.82 ms median ITL. The earlier serial baseline was 38.06 aggregate
tok/s, a 0.9% difference, and C=1 never enters the session-batch function.

The final post-change serving run used the canonical 411-token prompt and the
same 32-token output, context, warmup, and repetition count:

| C | Plan | Prefill tok/s | Per-request decode tok/s | Combined active decode tok/s | Output-only end-to-end tok/s | Total processed tok/s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | serial | 291.90 | 17.04 | 17.04 | 9.73 | 134.77 |
| 2 | W2 | 292.00 | 12.14 | 24.28 | 11.59 | 160.50 |
| 4 | W4 | 292.29 | 8.20 | 32.81 | 13.13 | 181.70 |
| 6 | W6 | 292.87 | 6.18 | 37.05 | 13.67 | 189.22 |
| 8 | W8 | 292.83 | 5.76 | 46.12 | 14.63 | 202.48 |

`C` counts concurrent requests and `W` is the physical decode width reached by
the scheduler. Exact W2-W8 plans mean C6 runs a six-row kernel rather than a
padded W8 kernel. Per-request decode excludes prefill. Combined active decode
is that rate multiplied by `C`, so W2 gives each of two users 12.14 tok/s while
the GPU emits 24.28 tok/s across both. `W2` names that two-row physical plan;
it is not itself a throughput measurement. Output-only end-to-end includes
prefill, TTFT, queueing, and scheduling gaps. Total processed also counts
prompt tokens; it must not be reported as decode speed.

The final C8 sorted gate/up route improved per-request decode from 5.46 to
5.76 tok/s (+5.6%) and combined active decode from 43.71 to 46.12 tok/s while
C1 remained at 17.04. A matching Q2 down dedup kernel regressed C8 to 5.23 and
was rejected. Expanding the earlier IQ2 kernel from 128-row to 32-row blocks
was neutral while launching four times as many blocks, so both experiments
were removed.

The September 6 decode-heavy closeout uses a 45-token prompt, 64 generated
tokens, context 512, one warmup, and one measured round. `Combined decode` is
the sum across active users, while `per request` is what each user receives:

| C | Physical plan | Per-request decode tok/s | Combined decode tok/s |
| ---: | --- | ---: | ---: |
| 1 | serial | 16.93 | 16.93 |
| 2 | W2 | 13.17 | 26.34 |
| 4 | W4 | 9.64 | 38.56 |
| 6 | W6 | 7.70 | 46.20 |
| 8 | W8 | 7.40 | 59.20 |

Packed Q8 weights now have exact W2-W8 specializations, ordered F16
projections reuse each weight across the live rows, C8 F16 projections execute
as two W4 groups, and routed IQ2 gate/up uses exact one- through four-pair
helpers. C1 still takes the serial route.

The attention compressor now batches its paired F16 KV and score projections
across the same physical width. One workgroup covers 32 output rows and keeps
one accumulator pair per live request, so each of the two weight streams is
loaded once for W2-W8. Every row retains the serial lane reduction order and
continues into its own compressor state and KV cache. The C8 specialization
uses one W8 launch instead of two W4 launches.

Matched release qualification with the batched compressor disabled and enabled
produced identical completion-hash multisets:

| C | Compressor off | Compressor on | Change |
| ---: | ---: | ---: | ---: |
| 1 | 17.09187 tok/s | 17.09251 tok/s | control |
| 2 | 13.69222 tok/s | 13.90365 tok/s | **+1.54%** |
| 4 | 10.32088 tok/s | 10.57844 tok/s | **+2.50%** |
| 6 | 8.12098 tok/s | 8.36513 tok/s | **+3.01%** |
| 8 | 7.86722 tok/s | 8.23946 tok/s | **+4.73%** |

Replacing the two W4 compressor launches with one W8 launch raised C8 again to
8.29156 tok/s (+0.63%). A matched C2 profile reduced total GPU kernel time from
4,051.23 to 3,861.43 ms (-4.7%); the compressor projections themselves fell
from 253.62 to 214.48 ms (-15.4%). The model-backed session gate remained
28/29 top-1 with 0.32 worst RMSE, 1.00 worst cosine, 1.81 worst maximum error,
and serial winner rank 2. Unified-memory startup remained in the normal
21.0-21.2 second band. Set
`GUFO_DEEPSEEK_ROCM_SESSION_COMPRESSOR_BATCH=0` to restore the previous route.

The concurrent routed gate/up kernel previously staged four activation rows in
18,688 bytes of LDS. At C2-C8, direct reads through MALL are faster and allow
more resident workgroups: in a matched C8 profile, gate/up fell from 5,235.69
to 4,198.79 ms (-19.8%) and total GPU time from 28,724.37 to 27,363.94 ms
(-4.7%). Prompt batches retain the staged route because their greater pair
reuse repays the copy. The final C8 profile attributes 15.3% of kernel time to
routed gate/up, 9.9% to the W8 Q8 projection, 8.1% each to dense prompt MMQ
and Q2 down, 8.0% to prompt-side shared-X Q8 work, 6.8% to hipBLAS, and 6.5%
to paired F16 projection.

A decode-heavy C2 profile attributes 22.9% of kernel time to the dense Q8
projection, 14.9% to routed gate/up, and 11.7% to Q2 down. Interleaved sweeps of
Q8 rows per block, Q2-down rows per block, and gate row span did not produce a
repeatable improvement and were rejected rather than adding C2 regressions.

Earlier concurrency experiments were rejected before the independent-request
verifier was introduced. Grouping the
DSpark support body across C2 changed the strict verified stream at token 12.
A ragged verifier recovered 74.8% acceptance but reduced per-request decode
from 13.31 to 12.57 tok/s and combined throughput from 30.53 to 25.79 tok/s,
with category completion hashes changing. The exact target route was retained
at that stage. On prompt prefill, forcing dense MMQ column tiles
from 16 through 80 did not beat the existing selector: at pp128 the default
measured 196.46 tok/s versus 186.89, 188.69, 194.02, 196.58, and 194.75. A
16-row Q2-down hot tile also regressed pp64 from 117.86 to 114.18 tok/s and
pp128 from 196.74 to 185.24 tok/s. Both ablations were removed.

Quality first isolates one W6 step, resets every session, then repeats the
existing three W4 steps, changed W2 membership, W8 step, and return to C1:

| Measure | Bound | Result |
| --- | ---: | ---: |
| top-1 agreement | descriptive | 28/29 |
| worst RMSE | <= 0.85 | **0.47** |
| worst cosine | >= 0.99 | **1.00** |
| worst maximum logit error | <= 4.5 | **3.03** |
| serial winner rank in batch | <= 3 | **3** |

The one free-running difference is a near-tie; the serial winner remains in
the batch top three. Cancellation, duplicate-session and invalid-token
preflight, and transition back to serial decode are covered by the same test.

### Adaptive DSpark serving

DSpark now works through the OpenAI-compatible server with
`--speculative dspark --dspark-model <path>`. The current request policy is:

- C1 uses DSpark speculative decoding and honors the requested draft limit and
  policy, including the default rolling policy.
- C2-C8 draft independently and verify the proposed rows together. Each request
  owns its accepted prefix, compressor state, support cache, and counters.
- Rolling draft tails start at three, grow after full acceptance, and shrink
  below 50% acceptance. The measured ceilings are three for C2/C4/C8 and two
  for C6.
- Target-only steps remain available when a request does not propose a tail.

The earlier implementation serialized one DSpark cycle per request and was
slower than target batching at every measured concurrent width. Its target-only
policy is superseded by grouped support execution and ragged verification.

The following target-only baseline used a 121-token prompt, 64 greedy output
tokens, one warmup, and two measured rounds:

| C | Physical plan | Per-request decode tok/s | Combined active decode tok/s | Prefill tok/s | Total processed tok/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2 | W2 | 13.86 | 27.73 | 178.48 | 61.92 |
| 4 | W4 | 10.04 | 40.16 | 178.44 | 81.41 |
| 6 | W6 | 8.15 | 48.90 | 178.74 | 93.06 |
| 8 | W8 | 8.01 | 64.09 | 178.62 | 110.24 |

`C` is the number of active requests and `W` is the number of rows advanced by
one physical target pass. Each of two C2 users receives 13.86 decode tok/s; the
GPU advances them at 27.73 tok/s combined. Total processed tok/s also counts all
prompt tokens and is not a decode-speed measurement. These target-only baseline
waves draft zero tokens, so speculative acceptance is not defined for them.

Before the September 9 C1 policy fix, the scheduler and memory changes retained
the legacy DSpark decisions. Repeated prompts produced identical text and the
same drafted/accepted counts:
40/16, 80/44, 80/47, and 40/19. The three-prompt quality corpus scored
58.5% aggregate acceptance with two of three outputs byte-exact; the remaining
near-tie trajectory matched the qualified DSpark baseline.
An earlier release-level lifecycle audit forced all eight resident sessions
through exact W8 with zero drafts, then issued the same lone request twice.
Both C1 requests resumed DSpark immediately, produced byte-identical text and
the same 40/17 drafted/accepted counts, and decoded at 21.41 tok/s.

Prompt capture now grows only while a DSpark session is prefilling and shrinks
to the verification width afterward. All DSpark sessions borrow one
engine-owned 4,096-row batch workspace because scheduler execution is
serialized. Eight resident sessions load with about 17 GiB still available,
instead of allocating roughly 4 GiB of duplicate prompt workspace per session.
Warm unified-memory model loads remain 21.4-21.6 seconds.

Moving greedy vocabulary sampling to a separate GPU argmax was also measured.
The selected tokens were identical and host readback took about 0.012 ms, but
the extra vocabulary scan and logits persistence slowed C2-C8. That experiment
was removed; the retained path keeps the existing fused/host selection behavior.

### Concurrent DSpark qualification by text type

Concurrent DSpark must be judged by both acceptance and proposal coverage.
Acceptance alone measures how many proposed support tokens the target kept;
coverage measures how often proposals were made. A route with high acceptance
over a small fraction of output tokens can still have little effect on serving
speed.

The serving harness now runs the shared ten-category corpus as synchronized
waves and reports drafted/accepted tokens, drafted tokens per output token,
per-category decode speed, and privacy-safe completion-hash counts. `distinct`
pairs unrelated cases to represent independent users. `homogeneous` runs C
copies of each case to expose text types where both requests qualify for a
shared speculative tail.

On a September 7 release C2 homogeneous run, the normal confidence scheduler
drafted 88 support tokens, accepted 73 (82.95%), and proposed only 0.072 tokens
per output token. Acceptance was concentrated in summarization (36/44),
structured output (18/18), repetitive text (13/16), and multilingual text
(6/6). Code, creative, expository, and instruction cases did not draft; the
reasoning case rejected all four proposals. This is why one blended acceptance
number cannot decide the route.

The retained C2 support-head kernel keeps support attention, KV, MoE, and final
transform session-local, then runs the tied target vocabulary projection once
over both five-row support blocks. The exact Q8 ten-row specialization reduced
the two output-head launches from 54.62 ms to 28.18 ms per eleven proposal
cycles in a matched profile. Forced full-tail C2 decode improved repeatedly
from about 10.63 to 10.73 tok/s per user. Under the normal category scheduler,
where proposals cover little of the output, median decode moved from 14.13 to
14.15 tok/s overall and from 14.57 to 14.69 tok/s for summarization. Draft and
accepted counts were identical, and every per-case completion-hash multiset
matched. The kernel became the default inside the opt-in C2
DSpark route; `GUFO_DEEPSEEK_DSPARK_SUPPORT_HEAD_BATCH=0` restores serial heads
for qualification.

Wider concurrent support was implemented and measured rather than inferred.
The custom C4/C6/C8 output kernels preserved the qualified trajectories, but
the support bodies remained session-local and verification expanded to 24, 36,
and 48 target rows. Against exact target batching, per-user decode regressed
56.3%, 65.1%, and 73.4% respectively. Those candidates were removed. The current
implementation batches the support body's routed work and verifies independently
sized request blocks instead of those fixed six-row blocks.

The September 9 validation found and fixed a state-corruption bug in the
grouped compressor path. Sessions share the same projection workspace. Copying
a later request's rows to offset zero overlapped the source when, for example,
the first request had two rows and the next had three. The corruption could
remain invisible until the next compressed row was emitted. A repeated baseline
run first diverged at compressed-attention layer 2, then changed later tokens.
Request-specific views now keep compressor and indexer scratch writes within
each request's rows and eliminate those copies. Aligned compressor blocks also
use the row loop when partial acceptance needs intermediate prefix snapshots.

The new model-backed `deepseek_v4_flash_dspark_down_routes_test` compares
independent per-request compressor projections and the original Q2 kernel
against grouped projections and the optimized Q2 kernel. It changes request
order and token budgets at C1/C2/C4/C6/C8, checks every frontier logit, and requires
identical emitted tokens, positions, and all speculative counters across 126
observations. The longer requests cross the ratio-128 compression boundary.
Finite logits, progress, budget compliance, and actual support proposals are
hard gates. `GUFO_DEEPSEEK_DSPARK_VALIDATE_DOWN=1` also reruns the original Q2
kernel on each candidate launch's actual input and compares every F16 output
bit. This diagnostic adds synchronous readbacks and must be off for timing.

The Q2 change loads each staged activation once for both independent output
rows. Each row preserves its original dequantization, accumulation order, warp
reduction, and F16 output boundary. Eight isolated rectangular/ragged shapes
were bit-exact under the production `-O3 -ffast-math -fno-finite-math-only`
flags, with 1.19–1.28x kernel speedups. A four-output-row variant was rejected:
it did not improve throughput and spilled 80 bytes per lane at four tiles.

Release HTTP measurements use the pinned target/support artifacts, context
2048, eight server slots, rolling drafts, homogeneous `repetition_sequence`,
64 greedy output tokens, and one warmup per width. Each arm ran three times in
the order off/on, on/off, off/on. Both arms contain the state fix; only
`GUFO_DEEPSEEK_DSPARK_DOWN_REUSE_MID=0/1` changes. These measurements preceded
the C1 policy fix described below: their C1 row is a legacy fixed-width control,
despite the server being configured for rolling drafts.

| C | Original Q2, per-user tok/s | Reused mid, per-user tok/s | Change | Accepted/drafted per wave |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 18.809 | 18.822 | +0.07% control | 17/40 |
| 2 | 22.928 | 23.611 | **+2.98%** | 88/100 |
| 4 | 14.568 | 14.973 | **+2.78%** | 176/192 |
| 6 | 7.337 | 7.619 | **+3.83%** | 204/276 |
| 8 | 6.652 | 6.856 | **+3.07%** | 328/416 |

Every completion-hash multiset and accepted/drafted count matches across all
six runs. These rates measure completion tokens over server decode time.
They exclude prefill; combined active decode is the per-user rate multiplied
by C. The optimized route is the default; setting
`GUFO_DEEPSEEK_DSPARK_DOWN_REUSE_MID=0` restores the original Q2 kernel.

Mixed `distinct` HTTP waves need a separate interpretation. At C2/C8 on the
ten-case, 32-token corpus, completion hashes were not all stable across the
off/on runs. Repeating the original kernel reproduced the same three C2 text
changes (creative, Italian, and JSON); C8 acceptance counts also varied in the
A/A repeat. This makes HTTP hash equality alone an unsuitable kernel oracle
for that workload. The deterministic session test and direct same-input Q2
comparison provide the arithmetic check. These results do not establish
batch-composition-independent text generation.
The mixed corpus then completed all 26 C2/C8 requests with direct Q2 validation
enabled and zero bit mismatches (168/254 and 284/412 accepted/drafted tokens).
That diagnostic run is correctness evidence only.

Raw per-user samples, in run order:

| C | Original Q2 | Reused mid |
| ---: | --- | --- |
| 1 | 18.809, 18.816, 18.766 | 18.821, 18.840, 18.822 |
| 2 | 22.919, 22.933, 22.928 | 23.611, 23.610, 23.619 |
| 4 | 14.551, 14.570, 14.568 | 14.951, 14.973, 15.030 |
| 6 | 7.337, 7.342, 7.325 | 7.619, 7.604, 7.621 |
| 8 | 6.647, 6.652, 6.653 | 6.861, 6.856, 6.847 |

The original PR head reproduced 22.94/14.49/7.31/6.92 tok/s at C2/C4/C6/C8.
Its C8 result used a different accepted trajectory (336/400), affected by the
state bug. The matched table above isolates the kernel gain after that fix.
It does not establish equivalence to autoregressive decoding or a DSpark win
over target-only batching on every workload; C6/C8 still need lower verifier
cost to beat their previously measured target-only rates.

Matched C8 profiles used 32 output tokens and no warmup. Both recorded 197,374
dispatches; the repeated gate and down kernels each ran 387 times:

| Kernel | Original Q2 run | Reused-mid run | VGPRs before/after | Scratch |
| --- | ---: | ---: | ---: | ---: |
| Grouped IQ2 gate/up | 944.61 ms | 947.20 ms | 136/136 | 0 |
| Grouped Q2 down, four tiles/two rows | 809.15 ms | **665.20 ms** | 64/72 | 0 |
| All GPU kernels, including prefill | 9,582.48 ms | 9,408.63 ms | — | — |

Q2 down uses 1,024 threads, 16 KiB dynamic LDS, and 128 SGPRs in both runs.
The isolated production-flag kernel occupancy query admits two workgroups per
processor (64 wave32 waves), with zero local-memory spills at both two and four
tiles. The matched full-width C8 cycle medians are:

| Stage | Original Q2 | Reused mid |
| --- | ---: | ---: |
| Support draft | 50.03 ms | 49.97 ms |
| Target verification | 467.50 ms | **452.05 ms** |
| Commit | 12.12 ms | 12.22 ms |

Verification remains the bottleneck. These traces are separate from the
unprofiled serving rates above.

A further C6 tile sweep retained three tiles per pass: 7.659 and 7.665 tok/s
around a four-tile control of 7.624, with identical completion hashes and
204/276 acceptance. This adds about 0.5% over the reused-mid four-tile route.
Other widths keep their existing tile counts.
`GUFO_DEEPSEEK_DSPARK_DOWN_TILES_PER_PASS=4` restores the C6 control.

The C1 investigation found a separate policy wiring bug: HTTP `DecodeStep`
called the legacy session API, ignoring both `--draft-tokens` and
`--draft-policy`. It always verified the full five-token support tail, whereas
concurrent requests honored the configured rolling width. Low C1 acceptance
then triggered the serial backoff controller. The one-request path now passes
the same configuration through the model API, keeps the qualified support
computation, and selects the requested prefix before target verification.
The legacy session API and explicit fixed/full-width policy remain available.

Three interleaved release repetitions of the same 64-token workload confirm
the C1 correction:

| C1 policy, requested draft limit 7 | Raw per-user tok/s | Median | Accepted/drafted |
| --- | --- | ---: | ---: |
| Fixed, legacy behavior | 18.825, 18.804, 18.813 | 18.813 | 17/40 |
| Rolling, configuration honored | 31.432, 31.461, 31.454 | **31.454 (+67.2%)** | 45/57 |

Completion hashes and support counts match within each policy across all three
runs. C1 now exceeds C2's 23.611 tok/s per user on this workload. Exploratory
C1 controls measured 29.09 tok/s at fixed width 3, 27.81 at width 2, 19.28 at
width 4, and 29.79 for rolling capped at 3. Disabling backoff was neutral
(31.41 tok/s, same 45/57), so the existing controller remains enabled.

A separate fresh-process C1 corpus sweep used all ten cases, 64 output tokens,
no warmup, and one measurement per case. The across-case median rose from
18.80 to 26.76 tok/s. These are exploratory per-case samples, not repeated
speed guarantees; expository text was 3.1% slower with rolling.

| Case | Fixed tok/s | Rolling tok/s | Fixed accepted/drafted | Rolling accepted/drafted |
| --- | ---: | ---: | ---: | ---: |
| Repetition | 18.79 | 31.37 | 17/40 | 45/57 |
| Expository | 27.73 | 26.87 | 47/80 | 44/70 |
| C++ | 25.46 | 27.20 | 45/85 | 44/76 |
| Reasoning | 18.80 | 28.77 | 19/40 | 46/64 |
| Summary | 21.97 | 26.84 | 26/55 | 27/44 |
| Italian | 17.61 | 18.15 | 14/40 | 12/27 |
| Chinese | 16.68 | 17.56 | 7/30 | 9/20 |
| JSON | 19.29 | 26.68 | 20/45 | 43/66 |
| Creative | 16.04 | 17.61 | 2/20 | 9/20 |
| Instructions | 18.08 | 18.67 | 15/40 | 17/37 |

The new test asserts C1 draft limits of 1/3/7 and exact full-width fixed-policy
equivalence to the legacy API. It also checks four 32-token chat continuations
(repetition, JSON, creative writing, and Chinese) by replaying each emitted
token through a separate scalar target session:

| C1 policy | Scalar top-1 agreement | Worst token rank | Worst frontier RMSE | Minimum cosine | Maximum logit error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy fixed full width | 123/128 | 6 | 0.65 | 0.99 | 3.94 |
| Rolling | 125/128 | 2 | 0.60 | 0.99 | 3.23 |

These continuations can differ between policies because verification width
changes numerical routes. The additional test freezes the measured legacy
agreement floor and preserves the existing full-logit envelope; it does not
claim byte-identical scalar generation. The separate immutable pinned
trajectory still scores 116/128 top-1, rank sum 142, and worst rank 3, with
its original thresholds unchanged.

Reproduce the model-backed checks with the pinned target and support paths:

```sh
GUFO_DEEPSEEK_V4_FLASH_MODEL="$MODEL" \
GUFO_DEEPSEEK_V4_FLASH_DSPARK_MODEL="$SUPPORT" \
  nix develop -c ctest --preset gpu-full --output-on-failure \
  -R '^deepseek_v4_flash_dspark_down_routes_test$'
env -u GUFO_DEEPSEEK_V4_FLASH_DSPARK_MODEL \
  GUFO_DEEPSEEK_V4_FLASH_MODEL="$MODEL" \
  nix develop -c ctest --preset gpu-full --output-on-failure \
  -R '^deepseek_v4_flash_engine_test$'
```

For release timing, start a fresh `result/bin/gufo serve` process for each
policy and keep profiling and validation readbacks disabled:

```sh
./result/bin/gufo serve --host 127.0.0.1 --port 19229 --sessions 8 llm \
  --model "$MODEL" --context 2048 --speculative dspark \
  --dspark-model "$SUPPORT" --draft-tokens 7 --draft-policy rolling
# In another terminal:
nix develop -c python3 tools/serving/gufo-serving-bench.py \
  --base-url http://127.0.0.1:19229 --gufo ./result/bin/gufo \
  --suite benchmarks/qwen3.8-27b/speculative-corpus.json \
  --case repetition_sequence --corpus-layout homogeneous \
  --concurrency 1,2,4,6,8 --max-tokens 64 --warmup 1 \
  --repetitions 1 --output /tmp/dspark-serving.json
```

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
| Retained default (September 3) | 455-462 | 16.6 | 116/128, 142, 3 | 0.41 | pass |
| Plus the fused hyper-connection pre-block chain | 496.1 | 16.3 | 116/128, 142, 3 | 0.42 | pass |
| Retained default, MMQ disabled | 270.1 | 16.6 | 116/128, 142, 3 | — | pass |
| Plus the attention output-B hipBLASLt fallback | 410.5 | 16.6 | 114/128, 150, 5 | — | fail |

The September 4 row is the mean of three interleaved rounds against `a8158c7` on
one host, which read 483.4 over the same rounds; across sessions only that paired
number is comparable, not the absolute. See "The hyper-connection pre-block
chain, fused into one pass".

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

Each of those routes was reachable individually through an environment switch
while it was being evaluated. All of them are now compiled in unconditionally;
see "Every DS4 tuning switch is now compiled in" below.

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

#### The LDS access width, which is what finally moved it

The counters that priced the routed loader at 19.7% LDS bank-conflict cycles led
somewhere after all, but not to the tile layout. `tools/prof/isa_mix.py` on the
production instantiation showed the two adjacent 4-byte stores of the decoded
pair being fused by the compiler into **`ds_store_2addr_stride64_b32`** -- whose
two addresses are 64 dwords apart, exactly 32 banks, hence *the same bank*. Every
one of those instructions self-conflicts before cross-lane conflicts are even
counted.

Writing the pair as one `int2` gives `ds_store_b64` across banks b and b+1
instead. The pair is always 8-byte aligned (the row stride is even, and so are
`8*kqsx` and `2*l`), so it is bit-identical, costs no shared memory, and took the
conflict rate from **19.7% to 11.7%** with total LDS cycles 40.5 G to 34.0 G.

The same idiom appeared in this repository's own kernels, and the wins there were
larger than in the vendored one:

| Change | Effect |
| --- | --- |
| IQ2 decoded pair as one 8-byte store | conflicts 19.7% to 11.7%; +1.7% end to end |
| Q2-down dequantized pair as one 8-byte store | that kernel **1,192 to 812 ms**, conflicts to 6.7% |
| Attention KV tile staged four values per lane instead of one half | the window producer 259 to 206 ms |
| Indexer and remaining Q2 dequant stores widened the same way | neutral; kept because it is strictly fewer, wider accesses |
| Attention row-table setup taken off lane 0 (512 dependent `topk` reads and 256 serial writes per block) | neutral; kept for the same reason |

Two things in the same family were measured and **rejected**:

| Rejected | Why it looked right | Measured |
| --- | --- | ---: |
| Nine-dword group pitch in the IQ2 tile, so the eight groups land on eight distinct banks | conflicts are structural at an 8-dword pitch: lanes `kqsx` and `kqsx + 4` are 32 dwords apart whatever the row stride | conflicts 19.7% to 7.7%, but 9 dwords breaks 16-byte alignment for `ds_read_b128`, so total LDS cycles went 33.9 G to 44.8 G and the kernel got slower |
| `int4` copy in the `tile_y` staging loop, the hottest staging in the model | one dword per lane is four times the loads and stores it needs | **7% slower** (428 against 462 tok/s). It also needs a scalar remainder: a full vector step reaches past `GGML_PAD(mmq_x*MMQ_TILE_Y_K, nwarps*warp_size)`, which is where `tile_x` starts -- at mmq_x 80 it would run to 3,071 against a 2,944 pad and corrupt the weight tile |

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

Both kernels leave 85% of VALU capacity unused, so the 15% ceiling is a
dependency-stall property of the inner loop rather than a throughput limit.

#### But reducing the instruction count does not reduce the time

Three interventions on the expansion, all bit-identical (gate output 116/128,
142, 3 with `rmse` 0.41 and `max_error` 1.95 in every case):

| Intervention | Effect on the inner loop | Measured |
| --- | --- | ---: |
| Sign masks derived arithmetically instead of loading `ksigns64` | -1 load, +4 VALU per 8 weights | 435.1 against 439.6 tok/s |
| Whole sign-applied codebook precomputed into a 2^15-entry table | -1 load and -6 VALU per 8 weights, one 8-byte load for the pair | 412.1 against 418.7 tok/s |
| Row-loop unroll factor | more independent decode chains | already tuned: `#pragma unroll 2` is deliberate, "limit concurrent row decompression while retaining two-row ILP" |

So the descriptive correlation (2.62x the time for 3.00x the instructions) is not
causal in the direction that helps: the loader's time is insensitive to its own
instruction mix. Removing six operations per eight weights, or trading a load for
arithmetic, or the reverse, all land inside noise or slightly negative.

The instruction listing and the remaining counters say why there is no single term
to attack. `tools/prof/isa_mix.py` on the production instantiation gives 4,063
instructions with **568 `s_delay_alu`** -- the compiler's explicit dependency
stalls, which is the 15% issue ceiling made visible -- and 728 instructions (18%)
of 64-bit vector address arithmetic. That address arithmetic is mostly the fused
SwiGLU epilogue's four separate pointer computations per output, and the epilogue
turns out to be dynamically cheap: disabling it
(`GUFO_DEEPSEEK_ROCM_MMQ_FUSED_SWIGLU=0`) moves `mul_mat_q` only 2,791.68 to
2,738.08 ms, about 2%. Statically prominent, dynamically small.

Against the dense path the routed one issues about three times as much of
*every* instruction class, not one:

| | IQ2 routed | Q8 dense | ratio |
| --- | ---: | ---: | ---: |
| VALU | 45.5 G | 15.2 G | 3.0x |
| SALU | 8.3 G | 1.0 G | **8.3x** |
| LDS | 8.9 G | 3.3 G | 2.7x |
| LDS bank-conflict cycles | 6.66 G of 33.9 G (**19.7%**) | 0.66 G of 9.3 G (7.1%) | 2.8x |

The bank conflicts are structural rather than a padding mistake, and worth about
4% of the kernel if eliminated. Within a wave `kqsx` and `kqsx + 4` write exactly
`8 * 4 = 32` dwords apart, the same bank, whatever the row stride is -- and the
row stride is already padded for the row case (`MMQ_MMA_TILE_X_K_Q8_0 % 8 == 4`,
76 dwords). Fixing it means interleaving the K axis inside the tile, which the
consumer's `load_ldmatrix` layout in `vec_dot_q8_0_q8_1_mma` also has to mirror --
a relayout of vendored MMQ internals for ~1.2% end-to-end.

So the honest position after all of this: the routed gate/up's cost is the volume
of decode work the IQ2_XXS format implies, it is spread across VALU, SALU and LDS
in proportions that no single source change shifts, and the schedule around it is
already at the layout's limits.

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

### Against antirez/ds4 PR 887 (September 4, 2026)

[PR 887](https://github.com/antirez/ds4/pull/887), "ROCm: optimize DeepSeek V4
Flash prefill and DSpark on gfx1151", was pulled and compared change by change.
It takes 4,096-token prefill from **187.26 to 292.86 tok/s**. This branch takes
it from **188.5 to 455-462 tok/s mean, 478.2 peak**. The two baselines agree to
0.7%, which makes the comparison a controlled one despite the different ROCm
(7.2.3 here, 10.0 there): **the same starting point, and this branch ends 57%
ahead.**

Every prefill lever in that PR is already present here:

| PR 887 prefill change | Status here |
|---|---|
| Four-wave wave32 `y64` IQ2 MMQ tile (their largest single item, 33.2% on the isolated gate/up test) | Present: `DS4_ROCM_WMMA_MMQ_Y=64` with `nwarps = mmq_y/16`, which the profile confirms as `wg=128` |
| `mmq_x` clamped to 64 on gfx1151 | Tested, **dead tie**: 453.5 against 454.0 tok/s mean over three interleaved pairs via `DS4_CUDA_MMQ_X_MAX=64` |
| 32-head wave32 rocWMMA prefill attention | Present |
| Dedicated `4096 x 2048 x 8192` F16 rocWMMA attention-output-B kernel (25.89 -> 11.88 ms there) | Present, 13.6 ms live and 11.6 ms standalone, best of 17 geometries |
| Vectorized F32-to-F16 KV staging | Present |
| Cooperative Q2_K weight staging once per 256-value K slab | Present (`shW` in the wide down kernel) |
| `ksigns64` `uint2` sign-mask lookup in the IQ2 tile loader | Present; the arithmetic alternative was measured here and was neutral |
| Halved down-kernel accumulator LDS plus a raw Q2_K window | Present |
| Routed grid bounded by the largest expert bucket rather than the whole chunk | Present (`ds4_mmq_set_routed_max_expert_rows`) |

What is *not* ported, and why:

- **rocBLAS solution-index overrides.** Those indices are pinned to two exact
  library builds (`5.5.0.cd957402` in ROCm 7.14 uses `-217/-216`,
  `5.6.0.8d1ae90e` in ROCm 10.0 uses `-50/-49` plus `-401`). This box runs
  neither, and PR 887's own code falls back silently on an unknown build, so
  porting the table would be inert. Retuning for this build is open work; the
  largest library calls left are `Cijk_Alik_Bljk_HHS_BH_MT128x128x32` at 554 ms
  and `MT96x96` at 365 ms.
- **The tiny-M matmul specializations** (`matmul_f16_tinym24_wmma_kernel` at
  `M=24, N=2048, K=16384`, `matmul_f16_smallm_wmma_kernel`,
  `matmul_q8_0_f32_batch_sharedx_exact8_kernel<2..5>`) and the `down_tile == 2`
  / `expert_tile_m = 4` / `compact_active` MoE paths. All of these are gated to
  `n_tokens <= 8`: they are DSpark verifier and decode shapes, not prefill.
- **The 16-cycle DSpark scheduler policy and the support-model cache fix.**
  Orthogonal to prefill throughput.

Consistent with that, PR 887 reports ordinary decode unchanged at 16.87 tok/s
before and after, which matches the 16.6 tok/s measured here: neither branch
moves tg, and for the same reason -- see the decode note below.

### Four more levers, all falsified (September 4, 2026)

Each of these follows from a mechanism that had already paid somewhere else in
this model, which is why they were worth measuring; none of them transfers.

- **LDS pad on the attention output GEMM.** `ds4_gemm_f16_wmma_kernel` stages
  its A tile at a 256-half pitch -- 128 dwords, exactly 0 mod 32 banks -- so
  every fragment row starts on the same bank. Padding the pitch is the textbook
  fix and measured `11.52` (AP16) and `11.54` ms (AP8) against `11.63` base, or
  0.8% of one kernel and 0.06% end to end. Padding B instead is neutral to
  slightly worse. The base geometry stands. (First attempt at this reported
  `maxdiff inf`: the pad widened `A_RUN` as well, so the staging loop ran off
  the end of its source. The pad is only a pitch, never a data extent.)
- **A 16-byte dequant store in the wide Q2-down staging.** Going from two
  4-byte LDS stores to one 8-byte store in this kernel was worth `1,192` ->
  `812` ms, so `KG = 8` and a `uint4` -- which also halves how often the block
  and group scales are re-decoded -- looked like the same win again. It is
  worth nothing: `1,167.03` -> `1,164.80` ms at kernel level, and the +1.2%
  end-to-end mean sat inside a +-12 tok/s spread. **The 8-byte win was never
  about store width.** Split stores get paired into
  `ds_store_2addr_stride64_b32`, whose two addresses are 64 dwords apart and so
  land in the same bank; once that pairing is broken the kernel is back to
  being latency bound at 9-18% issue utilisation, and LDS store bandwidth is
  not what it is waiting on.
- **float4 in the hyper-connection kernels.** `hc_expand4` keeps eight streams
  in flight at one dword per lane and looked under-vectorized. Sixteen bytes
  per lane made all three *worse*: `hc_expand4` `116.75` -> `118.48`,
  `hc_expand4_add_moesum` `159.20` -> `163.18`, and
  `hc_split_weighted_sum_fused` `148.61` -> `170.75` ms. A wave already covers a
  full cache line per stream at one dword per lane, so there was no coalescing
  to win, and quartering the block count while raising register pressure costs
  more than the wider access buys.
- **Moving sampling to the GPU.** Worth 24 microseconds. See below.

### No requant can reduce matrix ops on gfx1151 (compiler-verified)

The second form of the requant argument is not about LDS at all: get *fewer
matrix ops per output* by carrying more k per WMMA instruction. On this hardware
that is not available. Probing the compiler directly:

```
wprobe.hip:7:8: error: use of undeclared identifier
  '__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32'; did you mean
  '__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32'?
```

`16x16x32` does not exist for gfx1151; the only integer WMMA shapes are
`16x16x16` in `iu8` and `iu4`. **Matrix ops per output is therefore
`M*N*K/4096`, fixed by the GEMM dimensions and completely independent of the
weight format.** No routed-expert requant can reduce it. `iu4` would only halve
LDS, which the occupancy table above already shows is not the lever -- and it
cannot represent IQ2_XXS's grid magnitudes anyway, which are the odd values 1..15
and need sixteen signed levels against `int4`'s -8..7.

### The last two small items, sized properly

- **`QK_WG = 4` in the mixed attention kernel is worth zero, not 0.4%.** The
  isolated harness shows `QT WG4 FUSEP` at 3.920 against `WG2 FUSEP`'s 4.068 ms,
  a 3.6% kernel win, and scaling that by the kernel's 10.9% share suggests 0.4%
  end to end. It does not translate: measured **433.1 against 433.2 tok/s**. It
  also reassociates the score sum and moves the prefill envelope from rmse 0.41
  to 0.48, which is no longer clearly better than the `16d5e30` baseline's
  0.478146. Both numbers were already recorded at the `QK_WG` declaration; the
  0.4% estimate was the error.
- **`quantize_mmq_q8_1` has about 0.7% in it, not 1.7%.** Its 203 ms splits into
  127 ms on a `4096 x 4096` shape at **115 GB/s** (48% of ceiling) and 66.6 ms on
  the routed `24576 x 4096` shape at **329 GB/s** -- above DRAM peak, so
  MALL-served, because the routed activations are the same 4,096 tokens
  replicated per expert. Only the first is winnable, and the kernel already reads
  `float4`, writes `char4` and reduces by shuffle; 48% is about what a 4:1
  read/write asymmetry gives.

### Status: the search for the remaining 10% is exhausted

Superseded on September 4, 2026: it was exhausted for *kernels*, not for
*chains*. Fusing the hyper-connection pre-block chain took 445 ms of dispatches
to 199 ms for +2.6% paired; see "The hyper-connection pre-block chain, fused
into one pass" below. The per-kernel table here still stands.

pp4096 stands at **441-480 tok/s, mean ~455**, from a 188.5 baseline (+142%),
with the pinned trajectory at 116/128 rank sum 142 worst rank 3 and prefill rmse
0.41 -- better than baseline on both. Reaching 500 needs about 880 ms of the
8,898 ms profile. Every mechanism identified for it has now been measured rather
than argued:

| Candidate | Outcome |
|---|---|
| All of antirez PR 887's prefill levers | already present; branch is ~55% ahead of its result from a matching baseline |
| `mmq_x` 24/32/40/64/96/112/128 | tie at 64-128, monotonically worse below 64 |
| MMQ activation-tile register prefetch | -2.7% split loops, -7.7% interleaved |
| Own kernel vs rocBLAS/hipBLASLt on every hot GEMM | parity (best of 27 geometries ties rocBLAS at 21.4 TFLOP/s) |
| More resident waves (the point of shrinking `tile_x`) | halving residency costs only 7.8%; 4->3 workgroups is free |
| Fewer matrix ops via requant | impossible: every gfx1151 WMMA shape is K=16 |
| `QK_WG = 4` | 433.1 vs 433.2, and costs the rmse margin |
| `quantize_mmq_q8_1` | ~0.7% ceiling, already vectorized |
| GPU sampling | 24 microseconds of a 59 ms token |

### The IQ2 byte-expansion / occupancy program, measured rather than assumed

The standing argument for a routed-expert format change was: `tile_x` byte-expands
2-bit IQ2 codes for `iu8` WMMA, that is 19.5 KiB of the kernel's 30.8 KiB, and it
pins residency to 8 of the hardware's 32 waves. That argument only pays if this
kernel is actually residency-starved, and both earlier attempts to test it were
confounded -- narrowing `mmq_x` also multiplies column tiles and weight re-reads,
and the NCW column split also halves matrix ops per A-fragment load.

The measurement that settles it padded `mmq_get_nbytes_shared` with N KiB of
*unused* dynamic shared memory. MMQ's shared block is dynamic and the kernel
indexes only what it needs, so the grid, the tile shape and the work per wave
stay byte-for-byte identical and only workgroups per CU move:

| LDS | workgroups/WGP | pp4096 mean |
|---|---:|---:|
| 30.8 KiB (pad 0) | 4 | 453.6 |
| 32.8 KiB (pad 2) | 3 | 445.0 |
| 38.8 KiB (pad 8) | 3 | 454.1 |
| 54.8 KiB (pad 24) | 2 | 418.2 |

**Halving residency costs 7.8%, and going from four workgroups to three costs
nothing.** The kernel is already past the point where more resident waves buy
anything, which is also why the 31x stall factor never responded to occupancy
work: it is latency that additional waves do not hide.

So the payoff ceiling for un-expanding the IQ2 codes -- roughly doubling
residency -- is *below* 7.8%, before paying for the on-the-fly fragment
construction it would require (packed 2-bit codes in LDS cannot be fed to
`load_matrix_sync`; the A fragment would have to be built in registers in the
exact `iu8` matrix_a lane layout). Reaching 500 tok/s from a 455 mean needs about
10%. **The routed-expert format change does not get there**, and that is now a
measurement rather than an inference. `iu4` is not an escape either: IQ2_XXS grid
magnitudes are the odd values 1..15, which need 16 signed levels and do not fit
`int4`'s -8..7 without a non-linear remap that breaks the dot product.

Narrowing `mmq_x` for the same purpose, for completeness: 80 (default) 465.2,
40 410.1, 32 411.6, 24 397.4 tok/s. Monotonically worse.

### rocBLAS solution-index retuning: closed by measurement, not declined

This was the last lever named as sized-but-untested, so it was measured.

Two things had to be separated first. The call site for the strided-batched
attention output-A carries the note "rocBLAS reaches 2.7 TFLOP/s on this
strided-batched shape", which reads like 5% of peak on a 554 ms call. That
figure is the **small-n** instance of the shape -- the DSpark verifier at
`n_tokens` of 5 or 6, where the call is latency bound. In prefill the same
`Cijk_Alik_Bljk_HHS_BH_MT128x128x32` dispatch is `m=1024 n=4096 k=4096 batch=8`,
and 43 calls at 12.886 ms each is **21.3 TFLOP/s, about 39% of peak.** This is
also why PR 887's `-401` solution override is worth 1.30x for them and would not
be here: it targets their `4096 x N x 8192` projection at N=5 and N=6.

Swept against it, `tools/bench/dsv4_attn_out_gemm_bench.hip` puts the best of
**27 own-kernel geometries** at `BM128 BN128 BK32 W4x2 SWZ4`, **12.857 ms /
21.38 TFLOP/s** -- a dead tie with rocBLAS's live 12.886 ms, and `maxdiff 0`.
(The earlier 855.99 ms figure recorded for this shape was the `BM256` tile;
`BM128` matches rocBLAS rather than beating it, so there is still nothing to
gain by switching.) rocBLAS's heuristic is already choosing a near-optimal
kernel for the prefill shape, so there is no solution index to find.

The other library block, `MT96x96` at 365 ms over 236 calls, is hipBLASLt at
roughly 22 TFLOP/s on a `K`-short shape where our own kernel measures *worse*
than on the two longer-K shapes above (23.6 at K=8192, 21.4 at K=4096). Not a
candidate either.

Also swept and neutral: the one hipBLASLt route still disabled by default,
`DS4_ROCM_LT_ROUTE_ATTN_B`. `GUFO_DEEPSEEK_ROCM_HIPBLASLT_ROUTING=31` against
the default 23 measured 457.5 against 455.2 tok/s -- the bit is a fallback that
the own rocWMMA output-B kernel already serves, so it rarely fires. The
candidate-index picks (4, with 5 and 6 for two `in_dim == 4096` projections) are
already enabled and already width-scoped to `n_tokens >= 128`, which is what
keeps them out of the pinned trajectory's narrow decode path.

### Prefetching the MMQ activation tile: the one non-requant lever on `mul_mat_q`, and it loses

`mul_mat_q` is 34% of prefill (3,031 ms) and every throughput explanation for it
has now been falsified: 25.8 GB/s of a 240 GB/s ceiling, 11% VALU and 5% LDS
issue, tile width 64 through 128 a dead tie, tile height 128 no gain, resident
waves pinned at 8 of 32 by LDS. What is left is a 31x stall factor, and
`L2CacheHit` says the kernel gets **83.7%** of its loads from L2, so what it
waits on is L2 latency, not DRAM.

The K loop stages `tile_y` in two halves through a single LDS buffer, so the
second half's global loads were necessarily issued *after* the barrier that ends
the first `vec_dot` -- their latency fully exposed, with eight waves to hide it.
Issuing those loads before the first `vec_dot` and replaying them from registers
afterwards is the textbook fix, it changes no arithmetic, and it looked free:
1,536 VGPRs per SIMD against 176 allows eight waves per SIMD while LDS pins the
kernel to two, so 23 more registers cost no residency. **It is 7.7% slower**
(419.5 against 454.3 tok/s mean over three interleaved pairs).

Not a spill and not occupancy -- the prefetch build reports `scratch=0` and
VGPR 144 (IQ2) / 216 (Q8), both still allowing seven or more waves per SIMD.
Interleaving the two streams in one loop makes every first-half LDS store wait
on the second half's loads as well, because `s_waitcnt vmcnt` counts all
outstanding loads rather than a specific one. Draining the first half in its own
loop before issuing the second recovers most of that -- and is still 2.7% slower
than no prefetch (446.6 against 458.8). Holding 23 values live across `vec_dot`
constrains the scheduling inside it by more than the covered latency is worth.

**Do not retry this.** What remains on this kernel is either the routed-expert
format change (a requant) or rocBLAS solution-index retuning for this exact
library build, which targets the 554 ms `Cijk_Alik_Bljk_HHS_BH_MT128x128x32` and
365 ms `MT96x96` calls rather than `mul_mat_q` itself.

### Sampling is on the CPU and that is not costing anything

DS4 selects tokens on the host: `ds4_session_argmax` for greedy decode and a
bounded top-k insertion scan (capped at 1024) when sampling, both in
`runtime/sampling.cpp`. Qwen has a full GPU sampler
(`src/models/qwen/hip/kernels/sample.hip`), and DS4 already has a bit-exact GPU
argmax -- `spec_row_argmax_kernel` resolves ties to the lower index precisely so
that it matches the host, and NaN never wins in either -- so porting the greedy
path is a small, exactly equivalent change.

It is not worth making. `GUFO_DEEPSEEK_ROCM_GRAPH_TOKEN_PROFILE=1` puts the
full-vocabulary device-to-host read at **0.024 ms of a 59 ms token**:

```
ds4: ROCm graph token pos=22 encode=6.395 ms execute=52.567 ms read=0.023 ms total=58.986 ms
```

129,280 floats is 517 KB, and on an APU with unified memory that is a memcpy,
not a transfer. Host argmax over the same buffer adds a comparable amount, so
all host-side token selection is about 0.1% of the decode budget.

Two things this ruled out at the same time:

- **The 6.4 ms encode is not dead GPU time.** `ds4_gpu_begin_commands` is a
  no-op and `ds4_gpu_end_commands` is just `hipDeviceSynchronize`, so there is
  no graph capture and launches issue eagerly; the GPU is already working on
  early layers while the host is still issuing later ones. `execute` is the tail
  wait, not the whole of it.
- **Decode is not DRAM bound either.** The obvious story -- 52.7 ms is just the
  time to stream the active weights -- does not survive `FETCH_SIZE`: decode
  moves roughly 2.75 GB per token, which against 52.7 ms is far under the
  240 GB/s ceiling. tg has headroom, but it is latency headroom and a separate
  work item from prefill. (Do not read achieved bandwidth off a `--pmc` run's
  own timestamps: counter collection serializes dispatches and inflates every
  duration. Take bytes from the counter pass and time from a plain kernel
  trace.)

### DSpark re-validated after the prefill work (September 4, 2026)

Ten-category corpus, 128 tokens per prompt, run on the build before and after the
indexer change so the comparison attributes rather than guesses:

| | before | after | recorded 2026-09-01 |
|---|---:|---:|---:|
| AR | 15.82 | 15.76 | 16.32 |
| DSpark | 17.69 | 17.82 | 18.00 |
| mean speedup | 1.12x | **1.13x** | 1.10x |
| median speedup | 1.00x | 1.01x | 1.00x |
| support acceptance | 61.2% | 61.2% | 60.7% |
| attempts / skipped | 164 / 505 | 164 / 505 | 163 / 541 |

**The two builds produce byte-identical speculative behaviour.** Support
acceptance, positional agreement, full-block rate, attempts and skips match
exactly in all ten categories, and the exact-vs-AR set is the same 4/10. So the
indexer change does not touch the draft or verify streams, and DSpark needs no
re-tuning after it.

What *does* move between runs is the per-category speedup, by up to +-0.15x on
identical token streams: summarization 1.01x -> 0.84x, but Italian 0.93x -> 1.02x
and creative 0.86x -> 0.95x in the opposite direction. These are single 128-token
generations with 5 to 8 speculative blocks each, so they are timing noise, not
signal. **Read the aggregate and the acceptance columns; treat a single
category's speedup as noise unless its acceptance moved too.**

The repetitive case, which is the stable throughput target, holds at **29.22
tok/s and 1.83x** (recorded 29.05 / 1.85x -- the ratio is slightly lower only
because AR itself is faster now). Its 96.4% acceptance and 90.9% full-block rate
are unchanged.

### Long context: the indexer is the whole degradation curve (September 4, 2026)

Prefill throughput against prompt length, one process, `result-clean`:

| prompt | tok/s |
|---|---:|
| 4,096 | 434.20 |
| 8,192 | 481.18 |
| 16,384 | 482.92 |
| 32,768 | 460.48 |
| 65,536 | 414.00 |

The peak is at 16K -- 4,096 is *below* it because a single chunk cannot amortize
per-chunk setup -- and 64K gives up **14.3%** against that peak. For a 16x
context increase that is a flat curve, which is the sparse-attention design
working: the raw KV window and the 512-row indexer top-k bound per-token
attention work independently of context length.

What does not stay bounded is the indexer's own scoring, which must compare each
token against every compressed row so far. Normalized per token against the
4,096 profile:

| kernel | 4K ms/token | 64K ms/token | ratio |
|---|---:|---:|---:|
| `indexer_scores_wmma128_kernel` | 0.0250 | 0.334 | **13.4x** |
| `mul_mat_q` | 0.740 | 0.653 | 0.88 |
| `moe_down_q2K_hotlist_wmma_wide` | 0.285 | 0.284 | 1.00 |
| `attention_mixed_heads32_wmma<true,true>` | 0.2368 | 0.1864 | 0.79 |

Everything except the indexer is flat or *cheaper* per token at 64K. The indexer
scoring plus its top-k machinery (`indexer_topk_chunk_pow2`,
`..._merge_pow2`, `..._8192_cub`) is **16.8% of a 64K prefill against about 1.5%
at 4K**, which is the entire degradation and then some.

**The scoring kernel was loading one operand the expensive way.** It computes
`scores[token][comp] = sum_h relu(q_h[token] . k[comp]) * w_h`, staging the
shared compressed key tile as `b_sh[comp][d]` and loading it as a **`col_major`
matrix_b** -- the exact pattern that cost this model's attention score pass about
8x, recorded above. Staging q transposed as `qt[d][token]` turns the product into
`scoresT[comp][token] = K . Qt` with **both fragments row_major**:

| variant | `indexer_scores_wmma128` at 16K |
|---|---:|
| baseline | 1,413.49 ms |
| + double-buffered q staging | 1,358.13 ms (-3.9%) |
| + transposed q, both operands row_major | **952.37 ms (-32.6%)** |

Two changes, and the second is by far the larger:

- **Double-buffered q staging.** Each head's 16x128 q tile is a scattered global
  read (consecutive token rows are `n_head * head_dim` floats apart, so 16 cache
  lines) and it sat between two barriers with only eight `mma_sync` after it, 64
  heads deep -- 128 barriers per block with the load latency exposed. Staging head
  h+1 into the other buffer before consuming head h leaves one barrier per head.
  Worth -3.9%, so barriers were not the main cost.
- **Transposed q.** -30.9% on its own. As a bonus the token is now fixed per lane
  rather than varying with the accumulator element, so each head's weight is one
  load instead of eight.

`FETCH_SIZE` was measured first and ruled out the tempting explanation: q is
logically re-read once per comp tile, which at 64K is 16.8 GB per call against a
134 MB buffer, and 65 ms per call would put that at 258 GB/s -- apparently at the
DRAM ceiling. The counter says the kernel actually fetches **13.88 GB over
1,405 ms at 16K, 9.9 GB/s, 4% of the ceiling**: the 32 MB MALL absorbs
essentially all of the re-reads. It is latency bound, not bandwidth bound, and
acting on the analytical estimate would have been wrong.

End to end, measured with the new build bracketing the old one so thermal drift
cannot favour either:

| prompt | before | after | change |
|---|---:|---:|---:|
| 16,384 | 463.00 | 472.07 | +2.0% |
| 65,536 | 411.03 | 431.13 | **+4.9%** |

The gain grows with context because the kernel's share does. **Degradation from
the 16K peak out to 64K falls from 14.3% to 8.7%.**

`nix build .#checks.x86_64-linux.pr` stays green at 81/81 with both changes, which
matters here because swapping the two WMMA operands can reorder the intra-fragment
summation.

What is left on the long-context curve, in order: the scoring kernel is still
about 9% of a 64K prefill, and the top-k stages (`indexer_topk_chunk_pow2` 2.1%,
`..._merge_pow2` 0.6%, `..._8192_cub` 0.4%) another 3%. The remaining scoring cost
is irreducibly O(context) per token -- exact top-k over all compressed rows -- so
further work there means either a cheaper score (fewer than 64 heads contributing,
which changes selection and therefore quality) or a hierarchical/approximate
top-k.

### Every DS4 tuning switch is now compiled in (September 4, 2026)

The backend had accumulated **35 behaviour-selecting environment variables** from
successive rollouts -- kill switches for routes that have long since become
unconditional, sweep knobs for settled questions, and debug instruments whose
non-default branch had not run in months. Every one of them defaulted to the
shipped behaviour, so none was needed to get it, and each one was a live path
that could silently change numerics.

All 35 are gone and their defaults are compiled in. Removed, by category:

- **Kill switches for now-unconditional routes** (default on): `..._MMQ`,
  `..._MIXED_WINDOW_WMMA`, `..._MMQ_FUSED_SWIGLU`, `..._FUSED_MOE_SUM`,
  `..._FUSED_INV_ROPE_PACK`, `..._WIDE_DOWN_TILE_MAP`, `..._TILED_Q8_TRANSPOSE`,
  `..._F16_INPUT_MIRROR`, `..._VEC_CONVERT`, `..._Q8_DECODE_SHAREDX_64K`,
  `..._HIPBLASLT_TUNE`, `..._DISABLE_SHARED_GATE_UP_SWIGLU_FUSION`,
  `..._NO_PREFILL_KERNEL_WARMUP`, `DS4_MMQ_D2R`, `DS4_MMQ_D2R_IQ2`,
  `DS4_MMQ_NO_YIND`, `DS4_MMQ_ROUTED_TILE_BOUND`, `DS4_CUDA_NO_MOE_DEDUP`,
  `DS4_MMID_LARGE`, `DS4_MMID_CASE1`.
- **Seven more of the same, hidden behind a helper** that took the variable name
  as a parameter, so a `grep` for `getenv("` did not find them:
  `..._DISABLE_HC_FUSION`, `..._DISABLE_KV_FUSION`,
  `..._DISABLE_QKV_NORM_FUSION`, `..._DISABLE_COMPRESSOR_PAIR_PROJ`,
  `..._DISABLE_HC_NORM_FUSION`, `..._DISABLE_SHARED_DOWN_HC_FUSION`,
  `..._DISABLE_ATTN_OUT_HC_FUSION`.
- **Sweep knobs for settled questions**: `DS4_CUDA_MMQ_X_MAX`,
  `DS4_MMQ_COL_SLOTS`, `DS4_MMQ_LDS_PAD`, `..._HIPBLASLT_CANDIDATE`,
  `..._HIPBLASLT_ROUTING`.
- **Numeric thresholds** now the constants they defaulted to:
  `..._PREFILL_CHUNK` (4,096), `..._RESUME_PREFILL_MIN` (4),
  `..._GRAPH_RAW_CAP`, `..._DECODE_INDEXER_SPARSE_THRESHOLD` (1,024),
  `..._GRAPH_TOKEN_SPLIT_LAYERS` (4), `..._GRAPH_OUTPUT_ROW`,
  `..._DENSE_SMALL_BATCH_ROWS` (24), `..._MOE_SMALL_BATCH_ROWS` (16),
  `DS4_MMQ_D2R_MIN_COLS` (1,024), `DS4_ROCM_WEIGHT_PRELOAD_SPAN_MB` (1,024).
- **Debug instruments** whose default-off branch was dead: `DS4_MMQ_OUT_MEMSET`,
  `DS4_MMQ_YBUF_MEMSET`, `DS4_MMQ_YIND_VERIFY`, `DS4_Q8_FOLD_SELFTEST`,
  `DS4_CUDA_MMQ_Q81_PERSISTENT` (and its eight dead fast paths),
  `..._MOE_EXPERT_SPREAD`, `DS4_CUDA_NVTX` / `DS4_CUDA_NSYS_PREFILL_START_POS`
  (nsys is CUDA tooling; `DS4_MMQ_HAS_NVTX` is 0 on this HIP build),
  `GUFO_DEEPSEEK_DSPARK_SELFTEST_REPS` (4).

**Two of the 35 were not behaviour-preserving, and the structural check could
not see it.** `..._DENSE_SMALL_BATCH_ROWS` (24) and `..._MOE_SMALL_BATCH_ROWS`
(16) were not read directly. Both returned
`ds4_rocm_small_batch_limit(configured)`, which is
`g_small_batch_mode ? configured : 0u`, so compiling in the constants dropped
that gate: the dense and routed-MoE small-batch routes went from off outside a
DSpark verification block to unconditionally on, and one-token decode began
taking the verifier's kernels. The pinned trajectory fell from 116/128 rank sum
142 to **104/128 rank sum 160**. Bisected to `5f231f8` and fixed by restoring
both calls.

The structural check was blind to this by construction. At 24 and 16 rows those
routes never fire on prompt-width batches, so a **prefill** trace really is
identical -- 133 kernel groups and 3,926 dispatches, identical in name, grid,
workgroup size and count -- while the one affected shape, single-row decode, was
never traced. Two lessons, both cheap: **when removing a tuning switch, check
whether its value passes through a helper before use**, and **trace the shape the
switch actually governs, not the one you are optimizing**.
`nix build .#checks.x86_64-linux.pr` stays green at 81/81 either way -- it sets
`BUILD_TESTING=OFF` and never builds this gate.

Restoring the gate costs nothing. Interleaved against the unfixed head, both
arms `nix build` Release: `pp4096` 478.1 against 480.2 tok/s, `tg16` 16.65
against 16.69, `tg128` 16.63 against 16.63, `pp512` 240.5 against 244.9 -- every
one inside the +-5% cross-binary noise, and the trajectory back at 116/128, 142,
3 with the 273-token envelope at `rmse` 0.41, `max_error` 1.95. The always-on
small-batch routes were buying no throughput whatever; they only changed the
arithmetic. Timing agreed once the thermal confound was removed: a first A/B put the
cleaned build 10% down, but both pairs had run the old binary first, and
reversing the order within each pair gave 450.2 against 452.9 tok/s. **Alternate
the order within pairs, not just between them** -- ordering alone is worth 10% on
this APU.

What is deliberately kept, none of which changes behaviour:

- Seven observability switches: `..._GRAPH_TOKEN_PROFILE`,
  `..._GRAPH_PREFILL_PROFILE`, `..._GRAPH_PREFILL_SPLIT_PROFILE`,
  `..._LAYER_STAGE_PROFILE`, `..._DECODE_STAGE_PROFILE`,
  `..._INDEXER_STAGE_PROFILE`, `..._Q_STAGE_PROFILE`, plus
  `GUFO_DEEPSEEK_DSPARK_TRACE`. These only print timings, and the token profile
  is what settled the sampling question above.
- `DS4_LOCK_FILE`, an operational path override for the single-instance lock;
  hardcoding `/tmp/ds4.lock` would leave no escape in a container.
- `GGML_CUDA_DISABLE_GRAPHS` in `mmq/common.cuh`, which is verbatim upstream
  llama.cpp and pinned by `mmq/VENDOR.md`.

### The hyper-connection pre-block chain, fused into one pass (September 4, 2026)

The "exhausted" verdict above held for the kernels it examined. It missed a
*chain*: every layer half ran norm -> 24-wide mix projection -> Sinkhorn split ->
weighted sum as three dispatches over the same 16,384-wide hyper-connection row,
and re-read that row at every step. At a 4,096-token chunk, per call:

| Pass | Bytes | Measured |
| --- | ---: | ---: |
| `rms_norm_plain_regs<64>`, F32 row in, F16 mirror out | 268 MiB read, 134 MiB written | 2.01 ms (199 GB/s) |
| `Cijk_..._MT32x32`, the 24-column projection | 134 MiB read | 1.24 ms (**108 GB/s**) |
| `hc_split_weighted_sum_fused` | 268 MiB read, 67 MiB written | 1.58 ms (187 GB/s) |

The projection is the tell: 24 output columns against a 16,384-deep K gives
Tensile an `MT32x32` tile and 256 workgroups of 128 threads for the whole chunk,
which is under half the DRAM ceiling because there is not enough of the grid to
cover latency, not because there are too many bytes.

`hc4_norm_mix_split_weighted_sum_kernel` holds the row in registers instead. A
lane owns sixteen *contiguous* embedding columns in all four streams -- 64
floats, four `float4` loads -- and that choice is what makes the weighted sum
thread-local: the four values a column needs are four of the lane's own
registers, so no stage after the first read touches memory again. The projection
is linear in the row, so `rsqrt` comes out of the sum (`mix[j] = nscale * sum_k
w[j][k] x[k]`), which also skips the F16 mirror's rounding entirely. Four tokens
share a workgroup so the 786 KiB projection weight -- the one operand still
re-read, and L2-resident across the grid -- is amortized over four rows.

Both reductions are a wave shuffle plus one combine over the eight wave totals
rather than a 256-lane LDS tree. That is not cosmetic: the workgroup is 32 waves
wide, so eight barriers per reduction cost more than the pass being replaced. The
first version used LDS trees and ran 263.1 ms; the same kernel with shuffles runs
198.5 ms.

| | Separate | Fused |
| --- | ---: | ---: |
| dispatches per layer half | 3 | 1 |
| kernel time for the chain | 444.7 ms | **198.5 ms** |
| dispatches in the chunk | 3,842 | 3,670 |
| profile | 8,890.6 ms | 8,259.5 ms |

The chain accounts for **-246 ms, or -2.8% of the profile**; the profile sums
carry more spread than that between runs, so the mechanism is the 246 ms. Three
interleaved rounds of the two release binaries on one host read

| round | `a8158c7` | fused | delta |
| --- | ---: | ---: | ---: |
| 1 | 479.64 | 495.29 | +3.3% |
| 2 | 487.77 | 493.50 | +1.2% |
| 3 | 482.87 | 499.58 | +3.5% |
| mean | **483.4** | **496.1** | **+2.6%** |

152 VGPRs, no scratch, under 8 KiB of shared memory. Two and eight tokens per
workgroup measured 494.0 and 493.1 tok/s against four's 498.6: the weight
amortization and the register room for a second resident workgroup trade off
almost exactly, and four is the flat optimum.

It holds across the whole depth curve, one process per arm, arms alternated:

| prompt | `a8158c7` | fused | delta |
| ---: | ---: | ---: | ---: |
| 4,096 | 439.25 | 456.01 | +3.8% |
| 8,192 | 478.16 | **509.06** | +6.5% |
| 16,384 | 489.70 | 500.41 | +2.2% |
| 32,768 | 476.40 | 481.50 | +1.1% |
| 65,536 | 436.03 | 440.67 | +1.1% |

The saving is per chunk and fixed, so it dilutes as the indexer's O(context)
scoring grows: +3.8% at 4K down to +1.1% at 64K. The peak moves from 16K to 8K
because the chain was a larger share of the shorter chunks.

Not bit-identical, and scoped to `n_tokens >= 128` for that reason: `mix`, and
therefore `split`, reassociates the K reduction into a shuffle tree where Tensile
used a tile. The gate is unmoved and the envelope barely is:

| | `a8158c7` | Fused | Fused, plus the block-norm fold |
| --- | --- | --- | --- |
| pinned trajectory | 116/128, 142, 3 | 116/128, 142, 3 | 116/128, 142, 3 |
| 273-token `rmse` | 0.41 | **0.42** | 0.48 |
| `max_error` | 1.95 | **2.00** | 2.04 |
| `cosine` / `top1_match` / rank | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 |

The pinned trajectory cannot move: its prompt is about twenty tokens and its 128
steps are one row each, so a route scoped to 128 rows never fires there. All three
readings are the same `build/gpu-test` binary, and the whole
`deepseek_v4_flash_engine_test` suite exits 0.

**Folding the following weighted block norm in as well was measured and
rejected.** `rms_norm_weight_kernel` reads the row this kernel produces back
twice and the values are already in registers, so the fold removes 134 MiB per
call and a dispatch, and it is worth 48 ms. It also takes `rmse` 0.42 to **0.48**
-- the same value that got `QK_WG = 4` rejected, and no longer clearly better
than the `16d5e30` baseline's 0.478146. The cost is entirely the reduction: the
separate kernel sums each lane's strided pair into a 256-lane LDS tree, this one
sums sixteen contiguous columns into a shuffle tree, and reproducing the former
from the latter's register layout would mean redistributing the row through
shared memory. 48 ms is not worth the margin, so the separate norm pass stays.

**Reduction order matters where a norm scale is involved and not where a
projection is.** The same reassociation cost 0.01 rmse on `mix` and 0.06 on the
block norm, because the envelope measures agreement with the sequential path,
which still uses the old order. Price the two separately before fusing.

#### DSpark is untouched, and the long-context curve only improves

The ten-category corpus, both arms alternated on one host:

| | `a8158c7` | fused |
|---|---:|---:|
| exact vs AR | 3/10 | 3/10 |
| support acceptance | 60.7% | **60.7%** |
| positional agreement | 74.4% | **74.4%** |
| full-block rate | 38.0% | **38.0%** |
| attempts / skipped | 163 / 541 | **163 / 541** |
| AR / speculative tok/s | 16.15 / 17.85 | 16.12 / 17.40 |
| aggregate / median speedup | 1.11x / 1.04x | 1.08x / 0.98x |

**Every behavioural column is identical, per category as well as in aggregate**,
and for a structural reason rather than by luck: these prompts are chat framings of
a few dozen tokens, so the prompt prefill never reaches the 128-row scope and the
fused route does not run at all in this suite. The speedup columns move by the
documented +-0.15x of single 128-token generations with no acceptance change
behind them, which the section above already says to read as noise.

The stable throughput target is the one case with enough blocks to be signal, and
it improves: `repetition_sequence` in raw mode goes **27.48 to 29.00 tok/s, 1.67x
to 1.77x**, at 100% acceptance, 100% positional, 100% full blocks and 22 attempts
on both arms.

**A note on the DSpark table two sections up.** Its "before"/"after" columns read
61.2% acceptance and 164/505 attempts/skips against the 2026-09-01 record's 60.7%
and 163/541. Both were measured with the `5f231f8` small-batch regression live, so
that difference was the regression, not the indexer change: with `08584bd` in
place the corpus returns to 60.7% and 163/541 exactly. The attribution in that
section -- that the indexer change leaves the draft and verify streams alone --
still holds, since both of its arms carried the same regression.

### Two more prefill levers, both falsified (September 4, 2026)

| Lever | Predicted | Measured |
| --- | --- | --- |
| Route the routed-expert Q2_K **down** projection through the vendored MMQ tier, the way gate/up already goes, reusing the same `ids_dst` map and the SwiGLU epilogue's `mid` | the native wide WMMA kernel sustains 12.2 TFLOP/s against gate/up MMQ's 18.7 on comparable arithmetic, so about -500 ms | **19% slower at every tile width.** 461.4 tok/s native against 370.0 / 374.4 / 370.0 / 368.8 at `mmq_x` 16 / 24 / 32 / 40. Q2_K's MMQ tile carries a second accumulator set for its per-block minima: at the width the selector picks it needs **256 VGPRs plus 2,768 bytes of scratch** and 37,696 bytes of shared memory -- one workgroup per CU and a spilled inner loop -- against IQ2_XXS's 176 VGPRs, no scratch and 31,552 bytes. Capping the width removes the spill and the column tiles then multiply; neither end wins. The composition itself is sound and cheap (`swiglu_epilogue` and `fused_down` now share one `mid`, no second `mm_ids_helper`), so the finding is about the Q2_K tile, not the plumbing |
| Compact the cold-expert Q2 down grid, which is sized by the whole 256-entry expert table while the router skew leaves about 83 experts cold at prompt width | 65,536 workgroups down to 21,248, so most of 138 ms | **7 ms.** 138.3 to 131.1 ms at kernel level with `grid.y` 256 to 83. This is the third independent confirmation that empty workgroups are free on this GPU (see "Nothing in this prefill is bandwidth bound"): the 131 ms is real work, 83 experts' full 4,096 x 2,048 weight streams for a few hundred assignments, at about 1.3 TFLOP/s. Reverted -- it also added a synchronous host copy per layer for less than its own noise |
| `PAIR_TILE` 8 instead of 4 on that same cold route, so a five-to-seven-row bucket stops paying a second full weight dequantization pass | about -30% of 133 ms | **+64%**, 132.6 to 218.0 ms. The mid staging loop covers the whole tile whatever the bucket holds, so eight stages 2,048 floats per K block to use three of them, and that costs more than the weight pass it saves. The default stays 4 |

### The conversational prompt widths, which were at half speed (September 5, 2026)

Every prefill number above is a 4,096-token chunk. A chat turn is not one: the
prefix cache already holds the conversation, so the chunk is whatever is new --
a few hundred tokens. Measured in one warm process on `9719ae2`, with a
4,096-token point first so the one-time weight mirrors and plan selection land
outside the rows being compared:

| prompt | tok/s | of the 4,096 rate |
| ---: | ---: | ---: |
| 128 | 142.69 | 27% |
| 256 | 211.94 | 40% |
| 512 | 293.73 | 55% |
| 768 | 343.98 | 65% |
| 1,024 | 373.62 | 70% |
| 1,536 | 407.52 | 77% |
| 2,048 | 494.19 | 93% |
| 4,096 | 530.33 | 100% |

`GUFO_DEEPSEEK_ROCM_LAYER_STAGE_PROFILE` puts essentially all of the deficit in
two stages. Per token, 512 against 4,096 (these readings carry a per-stage
synchronize, so read the ratios and not the absolute values):

| stage | 512 | 4,096 | ratio | excess per 512-token chunk |
| --- | ---: | ---: | ---: | ---: |
| `attn inv_rope`, the attention output projection | 1,298 us | 298 us | **4.36x** | 512 ms |
| `ffn routed_moe` | 1,904 us | 785 us | **2.43x** | 573 ms |
| the other fifteen stages, summed | 972 us | 823 us | 1.18x | 76 ms |

The two own 1,085 ms of a 2,137 ms chunk. Nothing else is worth more than 40 ms,
and four stages -- `attention`, both norms, `attn hc_post` -- are *cheaper* per
token at 512, which is the sparse-attention window working as designed.

#### The routed column tile was sized by the chunk, not by the bucket

`mul_mat_q` is 39.7% of a 512-token chunk (729.91 ms of 1,840.76 ms of kernel
time), of which 599.76 ms over 86 dispatches is the routed IQ2 gate/up pair. Its
grid is `(nty, ntx, 256)`, and `ncols_grid_max` already bounds `ntx` by the
largest expert bucket -- but `mmq_x`, the width each surviving column tile
actually executes, is still chosen from `ncols_max`, the chunk width. That yields
the 80-column tile at every width from 256 up.

A tile executes all of its columns whether the bucket fills them or not, so the
cost is `sum_e ceil(c_e / w) * w` column-slots plus one weight-panel reload per
tile, and what decides it is the whole distribution rather than its maximum:

| chunk | live experts per layer | mean bucket |
| ---: | ---: | ---: |
| 256 | 116 | 13 |
| 512 | 131 | 23 |
| 1,024 | 147 | 42 |
| 2,048 | 161 | 76 |
| 4,096 | 170 | 145 |

and the router skew is large: at 512 tokens the largest bucket is 358 to 483 of
3,072 pairs against a mean of 23, and at 4,096 it reaches 3,496 of 24,576.

At 4,096 tokens `w = 80` is nearly full. At 512 it spends about 85% of its
matrix-core issue on padding -- and the router skew means the maximum bucket
cannot tell you that. The host already copies the per-expert counts back to build
the Q2 down hot list, so `ds4_mmq_set_routed_tile_cols` now takes the width that
minimizes the cost above.

The change is bit-identical by construction: the macro tile only partitions
output columns, and each element still walks K once in the same order.

A forced-width sweep, one warm process per width, says the model is worth having
below about a thousand rows and nothing above it:

| width | pp256 | pp512 | pp1024 | pp2048 |
| ---: | ---: | ---: | ---: | ---: |
| 16 | **261.47** | **354.44** | 426.59 | 472.36 |
| 32 | 243.66 | 343.55 | 420.91 | 476.07 |
| 48 | 240.51 | 346.71 | **435.80** | **499.01** |
| 64 | 228.26 | 332.74 | 429.85 | 496.93 |
| 80 | 219.74 | 320.62 | 420.49 | 493.31 |
| default rule | 228.80 | 321.43 | 421.62 | 493.80 |

The default gives up 14% at 256 and 10% at 512, and is within 1.1% of the best
measured width at 1,024 and 2,048. The reload constant was then fitted rather
than assumed: the real per-expert counts were dumped for all 43 layers at each of
those four widths and scored against this sweep. `P = 16` is the single best
constant -- 0.85% mean and 3.4% worst-case shortfall against a per-width oracle,
where `P = 8` and `P = 32` both give 1.7% and `P = 48` gives 2.6%.

**No constant reproduces the whole curve.** The optimum is 16 columns at 512 and
48 at 1,024, and any cost of the form `a * (column-slots) + b * (tiles)` collapses
to `tiles * (w + b/a)`, which cannot separate those two: 16 beats 48 at 512 only
for `P < 29.5`, 48 beats 32 at 1,024 only for `P > 33.7`. So the model runs only
where the sweep validates it, under `DS4_ROCM_ROUTED_TILE_MODEL_ROWS` (1,024).
Left on the table: a fixed 48-column tile is worth +3.4% at 1,024 and +1.1% at
2,048 and is the obvious next step, not taken here only because the same sweep's
4,096 point is a cold-cache reading that cannot settle the wide end.

#### The attention output-B projection had fallen off its own kernel

`DS4_WMMA_GEMM_MIN_BLOCKS` is 240 tiles, "six workgroups per CU", and it was
calibrated to keep the 256x128 tile off the small-`m` `opA=T` projections where 32
to 64 workgroups lose to Tensile. At `m=4096 n=chunk k=8192` -- the one shape the
kernel was written for -- 16 `m`-tiles means the floor is not met until
`n >= 1,920`, so every chunk under 2,048 rows took the fallback. And the fallback
for this shape is not Tensile's good `opA=T` tile; it is rocBLAS on `opA=N`, which
holds 5.6 to 6.6 TFLOP/s at every width while both macro tiles reach 16 to 20.

`tools/bench/dsv4_attn_out_gemm_bench.hip`, three rotating operand copies so the
32 MB MALL cannot hide a re-read the production call pays cold:

| n | rocBLAS `opA=N` | 256x128 W4x2 | 128x128 W2x2 SWZ8 | 64x64 W1x2 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1.533 ms (5.60) | 0.934 (9.20) | 0.653 (13.16) | **0.600 (14.32)** |
| 256 | 2.879 (5.97) | 1.018 (16.87) | **0.949 (18.11)** | 1.198 (14.34) |
| 512 | 5.202 (6.61) | 2.094 (16.41) | **1.887 (18.20)** | 2.352 (14.61) |
| 1,024 | 10.521 (6.53) | **3.505 (19.61)** | 4.042 (17.00) | 6.508 (10.56) |

TFLOP/s in parentheses. Two things fell out of the sweep. **All 26 WMMA
geometries agree bit for bit** -- `maxdiff 0` throughout -- so the macro tile is a
scheduling choice with no numerical consequence, and only leaving rocBLAS changes
anything at all (the two disagree by 3e-4 at these widths). And the crossover is
sharp: 128x128 below n=1,024, 256x128 above.

The remaining obstacle was that a real chunk width is not a multiple of the tile.
The caller owns both the staged activation and the result, so it stages them for
the width rounded up to 128, zeroes the activation pad, and lets the extra result
columns be computed from zeros and never read -- no guard in the K loop, and the
real columns stay bit-identical to the aligned case. It is held to
`DS4_ROCM_WIDE_PREFILL_ROWS` like every other accelerated prefill route, which
keeps the pinned trajectory's own twenty-token prefill on the library its envelope
was measured against.

Per 512-token chunk, at kernel level: **229.87 ms of rocBLAS becomes 79.73 ms** of
`ds4_gemm_f16_wmma_kernel<128,128,32,2,2,8>`, 1.854 ms per layer against the
harness's 1.887.

#### Result

Two arms of `nix build`, alternated within each pair so ordering alone cannot
decide it, one warm process per arm, a 4,096-token point first:

| prompt | `9719ae2` | this change | delta |
| ---: | ---: | ---: | ---: |
| 128 | 143.04 | 180.38 | **+26.1%** |
| 256 | 212.58 | 260.91 | **+22.7%** |
| 512 | 293.84 | 354.77 | **+20.7%** |
| 768 | 343.15 | 391.80 | **+14.2%** |
| 1,024 | 374.01 | 421.63 | **+12.7%** |
| 1,536 | 405.24 | 467.96 | **+15.5%** |
| 2,048 | 494.35 | 493.66 | -0.1% |
| 4,096 | 529.49 | 529.54 | +0.0% |
| `tg16` | 16.90 | 16.95 | +0.3% |

Both mechanisms are scoped away from single-row work, so `tg16` is the control
rather than a result: decode takes the same kernels it did before.

Kernel time for a 512-token chunk falls from 1,840.76 ms to 1,527.85 ms, and the
two mechanisms account for all of it: routed gate/up 599.76 -> 447.4 ms, output B
229.87 -> 79.73 ms.

#### What it is worth on the real server

`tools/serving/gufo-serving-bench.py` against `gufo serve`, a 223-token prompt,
32 greedy output tokens, one warmup and three measured repetitions per
concurrency level, arms alternated within each pair. This prompt-only comparison
predates the native session-batch route above; physical width was pinned to one
so C=2 and C=4 are the fair round-robin serial fallback and their queueing is
included:

| | `9719ae2` | this change | delta |
| --- | ---: | ---: | ---: |
| prefill tok/s | 155.4 | 205.2 | **+32.1%** |
| TTFT p50, C=1 | 1,429.5 ms | 1,087.3 ms | **-23.9%** |
| TTFT p50, C=2 | 3,088.6 ms | 2,562.2 ms | **-17.0%** |
| TTFT p50, C=4 | 6,401.7 ms | 5,516.0 ms | **-13.8%** |
| whole request p50, C=1 | 3,303.4 ms | 2,957.3 ms | -10.5% |
| aggregate tok/s, C=1 | 76.9 | 86.2 | **+12.1%** |
| aggregate tok/s, C=2 | 76.8 | 86.3 | **+12.4%** |
| aggregate tok/s, C=4 | 76.7 | 86.3 | **+12.5%** |
| decode inter-token p50 | 60.40 ms | 60.29 ms | -0.2% |

The aggregate gain is the same at every concurrency because the thing that got
faster is the per-request prefill each of them pays, and the inter-token latency
is the control: it does not move, which is the check that neither route reached
single-row decode.

#### The envelope does not move

`build/gpu-test/deepseek_v4_flash_engine_test`, same binary and model:

| | floor | `16d5e30` baseline | this change |
| --- | --- | --- | --- |
| pinned trajectory top-1 | >= 116/128 | 116/128 | **116/128** |
| aggregate rank | <= 142 | 142 | **142** |
| worst rank | <= 3 | 3 | **3** |
| 273-token `rmse` | <= 1.12 | 0.478146 | **0.47** |
| `cosine` | >= 0.979 | 0.99617 | **1.00** |
| `max_error` | <= 5.0 | 2.39995 | **2.13** |
| sequential choice rank | <= 3 | 1 | **1** |

The pinned trajectory cannot move: its prompt is about twenty tokens and its 128
steps are one row each, so neither route fires there. The 273-token comparison
does exercise both, and lands below the immutable pre-optimization `rmse`.
`nix build .#checks.x86_64-linux.pr` is green.

#### One falsified item, re-confirming a recorded one

Compacting the cold-expert Q2 down grid -- `grid.y` from the whole 256-entry
expert table to the populated experts -- was measured again at 512 tokens, where
about 130 of 256 experts are live and 83 of them land on that route: **156.15 ms
before against 145.39 ms in a build that also carries both changes above, which
do not touch this kernel's work**, with `grid.y` 256 to 83. That reproduces the
138.3 to 131.1 ms recorded for the same experiment at 4,096 tokens, at a second
chunk width and a 3x grid reduction, and it stays reverted for the same reason:
0.7% of the chunk, inside the noise of an end-to-end pair, for a host copy per
layer. **Empty workgroups really are free on this GPU.**

### What is left, and why it was not attempted

The attention producer writes its 32,768-wide F32 output and
`attention_pack_group_heads_rope_f16_vec4_kernel` immediately reads all of it
back to transpose `(token, group)`, apply the inverse rope and narrow to F16:
537 MiB written, 537 MiB read, 268 MiB written per layer, of which the fused form
would keep only the last. That is about **190 ms**, the largest single chain left.

It does not fit the producer's epilogue. The accumulators leave WMMA through
`rocwmma::store_matrix_sync` with one leading dimension, and the packed
destination is `[group][token][group_dim]`, so a 16-row fragment spanning two
groups needs two different strides -- the fragment would have to land in shared
memory first and be rewritten per head. That kernel already holds 57,992 of
65,536 bytes of shared memory, which is exactly what pins it to one workgroup per
CU, so the staging buffer would come out of the operand tiles that make it fast.
Worth revisiting only together with a rewrite of that producer's LDS budget.

## To Do

- Add model-owned thinking/reasoning mode and effort controls; the current chat
  template intentionally uses the no-thinking path.
- Retain and compare the first four-case `gufo eval` HTTP regression baseline;
  Pi/coding-agent evaluation remains deferred under #153.
- Profile and optimize the model-owned gfx1151 kernels under #155.
- Continue improving general-workload DSpark acceptance and verifier cost under
  #156 and #222. The retained fast route exceeds the 24--25 tok/s target on
  sustained high-acceptance decoding; see "DSpark speculative decoding" below.
- Profile sharing Q8 activation loads between the concurrent IQ2 gate/up dot
  products while preserving each request's arithmetic. Gate/up is now the
  largest repeated routed kernel; four-output-row Q2 widening was rejected for
  spilling, and C6 already uses the measured three-tile down route.
- Add model-owned offline calibration/imatrix tooling only when a new
  quantization recipe requires it.
- Continue restart-safe SSD state reuse through the serving milestones.

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
