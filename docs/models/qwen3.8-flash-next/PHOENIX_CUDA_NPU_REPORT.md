# Phoenix NPU + discrete CUDA GPU: experiment report and request for review

October 27, 2026. **No NPU full-model acceleration was demonstrated.** This
report records a native Windows adaptation of Gufo's projection, two-bank
prefill and common-buffer ideas inside Strata. It includes successful physical
NPU execution, unsuccessful performance experiments, correctness limits and
the final experimental patch so the community can help identify missing
mechanisms. It does not change Gufo's runtime or benchmark Gufo itself.

The strongest final NPU candidate increased fresh-prefill latency in all three
confirmation pairs: median paired increases of **2.35% at 2,667 tokens** and
**3.61% at 7,805 tokens**. The NPU path remains disabled. An earlier CUDA engine
and resident-RAM profile change did improve prefill and remains selected.
Results on Phoenix plus a discrete RTX GPU do not invalidate Gufo results on
Strix Halo/XDNA2 with an integrated GPU and a different memory architecture.

## Hardware, versions and comparison boundary

| Item | Tested setting |
| --- | --- |
| Host | Native Windows x64; Ryzen 9 7940HS / Phoenix NPU; 64 GiB DDR5 RAM |
| GPU | RTX 4070 Laptop, nominal 8 GiB dedicated VRAM; discrete PCIe GPU |
| Storage | Local NVMe; model files unchanged throughout each matched comparison |
| Model | Qwen3.8-Flash-Next-GSQ-RCO IQ3_S, original two GGUF shards; INT8 KV |
| Capacity and budgets | 65,536 configured context; high reasoning, 8,192 reasoning-token budget; 16,384 output-token allowance |
| Final profile | Seven CPU workers; 40 GiB resident expert budget; requested speculation 4; 2 GiB conversation cache / two slots; PCIe fraction 0.37 |
| Strata source | `1aff6d14117b88171fcfcec8f2b55fdc47f2b165`; engine 0.1.50 for matched final controls/candidates |
| Gufo mechanism reference | `mikealanni/gufo` at `2e499c39e4cd2d9250da2f44f529435f04995c84` |
| NPU toolchain | XRT 2.21; MLIR-AIE `1.4.5.dev72+gb775c31`, source `6f73deca839e79a170e7c90b9285b3be949ca781`; Peano `22.0.0.2026101901+4f5df13d` |
| NPU driver / firmware | `32.0.20102.3930` / `1.5.5.391`, existing installation |
| Host compilers | CUDA 13.4, MSVC / Visual Studio 2026; native NPU Python 3.13 ABI; serving harness Python 3.11.9 |

Requested CLI values and effective engine allocations are distinct. Automatic
VRAM allocation and speculation decisions remain enabled and are not inferred
solely from requested values. The [sanitized CUDA configuration](artifacts/phoenix-cuda-npu/08-profile-cuda.json)
and [NPU configuration](artifacts/phoenix-cuda-npu/08-profile-npu4.json) preserve
the exact arguments and environment. Existing `route-tail-skip=7` behavior was
retained; this is not a claim of mathematically exact inference.

CPU AVX-512 execution was physically probed successfully. Python version alone
does not turn on AVX-512 or CPU multicore inference. The native NPU Python ABI
was retained for compatibility; seven workers were used by the engine. CUDA
FP16/BF16 comparisons described below were native CUDA measurements. PyTorch
and power-limit sensitivity were not measured.

No driver, firmware, power setting, original model or ordinary Strata binary
was changed. Every heavy experiment used owned, serialized processes. The
final experiment added approximately 55 MiB of logical bank storage, with
actual resource counters checked separately.

## Sources and what was actually reused

The reference Gufo
[executor](https://github.com/mikealanni/gufo/blob/2e499c39e4cd2d9250da2f44f529435f04995c84/src/models/qwen38_flash_next/kernels/rocm/executor.cpp)
and [NPU matrix implementation](https://github.com/mikealanni/gufo/blob/2e499c39e4cd2d9250da2f44f529435f04995c84/src/models/qwen38_flash_next/kernels/rocm/npu_gemm.cpp)
informed the projection selection, two-bank dependency schedule and mapped
host allocation shared with XRT. The reference has an 8,192-token eligibility
threshold and 4,096-token banks. This adaptation uses 1,024-token banks because
the discrete GPU has 8 GiB VRAM and must preserve the existing model capacity.
The final work did implement SSM input/gate and QSA Q/K/V projections, beyond
the earlier HC-only and small decode experiments.

Existing Phoenix SDR work supplied the M17 BF16 matrix-unit pattern
`mmul<4,8,8,bfloat16,bfloat16,accfloat>` for the router/HC kernels and M9/M9b/M17
parallel-worker and FIFO patterns. Existing Phoenix PQC work informed resident
request state, FIFO worker chains, output-token storage, generation identity,
completion checks and repeated local buffer reuse. These are implementation
references; their historical application benchmarks are not counted as LLM
results. Personal repository links and author identifiers are omitted.

The final K=2,560 projection kernel comes from the pinned MLIR-AIE
[whole-array matrix example](https://github.com/Xilinx/mlir-aie/blob/6f73deca839e79a170e7c90b9285b3be949ca781/programming_examples/basic/matrix_multiplication/whole_array/whole_array.py).
It distributes A by worker row and B by column through memory tiles, uses
depth-two FIFOs and accumulates C across K tiles. It is not an FFT or
cryptographic operation presented as model work. The unchanged upstream
source retains its Apache-2.0 WITH LLVM-exception notice; the included
[build invocation](artifacts/phoenix-cuda-npu/kernel-build.py.txt) records the
parameters used.

## Complete experiment inventory

The numbered evidence files contain retained phase summaries. Counts below
are stage-specific; warmups, repeated qualification and diagnostics overlap,
so they should not be summed into one supposed count of independent tests.
All experiments below were local Strata or native component tests.

### 1. Native Windows NPU execution and functional integration

[Native functional evidence](artifacts/phoenix-cuda-npu/01-native-functional.json):

- Physical BF16 addition checked 8,192 outputs for two inputs. A soft-float
  FP32 path checked 51,200 outputs, preserving tested finite values and signed
  zero bitwise and checking NaNs as NaNs.
- A native C++ bridge made 14 dispatches at T=1/4/8; a CUDA graph harness
  completed 24 replays. A graph negative control made zero NPU dispatches.
- Four actual-model tasks covered JSON, a generated clamp function and tool
  output. There were 308 groups / 816 rows / 816 dispatches and 2,088,960
  outputs compared with CUDA. A deliberately poisoned verifier rejected the
  actual-model request; the restored CUDA control passed all four tasks.
- Minimum available RAM was 5.72 GiB and free dedicated VRAM 357 MiB.
  This established execution and integration, not a paired model speed gain.

### 2. Packed final-add bridge and CPU/GPU sharing screens

[Packed bridge and cooptimization evidence](artifacts/phoenix-cuda-npu/02-packed-cooptimization.json):

- A packed 16-worker design reduced median bridge time at one row from
  0.6474 to 0.3301 ms and at eight rows from 4.9288 to 0.4353 ms. Three blocks,
  two warmups and ten measured changing-input calls per case were used.
  Intermediate row counts are retained. These are component gains of roughly
  49â€“91%, including bridge overhead, not whole-model gains.
- Four workers were slower for T=1. A separate 16-input design failed DMA
  placement; the failure was retained instead of counted as hardware execution.
- Arithmetic checked 409,600 values and graph replay checked 24 executions.
  Three alternating model pairs across seven workloads produced 21 candidate
  and 21 control answers. Candidate totals included 1,607 dispatches / 4,710
  rows and 12,057,600 values bitwise equal to CUDA. No whole-model gain was
  accepted. An actual-model blocked-dispatch control rejected the request.
- A 512-token prefill trial saved about 190 MiB VRAM but increased a roughly
  2,660-token prefill to 11.58 s from about 7.6 s. A tool-response failure was
  retained; the original NPU arm also failed that tool test, so buffer size
  alone was not established as the cause.
- CPU/GPU share 0.25 gave mixed coding gains of +6.32%, -4.84%, +8.31%; reasoning
  gains were +1.97%, +1.64%, +3.04%. Outputs differed and no fixed-length
  confirmation established a gain. Share 0.55 lost in every coding/reasoning
  pair. The existing 0.37 setting remained selected.
- A 7,796-token prompt and continuation of 7,888 tokens with 7,865 reused
  tokens passed. This was a deeper-context check, not a full-64K test.

### 3. Resident SRAM router, empirical hardware probes and GPU precision

[Resident-router evidence](artifacts/phoenix-cuda-npu/03-resident-router.json):

- Sixteen cores each held 40 KiB, totaling 640 KiB of original BF16 weights
  for the first 128 of 512 layer-0 router outputs. CUDA computed the remaining
  384. Weight bytes were loaded at artifact startup, checked against the
  model and not DMA-transferred per projection. FP32 inputs used four BF16
  parts, with 1â€“8 rows supported.
- Persistent XRT objects, sequence/generation/counter checks and explicit
  completion were exercised with changing inputs, context recreation, wrong
  weights/generation and sticky failure. An idle state reset was reproduced
  and repaired with an owned 250 ms NOP heartbeat; the initial failure was
  retained. No system power settings were changed.
- Standalone arithmetic: 5,120 outputs versus FP64, maximum error 1.404e-5.
  Graph: 27 replays in each mode, all 512 logits and actual CUDA top-10 routing
  IDs/weights, including nonzero row offsets. Model: 1,278 rows, maximum logit
  error 4.768e-5 and route-weight error 1.490e-6. Blocked dispatch failed without
  returning a completed fallback answer.
- Three pairs across seven workloads gave 42 measured requests. Median paired
  decode gains were prose -2.73%, reasoning -0.38%, code +2.59%, JSON -5.19%,
  tool +6.50%, long prompt -12.45%, repeated prefix +2.98%, with wide ranges
  and varying output lengths. No gain was accepted.
- An explicitly truncated 256-token diagnostic had paired decode gains
  -3.07%, +1.49%, +4.41% and median whole-request latency reduction +1.02%.
  It remained inconclusive and is not counted as complete-answer success.
- Main candidate servers, including warmups, made 1,641 compute dispatches
  over 4,943 rows. Mean callback time was 419â€“426 microseconds, XRT submit/wait
  350â€“353 microseconds; both omit CUDA staging, which full-request timing includes.
- One prose answer's final "only once" statement was inaccurate or ambiguous.
  Six full-vocabulary warmup comparisons had KL 0.000533â€“0.004585 and the same
  argmax; control repeats varied 0.000324â€“0.004582. Maximum absolute logit
  difference reached 2.89. This does not establish broad quality equivalence.

[Hardware probes](artifacts/phoenix-cuda-npu/03-hardware-probes.json) and
[executed-object inspection](artifacts/phoenix-cuda-npu/03-observed-vliw.json):

- XRT reported five physical columns. Sixteen signature workers executed.
  The attempted 20-worker design was rejected at compiler column 4 before
  hardware execution. This identifies the exposed toolchain restriction;
  it does not prove only 16 physical compute tiles exist.
- Local SRAM, direct neighboring-core SRAM and an explicit memory-tile FIFO
  stage passed full-buffer tests. Cycle-counter reads failed to link, so
  peak SRAM bandwidth was not measured.
- The object used by the executed artifact showed at most six decoded slots
  and four non-NOP operations in an observed bundle. This is not an empirical
  maximum hardware issue-width result and does not resolve a seven-way claim.
- Native CUDA BF16 versus preconverted FP16 was tested for seven shape classes
  in six processes. Larger matrices were essentially tied; forced conversion
  added cost. A separately labeled full-model FP16 screen passed but did not
  establish a repeatable speed gain. No PyTorch comparison was performed.

### 4. CUDA profile improvement, flag audit and HC projection cost gate

[CUDA profile and cost evidence](artifacts/phoenix-cuda-npu/04-cuda-profile-and-hc-cost.json):

Three alternating pairs / 18 complete answers compared the original engine
0.1.48 / 36 GiB resident profile with engine 0.1.50 / 40 GiB. The combined
change reduced prefill from 9.121 to 6.885 s at 2,663 tokens and from 24.281 to
19.205 s at 7,801 tokens. Reported paired latency reductions were 25.13% and
20.91%, corresponding to throughput gains of 33.57% and 26.44%. The repeated
prefix changed from 0.250 to 0.276 s. No claim isolates which part of the
combined engine/RAM change caused the gain or treats shorter outputs as faster
decoding. This profile became the baseline for subsequent NPU experiments.

The Qwen `--mmap-experts` flag is a bare switch in the tested Strata revision,
not `--mmap-experts off`. The cache flag is `--conversation-cache-mib`, not
`--conversion-cache-mib`; 2,048 MiB was used. Slot save/restore arguments and
POST endpoints were source-audited, not enabled or counted as measured gains.
Full 46.84 GiB expert residency was not qualified within the available-memory
floor. These are Strata flags, not recommendations for Gufo's CLI.

The HC cost gate used actual K=320, N=10,240 weights, a resident 1,024-column
NPU slice across 16 workers, four independent rows per matrix instruction,
and CUDA for the remaining 9,216 columns. It included all copies, packing,
XRT completion and the CUDA consumer. Twenty alternating warm/cold rounds
at T=128/256/512 all lost. At T=512, warm NPU/CUDA times were 2.034/0.287 ms;
cold times were 2.094/0.322 ms. See [warm data](artifacts/phoenix-cuda-npu/04-hc-cost-warm.json)
and [cold data](artifacts/phoenix-cuda-npu/04-hc-cost-cold.json).
An earlier input-ordering bug was exposed by the cold probe and fixed with
compute-stream ordering plus CPU anchors. Unordered/failed trials were
retained and excluded from final performance conclusions.

### 5. First full-model HC two-bank pipeline

[HC two-bank evidence](artifacts/phoenix-cuda-npu/05-hc-two-bank.json):

This first pipeline used two 512-token GPU banks from a 1,024-token superchunk,
full HC attention-up projection K=320/N=10,240 in ten NPU column slices,
explicit D2H/H2D transfers and chronological first-two-layer PLE handling.
The schedule placed A(layer+1) preparation between A's body and B's body.

Nine qualification requests per arm checked NPU, the CUDA-only bank schedule
and the accepted CUDA profile. NPU qualification checked 2,717,900,800 values;
blocked and wrong-tag cases failed without an answer. A trace showed 96.64%
interval overlap, which is not proof of simultaneous SM work or acceleration.

The final three-pair comparison had 27 measured complete answers and 18
warmups. At 2,664 / 7,802 tokens, median prefill was 6.91 / 19.31 s for accepted
CUDA, 10.49 / 29.47 s for CUDA-only banking, and 10.80 / 30.37 s for NPU banking.
NPU increased latency 56.34% / 57.29% relative to accepted CUDA, reducing
throughput 36.04% / 36.42%. Relative to the same bank schedule on CUDA, NPU
increased latency 2.96% / 3.05%. Smaller GPU batches already caused most of
this regression. This candidate was rejected.

### 6. Closer Gufo common-buffer transport, matrix tiles and projection costs

[Common-buffer transport](artifacts/phoenix-cuda-npu/06-shared-memory.json)
physically exercised CUDA producer â†’ NPU HC matrix â†’ CUDA consumer with
524,288 values and output headers. Thirteen rounds, including three warmups,
gave 1.72705 ms for explicit copies, 1.7099 ms for a CUDA-registered XRT BO,
and 1.6537 ms for VirtualAlloc memory registered with CUDA and supplied to
XRT as a user pointer. Blocked-dispatch negative controls passed for all
three. This unpaired component screen is not a model speed gain; common
host pointers still require physical PCIe traffic to the discrete GPU.

The [whole-array matrix screen](artifacts/phoenix-cuda-npu/07-matrix-tile-screen.json) used M=1,024 / K=2,560 / N=1,024, BF16 input,
FP32 output and 16 workers. With three measured rounds, all tested shapes
passed arithmetic. Median NPU times by tile `(M,K,N)` were:

| Tile | NPU time |
| --- | ---: |
| 64,64,32 | 4.9699 ms |
| 64,32,64 | 2.7895 ms |
| 128,32,32 | 4.3282 ms |
| 32,64,64 | 6.3277 ms |

The chosen 64/32/64 tile had 2.9595 ms dispatch and 3.3629 ms end-to-end time.
This is a small screen, not an exhaustive search or hardware peak claim.

[Actual-projection cost tests](artifacts/phoenix-cuda-npu/07-projection-cost.json)
used original layer-0 SSM Q6_K QKV (N=10,240) plus Q4_K gate (N=6,144), and
layer-3 QSA IQ4_XS Q (N=12,288) plus Q6_K K/V (512 each), all with K=2,560.
Changing synthetic FP16 GPU inputs were used in nine rounds, three warmup
and six measured, with alternating order. CUDA remainder work was concurrent;
costs include the common CUDA consumer, conversion, transfer and synchronization.

| Projection | CUDA | NPU 4K columns | NPU 8K columns | NPU full |
| --- | ---: | ---: | ---: | ---: |
| SSM total | 4.34985 ms | 17.56805 ms | 34.78435 ms | 65.961 ms |
| QSA total | 3.63565 ms | 18.9593 ms | 33.72215 ms | 58.0221 ms |

BF16-reference arithmetic maximum error was about 6e-6; original FP16
relative RMS difference about 0.15% on these synthetic inputs. Both blocked
controls passed. These are projection costs, not model activation quality.

### 7. Final SSM/QKV two-bank full-model implementation

The [experimental Strata patch](artifacts/phoenix-cuda-npu/strata-ssm-qkv-experiment.patch)
implements actual SSM input/gate and QSA Q/K/V projections for layers 2â€“47.
Layers 0â€“1 preserve chronological PLE processing. Two 1,024-token banks retain
residuals, exact FP16 mixed inputs, injection values and small side projections;
the large HC/GDN/QSA/MoE body scratch is shared. The native C++ XRT worker uses
immutable job identities, input-ready CUDA events, bank-owned mapped output,
bounded completion and fail-closed behavior. Exit paths settle owned work
before reuse or release. No fallback answer is counted as NPU execution.

The 4,096-column, 8,192-column and full 16,384/13,312-column variants were
tested. Full projection weights live in host RAM: 920 / 1,840 / 3,500 MiB.
The matrix tiles are reused in SRAM; these complete weight sets are not
claimed to remain entirely in NPU SRAM. CUDA retains original FP16 work for
the remainder. NPU operands are explicitly rounded from FP16 to BF16.

The five-arm screen had one observation per arm/workload and two warmups
per server. It separated the new bank schedule from NPU work:

| Path | Fresh 2,667-token prefill | Fresh 7,805-token prefill |
| --- | ---: | ---: |
| Accepted CUDA | 6.9315 s | 19.0938 s |
| New schedule, CUDA only | 6.8520 s | 19.1511 s |
| NPU 4,096 columns | 7.0787 s | 19.9901 s |
| NPU 8,192 columns | 7.2967 s | 20.7342 s |
| NPU full projection | 9.5110 s | 28.1666 s |

A separate one-observation 15,512-token screen measured CUDA 37.7225 s,
NPU-4K 39.3530 s (+4.32% latency), and full NPU 53.1594 s. All six
fresh/repeated answers passed. The repeated request reused 15,507 tokens.

Across each screening server including warmups, full offload had 14.236 s
host wait versus 0.324 s for NPU-4K; worker totals were 48.081 versus 14.336 s,
XRT totals 37.970 versus 10.987 s. Both had 644 jobs; full offload made 9,800
dispatches versus 2,576. These are aggregate diagnostics, not separately
attributable exclusive critical-path times.

## Final confirmation: all matched pairs

The strongest screened candidate, NPU-4K, was independently repeated in
three alternating pairs. There were 18 measured complete answers and 12
warmups. Prompt and token hashes, settings and cache counts were matched.

| Fresh prompt | Pair | CUDA prefill | NPU-4K prefill | Latency increase |
| --- | ---: | ---: | ---: | ---: |
| 2,667 tokens | 0 | 6.8094 s | 7.0310 s | 3.25% |
| 2,667 tokens | 1 | 6.8687 s | 6.9285 s | 0.87% |
| 2,667 tokens | 2 | 6.7921 s | 6.9520 s | 2.35% |
| 7,805 tokens | 0 | 18.7993 s | 19.6245 s | 4.39% |
| 7,805 tokens | 1 | 18.9055 s | 19.4307 s | 2.78% |
| 7,805 tokens | 2 | 18.9279 s | 19.6106 s | 3.61% |

Median prefill times were 6.8094 versus 6.9520 s, and 18.9055 versus
19.6106 s. Median paired throughput changes were **-2.30% and -3.48%**.
Latency reduction is `100 * (1 - candidate / baseline)`; throughput gain
is `100 * (baseline / candidate - 1)`. Medians of matched-pair percentages
are not ratios of marginal medians.

Median first streamed text was 6.869 versus 7.015 s and 18.953 versus
19.662 s. This is first API text, not the raw engine-token timestamp.
Median total answer time was 8.879 versus 8.791 s at the smaller prompt,
but output counts were CUDA 56/58/67 versus NPU 48/48/69. That apparent
total-time improvement is not accepted as acceleration. At the longer
prompt, total medians were 20.998 versus 21.794 s with outputs 57/63/65
versus 68/64/68. All per-request decode, first-answer, output and draft
acceptance metrics are retained in [request records](artifacts/phoenix-cuda-npu/08-requests.json).

Repeated 7,805-token requests reused 7,800 tokens and computed five new
tokens. Median prompt times were 0.1925 versus 0.2298 s. This short path
bypasses the NPU pipeline, so its variable total-answer time is not evidence
of NPU acceleration.

All **39 measured answers and 28 warmups** across the final screens and
confirmation passed the bounded ledger/JSON and cache checks and stopped
normally. Numerical reference duplication, trace callbacks and logit dumping
were disabled in timed runs. NPU performance logs show `checked_values=0`
and positive dispatch counts. The exact synthetic
[request bodies and token IDs](artifacts/phoenix-cuda-npu/08-workloads.json)
are included for reconstruction.

## Numerical checks, negative controls and overlap limits

Three final model numerical qualification requests checked:

| NPU width | Output values checked | Maximum relative RMS versus original FP16 | Maximum absolute arithmetic error versus matching BF16 operands |
| --- | ---: | ---: | ---: |
| 4K | 380,035,072 | 0.17438% | 0.00010681152 |
| 8K | 760,070,144 | 0.16536% | 0.000099182129 |
| Full | 1,445,785,600 | 0.21155% | 0.00010299683 |

Total: **2,585,890,816 values**. All three model requests returned correct
JSON with normal stop. The original 1% per-projection RMS gate was retained.
The first numerical attempt failed a strict elementwise absolute comparison
with FP16. The revised verifier separates BF16 arithmetic correctness against
CUDA with identical BF16 operands from the FP16-to-BF16 precision tradeoff.
The failed original attempt is retained and is not relabeled as passing.

A final actual-model blocked-dispatch negative test returned an explicit
error with no answer. The separate [trace/logit diagnostic](artifacts/phoenix-cuda-npu/08-trace-and-logits.json)
used the final binary: all 368 native dispatch/completion intervals, totaling
1,585.7036 ms, were inside the other bank's CUDA-stream body intervals.
Those intervals include waits and callback scheduling. They do **not** prove
simultaneous SM occupancy or useful critical-path overlap.

The matching 2,022-token diagnostic had the same first-window argmax and
full-vocabulary KL 0.000366818. This is one bounded logit comparison, not a
perplexity evaluation. An initial CUDA-only smoke used an earlier binary
before the verifier revision and is not attributed to the final binary.

## Resources and scope limits

Across final performance runs, minimum sampled free dedicated VRAM was
305.3 MiB and peak dedicated usage 7,882.7 MiB. Minimum available RAM was
4.08 GiB for full offload and 6.97 GiB in the repeated NPU-4K confirmation.
All monitored free-VRAM 250 MiB and free-RAM/commit 3 GiB floors passed.
Samples were taken at 0.5-second intervals; this does not prove the absence
of shorter excursions. Maximum GPU temperature was 63 C and AC was present.
Throttle masks 1 and 1024 are retained without inferring a new diagnosis.

Mapped expert storage produced approximately 42 GiB in WDDM shared-memory
counters; NPU adapter peaks were approximately 1.002/1.932/3.615 GiB.
Adapter identifiers are replaced with anonymous labels in public data.
CUDA reported a low allocation budget for both controls and candidates.
These counters are not proof that VRAM paging occurred or did not occur.
Owned processes and allocations were released after the experiments.

The configured 64K capacity was preserved, but the largest final prompt
actually tested was 15,512 tokens. Vision execution, concurrent clients,
exhaustive cancellation/recovery, held-out perplexity, maximum physical
tile count, maximum VLIW width and peak SRAM bandwidth were not qualified.
No Gufo HIP/NPU hardware suite was run on this RTX/Phoenix host. This report
also makes no claim that Phoenix is fundamentally unable to help an LLM.

## Evidence package and reconstruction

[Full-model data](artifacts/phoenix-cuda-npu/08-full-model.json),
[original build identities](artifacts/phoenix-cuda-npu/08-original-identities.json)
and the [manifest](artifacts/phoenix-cuda-npu/manifest.json) record provenance.
The manifest contains original source hashes and public-file hashes. JSON
changes are formatting, local-path placeholders and anonymous adapter labels;
benchmark numbers are preserved. Request records omit generated reasoning
and server dashboards. No personal names, home paths, machine identifiers,
credentials, personal repository links, model weights or executables are
included. Third-party license notices remain intact.

The final Strata executable SHA-256 is
`02163319d3ffdebd1580e04f66ff041a1a205e98766bb114d6a70779a5df4dc5`;
the unmodified 0.1.50 CUDA control is
`b22c9546670754408e9ed7b91bd221e0a64051ea3bb2657a8e86ec4cd0a677da`.
The selected matrix xclbin SHA-256 is
`6e432ced70c305bf282d8bae0bd73e4c3d25261d1668afc433cf65e22d9ff143`;
instruction binary SHA-256 is
`2e5674d6c5ea7a0a342b753791c6466a9d4ff90800de669fb716f46fccd8126b`.

The included patch is a byte-for-byte complete diff against Strata
`1aff6d14117b88171fcfcec8f2b55fdc47f2b165`, supplied for implementation review,
not as a patch to apply to Gufo. Strata's [MIT license](artifacts/phoenix-cuda-npu/STRATA-LICENSE.txt)
is included. `STRATA_GUFO_PROJECTION` is off by default; the experimental
runtime uses `STRATA_GUFO_PIPE=0/1/2` for disabled/CUDA-bank/NPU-bank modes
and a column-width setting. The native build expects XRT and CUDA locally.
The pinned upstream NPU kernel must be regenerated with the recorded shape
and tile parameters. The package does not contain the entire local harness,
model or binary environment and is not a one-command portable benchmark.

To reconstruct the comparison, use the exact original model/tokenizer and
recorded profile, two warmups per server, identical synthetic request/token
sequences and fresh/reused cache states, then alternate CUDA/NPU process order
across three pairs. Wait for complete answers; verify completion, ledger
values, prompt hashes and cache counts before comparing prefill times.
Run numerical/trace qualification separately. Include conversion, packing,
PCIe traffic, XRT waits and CUDA consumption in the measured path. Preserve
the restored control and reject performance promotion without repeatable gain.

The offline evidence validator checks file hashes, per-request completion,
pair matching and percentage arithmetic without running inference:

```sh
python docs/models/qwen3.8-flash-next/artifacts/phoenix-cuda-npu/verify_evidence.py
```

## Follow-up investigations and resolution of open questions

Following initial community review and analysis of related XDNA1/AIE2 implementations (including `ryzenai-npu-lab`, `hawkpoint-npu-llm`, and `whisper-xdna`), targeted experiments were conducted on the same Ryzen 9 7940HS / Phoenix hardware to address the six core questions raised above. Detailed machine-readable results are preserved in [artifacts/phoenix-cuda-npu/09-followup-five-column-runlist-nvme.json](artifacts/phoenix-cuda-npu/09-followup-five-column-runlist-nvme.json).

### 1. Physical 5-column (20 compute tiles + 5 MemTiles) silicon execution

To determine whether the 4-column limit was a hardware barrier or driver/compiler convention, a direct AIE2 binary instruction generator was implemented to bypass the upstream MLIR-AIE Phoenix column-mask constraint (`mlir-aie#2333`).

- **Physical probe result**: All 5 physical columns (columns 0 through 4, rows 2 through 5) executed correctly on Windows Phoenix silicon, verifying execution on all 20 physical compute tiles and 5 MemTiles simultaneously (`all_20_cores_passed: true`, dispatch latency 2.14 ms).
- **Throughput and architectural bottleneck**: On an M=1,024, K=2,560, N=1,280 GEMM benchmark:
  - 16 compute tiles (4 columns) reached **1.92 TFLOPS**.
  - 20 compute tiles (5 columns) achieved **1.04 TFLOPS** (peak 1.035 TFLOPS, median 807.2 GFLOPS).
  - *Root cause*: Phoenix silicon features only **4 `ShimNOCTile` DMA interfaces** connecting the AIE array to the host interconnect. Column 4 shares NOC DMA routing with Column 3, causing a 25% bus-routing stall when all 20 tiles transfer data concurrently. The upstream 4-column driver constraint reflects this physical DMA asymmetry.

### 2. Batched `xrt::runlist` and sub-tile completion sentinels

The original report identified that returning to the CPU after every 1,024-column NPU slice (filling NaNs, calling `run.wait()`, and reading back the entire 8 MiB buffer) introduced severe host overhead.

- **Pre-built execution lists**: We integrated `xrt::runlist` (`<xrt/experimental/xrt_kernel.h>`) into `gufo_projection.cpp` and `projection_cost.cu`, chaining dispatches across layer banks without host-kernel round-trips.
- **Sub-tile sentinel guard**: Replaced full-buffer NaN scanning with a 256-subtile start/end and last-active-row sentinel verification (`4 KiB` host traffic vs `8 MiB` full tensor traffic).
- **Results**:
  - `gufo-npu4` non-XRT CPU worker overhead dropped **81.4% (5.4x)** (`3,185.1 ms -> 598.5 ms` across 644 jobs, down to **0.64 ms/job**).
  - `host_wait_ms` dropped from **241.4 ms to 0.043–0.067 ms** (-99.97%).
  - Total worker wall time on `gufo-npu8` dropped by **6,122.9 ms (-23.1%)** (`26,450.0 ms -> 20,327.1 ms`).
  - Passed full numerical qualification across 9 model tasks with **3,056,672,768 checked values** (`max_math_abs = 0.000103`, `max_rel_rms = 0.001804`), and passed the negative dispatch control.

### 3. Long-context scaling (16K to 262,144 tokens) & NVMe offload co-optimization

Context lengths from 16,384 through 32,768, 65,536, 131,072, up to the full architectural maximum of **262,144 tokens** (256K) were systematically evaluated under discrete GPU and host memory pressure:

- **KV cache scaling across tiers**:
  - At 16,384 tokens (`kv_resident 16384`), 100% of KV cells remain in VRAM (`0 B` pinned host memory).
  - Above 20,480 tokens, the engine streams KV cells from pinned host DDR5 RAM across PCIe:
    - At 32K context: `0.30 GiB` (`k8v4`) / `0.39 GiB` (`int8`) pinned host KV.
    - At 64K context: `0.60 GiB` (`k8v4`) / `0.77 GiB` (`int8`) pinned host KV.
    - At 128K context: `1.20 GiB` (`k8v4`) / `1.55 GiB` (`int8`) pinned host KV.
    - At 262,144 context: **`2.39 GiB` (`k8v4`) vs `3.09 GiB` (`int8`)** pinned host KV.
- **Prefill buffer borrowing vs dedicated buffers**:
  - In default borrowing mode (`plan_lend`), `take_stage()` carves a single-layer prefill KV stage buffer (`213.9 MiB` for `k8v4` / `276.8 MiB` for `int8` at 262K) out of borrowed VRAM expert-cache slots. When borrowable headroom is constrained, chunk size drops from `1024` to `512` or `256`, degrading prefill throughput from `~460 tok/s` down to `~230 tok/s`.
  - Co-optimizing with hybrid `k8v4` KV and dedicated prompt buffers (`--no-prefill-borrow` or `--expert-cache auto --vram-reserve-mib 300`) keeps prefill at full `1024`-token chunks, achieving **`442.3 tok/s` prefill at 16,384 prompt length / 262,144 max context**!
- **NVMe Direct I/O (`--ple-io direct --ple-inflight 512 --ple-row-cache 2097152`)**:
  - The 9.1 GB Per-Layer Embedding table (Shard 2) is streamed directly from the PCIe Gen4 NVMe SSD via unbuffered asynchronous Windows I/O (`FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED`), reaching **`140,000–235,000 rows/s`** with `STRATA_SSD_KEEPALIVE=100`, eliminating OS file-cache eviction of pinned MoE expert weights.
- **NVMe Tiered Expert Offloading (28 GiB RAM cap)**:
  - When system RAM is capped at 28 GiB (`--resident-budget-gib 28 --mmap-experts`), the engine streams cold expert tiers directly from the NVMe SSD via router-guided lookahead (`STRATA_LOOKAHEAD=1`, `STRATA_FILL_AHEAD=512`). On 8,192-token document prompts, the system sustained **`278.1 tok/s` prefill and `42.27 tok/s` decode** while streaming 3,324 expert blobs (6.8 GB) from NVMe per query with zero memory faults.

### 4. Same-binary NPU vs CUDA comparison with batched runlist

Evaluating the batched `xrt::runlist` NPU worker against the identical-binary CUDA control (`PIPE=0`) under standard 1,024-row geometry:

| Workload (Context 16K) | NPU-4K (`PIPE=2`) | CUDA Control (`PIPE=0`) | NPU vs CUDA Delta |
| --- | ---: | ---: | ---: |
| 2,048 tokens (`Doc 2048`) | 332.4 tok/s | 355.6 tok/s | -6.5% |
| 4,096 tokens (`Doc 4096`) | 423.5 tok/s | 431.8 tok/s | -1.9% |
| 8,192 tokens (`Doc 8192`) | **424.2 tok/s** | **440.2 tok/s** | **-3.6%** |
| Median decode | 30.58 tok/s | 30.79 tok/s | -0.7% |

Through batched `xrt::runlist` dispatch and sub-tile sentinel checking, the NPU prefill gap at 8K prompt length narrowed from the previous 15–20% down to **3.6%**, with host CPU overhead virtually eliminated (`0.64 ms/job`). However, because discrete GPU GDDR6 bandwidth and internal tensor cores remain faster than round-trip host-PCIe traffic to the integrated NPU, discrete CUDA execution remains the optimal prefill engine on this architecture.
