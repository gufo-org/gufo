---
name: optimize-kernel
description: Optimize Gufo inference on Strix Halo gfx1151 using focused profiles, independent quality checks, and matched end-to-end measurements.
metadata:
  origin: gufo
---

# Optimize gfx1151 kernels

Read the affected model's `docs/models/<model>/{BENCHMARKS,EVALUATION,EXPERIMENTS}.md`
and its launch code. Establish the timed scope and arithmetic contract before
changing dispatch. Follow the user's machine, time, and Git instructions.

## Fast experiment loop

1. Keep baseline and candidate production binaries/objects. Use `nix build`, or the
   `release` CMake preset with the same compiler/dependency versions. Test
   binaries keep assertions and are for correctness, not headline timings.
2. Start with one affected shape and one control. Use pp2048/tg128 for retained
   end-to-end results; tiny prompts/outputs suffice to debug launches. For a
   depth issue, try d16K/d32K first. Expand to d64K/d128K only when justified.
3. Profile separately with `tools/prof/prof.py`; compare stage totals, launch
   counts, GPU-busy time and wall span. Inspect allocator, synchronization,
   sampling and cache I/O when GPU time does not explain latency.
   Exclude model loading from request-time GPU utilization: a server trace here
   looked 18% busy overall but was 87% busy during inference.
   Warm each batch shape before calibration: large lazy allocations caused
   100 ms stalls here. Use a median complete cycle when timings remain noisy.
4. Change one mechanism, compile the affected target, then run its existing
   analytic/operator check. Use `tools/bench/build.sh <name>` for standalone
   HIP experiments; each result needs correctness alongside throughput.
5. Compare unprofiled baseline/candidate under identical model, prompt, depth,
   quantization, sampling, cache history and concurrency. Repeat only to resolve
   noise. A microbenchmark gain must survive the full inference path.
6. Keep the winning implementation as default. Delete rejected/dead routes;
   do not add environment switches or duplicate tests to preserve experiments.
   Record one concise retained/rejected row in `EXPERIMENTS.md`, current speeds
   in `BENCHMARKS.md`, and actual quality evidence in `EVALUATION.md`.

## Techniques that worked here

- **Bandwidth and cache:** measure `tools/bench/gfx1151_peak.hip`, not a spec
  sheet. Previous large-buffer controls reached roughly 240 GB/s. The 32 MiB
  MALL cache can make repeatedly reused small matrices look unrealistically
  fast: rotate weights beyond cache capacity or reproduce full-model traffic.
- **GEMV:** coalesce packed weight loads, stage shared activations in LDS,
  fuse QKV or gate/up reads where useful, and preserve each dot product's
  accumulation order. More fusion can increase VGPR pressure and spills.
  Two-iteration prefetch helped the small 320×10240 HC projection; wider
  prefetch and applying it to larger matrices did not. Preserve the actual
  rounded products/FMA sequence, not just the algebraic formula.
  Integer WMMA helped wider Q8 verification only after retaining four K8
  partials and the original reduction tree. Reducing one result per thread
  avoided duplicate wave sums. DPP xor/add helped GDN; fewer waves or smaller
  scratch alone did not help.
  Repeat identical inputs at different tile columns: fast-math gave routed
  Q8 minitiles different contraction until the product/FMA order was explicit.
- **Verification and concurrency:** reuse quantized weights across token and
  request rows. Tune real ragged shapes, including 3–8 rows and partial tiles.
  Batch the complete draft transformer body, not only the vocabulary head;
  keep each request's KV, recurrence, acceptance and RNG state independent.
  A request dimension in GDN's grid reduced launch overhead for shallow
  verification. Use disjoint scratch and rollback pointers, skip cancelled
  rows before touching state, and keep descriptor storage alive until completion.
  Test both shared and disjoint expert routing: weight reuse benefits shared
  experts, but single-request experts need a compact path.
  Qualify draft controllers on both ordinary and perfect-acceptance prompts.
  Context-dependent attention cost favored shorter blocks here, but treating
  a capped accepted-run estimate as uncensored slowed perfect acceptance.
  Compare completed-token rates; raw acceptance fractions depend on draft width.
  Group independent small projections in the launch grid and quantize batch
  activations once in idle prefill scratch. Preserve each row's original
  dot-product specialization; larger generic GEMV tiles can be slower.
  Ragged Q8 groups benefited from masking the final group's loads/stores,
  avoiding a separate weight pass. Profile mixed cohorts too: saturated
  full-block costs miss repeated partial verification and odd-cohort fallbacks.
  For Q4 experts, grouping across the full batch improved weight reuse.
  A bounded grid can consume a compact device list without downloading its
  count. Qualify disjoint routing too; the same approach slowed Q5 mixed work.
  Exact packed-integer transforms can help both: spreading Q5 high-bit
  nibbles with multiply/mask removed shifts without changing dequantization.
- **Wave mode and compiler:** selected quantized kernels need wave64 while
  other paths use wave32. Match helper/caller wave modes per translation unit.
  One wave64 helped batched Q4 gate/up while preserving separate 32-lane
  reductions; Q5 down, wider output tiles and 16-input Q4 groups were slower.
  Iterative ILP scheduling helped selected 16-row Q4/Q5 kernels; applying it
  globally was not a win. Inspect VGPR/LDS/private scratch and ISA with
  `tools/prof/isa_mix.py`; occupancy alone is not the optimization objective.
  For wide register-cached softmax, a scheduling barrier after each exponential
  removed spills while preserving the sum order. Cache only bounded row sizes.
  FP32 selector queries also benefited from bounded load scheduling: grouping
  16 dot-product chains removed scalar-register spills without changing scores.
  Verify generated register use and full-model time; barriers elsewhere lost.
- **Attention and selection:** exact partial top-k can avoid sorting the full
  context. Preserve tie ordering and FP32 ranking. Pack existing KV bytes into
  bounded scratch/head groups without changing persistent precision. Do not
  change softmax reduction order without the model's numerical qualification.
  Two 16-lane query heads per wave helped verification after preserving both
  original 32-lane dot partials and their offset-16 addition. Scalar AR stayed
  faster at 32 lanes; sharing KV across different query rows was slower.
  In draft attention, prefetching 32 V rows before their original sequential
  FMAs hid load latency; advance the ring slot instead of dividing each time.
  Pairing query heads helped wider blocks. Wider prefetch and four-head groups
  regressed; the component gain translated to about 1% in the complete model.
  For long H3 attention, packing V once into dead projection scratch replaced
  repeated scalar LDS gathers. Sharing K/V LDS in sequential phases retained
  occupancy; transposing V anew inside every attention tile was slower.
  Smaller vision attention tiles saved scratch and time at 4096 patches;
  ragged BLAS tails changed rounding. Qualify complete GEMM/encoder outputs
  and specialize only the shapes that retain both quality and speed.
- **Fusion and memory:** split repeated embedding/hidden projections instead
  of concatenating duplicate inputs. Respect full-width HC normalization.
  Reuse scratch, allocate rollback depth on demand, and avoid carrying entire
  prefill chunks into draft catch-up. Fuse short convolution/history saves
  while values are in registers; require one owner per channel so another
  token tile cannot overwrite history still being read. UMA has placement
  costs: mapped quantized weights and copied reusable audio/DiT weights behave
  differently.
  Match production allocation in GEMV microbenchmarks: gate/up gains on
  `hipMalloc` buffers disappeared with read-only mapped weights. Anonymous
  huge pages improved Q4 AR by about 2%; bounded parallel copying kept warm
  readiness below one second. Include startup cost and separate anonymous
  model memory from the reclaimable file cache when judging that tradeoff.
  Cold HIP registration/upload can serialize page faults. Prefault existing
  mappings in bounded parallel chunks; this improved text/audio/image startup
  without another weight copy. Measure launch-to-ready and the first request,
  since image components load lazily. Fault only active component ranges.
  Size snapshots by valid KV rows and recurrent layers, not context capacity;
  Qwen's FP16 KV is token-major but its FP32 reference KV is head-major.
  Qualify dirty tails, cancellation, disk reload and image continuation.
  Recurrent rollback can retain one full state plus exact update operands,
  replaying only rejected prefixes; this cut Flash-Next's seven-draft reserve
  to about 147 MiB/session. Recording an intermediate changed fast-math
  contraction here. Compare the original kernel directly, preserve each
  rounded product/FMA (including lane tails), and test every rollback prefix.
  After catch-up writes all attention KV rows, only final request rows need
  the remaining predictor projections/FFN; reuse idle scratch and retain
  scalar-tail arithmetic.
  Repeated voice cloning can reuse an exact waveform-reference frontier:
  preserve convolution history and attention KV, key it by actual codec IDs,
  and qualify replacement/cancellation plus the decoder's context reset.
  This saved more complete-request time than faster individual TTS GEMVs.
  Finish a norm's existing descending sum tree within one wave to remove
  barriers without reordering additions. Check the BF16 output boundaries.
  Unrolling cached RMSNorm changed FMA contraction despite passing greedy
  replay. Explicit fused square accumulation restored scalar/batched full-logit
  agreement; test small unsaturated inputs and every verification width.
  For BF16 image kernels, branchless round-to-nearest-even conversion and
  fused gate/up, SiLU and output packing helped; retain NaN payload handling.
  A BF16 activation has only 65,536 inputs: exhaust that domain before
  substituting an intrinsic, including signed zero and subnormals.
  Reuse overlapping convolution windows in LDS while keeping the original
  channel/tap accumulation order. Fuse normalization/RoPE or activation only
  after preserving every intermediate BF16 rounding boundary.
  H3's fused gate/up projections retain two BF16 results before FP32 SwiGLU;
  AdaLN and SwiGLU then write directly in the next projection's packed layout.
  This removed a large gate/up tensor and a packing pass. Pairing even/odd
  WMMA output lanes enabled contiguous vector stores with identical bytes.
  Split large projection interiors from their single ragged tail to remove
  repeated bounds work. Paired-lane vector stores also sped up row-major
  output projections; benchmark the complete block after each change.
  Count library reservations too: H3 retained 32 MiB per rocBLAS handle.
  Delete unused handles and share handles only across sequential work.
  Reuse tensors after their last read; remove `restrict` where in-place
  residual writes now alias an input.
  rocBLAS can rotate the K reduction per output tile: matching that rotation
  made native ASR convolutions byte-exact to PyTorch. For attention, lane
  swizzles change softmax sum groups; preserve those groups and explicit FMA
  operands. A near-identical single block can still drift over 50 blocks.
  Lossless weight packing can still be a poor trade: 2% TTS speed for over
  2 GiB extra copies was rejected. Non-temporal loads won cold microbenchmarks
  but lost complete TTS requests; always reproduce model cache traffic.
- **Graphs:** verify replay actually launches every new operation. Side-stream
  work during capture is not automatically included in the graph. Check the
  relevant shallow/deep dispatch boundaries and unprofiled wall time.
  Independent snapshot workers need nonblocking streams and thread-local
  capture validation; global capture failed when a peer copied its frozen
  state. Test that overlap and compare snapshot bytes and replay logits.

## Quality and reporting

Use an independent operator formula or pinned official teacher; two Gufo
paths agreeing does not prove upstream parity. Compare matched token histories
and find the first changed layer if logits drift. Never relax tolerances or
replace goldens to accept a speedup. Test finite outputs and awkward tails
(e.g. 1/8/9/32/33 rows), not only aligned shapes. End ragged inputs at the
allocation boundary; oversized shared scratch can conceal invalid reads.
Moving FP32 expressions into a device helper changed contraction here.
Capture the first failing operator's inputs for a small replay; include
unsaturated activations, since large synthetic values can hide the difference.
Repeat across process/runtime reloads as well as cached requests. TTS reference
caching concealed a speaker-softmax race: all waves must consume a shared
maximum before scratch is reused for sums. The small shape passed; the real
1536-channel shape failed an independent FP64 check.

For state or speculative changes, cover greedy output, sampled p/q rejection
and residual correction, seeded replay, EOS, multi-turn continuation and
snapshot restore. Add images/cancellation when those paths change. Timing-based
controllers are for greedy decoding; sampled execution must retain seeded
replay. Full model sweeps are final qualification, not each iteration.

Measure C1 plus affected C2/C4/C6/C8. Keep prompt and cache state identical when
comparing CLI single-user and HTTP C1. Model serving tables report the sum of
individual request decode rates; define the timed scope and retain request
latencies in raw profiling artifacts. Time execution directly: an asynchronous
snapshot outside the decode timer must not be subtracted again at completion.
Raise the benchmark server's per-client queue limit for
C8 from one host; a rejected request is not a throughput result. For H3,
prefer an analytic case or one block/forward pass;
do not generate full videos during routine kernel work.
