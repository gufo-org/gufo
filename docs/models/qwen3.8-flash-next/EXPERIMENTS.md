# Qwen3.8 Flash-Next experiments

| Experiment | Decision / evidence |
| --- | --- |
| Full-width MTP RMSNorm and split projection | Retained after independent CPU stage audit; one 10240-wide normalization, embedding projection shared across HC branches. |
| Full Q8 vocabulary head | Retained; private Q4 shortlist removed. Sampled top-64 proposals use exact target verification. |
| Batched MTP transformer and heads | Retained; independent body/head comparisons, private KV/recurrent/rollback/RNG state. |
| Batched decode mixers, residual epilogues and MTP norms | Retained; each request keeps its scalar reduction and private state; exact C2/C4/C6/C8 logits, acceptance and RNG. |
| HC Q8 weight prefetch | Retained for 1–8 rows of the 320×10240 projection; exact original products/FMA order, no extra allocation. |
| Q4 shared-expert weight reuse | Retained with a separate compact kernel for single-request experts; faster repetitive C1/C4/C8, no mixed-work regression, exact projection/model replay. |
| Q5 high-bit expansion | Retained; exact integer multiply/mask replaces repeated shifts, with unchanged dot products and faster serving. |
| Short convolution/history fusion | Retained for 1–8 tokens; exact output, rolling state and rollback snapshots, fewer launches and no additional allocation. |
| Batched GDN recurrence | Retained; private ragged state/rollback rows, cancellation isolation and exact session replay. Helps shallow batches most; end-to-end gains are modest. |
| Batched small projections and MoE preparation | Retained; group independent rows, quantize activations once in existing scratch, and batch router/shared-expert work. Exact scalar/batch outputs and sampled state; C1 unchanged. |
| Q4 expert grouping across the full batch | Retained; bounded groups share weights across request boundaries, with original scalar arithmetic and exact session replay. Q5 down grouping/reordering was slower on mixed routing and was rejected. |
| Expert-ordered prefill intermediates | Rejected: exact gate/up → down results with captured routing at 1024/2048 tokens, but no useful projection-chain gain for Q4_K/Q5_1 or Q5_K/Q8_0. |
| Wave64 batched Q4 gate/up | Retained above eight rows; independent 32-lane reductions preserve exact outputs and sampled state. Repetitive serving improves, C1 remains stable. Four output rows, 16-input groups and replacing the slot scan did not improve performance. |
| Register-cached vision softmax | Retained for bounded row sizes; unchanged reduction order, byte-identical Flash-Next/Qwen27B embeddings, no additional allocation. |
| 4096-patch vision attention tiles | Retained; byte-identical GEMMs/embeddings, lower latency and 24 MiB less attention scratch. Other shapes keep their original tiles. |
| Partitioned BF16 WMMA vision value projection | Rejected: failed the full-encoder reference gate despite lower isolated FP64 error. |
| Integer WMMA Q8 verification | Retained through 48 input rows for wide target projections/heads; preserves K8 partials/FMA order and reduces each result once, with exact session replay. Small batches retain vector kernels. |
| Q8 activation reuse across output rows | Rejected: no repeatable gain on the real projection shapes. |
| Wider Q8 decode and Q5 expert tiles | Rejected: 56–64 dense rows, phased/packed token tiles, Q5 wave64 and 16/32 Q5 expert rows were slower despite exact output. |
| Compact expert launch groups | Rejected: improved shared routing but negligible mixed-routing gain. |
| Sparse attention register/cache retuning | Rejected: exact d128K output, but register scheduling/occupancy gave no material gain and reloading queries was slower. |
| Shared attention selection lists | Rejected: exact outputs, but roughly 1% isolated gain did not justify another buffer and setup kernel. |
| Skip unused attention Q8 staging | Rejected: exact chunk/full-logit checks passed, but no clear end-to-end prefill gain justified the extra dispatch logic. |
| FP32 selector load scheduling | Retained; bounded scheduling removes scalar-register spills, preserves every score bit and lowers d32K selection time to 22.4 ms per pp2048. Matched d128K AR prefill improves about 1.5%. |
| Integer WMMA value transpose and paired FP32 selector lanes | Rejected: bit-preserving transpose and exact selector scores, but both were slower on deep-context inputs. |
| Packed Q8 prefill staging | Rejected: exact output, but extra decode/register/transpose costs outweighed reduced LDS use. |
| Persistent packed Q8 SSM weights | Rejected: small prefill/1–4-row gains would cost roughly 14% on eight-row decoding; two-step prefetch did not recover it. No second weight copy or private format retained. |
| Transient F16 SSM weights and parallel HC branches | Rejected: F16 staging was exact but slower overall; parallel HC branches changed quantization ties. |
| Smaller-LDS SSM projection, unrolled HC expert sum and wave64 HC combine | Rejected: exact output but no useful prefill speed gain. |
| HC quantized-output store layouts | Rejected: row-major staging plus transpose is slower; cooperative stores within one kernel save only about 2% in isolation. Residuals, normalized activations and Q8 bytes remain exact, including ragged rows. |
| Transient key transpose and mixed expert tiles | Rejected: exact outputs; key transpose slows deep selection, mixed tile sizes provide no useful prefill gain. |
| Query sharing, paired-key FP32 scoring, query LDS caching, MoE prefetch barriers/unrolling | Rejected: exact outputs, but no useful speed gain. Four-wave Q8 matrix reduction also lost to eight waves. |
| Approximate selector screening followed by exact rescoring | Rejected: GPU thresholding and compaction erased the isolated gain, before accounting for runtime error bounds. Remains a synthetic experiment; no approximate production selector added. |
| Ratio-four predictor QSA, FP32 ranking queries | Retained with sparse state, rewind and deep selector checks. |
| Greedy batch cost controller | Retained only for all-greedy C>1; separate occupancy/context bins, stable plain controls, no transition timings. Sampled replay uses fixed curves calibrated from median warmed cycles on 2026-09-20. |
| One-row MTP prefill lag | Retained; 320 KiB kept hidden state/session, avoids replaying a final prefill chunk. |
| Final-tile MTP catch-up | Retained; preserve every KV/indexer row, rank only the final attention tile, then restrict output projection, mixers and shared experts to the final aligned tile(s), with one routed Q8 expert row. Full predictor/catch-up stages, candidates and recursive carry match exactly at 224/257/2047/2048 rows. |
| Batched MTP catch-up tail | Retained; after writing every attention cache row, compact final request rows into existing scratch and skip unused output projections/FFNs. Preserve the scalar arithmetic of one-row tails; full-head candidates and recursive carry match independent execution. |
| Routed Q8 accumulation order | Retained; explicit rounded product/FMA prevents identical rows changing with column placement. Independent FP64 dot and predictor carry checks pass. |
| Isolated MMQ quantizer rounding change | Deferred: it changes half-integer tie behavior shared with W8A8 and fails their existing agreement gate. No quantizer change retained. |
| SSM tile, wave and compiler scheduling variants | Rejected: four/sixteen-wave groups, wave64, wider token tiles and iterative ILP scheduling retained exact output but did not beat the existing eight-wave projection. |
| SSM fragment lifetime and prefetch pipeline | Rejected: shorter-lived K16 fragments reduce register use without a useful gain; delayed prefetch and double-buffered LDS are slower. Projection/convolution outputs remain exact at 2048/2049 tokens. |
| Captured shared stages in batched target decoding | Rejected: exact C2/C4/C6/C8 logits, sampled state and cancellation checks, but no serving or warmed-cycle gain justified graph metadata/capture overhead. |
| Compact recurrent rollback | Retained; one full state plus exact FP32 update operands, with the original product/FMA order. All rollback prefixes match fresh execution. Depth grows on demand; seven-draft cap about 147 MiB/session, released on reset. |
| Live final frontier and async prompt snapshots | Retained; immutable branch snapshot, worker capture, bounded persistence outside the lookup lock. |
| Thread-local decode graph capture | Retained; independent snapshot workers no longer invalidate a peer's capture. Snapshot bytes and captured/replayed logits are exact; no request serialization added. |
| Chunk-equivalent projections/attention | Retained with exact continued-image/cache/full-logit gates; one-token tails keep prefill arithmetic. |
| More Q8 vocabulary rows/block | Rejected: no C1 improvement. |
| Private shared-expert F16 rows | Retained; the shared expert's SwiGLU rows for its F16 down projection get their own buffer instead of overwriting the routed experts' narrowed token rows, which removes one full-batch narrowing pass per layer. Every GEMM reads identical bytes; greedy output, chunk-boundary logits and the C2–C8 batch checks are exact. Interleaved Nix A/B at d0: 1559.6 → 1568.1 tok/s (+0.6%); the profiled kernel time fell 1362.8 → 1333.1 ms. |
| 256-row routed down-projection blocks | Rejected: exact output, Q5_1 unchanged and Q8_0 slower. The routed kernels already stream expert weights at roughly two thirds of DRAM bandwidth, so per-block prologue/epilogue and activation restaging were not the bound. |
| 64-token HC down-projection tiles | Rejected: exact output, 23% slower than the 128-token tile despite twice the resident blocks. |
| One-tile-ahead LDS fragment prefetch (dense and routed WMMA) | Rejected: exact output, but the explicit double buffer raised register use (dense 221 → 253 VGPRs; the 128-token pair reached 256 with spills) and every GEMM slowed 3–14%. |
| hipBLASLt F16 dense projections | Rejected: the pinned library reaches 19–26 TFLOPS on the 2048-token dense shapes against 30–34 TFLOPS for the Q8→F16 WMMA kernels; `tools/qwen-flash/dense_blaslt_sweep.hip` reproduces the sweep. |
| Side-stream inject/shared-expert overlap | Rejected: exact output and real kernel overlap in the trace, but the co-running kernels slowed each other and interleaved wall-clock runs were 0.5–0.9% slower. |

Separate d32K pp2048 profiling attributes 29.1% of kernel time to MoE, 34.9%
to dense projections and 12.6% to attention/indexing. Final-tile catch-up
takes 20.9 ms of MTP kernel time; the target takes 1450.2 ms.

A d0 pp2048 profile (2026-09-21) is GPU-bound: 1362.8 ms of kernel time in a
1380 ms span. Routed expert GEMMs take 35%, dense F16 projections 29%,
hyper-connection combines 10.5%, the GDN recurrence 9.5%, HC down/inject/shared
projections 7% and attention 5.5%. The dense projections run at 30–34 TFLOPS
against the measured 59 TFLOPS matrix ceiling and the combines at about
200 GB/s; the routed gate/up and down kernels stream 0.9–1.1 GB of expert
weights per layer at 160–190 GB/s.

A C4 mixed MTP trace with 32 output tokens per request is 84.9% GPU-busy
during inference, excluding model loading. Batched GDN updates take 40.9 ms,
rollback replay 14.5 ms and lazy allocations 31.8 ms. The scheduling thread
spends 5.4% of the window outside HIP API calls, including model-side CPU
work; this is not pure scheduler overhead. These are profile observations,
not unprofiled throughput measurements.

Next: improve prefill at depth and target/draft batch projection reuse while
preserving [quality](QUALITY.md). The 1700 tok/s PP and flat d0–d128K
objectives remain unmet; see [current benchmarks](BENCHMARKS.md).

## TP2

Two-host tensor parallelism over InfiniBand ([TP2.md](TP2.md)). Measured on two
Strix Halo hosts with ConnectX-3 FDR cards at PCIe Gen3 x4, Q4 unless noted.
Development measurements, not published cells.

| Experiment | Decision / evidence |
| --- | --- |
| Fabric ceiling | Measurement: `ib_write_bw` at 5 MiB gives 3.27 GB/s one way and 2.7 GB/s each way in both directions, so a 5 MiB prefill exchange cannot go below about 1.9 ms. The card's two ports share one Gen3 x4 link. |
| Header-free RDMA exchange | Retained; replaces a TCP header, a one-sided RDMA read and a TCP acknowledgement. One RDMA write with immediate data carries header and partial into one of three rotating receive windows. One-row exchange 46 → 24 µs; AR +2.5%, MTP +4.3%. |
| Queued exchanges | Retained. A stage kernel publishes a ready flag, a communicator thread writes and waits, and a one-wave kernel holds the stream until the arrival flag, so the host queues a whole forward. GPU cost per exchange 53 → 21 µs; AR +1.3%, MTP +1.2–1.9%, outputs unchanged. |
| Bounded completion spin | Rejected: a 5 µs spin on the completion queue raised empty polls from 5,728 to about 663,000 per rank without reducing sleep time. The thread now spins with a pause for 5 ms, then sleeps. |
| One-host eager control | Measurement: forcing eager execution on one host decoded tg128 at 26.39 against 26.73 tok/s with graphs. Graph capture is worth about 1% on one host; TP2 runs without it. |
| Intra-expert split | Retained; replaces whole-expert ownership. Each rank holds half (320 of 640 units) of every routed and shared expert's intermediate dimension, on quantization-block boundaries, and computes every selected expert on it. AR +5% (Q4) and +6% (Q8); no load imbalance between the ranks. |
| Shared expert on both ranks | Retained fix. Full Q8 ranks diverged from the second layer on: running a Q8_0 projection on one rank only re-keyed that rank's activation staging cache, so its later projections read a differently rounded copy. Both ranks now run their half of the shared expert; ranks agree bit for bit on Q4 and Q8. |
| Dense split | Retained. Every trunk layer's GDN (8 of 16 key heads, the 24 value heads that read them) or attention heads (one KV head, twelve query heads) are split, with a second exchange per layer; per-rank KV cache and recurrent state halve. Split kernels equal the full ones bit for bit. AR 28.2 → 34.0 tok/s, MTP +17–24%, prefill +21–24% at 4K–32K; Q8 MTP +15–17%. |
| Prefill overlap | Retained. A prefill step of up to 4096 tokens runs as two trunk batches a layer apart, so each batch's exchange crosses the link while the other computes; results equal two steps bit for bit. Q4 +33–46% at 4K–32K, Q8 +21%. |
| Rank 1 as executor | Retained. Rank 0 runs the ordinary scheduler and sends each state-changing model call to rank 1, including multi-token MTP cycles with the sampler's draw state and cache operations; both ranks digest every call. Sampled MTP 35.5/38.5 tok/s mixed/repetitive against about 24 token by token; a follow-up turn on 5.9K tokens of history reaches its first token in 124–141 ms instead of 4.9 s. |
| Concurrent requests | Retained. Rank 1 keeps every open request, and concurrent single-token decoders advance in one batched instruction. Greedy aggregate C1/C2/C4/C8: AR 35.2/57.4/89.7/121.9 tok/s; every concurrent output equals its serial one. |
| Batched MTP | Retained. Concurrent multi-token cycles run as one mirrored batch: the instruction carries each member's budget and draw state, and the batch's draft count, which rank 0 chooses from its own cycle timings for an all-greedy batch. Greedy aggregate C2/C4/C8, Q4: 70.4/98.9/127.3 tok/s against 56.7/86.1/116.4 in batched single-token steps; Q8: 63.4/86.3/98.9 against 48.5/74.2/99.7 (at C8 the draft-count controller finds no gain). Every concurrent output equals its serial one, and seeded sampled requests equal their one-at-a-time outputs beside greedy and sampled ones. |
| Images | Retained. A request's images (resized pixels, grid, RoPE layout) travel with it to rank 1, each rank runs the replicated vision encoder, and binding a state to a request's images is a mirrored call. Q4, greedy: a striped flag, a disc, two images at once and an image beside a text request give answers byte-identical to one host, AR and MTP alike; the repeat restores all 105 prompt tokens and a follow-up 138 of 160. |
| Disk cache | Retained. Each rank keeps its half of a snapshot on its own disk; rank 0's store decides what is saved and restored. Q4 MTP: upstream's `check-continuation.py`, text and images, left 46 files (3.0 GB) per host; after both ranks restarted, every checked request was restored from disk and repeated its output, except one seeded sampled case's next turn. Memory had replaced that prompt's first capture with a later cold prefill's, whose state differs slightly (on one host too), while the disk kept the first. |
| Prefill steps and reuse timing | Retained fixes, found with upstream's `check-continuation.py`. A TP2 rank's recurrent-gate projection (48 rows) fell outside the pinned F16 GEMM that one host uses for its 96, so hipBLASLt's tiles made a prefill depend on how the prompt was split into steps; pinned, `tp_probe --split` is bit-identical at 7/40/150/200 of 251 tokens and at 600/1,100 of 1,115. A released state or snapshot was reused only after rank 1's verdict, so a follow-up sent right after a client disconnect missed the cache; captures are now checked against rank 0's digest, and reuse before the verdict makes the reusing request depend on it. The check's default cases pass 8/8 on TP2, AR and MTP. Its remaining failures (a sampled replay that restores a cold-prefilled snapshot, whose decoded tokens were prefilled) occur on one host and on upstream `main` alike: decoding a token differs slightly from prefilling it (`tp_probe --decode-tail`). |
| TP2 against one host | Measurement. Reduction order differs, so logits are compared with a tolerance: 216 tokens RMSE 0.134, top-1 63/64, mean KL 0.0055; 5,713 tokens RMSE 0.51, KL 0.029. WikiText-2 (8 × 1024 tokens) perplexity 2.1093 on TP2 against 2.1168 on one host, KL 0.019, 97.1% same top token. |
