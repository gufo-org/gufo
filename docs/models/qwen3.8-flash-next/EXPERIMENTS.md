# Qwen3.8 Flash-Next experiments

| Experiment | Decision / evidence |
| --- | --- |
| Cheaper expert selection, direct MTP normalization and partial SSM tiles | Retained: exact dequantization and live matrix operations, no added allocation. Mixed pp2048 d0/d32K: AR 1697.86/1534.50 tok/s; MTP 1650.29/1497.00. Separate d128K code continuation improves 1381.32 → 1400.23 tok/s AR and 1341.04 → 1361.96 MTP. Default sampled MTP PP improves 1543.99 → 1584.22 tok/s; TG 38.79 → 39.04. One sample per control; outputs, counts and acceptance match. [Evidence](artifacts/prefill-long-context.json). |
| Four-phase expert loop and rolling boundary convolution | Rejected: exact microbenchmarks improved, but full HTTP did not establish a useful gain. Reverse d0 AR: four-phase 1690.95 versus simpler loop 1692.07 tok/s; deep AR/MTP were slightly slower. Both specializations removed. [Evidence](artifacts/prefill-long-context.json). |
| Further barriers, selector scheduling and tail launches | Rejected: LDS-only barriers, code-cache rotations, smaller selector histograms, dot-product ILP groups and separate tail launches did not improve complete operations. Skipping ineffective tail checks also remained exact but slowed projection/convolution at 2040/2048/2060/4096 tokens. |
| Skip unused SSM activation copies and routed-row clears | Retained: 96 fewer launches per pp2048. d0 AR 1675.26 → 1681.77 tok/s; MTP 1622.87 → 1629.90 tok/s against the preceding private increment. d32K stays within 0.6%, TG within 0.2%; no material depth gain. Full logits, sampled replay and all six greedy completions remain exact. [Evidence](artifacts/prefill-long-context.json). |
| Fixed/split HC residual blocks | Rejected: the faster fixed-width variant changed F16 normalization bytes; exact variants and smaller blocks did not improve speed. |
| Wider SSM projection launch groups | Rejected: groups 1/4/8/16/32 preserve output bytes but are 1.3–4.0% slower than the existing group of two at 2048/4096 rows. |
| Canonical pp2048 HTTP control | Complete branch versus matched main: MTP d0 1581.01 → 1623.99 tok/s; d32K 1490.59 → 1524.10 tok/s. All 12 mixed/repetitive completions and draft counts match; TG stays within 0.5%. One sample per workload/depth through 32K. [Evidence](artifacts/prefill-long-context.json). |
| Packed selector loads and grouped MTP Q8 projections | Retained: exact arithmetic, fewer narrow loads and projection launches, no added allocation. A separate default-sampling code continuation (2217 new tokens at 32K) gains 1.9% PP; d0/128K and TG are essentially unchanged. [Evidence](artifacts/prefill-long-context.json). |
| MTP catch-up at the sparse boundary | Fixed: final-tile execution now uses the full predictor's dense/sparse split and query grouping. Exact stage/candidate/carry checks pass; target arithmetic and sampling policy are unchanged. |
| First decode after cold 32K prefill | Timing remains inconclusive: initial 4.3 ms flags did not recur in the reverse-order pair (50.92 → 51.29 ms); unchanged baseline itself varied by 4.7 ms. GPU work and graph setup match. Retained separately from tg128 results. [Evidence](artifacts/prefill-long-context.json). |
| Q8 vocabulary-head prefetch | Rejected: exact isolated kernels improved, but sampled tg128 stayed at 38.54 → 38.50 tok/s. No prefetch specialization retained. [Evidence](artifacts/prefill-long-context.json). |
| Direct mapped n-gram input | Retained: GPU reads the existing coherent pinned buffer, removing one upload per target forward and 40 MiB of shared scratch at 4096-row capacity. AR/MTP, images, cancellation and C2/4/6/8 sampled replay stay exact. No material d0 TG gain. [Evidence](artifacts/prefill-long-context.json). |
| Pinned weights and aligned Q8 head layouts | Rejected: coherent/noncoherent pinned weights and registered huge-page storage did not improve decode. A lossless planar head improved isolated kernels but only 0.8% in full MTP decoding while retaining another 644 MiB; no second head or private weight format retained. [Evidence](artifacts/prefill-long-context.json). |
| GDN row reuse, padded SSM rows and full selector scratch utilization | Retained: pp2048 d0 1648.38 tok/s (+2.1%), MTP 1592.39 (+2.4%) in matched single-run HTTP controls. Cold 131K AR +1.3%; late AR prefill after 196K +2.0%. Cold 194K MTP +0.9%; its short continuation was 0.9% slower, so no deep-MTP continuation gain is established. No added model/session allocation; exact logits and state checks. [Evidence](artifacts/prefill-long-context.json). |
| Cooperative FP32 selector scoring and other GDN layouts | Rejected: sharing score lanes or keys preserved bits but was slower. Two GDN state rows/four waves beat the other tested row/wave layouts. [Evidence](artifacts/prefill-long-context.json). |
| Final-layer AR prefill pruning | Retained: pp2048 reaches 1501.84 tok/s at 128K (+1.8%) and 1713.11 at d0 (+1.2%) in matched single-run controls. Only unused output rows are skipped; target logits and cache replay remain exact. MTP needs the full hidden sequence. [Evidence](artifacts/prefill-final-rows.json). |
| Further 128K selector/attention tuning | Rejected: FP32 key staging, lane/register scheduling, histogram aggregation and attention LDS/row tiling retained exact outputs but did not improve complete operations. [Profile](artifacts/prefill-final-rows.json). |
| Bounded selector score scratch | Retained: exact deep-context logits; pp2048 gains about 1.4% AR / 1.2% MTP at 128K in single-run controls, with no measured shallow-context regression. Shared scratch falls from 128 to 20 MiB. [Evidence](artifacts/prefill-deep-context.json). |
| Deep attention packing and skipped work | Rejected: shared selection lists and skipping masked matrix rows were slower; rescaling and fragment prefetch gave no compelling model-level gain. [Experiments](artifacts/prefill-deep-context.json). |
| Skip unused final HC normalization | Retained: residuals and 64 full-vocabulary logit rows are exact. PP is within 0.3% of main at d0/d32K, AR and MTP; no significant speedup demonstrated. [Controls](artifacts/prefill-normalization.json). |
| Prefill expert tiles, load scheduling and GDN stores | Rejected: mixed expert tiles slowed PP about 1.1%; smaller dense tiles and streaming loads were slower; GDN changes had no repeatable gain. [Experiments](artifacts/prefill-normalization.json). |
| Conversation checkpoints and asynchronous capture | Retained: edits, growing histories and rotation with bounded RAM; AR/MTP cancellation, image replay and disk restart pass, including four execution sessions. Intermediate copies allow peers to continue, and coincident RAM/disk boundaries share one copy. [Functional checks](../../../tests/functional/README.md). |
| Prefix-independent MTP cache projections | Retained: exact seeded replay across prompt splits and checkpoint replacement, using shared Q8 row arithmetic. [Checks and timings](artifacts/mtp-cache-replay.json). |
| Skip discarded MTP outputs | Retained: K/V-only prefill, compact catch-up and wider projection tiles; C1 costs remeasured. Prefill is within 0.3% of main; d0 TG remains 1.1% slower, d4K TG is 0.4% faster. |
| Group vocabulary rows per block | Rejected: no measurable end-to-end gain. |
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
| Dense split at the indexer budget | Retained; pre-budget queries always take the dense tiles, so prefill logits no longer depend on the prefill batch. Before: batches 4096 and 8192 differed from 2048 on 4/4 repository documents; after: byte-identical decode dumps at 2048/4096/8192. |
| 4096-token prefill chunks | Retained together with 128 readers and the next-chunk prefetch; exact given the dense split (byte-identical dumps on 4/4 repository documents, identical served greedy outputs). Served repository prompts on two gfx1151 machines, ABBA, 8+ samples per prompt: 24K–102K-token prompts +6–11% (median per prompt), 7K–8K prompts −0.1% to +0.6% in dedicated rechecks; pp4096 +5.3%/+5.4% at d0 and +5.8%/+5.9% at d32K; TG unchanged. Peak memory at 262K +1.3 GiB. Prefill beside an active decode stays at `--prefill-chunk` (512); two-session gaps unchanged. |
| Next-chunk n-gram prefetch | Retained; a prefill chunk reads the next chunk's rows once its own have arrived, and a batch uses them only with identical tokens and history, so output is unchanged. With 4096-token chunks the gather then overlaps a whole chunk's forward instead of only layer 0. Adds one row buffer (≤ 40 MiB pageable, allocated on first use). |
| More Q8 vocabulary rows/block | Rejected: no C1 improvement. |
| Private shared-expert F16 rows | Retained; the shared expert's SwiGLU rows for its F16 down projection get their own buffer instead of overwriting the routed experts' narrowed token rows, which removes one full-batch narrowing pass per layer. Every GEMM reads identical bytes; greedy output, chunk-boundary logits and the C2–C8 batch checks are exact. Interleaved Nix A/B at d0: 1559.6 → 1568.1 tok/s (+0.6%); the profiled kernel time fell 1362.8 → 1333.1 ms. |
| 256-row routed down-projection blocks | Rejected: exact output, Q5_1 unchanged and Q8_0 slower. The routed kernels already stream expert weights at roughly two thirds of DRAM bandwidth, so per-block prologue/epilogue and activation restaging were not the bound. |
| 64-token HC down-projection tiles | Rejected: exact output, 23% slower than the 128-token tile despite twice the resident blocks. |
| One-tile-ahead LDS fragment prefetch (dense and routed WMMA) | Rejected: exact output, but the explicit double buffer raised register use (dense 221 → 253 VGPRs; the 128-token pair reached 256 with spills) and every GEMM slowed 3–14%. |
| hipBLASLt F16 dense projections | Rejected: the pinned library reaches 19–26 TFLOPS on the 2048-token dense shapes against 30–34 TFLOPS for the Q8→F16 WMMA kernels; `tools/qwen-flash/dense_blaslt_sweep.hip` reproduces the sweep. |
| Side-stream inject/shared-expert overlap | Rejected: exact output and real kernel overlap in the trace, but the co-running kernels slowed each other and interleaved wall-clock runs were 0.5–0.9% slower. |
| Sparse attention tiles cut across selection windows | Retained; distribute tiles across splits using a 64-block carry, keeping four resident blocks/CU. Independent review: AR +6.7% at 32K and +16.5% at 128K, d0 unchanged. Same keys, reassociated FP32 sums; FP64 operator and model-level rounding checks pass. [Evidence](artifacts/attention-tiles-review.json). |
| Whole-tile carry across windows | Rejected: same output, but 16.5 KiB of LDS cost a resident block per CU and slowed d2K eight-row verification 8.5%. |
| GPU greedy penalties and linear CPU anchor selection | Retained; exact FP64 penalties, unchanged proposals and snapshots. Short heat-pump tg400: 32.12 → 33.80 tok/s; C2/4/6/8 improve 7.6/8.5/16.3/14.2%. Unpenalized control unchanged. [Evidence](artifacts/penalty-verification.json). |
| Prefill projection, attention and indexer kernels | Retained; bit-identical logit dumps and greedy hashes, plus a bitwise GEMM sweep against hipBLASLt for every routed n ≤ 4096. pp4096 +1.4% at d0 and +2.5–2.8% at 32K/115K (interleaved ABBA, `-r 6`, two machines); repository prompts 6.5K–102K +3.2% median; tg unchanged. The indexer projections use the own kernel only when hipBLASLt selects algorithm 4438 and otherwise stay on the library. |
| 128 n-gram readers | Retained with 4096-token chunks, which double each gather; readers block in `pread`, so the pool sets the queue depth. Readers are woken per read, so small gathers do not wake the pool; 96 more threads. |
| Paired routed GEMM code-cache blend | Retained: blend the four cached chunks with masks. A conditional over `uint4` lvalues becomes an address select, and clang 23 then keeps the cache in scratch (80 B, 0 spills). Clang 23 pp2048 +0.7%, pp8–pp64 +1.6–2.7%, MTP pp2048 +0.9%. Output is byte-identical. Clang 22 pp2048 is 0.7% slower. |

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
preserving [quality](QUALITY.md). The 1750 tok/s PP and flat d0–d128K
objectives remain unmet; see [current benchmarks](BENCHMARKS.md).
