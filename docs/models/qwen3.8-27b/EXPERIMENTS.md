# Qwen3.8 27B experiments

| Experiment | Decision / qualification |
| --- | --- |
| Quantized verification row groups | Retained per shape; scalar FP32 bits, full target logits and private acceptance/RNG must match. |
| Shared DFlash2 body/context injection | Retained across requests; independent attention, convolution, history and selector state. |
| Partial verification after rejection | Retained with the complete original proposal and unchanged consumed-prefix feedback. |
| BF16 draft gate/up reuse and tiled argmax | Retained; complete head, finite filtering and lowest-ID ties; no extra persistent buffer. |
| Long-context KV packing | Retained in idle FFN scratch with bounded head groups; exact attention output/log-sum-exp, cache bytes unchanged. |
| BF16 target projection reduction | Fixed per-row FP32 order retained for chunk/cache/continued-image equivalence. |
| Register-cached vision softmax | Retained; byte-identical embeddings with the shared Q4/Q8 projector, unchanged reduction order and memory allocation. |
| 4096-patch vision attention tiles | Retained; byte-identical full embeddings, lower latency and 24 MiB less attention scratch. |
| Alternate tiles/waves/pipeline depths (2026-09-19) | Rejected: no release throughput improvement. |
| Dynamic verification chunk/controller alternatives | No new default retained; seeded private-acceptance policy remains. |
| Parallel mapped-weight reads and larger DFlash packing chunks | Retained: faster cold startup, unchanged encoded weights. |
| Compact active/saved recurrence and valid-prefix KV | Retained: unused attention-layer state and future KV rows excluded; Q4/Q8 sampling, rollback, image and disk replay pass. |
| Verification queries grouped by KV partition | Retained: unchanged arithmetic and storage; eight-token attention 2.95× faster at 32K and 3.66× at 64K. Matched d32K C1 HTTP TG improves 15.7% on Q4_K_XL and 20.2% on Q8_K_XL with Q4 DFlash2; shallow TG and PP remain comparable. |
| Split-K two-position KV prefetch and DPP score reductions | Rejected: small single-row component gains did not consistently help eight-row verification. |
| Shallow attention paired KV loads and DPP reduction | Retained: byte-exact; 12.6% less attention time in the complete AR profile. Matched d0 pp2048/tg128 controls improve TG by 1.4–2.6% across Q4/Q8 AR and Q4 DFlash2, with unchanged outputs and acceptance; PP remains comparable. No extra allocation. |
| Parallel attention from 128 tokens | Retained for Q4 C1 AR and Q4 DFlash2: greedy outputs match; lower error against FP64 in all 12 component controls. Other targets/concurrency remain to qualify. |
| Wider shallow pipeline, removed broadcast and separate score/value passes | Rejected: worse cold-KV or eight-row costs than the retained two-position pipeline. |
| Additional GEMV format specialization | Not promoted: Q8 cold components unchanged; Q6 gains at most 3% in the cold component test. |
| Paired-row activation reuse and shared input sums | Not promoted: dominant Q5_K cold components were unchanged or slower. |
| Two-block Q5 weight prefetch | Rejected: byte-exact, but cold-weight throughput is 8–12% lower. |
| Seven-row Q5 token scheduling and output-row groups | Not promoted: byte-exact across three projection shapes and input scales, but cold-weight gains are at most 1.5%; several variants regress. |
| Compact seven-row Q5 activation staging | Rejected: byte-exact and lower register/LDS use, but cold-weight gains are under 1% on gate/up; down and attention projections regress. Token-step and wave-count variants do not recover a useful gain. |
| Seven-row Q5 row-first FMA scheduling | Rejected: byte-exact on three cold-weight projection shapes; most variants are 1–3% slower. |
| Seven-row Q5 wave32 and row-first scheduling | Rejected: byte-exact on three cold-weight shapes and three input scales; no consistent useful gain over wave64. |
| Fused SSM format specialization and larger output-row groups | Not promoted: no useful cold-weight gain; larger groups were slower despite byte-exact outputs. |
| 64–256 attention partitions | Rejected: no improvement at 32K, with different FP32 rounding. |
| Split-attention graph replay | Rejected: matched Q4 AR C1 d0 remains 11.90 tok/s. The existing launch path is already 97.5% GPU-busy. |
| Graph identity includes target feature taps | Correctness fix: changing captured layers or their order must replace the graph's feature-copy operations. |
| Residual/RMSNorm fusion and fewer reduction barriers | Rejected: byte-exact component gains did not survive the full model. Q4 AR C1 remains 11.90 tok/s and profile GPU time slightly increases. |
| Multiple attention heads per thread block | Not promoted: byte-exact through 32K, but cold-KV gains are at most 1.5%. |
| Shared four-position KV tile across six query heads | Retained: byte-exact; Q4 AR C1 d32K TG improves 2.9%, with shallow TG and PP retained. Existing scalar/batched and full-logit replay checks pass. |
| Shared KV tile for verification rows | Retained: byte-exact; Q4 DFlash2 C1 d32K TG improves 4.4%. Shallow TG, PP, output hashes and acceptance counts are retained. |
| Two 16-lane verification heads per wave | Retained for multi-row attention: preserves both original lane partials and their reduction tree. Q4 DFlash2 C1 d32K TG improves 5.8%, with shallow TG, PP, greedy output and acceptance retained. Scalar AR keeps 32 lanes. |
| Draft-attention value prefetch and paired query heads | Retained: 32-value prefetch preserves the original FMA order; wider blocks share K/V across two query heads. Byte-exact through 32K; C1 d0/d32K TG improves 0.8–1.0%, with unchanged outputs/acceptance and comparable PP. No persistent allocation. |
| Wider draft prefetch and four query heads per block | Rejected: 64/128-value prefetch and four-head sharing lose to the retained 32-value/two-head layout. |
| Context-aware Q4 draft cost and censored full acceptance | Retained: measured attention growth improves C1 d32K TG by 3.0%; shallow TG, PP, greedy AR agreement and repetitive throughput are retained. Choices remain deterministic from private history and position. |
| Width-weighted acceptance feedback | Retained: removes block-width bias under the controller's geometric model. Q4 C1 d0 TG improves 7.7%, with modest d32K/d64K gains, matched PP and exact greedy/cached sampled replay. Repetition retains full acceptance. |
| Fixed three-proposal default | Rejected: helps the ordinary d32K prompt but slows d0. The adaptive controller remains the default. |
| Context cost without handling saturated acceptance | Rejected: improves ordinary d32K output but slows perfect-acceptance repetition; saturation must allow wider-block probes. |
| Eight lanes per verification head | Not promoted: higher register use and less consistent gains across draft widths than the 16-lane variant. |
| KV sharing between adjacent verification rows | Rejected: byte-exact, but slower at 2K/32K than independent row tiles. |
| Eight-position verification tiles and LDS-only barriers | Not promoted: byte-exact, but gains are small or mixed across 3/7/8 rows, with shallow regressions. |
| Next-tile KV prefetch | Rejected: 144 byte-exact controls through 64K; scalar AR is flat and verification is slower. |
| Query-row workgroup ordering | Not promoted: 144 byte-exact controls; component gains are at most 1.5%, and padded row groups regress. |
| FP32 KV staging in shared memory | Rejected: 192 byte-exact controls; sharing conversions does not offset the extra shared-memory cost. |
| Distributed encoded K/V loads | Rejected: 144 byte-exact partial/output comparisons through 64K; plane-, token- and vector-interleaved loads are slower. |
| Sixteen-row Q5 activation staging in smaller token groups | Rejected: byte-exact on three cold-weight shapes; halving or quartering shared memory adds more work than it saves. |
| Sixteen-row Q5 row-first FMA operand schedules | Rejected: byte-exact on three cold-weight shapes; no throughput gain. |
| Sixteen-row Q5 packed or lane-broadcast headers and scalar scales | Rejected: byte-exact on three cold-weight shapes, but slower than the existing shared-scale decoder. |
| Unsigned Q4/Q5 coefficient conversion | Rejected: byte-exact on three sixteen-row cold-weight shapes, but no useful speed gain or register reduction. |
| Sixteen-row Q5 wave32 | Rejected: byte-exact on three cold-weight shapes, but substantially slower than wave64; the smaller row group also spills registers. |
| Native 32-row Q5 tiles | Rejected: byte-exact on three cold-weight shapes, but slower than paired 16-row tiles across one-to-four output rows per lane; wider accumulators also spill or use private memory. |
| Compact sixteen-row Q5 staging and register activation sums | Rejected: byte-exact on three cold-weight shapes; smaller LDS allocations and alternate bank layouts do not offset the extra work. |
| Batched DFlash2 selector | Retained: exact private token chains/probabilities, two launches per saturated C4 cycle instead of 56, and reused FFN scratch. The focused release pair gains 1.3% TG; C1 PP/TG and output are retained. |
| Six fixed drafts at Q4 C2 | Rejected on perfect-acceptance repetition: exact output, but slower than adaptive. |
| Paired Q4 greedy verification costs | Retained: accounts for the projection cost jump above eight rows. C2 generation improves 20.0% on the Italian/Chinese pair, 12.1% on the pangram/train pair and 21.3% at d32K; repetition, AR equality, C1 PP/TG and private sampled replay are retained. |
| Four-request Q4 greedy verification costs | Retained: accounts for the projection jump above sixteen rows. Ordinary C4 generation improves 26.9% and 13.0%; repetition, d32K PP/TG, AR output and private sampled replay are retained. C1 DFlash2 remains unchanged. |
| Joint C4 draft allocation | Rejected: the Italian/Chinese control retains AR output but proposes 603 tokens for 297 accepted, versus 527 previously, and runs slower. Private sampled/RNG/snapshot tests pass; no production change retained. |
| Two interleaved ten-row Q5 groups | Rejected: byte-exact with production compiler settings, but slower than the existing sixteen-plus-four split on cold-weight FFN projections. |
| IQ4_XS staging both phases and compact shared memory | Not promoted: byte-exact on cold-weight FFN projections, but no consistent gain; the down projection regresses. |
| Exact FP16 integer coefficients with mixed FP32 FMA | Rejected: 2,097,152 instruction controls and both 32-row Q5 FFN projections are byte-exact, but packing coefficients and using mixed FMA runs 4–6% slower. Activations and accumulation remain FP32; no production change. |
| Draft blocks wider than seven proposals | Not attempted: the cached Q4 DFlash2 artifact declares an eight-token block, including the anchor. Its metadata limit is retained. |
| Six-request Q4 greedy verification costs | Retained: ordinary C6 controls improve 6.3% and 16.5%, with a focused 4.3% d32K gain. PP, full acceptance, AR output and private sampled replay are retained; saturated history keeps the full-block probe. |
| Eight-request Q4 greedy verification costs | Retained: ordinary C8 controls improve 10.4% and 13.6%; repetition, d32K, PP and AR output are retained. Other cohort/mode decisions and sampled replay are unchanged. AR remains faster on the harder pair, so profitable fallback still needs work. |
| Three fixed drafts at Q4 concurrency | Rejected as a default: helps C2 low acceptance but slows the higher-acceptance pair and C4 repetition. |
| Two/three query heads per shared KV tile | Rejected: byte-exact, but slower than six-head sharing at 2K and 32K. |
| Wave64 scalar projections and uniform row addresses | Not promoted: cold-weight gains are flat or mixed; some Q4/Q6 shapes regress. |
| Copied weights, including a complete GGUF allocation | Not promoted: the complete-copy control has 3–8% lower cold-weight throughput than mapped weights, with identical outputs. |
| Exact zero-exponent fast path, scalar score broadcast and LDS-only barriers | Not promoted: byte-exact, but no useful gain after shared KV staging. |
| Register-cached decode RMSNorm, tiled C1 argmax and snapshot sizing | Retained: original normalization arithmetic, existing exact argmax and no logits download merely to count them. Matched Q4 C1 AR gains 1.3–1.5% at d0/d32K, with unchanged output and comparable PP. |
| Streaming Q4/Q5 payload loads | Rejected: byte-exact cold-kernel gains regress full-model AR by about 1%. |
| Q6 format specialization and paired/split SSM projections | Not promoted: byte-exact, but negligible gains or regressions; split SSM adds launches. |
| Precomputed affine input sums | Rejected: byte-exact, but no consistent cold-projection benefit after including the preparation pass. |
| Gate/up input sharing, including four simultaneous dots | Not retained: exact kernel outputs and verification checks, but the paired implementation stays flat in full-model Q4 C1 AR despite cold-kernel gains. |
| Huge-page-backed immutable weights | Retained for the focused Q4 C1 phase: identical encoded bytes and greedy/cache results, approximately 2% more TG at d0/d32K, comparable PP and unchanged process RSS. Parallel chunk copies keep warm readiness below one second; cold loading and other modes still need qualification. |
| Residual/RMSNorm fusion after register caching | Rejected: byte-exact and faster as a component, but full-model AR is slower. |
| Separate gate/up waves, staged inputs, fixed FFN geometry and `-O3` | Not retained: byte-exact controls, but no useful isolated mapped-weight gain. Stage weights as production does; device-allocated microbenchmarks overstated earlier gains. |

Current focus: **Q4_K_XL, C1 autoregressive**, at shallow and long context.
First beat the pinned llama.cpp AR controls, then qualify the improvements with
Q4_K_M DFlash2. Next apply and qualify them on Q8_K_XL, first AR and then
DFlash2. Resume C>1 optimization only after those single-user steps.
Keep the work on one PR branch with incremental, reviewable commits.

Overall target: match llama.cpp generation speed, aiming for a further 10%,
on Q4_K_XL and Q8_K_XL with AR and **Q4_K_M DFlash2** at C1/C2/C4/C6/C8.
Cover both shallow (d0–d16K) and long-context (d32K–d128K) token generation:
screen d0 and d32K, then expand where needed to isolate or qualify the change.
Preserve prompt-processing speed, greedy AR/speculative agreement and sampled
replay. Check cross-engine differences against each engine's AR output.

Iterate with one affected Q4 shape and one control. Do not refresh the
full benchmark sweep during exploration. Repeat only to resolve noise or a
failure. Qualify the other configurations after the focused optimization phase,
or earlier only when a specific correctness concern requires it. Remote GPU time
is limited.

Published workload numbers live only in [benchmarks](BENCHMARKS.md); source and
model qualification live in [evaluation](EVALUATION.md).
