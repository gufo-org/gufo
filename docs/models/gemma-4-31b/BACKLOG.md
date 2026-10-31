# Gemma 4 performance backlog

Open performance work for the Gemma 4 engine (31B, 31B QAT, 26B-A4B), as of
2026-10-23 on branch `feat/gemma-q8-perf`. Measured results, including every
rejected attempt, are in [EXPERIMENTS.md](EXPERIMENTS.md) and the
[26B log](../gemma-4-26b-a4b/EXPERIMENTS.md); read the matching rows before
retrying anything listed here. Work kept on experimental branches is listed
under [Experimental branches](#experimental-branches-refused-or-on-hold).
Impact figures are estimates on 31B UD-Q4_K_XL tok/s unless stated.

## Where the time goes

Prefill, one 2048-token chunk (31B UD-Q4_K_XL, rocprof):

| Depth | Projections | Global attention | Sliding attention | Other kernels | Chunk |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3.30 s (91%) | 0.05 s | 0.10 s | 0.17 s | 3.6 s |
| 16K | 3.30 s (~77%) | ~0.73 s (~17%) | 0.14 s | 0.17 s | ~4.35 s |
| 32K | 3.40 s (~66%) | ~1.48 s (~28%) | 0.15 s | 0.17 s | ~5.2 s |
| 128K | ~3.4 s (~35%) | ~6 s (~60%) | 0.15 s | 0.17 s | ~9.6 s |

Projections cost the same at every depth; global attention grows linearly
(~46 ms per 1K of depth). Projections run Q4_K at ~38–39 TFLOP/s and global
attention at ~35, against a measured 55.4 TFLOP/s binary16 WMMA ceiling
(`tools/bench/gfx1151_peak.hip`).

Decode, autoregressive (31B Q4, per token): ~84 ms of weight reads (~94% of
the measured 239 GB/s), ~0.6 ms of attention at d0 growing to ~36 ms at 128K
(global-layer KV, 50 KiB per cached token), ~6 ms of small kernels and launch
gaps. MTP verify cycles at d0 (~101 ms GPU): target verification projections
~93%, drafter ~3–5%, attention ~2%.

UD-Q8_K_XL (2026-10-23): a pp2048 chunk at d0 takes ~3.37 s, the projections
at 37–44 TFLOP/s (66–79% of the ceiling; Q8_0 decodes to binary16 more
cheaply than the K-quants, so Q8 prefill runs ~6% ahead of Q4). AR decode
reads ~35 GB per token at ~229 GB/s (~96% of the measured bandwidth); MTP
verification takes 1.72–1.78 × the UD-Q4_K_XL cost at d0.

## Open items

| # | Item | Area | Effort | Est. impact | Confidence |
| ---: | --- | --- | --- | --- | --- |
| 1 | MTP verification projections: close the remaining gap of the WMMA verification GEMV to the AR GEMV (after the 2026-10-21 grouped layout; Q8_0 joined 2026-10-23) | Decode MTP, all depths | Very hard | MTP +0–2% | Low |
| 2 | 8-bit KV cache | Decode at depth, memory | Very hard, quality risk | AR +5% at 32K, +14% at 128K; MTP more; about half the KV memory | Speed medium, quality unknown |
| 3 | MTP verification attention at five or more rows | Decode MTP at depth | Hard | +2–5% at 32K, more at 128K | Low |
| 4 | Prefill projections past ~70% of the WMMA ceiling; Q/K post-processing in the QKV GEMM epilogue | Prefill, all depths | Hard | +2–5%; the epilogue ≤2.2% (not bit-identical) | Low |
| 5 | Global prefill attention: the shared-LDS-arena variant of two heads per wave (the plain variant spilled and was rejected) | Prefill at depth | Hard | ~1.5% at 16K by the PV probe, more at 128K | Low |
| 6 | AR decode GEMV | Decode AR | Hard | +2–4% | Low |
| 7 | Measure and validate MTP cycle cost estimates | Decode MTP, 31B and 26B | Medium | Remeasured tables (2026-10-05, -20), byte-scaled Q8 prices and the Q8 replay (2026-10-23) found nothing above 0.2%; the 7.5% oracle gap needs information only verification gives | Very low |
| 8 | MTP siblings: done for sampled dense targets (drafts skip top-p, +1.8–3.2% on 31B); a second sibling per depth replays at +0.5–1.5% sampled, greedy unmeasured (paused) | Decode MTP | Hard (executor and kernel sibling layout); replay first | 31B sampled +0.5–1.5% | Medium |

### 1. MTP verification projections

`wmma_gemv.hip.cpp`. The kernel stays 5–13% behind the AR Gemma GEMV on equal
bytes (cold, one row: 5376×16384 247 vs 220 µs). The 2026-10-16 "16 cache
lines per instruction" diagnosis is disproven: load-only probes at the
kernel's launch shape measure 209–212 GB/s for every pattern — per-lane row
loads, cooperative 4 × 128 B full lines, 16 × 32 B or 8 × 64 B pieces — and
k-splits lose 5–10%, so the load instructions, request size and extra waves
are all dead levers (2026-10-19, the [experiment
log](EXPERIMENTS.md)). The cap is the walk geometry: the launch marches
through the super-blocks column-locked, while the AR GEMV's waves walk
contiguous k slices of four rows (225–237 GB/s; a linear stream 240). Waves
sharing the 16-row group over contiguous ~8-super-block slabs recover
216–225 GB/s, but the full kernel on that walk loses 3–83% everywhere except
5376 × 16384 at one to five rows (−0.8 to −2.4%): with waves at different
super-blocks, the shared LDS activation tile becomes per-wave L2 traffic,
about 3.5 × the weight bytes. Per-wave LDS tiles do not fit 64 KB, so what
remains is a walk that does not multiply A reads (unknown) or a repacked
weight layout, which the AR GEMV shares (doubles projection memory). The
~7% math term is untouched. 2026-10-21: the repacked layout landed at no
memory cost by replacing the GGUF order whenever a drafter is loaded (the
AR GEMV is then unused): 128-row groups storing each super-block's rows back
to back lift the walk to 220–230 GB/s, the kernel 0–9% per shape, MTP tg128
+3.4% at d0 (EXPERIMENTS.md). What remains is the math term and the last
~2–5% to the AR walk. Verification must keep one arithmetic per format
so it still equals decode bit for bit (`gemma4.projection_ops`); the slab
rewrite showed the accumulation may be restructured (slab partials in slab
order) while keeping width invariance and the FP64 bound.

Tried since (EXPERIMENTS.md): a fused q/k verification walk (rejected,
2026-10-21: fewer, taller blocks fill the WGPs worse, MTP −0.6%) and split-K
on the short-row shapes (rejected: the reduce pass and launches cost more).
2026-10-23: Q8_0 runs on the WMMA kernel from 2048 outputs and 4096 inputs,
grouped like the K-quants (31B UD-Q8_K_XL code-edit and repetitive +22–23%,
two users +10%); shorter Q8_0 shapes stay on the shared small-batch kernel,
which reads them faster at one to three rows.

### 2. 8-bit KV cache

The only large lever left for long contexts: decode streams the global-layer
KV once per token (or verify cycle, plus once per draft step in the drafter's
global layer). It changes numerics, so it needs a quality decision first:
target KL against the scalar oracle (`gemma4.target`), long-context and
functional suites. The format touches decode and verification attention
(`RowSplitAttentionKernel`), prefill attention (`attention_wmma.hip.cpp`),
QkvPost's cache writes, snapshots and the disk cache. Prefill attention is
compute-bound, so it gains little there.

### 3. MTP verification attention, five or more rows

`kernels.hip.cpp`, `RowSplitAttentionKernel` and `LaunchSplitAttention`. Two
to four rows now run on one wave per head (−28% for two rows at 32K). From
about four rows the kernel is compute-bound at ~270–300 µs per added row and
layer at 32K, ~50% of the unpaired FP32 FMA ceiling. Paired FMAs (VOPD
through opaque converted floats) were neutral, so the cost is the per-row score
reductions (`ReduceScatterN`), softmax statistics (`WaveMax`, `WaveSum`) and
the per-key `readlane` broadcasts. On 2026-10-19, foldable DPP exchanges
removed old-value copies from those reductions: five/eight/sixteen rows at 32K improved by 3.7/5.9/6.3%, with
bit-identical outputs, and both draft-policy tables were adjusted from matched
measurements through 64K ([experiment log](EXPERIMENTS.md)). On 2026-10-20,
coalescing the hd512 score reductions across four keys and two rows per
butterfly, folding `WaveMax` into `v_max_f32_dpp`, specializing `kStaged` for
zero ring/64 rotated pairs, and running 9–12 rows in one block of 3–4 waves per
head cut another 6–21% at 32K–64K (5/8/9/12/16 rows at 32K: −6.8/−7.8/−21.3/−9.1/−7.0%),
again bit-identical. Rows must keep
the arithmetic of single-row decode (the op test compares every verification row with its
single-row evaluation bit for bit). The draft cost tables in
`draft_policy.cpp` hold verification costs by rows and depth. Kernel A/B at
each table depth, times the number of global layers, estimates the affected
component; validate complete operation costs and any resulting scheduling
change separately (open item 7).

### 4. Prefill projections

`moe_prefill.hip.cpp`, `RoutedHalfKQuantKernel` (dense and routed). Drop-work
probes on the Q4_K gate/up (25 ms): without WMMAs 9.5 ms; without the weight
decode −12.5%; without the global → LDS staging −8%; without barriers −3%.
The decode only costs where it shares the SIMD issue with the WMMAs, and the
WMMA A operand must be replicated in both half-waves, so each row is decoded
twice. Rejected: 128-token tiles (three blocks per WGP), unpadded XOR-swizzled
activation rows (registers), 16-wave 256-row blocks, a cross-half decode swap,
scale-conversion splits for Q5_K and Q6_K (registers). A real gain probably
needs a different decode scheme.

Load-time weight copies trade memory for this work (`--prefill-weights`,
2026-10-21, opt-in): `compact` +3.3% pp2048 for +18.9 GiB, `decoded` +6.5% for
+43 GiB on UD-Q4_K_XL. No variant under ~4 GiB gained reliably (direct GGUF
register loads −0.7%, a binary16 scale sidecar +1.5%), and decode cannot read
the compact copy without ~10% more bytes per token. Binary16 copies of the
k >= 8192 projections (down, o) stay rejected: exact K chunks, 2D launch
regions and a register prefetch all still lose 7–55%, and their ceiling over
compact is ~1–1.5% of the projections (EXPERIMENTS.md). On the 26B-A4B the
modes copy the K-quant expert [gate | up] weights (+1.0% pp2048, +9.0 GiB).

Q8_0 (2026-10-23): Flash-Next's binary16 GEMM serves it. A per-shape plan
sweep gave the 31B's 5376-output projections a seven-tile walk (+4.4%
pp2048), and GeGLU now runs in the [gate | up] epilogue on the FP32 sums
(+2.8%, bit-identical), which made the `decoded` Q8_0 copies pointless
(dropped). Every other plan was slower on [gate | up], QKV and global
[q | k]. What is left on Q8 is the Q/K post-processing (~77 ms per chunk,
~2.2%): folding its per-head norms, rope and KV writes into the QKV GEMM's
epilogue would save the FP32 round trip, but cannot reproduce the separate
pass's summation order, so it needs a quality check; on 2026-10-23 it was
set aside in favour of bit-identical work. The six binary16 MLP
layers of UD-Q8_K_XL fuse their GeGLU too (−10.6 ms per chunk).

### 5. Global prefill attention redesign

Mostly explored. Two heads per wave over 128 dims was built and rejected
(2026-10-05: 256 VGPRs with spills, pp2048 at d16K 437 → 354 tok/s); the
64-dim variant with one shared LDS arena was gated by a PV probe at ~9% of
the layer (≈1.5% of pp2048 at d16K) and not pursued; the 2026-10-17 layout
sweep changed nothing (EXPERIMENTS.md). Only the arena variant remains.

`attention_wmma.hip.cpp`. After the 2026-10-17 rows the tile loop issues ~350
instructions per wave and 16-key tile, 16 of them WMMAs; VALU and WMMA share
the SIMD issue, so cutting instructions maps almost one to one to time. Probes
cap the obvious redesign: loading every K/Vᵀ fragment once per two products
saves only 7%, removing all barriers 5%. Two heads per wave would halve the
fragment reads but needs eight score partials, beyond 64 KB of LDS without
reusing one arena for K, the partials and Vᵀ. Larger key tiles do not fit in
64 KB. The kernel's outputs have been kept bit-identical; keep doing so or
revalidate quality.

### 6. AR decode GEMV

Already at ~94% of the measured DRAM read bandwidth; past work is in the
decode GEMV rows of EXPERIMENTS.md.

### 7. Measure and validate MTP cycle cost estimates

Added 2026-10-20. The controller verifies another proposal while its expected
tokens per millisecond of extra work match the cycle's rate, using drafter
and verification prices from `draft_policy.cpp`. Only price ratios steer it.

What is known. The essay maximum-gap flag after `422fbe4d` is one cycle that
adds two sibling rows and emits two tokens instead of one (164.9/167.4 vs
152.1/152.3 ms); no other cycle grows
([per-cycle trace](artifacts/mtp-cycle-trace-2026-10-20.json)). At ~31K on
31B, measured verification is 0.92–0.95 of its price, flat over 2–16 rows,
and drafter steps 1.11–1.14: the table holds QAT-drafter steps; the
standard assistant's measured 8–12% slower from 4K on and 4% faster at d0
(EXPERIMENTS.md, 2026-10-05). Host time between cycles is ~0.02 ms. Swapping
in those remeasured tables then measured neutral end to end.

Next steps, in order:

1. Done 2026-10-20: drafter steps remeasured in place for the 31B assistant
   and the 26B Unsloth drafter; prices now within ~6% of measured ratios,
   throughput unchanged (EXPERIMENTS.md). Other target quantizations (Q8,
   QAT) and drafters keep these tables. Pricing by the loaded files' bytes
   was tried 2026-10-23 and rejected: accurate UD-Q8_K_XL prices made the
   policy draft more on prose for no more accepted tokens (−1.2%). The replay
   agrees (−0.3% on 1,954 Q8 chains scored at the measured costs): on those
   chains the calibration overestimates positions 2–6 by 0.03–0.05, and
   cheaper prices let those drafts through. No price set or replay estimator
   beats the shipped policy by more than 0.2% (EXPERIMENTS.md; patch in
   `artifacts/verify-price-scale-2026-10-23.patch`).
2. Bound the gain first: replay labelled chains (`gemma4_draft_replay
   simulate` over a `--draft-policy fixed --draft-tokens 15` trace) with each
   cycle's measured costs (oracle prices) and with tables perturbed by ±2/5/10%;
   count changed decisions and their expected tokens per millisecond. Stop
   if perfect prices change little.
3. Extend the trace to other depths, the 26B (text-dependent expert routing)
   and sampled decoding, probing both sides of launch boundaries; fit the
   smallest table that explains consequential errors.

Rules: hold acceptance calibration constant; qualify kernel changes with
prices unchanged, then new prices as a separate scheduling change, on both
models, with prose and high-acceptance code. Judge streaming gaps per
emitted token (or p99 alongside tokens per cycle), not as one maximum: a
longer cycle that emits more tokens is not a pause regression.

### 8. Sampled MTP siblings

Done 2026-10-23 for the dense family: sampled drafts skip top-p, which keeps
a sibling candidate beside each draft (31B UD-Q8_K_XL story +1.8–1.9%,
code-sampled +2.5–3.2%; EXPERIMENTS.md). The 26B-A4B keeps top-p: its
verification rows cost too much for siblings to pay. Replay any next
variant with `tools/gemma4/proposal_replay.py` over a fixed 15-draft
`--draft-trace` before building it. Candidate 3 is untouched. The original
analysis follows the next subsection.

#### Second sibling per depth (paused 2026-10-23)

After a rejected draft and a rejected first sibling, a second sibling drawn
from q without both is checked against the next residual
r2 = norm(max(r1 - q1, 0)) (recursive rejection sampling; output still
follows p exactly). Replay over 1,038 sampled 31B UD-Q8_K_XL cycles: the
first sibling is accepted after 49.5% of rejected drafts, a second only 6.7%
(the first already takes most of the mass the target moves away from the
draft). Best fixed 16.98 → 17.06 tok/s (+0.5%), per-cycle oracle
19.87 → 20.16 (+1.5%); the first sibling's replay (+1.3/+2.4%) became
+1.8–3.2% live, so expect +0.5–1.5%. Dense targets only (the 26B's siblings
do not pay).

The single-sibling layout is built into the GPU side:

- Executor segments place sibling i at depth i (position
  first_position + 1 + i, K/V at a spare key); two per depth need an
  explicit depth per sibling row.
- `QkvPost` derives a sibling row's position and spare key from its row
  index; split attention gives a sibling the chain prefix up to its depth
  plus its own key (added in a final step). Both would read the depth table,
  keeping verification rows bit-identical to single-row decode
  (`gemma4.attention_ops`).
- `MoveKey` commits whichever sibling was accepted.
- A second-sibling calibration table and `IncludeSibling` pricing for it.
- Tests: two-sibling frequency test, attention cases with two siblings per
  depth, trace and replay support.

Next step before building: greedy could use the drafter's third-ranked
token as its second sibling, unmeasured because the trace keeps only the
runner-up. Log the drafter's top three in greedy cycles, replay greedy and
sampled together, and build the per-depth layout only if the sum clears ~1%.

#### Original analysis

Added 2026-10-23. Sampled decoding is bound by acceptance, not overhead:
without `--draft-trace` the host adds nothing measurable (decode GPU idle
1.9% sampled against 1.8% greedy), the same kernels just run ~8.7% more
often, and the calibrated policy beats fixed and confidence draft lengths
on story and code-sampled (EXPERIMENTS.md, 2026-10-23). The lever left is
how many extra tokens a cycle can win when a chain draft is rejected.

A sibling is one extra verification row holding the drafter's alternative
at a depth; if the chain's draft there is rejected, the sibling can be
accepted for one more token. On UD-Q8_K_XL a row costs ~1.6 ms of a
~163–188 ms verification (2–16 rows), and the calibrated policy adds one
only when its estimated hit rate pays for it.

Greedy siblings are the drafter's runner-up and always exist. Sampled
output must follow the target's distribution exactly, so verification is
recursive rejection sampling (`VerifyWithSibling`, `engine.cpp`) and the
sibling must be drawn from the drafter's proposal q with the draft removed
(`SampledAlternative`). q is the drafter's distribution after the request's
filters; at top-p 0.95 it collapses to one token whenever the drafter is
fairly sure, leaving nothing to draw. Traced HTTP cycles on UD-Q8_K_XL
(greedy prose and code-write against sampled story and code-sampled, so
directional):

| | Greedy | Sampled |
| --- | ---: | ---: |
| Drafts whose proposal holds one token (q ≥ 0.999) | 40% | 68% |
| Drafts whose unfiltered drafter top-1 ≥ 0.999 | 40% | 41% |
| Cycles with a sibling | 71% | 25% |
| Sibling hit rate where its draft was rejected | 0.43 | 0.38 |
| Sibling tokens per cycle | 0.187 | 0.062 |

About 27% of sampled drafts lose their sibling to top-p truncation alone.
Matching greedy's 0.187 sibling tokens per cycle would add ~0.125 tokens to
~3.5 emitted per cycle: sampled +3–4% before the extra rows and before any
change in chain acceptance.

Candidates, all exact (rejection sampling stays exact for any q):

1. Draw drafts from a less truncated drafter distribution (its own top-k,
   no top-p, or a drafter-specific top-p or temperature). Siblings stay
   available more often, but chain acceptance, the overlap of p and q, may
   fall; the balance needs measuring.
2. Several siblings per depth drawn without replacement (recursive
   rejection sampling over k candidates, as in "Recursive Speculative
   Decoding"), priced per row by the policy.
3. Multi-candidate verification such as SpecTr's optimal-transport
   matching: a larger rewrite of verification.

Next steps, in order:

1. Bound it offline, with no engine change. The 2026-10-20 study's replay
   already labels sibling hits (`VerifyWithSibling`) and found siblings worth
   1.8% (31B sampled) as they are. Extend the cycle trace to log each
   verified row's target candidates and the drafter's logits, then extend
   `gemma4_draft_replay` to simulate expected tokens per cycle and
   per millisecond for each proposal choice above, at the measured row
   costs. Stop if the best choice gains under ~1%.
2. Build only the winner; check seeded output distributions against the
   target (`gemma4.kernel_ops` style) and A/B story and code-sampled with
   ≥48 seeds (12-seed runs swung ±4%), on 31B and 26B.

## Opt-in levers (quality trade, default off)

Ideas that give up a measured amount of quality for speed. Production paths
retain quality by default, so each lever is a product decision: one canonical
long option, off by default, shipped only after a quality decision
(`gemma4.target` KL against the scalar oracle, the functional suites, and
long-context tasks for anything touching attention). Gains are 2026-10-17
estimates, not measurements.

| # | Lever | Speeds up | Est. gain | Quality risk | Effort |
| ---: | --- | --- | --- | --- | --- |
| L1 | Relaxed MTP acceptance: built, refused (see below) | MTP decode | Measured +5–23% sampled | High: sampled reasoning loops | Done on a branch |
| L2 | 8-bit KV cache (open item 2) | Decode at depth, KV memory | AR +5% at 32K, +14% at 128K | Low | High |
| L3 | Target output-head shortlist | AR and MTP decode | ~+3% | Medium: tokens outside the list become impossible | Low–medium |
| L4 | Binary16 dot products in decode and verification attention | MTP verification at five or more rows | +2–4% at 32K, +5–8% at 128K | Low: beyond the default decode KL limit | Medium |
| L5 | Binary16 prefill intermediates | Prefill | +1–1.5% | Very low | Low–medium |
| L6 | Sparse long-context prefill attention | Prefill at depth | 1.5–2× at 128K, +10–15% at 32K | Medium–high | High (research) |
| L7 | W4A4 prefill projections (INT4 WMMA): prototyped, on hold (see below) | Prefill, all depths | Measured +53% pp2048 d0, +35% d32K | Medium: decode after an int4 prefill KL 0.10 | Done on a branch |
| L8 | NPU prefill offload | Prefill | 0–10% | Low–medium | Very high |

### L1. Relaxed MTP acceptance

Built and refused 2026-10-20; see
[Experimental branches](#experimental-branches-refused-or-on-hold). The
original proposal:

Greedy verification accepts a draft only when it equals the target's argmax
(`Session::FinishCycle`, `engine.cpp`); on prose at ~31K depth 42% of drafts
pass (235/554). Many rejections are near-ties. Rule: accept the draft when
p(draft) ≥ τ·p(top) and the target's entropy is above a threshold (Medusa's
typical acceptance), top-1 only otherwise. The GPU argmax (`ReadArgmax`) would
also return the top logit, the draft's logit and the log-sum-exp; the
calibrated draft policy then learns the higher acceptance by itself. Measure
before building: log per verified row the top-1, the draft's logit gap and the
entropy, then replay thresholds offline for tokens per cycle against
divergence from greedy. Sampled mode already uses exact rejection sampling; a
relaxed variant there changes the output distribution.

### L2. 8-bit KV cache

Open item 2. hipfire's default KV cache on every GPU is Q8_0 (32-value blocks,
one binary16 scale) and its quality gates are measured with it: a working
reference for the format.

### L3. Output-head shortlist

The 31B's tied head (`token_embd`, Q5_K 262144 × 5376) is ~970 MB of the
~18.8 GB read per token. A head over the first ~98K token ids plus a
per-request table of used and special tokens, with a full-head fallback when
the anchor falls outside it (the Qwen3.8 Flash-Next drafter's scheme,
[its README](../qwen3.8-flash-next/README.md)), saves ~3%. The 26B's head rows
are already ordered by merge rank, so its shortlist is a row prefix. Risk:
non-English text and rare tokens.

### L4. Binary16 attention dot products in decode and verification

Round Q to binary16 and score with `v_dot2_f32_f16` (prefill attention already
uses binary16 Q). Verification and single-row decode keep one arithmetic, so
their bit-identity contract holds; only today's FP32 outputs change. Assessed
earlier and not adopted as a default: it moves decode beyond the declared
decode KL limit (EXPERIMENTS.md), so only an opt-in remains.

### L5. Binary16 prefill intermediates

Store the projection outputs that feed the norms and Q/K post-processing as
binary16 instead of FP32; those passes are bandwidth-bound at 185–203 GB/s.

### L6. Sparse long-context prefill attention

Global layers attend to attention sinks, a local window and the top-k key
blocks chosen by a cheap block score. Research grade; needle and long-context
suites decide.

### L7. W4A4 prefill projections (INT4 WMMA)

Prototyped 2026-10-23 and on hold; see
[Experimental branches](#experimental-branches-refused-or-on-hold). The
original proposal:

hipfire runs dense and MoE prefill through `v_wmma_i32_16x16x16_iu4`: weight
nibbles feed the WMMA directly and activations are quantized to int4 per 128
values (MSE-clipped). On gfx1151 the instruction sustains 110.7 TOPS, twice
binary16 and int8 WMMA (57.1; four independent chains per wave, measured
2026-10-17). On Qwen3.8-27B it took Halo prefill from 450 to
581 tok/s (+29%); hipfire's later 27B figures are 737–796 tok/s against
Gufo's 656 (pp2048, binary16 WMMA). Cost: WikiText-2 KLD against BF16
0.057 → 0.076 (27B dense) and 0.057 → 0.088 (Flash-Next MoE, top-1 agreement
91.8% → 89.7%); hipfire's agentic task scores stayed within seed spread.

Two blockers for Gufo. hipfire's weights are FWHT-256-rotated offline (its MQ
formats), which keeps activation outliers small enough for int4; GGUF Q4_K
weights are unrotated, so A4 on them would likely cost much more, and rotating
needs a requantized artifact from the BF16 source (hipfire uses GPTQ). And
Q4_K's per-32 sub-block scales and minimums end the integer accumulation every
two WMMAs, where hipfire's symmetric groups span 128–256 values. First step,
before any kernel: simulate A4 on Gemma activations (per-128 groups, with and
without a Hadamard rotation) and measure the KL with `gemma4.target`. W4A8 is
no shortcut: int8 WMMA runs at the binary16 rate here, and Gufo's earlier W4A8
path was slower.

### L8. NPU prefill offload

hipfire ships one NPU route: on Qwen3.8 Flash-Next (MoE) the trunk input
projections of one 8192-row bank run on the XDNA2 NPU (int8 weights from its
6-bit trunk, per-row int8 activations, BF16 output) while the GPU runs the
other bank; 16K TTFT 7,878 → 7,085 ms (−10%). It needs Linux 7.3-rc7 with the
IOMMU PerfOpt patch, plus hipfire's own AIE2P toolchain and `amdxdna` runtime
(no XRT). Measured there: a concurrent NPU GEMM loop slowed GPU prefill by 10%
through the shared fabric, and the NPU reaches ~22–26 int8 TOPS of its 59
peak. Their dense co-op (27B gate/up and down columns split between GPU and
NPU) was killed at +1.2% per step, bound by the NPU kernel's DDR traffic.
For dense Gemma prefill the ceiling is a column split worth at most ~10%, and
Q4_K's sub-block scales do not map onto an int8 NPU GEMM without per-32 partial
sums. Revisit after the PerfOpt kernel, starting with the GPU slowdown under
an NPU load.

### Ruled out

- Binary16-accumulate WMMA: same rate as FP32 accumulate.
- A lower-bit drafter: −9% code acceptance, a net loss. No matching row in
  EXPERIMENTS.md; the
  [`feat/gemma-mtp-drafter`](https://github.com/franzmoca/gufo/tree/feat/gemma-mtp-drafter)
  branch proposes a 4-bit drafter quantized from the BF16 file (below), so
  recheck this before relying on it.
- Windowing the drafter's global attention: acceptance falls with the window.

## Experimental branches (refused or on hold)

Kept off the main line on the fork (franzmoca/gufo); read their docs before
reviving them.

- **[`feat/gemma-mtp-exp`](https://github.com/franzmoca/gufo/tree/feat/gemma-mtp-exp): relaxed MTP acceptance (L1), refused.**
  `--relaxed-acceptance GAP` (off by default) accepts a draft the exact rule
  rejects when it is among the target's top three and at most GAP nats below
  the top. Fast (sampled decode +9–12% prose and essay at 0.5, up to +23% at
  2), but sampled 26B reasoning loops to the length cap (DS4 first 24 cases:
  21 exact, 9 at 0.5, 7 at 1, 1 at 2) and greedy text leaves exact decoding
  within ~50 words (`artifacts/mtp-relaxed-acceptance-2026-10-20.json` on the
  branch). Before revisiting: accept with a probability (a relaxed ratio
  min(1, λ·p/q) that keeps the sampler's randomness), gate on the target's
  entropy, check chat and agent workloads, and pass the same 24-case
  thinking eval against exact.
- **[`feat/gemma-quant-shape`](https://github.com/franzmoca/gufo/tree/feat/gemma-quant-shape): int4 W4A4 prefill sidecar (L7), on hold.**
  The study harness is `tools/gemma4/quant_sim/` there; the sidecar GGUF is
  not in git and is rebuilt with its `sidecar.py`.
  `--prefill-sidecar` loads a BF16-built GPTQ sidecar (14.07 GiB, rotated
  inputs quantized per token and 128 values, iu4 WMMA GEMM) beside
  UD-Q4_K_XL; decode keeps the GGUF bit for bit. pp2048 575 → 881 tok/s at
  d0, 456 → 648 at 16K, 393 → 531 at 32K (`--prefill-weights decoded`: 616 at
  d0). Quality: decoding after an int4 prefill KL 0.10 against exact rows,
  simulated chat replies 0.027–0.028 against 0.021 for UD-Q4_K_XL alone
  (the branch's QUALITY.md and WEIGHT-FORMATS.md).
- **[`feat/gemma-mtp-drafter`](https://github.com/franzmoca/gufo/tree/feat/gemma-mtp-drafter): BF16 drafter study, no change.** BF16 and F16
  drafters now load there; Q8_0 measured as good as BF16 (2.271 against
  2.275 accepted greedy drafts per cycle, paired), so Q8_0 stays. Its
  follow-up proposal: quantize the drafter's layers and projections to 4 bits
  from the BF16 file and repeat the fixed-chain acceptance study (~200 MB
  per draft step; +1–2% MTP decode on 31B, more on 26B).

## Notes from hipfire (2026-10-17)

[warpfront/hipfire](https://github.com/warpfront/hipfire) v0.4.2, read at
`bfb90b9c8`: a Rust runtime with its own quantization formats, kernel toolchain
(PeaceMaker) and dispatch layer (Redline).

- Qwen3.8 Flash-Next on Halo: 2,156 tok/s synthetic pp8192, 2,306 tok/s on a
  32K prompt, 2,386 tok/s on a 16K prompt with the NPU; AR decode 34.1 tok/s;
  greedy MTP 60.2 tok/s on its code prompts. Gufo (2026-10-16): 1,737 tok/s
  pp2048 at d0, 1,624 at 32K, AR 28.3 tok/s. The prefill lead comes from lossy
  defaults (the symmetric IU4 MoE route at BF16-reference KLD 0.103 against
  0.066 on hipfire's own binary16 route, which ran ~1,455 tok/s at pp8192 when
  IU4 became the default; the int4 trunk-activation overlay; the NPU route's
  int8 activations) and from 8192-row chunks against Gufo's 2048-token
  benchmark prompt. The AR decode lead (+20%) comes from none of those (decode
  never takes them) nor from file size (125 GB against Gufo's 115 GB): worth a
  per-kernel Flash-Next decode profile against bytes per token.
- Redline retained PM4 replay of the decode graph: +12.6% on gfx1151 for
  Qwen3.6 35B-A3B (102 → 115 tok/s), a small-active MoE where launches weigh.
  Gufo's HIP-graph probe found the ~1.4 µs per removed launch GPU-side; PM4
  replay is a way past it, mainly for 26B and Flash-Next decode.
- Halo 16-row verify GEMMs (`HIPFIRE_GFX1151_XLDS`) stage activations in LDS
  per block and reach 186–208 GB/s, about Gufo's verification GEMV
  (207–217 GB/s): nothing new for open item 1.
- hipfire's Gemma 4 runtime (`hipfire-arch-gemma4`) runs the 12B, E2B/E4B and
  26B-A4B plus an opt-in EAGLE drafter; its plan lists the 31B as incoming.
  It loads only hipfire's own formats (quantized from BF16 safetensors, no
  GGUF), publishes no Gemma artifact or Strix Halo throughput, and its 0.4.2
  notes say its gfx1151 Gemma defaults change numerics with no KLD measured:
  no Gemma comparison exists yet. Its Gemma KV cache is Q8 by default.

## Measured, no change (do not retry without a new idea)

- Fusing the prefill norms and Q/K post-processing into one another: they
  move 154–230 MB per layer at 185–203 GB/s against a 210 GB/s
  read-plus-write ceiling. (Folding Q/K post-processing into the QKV GEMM's
  epilogue is a different idea; see open item 4.)
- Verification prices scaled by the target's bytes (2026-10-23, open item 7):
  UD-Q8_K_XL prose −1.2%.
- Sampled MTP draft length (2026-10-23): fixed 3 or 5 drafts and the
  confidence policy all lose to the calibrated policy on story and
  code-sampled; the host adds nothing measurable without a trace.
- Q8_0 `--prefill-weights decoded` copies (2026-10-23): slower than the
  fused-GeGLU default; dropped.
- Decode launch gaps through HIP graphs: a 733-kernel dependent chain takes
  7.9 µs per kernel launched on a stream and the same replayed as a graph; the
  gaps are GPU-side. A removed launch saves ~1.4 µs.
- Prefill attention LDS padding sweeps, packed probability stores, interleaved
  row scales, 128-bit Vᵀ stores, DPP-swapped score writes (see EXPERIMENTS.md).

## Housekeeping

- Model card refresh: BENCHMARKS.md's Gufo columns are from 2026-10-17 and
  hold none of the 2026-10-21 MTP or 2026-10-23 Q8 gains (31B UD-Q8_K_XL
  pp2048 557 → ~610, two-user MTP +10–23%); one full sweep (the
  `benchmark-model` skill) at the end of the round.
- `feat/gemma-q8-perf` is local; fork pull requests (#1–#7 so far) go to
  `feat/gemma` and are titled per [the release policy](../../RELEASING.md).

## Methods that worked

- Kernel A/B in one binary: compile HEAD's kernel file with its namespace
  renamed (`git show HEAD:<file> | sed 's/^namespace X {$/namespace X::base {/'`)
  next to the working copy, run both on production shapes, compare outputs
  bitwise and time them in interleaved rounds. Widen the inputs (for example
  queries ×10–30) to exercise underflow and large softmax maximum jumps.
- Drop-work probes: remove one mechanism at a time (results wrong, timing
  only) to see what a kernel's time is made of before redesigning it.
- Instruction counts of a kernel's hottest loop from `--cuda-device-only -S`
  (`tools/prof/isa_mix.py` for whole kernels); on RDNA3 VALU instructions and
  WMMAs share the issue, so the loop's VALU count predicts time.
- `gemma4.attention_ops` outputs hashed to prove bit identity across all its
  cases (a temporary local hash print; not committed).
- End to end: `tools/prof/prof.py` chunk profiles compared at matched
  projection times, because long prefills here drift 2–4% between runs with
  power and temperature; `gufo bench` in ABBA order for tok/s; an HTTP
  ~31K-token document with prose answers for MTP decode at depth (`gufo
  bench`'s synthetic prompt is repetitive and drafts 15 tokens every cycle).
