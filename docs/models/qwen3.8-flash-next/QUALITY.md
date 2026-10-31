# Qwen3.8 Flash-Next quality

## Decode rows at the sparse budget, October 28, 2026

A verification chain that crosses the 2,048-token sparse-attention budget
selects blocks for every row. Its rows before the budget then took the mask
walk, which attends the incomplete tail block (keys 4k..pos) in a tile of its
own on split 0, where the same row decoded alone sweeps those keys densely in
its last tile. The online softmax rounds differently, so positions whose key
count is not a multiple of four (2041, 2042, 2044, 2045, 2046) changed with
the draft widths, and with them with batch timing. Rows that see at most
`indexer_top_k` keys now sweep densely even with a mask
(`AttentionKernel`'s `dense_extent`). Prefill and plain AR rows take the same
route as before.

- `attention_ops_test` checks chains of eight rows starting at 2040..2047
  (one launch, eight splits, compact lists) bit for bit against each row
  launched alone, and that the mask walk differs on some of them (it does
  on 24 rows), so the check can tell the routes apart.
- End to end on a downstream branch (v0.23 base with 8-bit K/V; greedy
  decoding through the engine API with a frontier-logit hash per step):
  one prompt decoded AR to position P and then with MTP from P first
  differed from plain AR at position 2048 for every P = 2041..2046; with the
  change every frontier of the 130 tokens equals plain AR, and plain AR is
  unchanged. Eight IQ prompts decoded up to 3,000 tokens with eight-way
  batched MTP, staggered joins and single-session MTP equal eight-way
  batched AR at every common frontier. Two shuffled 149-question replays
  through `gufo serve` (eight sessions, MTP) are byte-identical on all 149
  answers (135,179 tokens each); before the change, answers diverged run to
  run, every divergence starting past about 2,017 tokens.

## Sampled draft-head costs, October 23, 2026

Based on main `41d93463`, including the expanded greedy-head eligibility.
The new C1 curve measures the smaller sampled draft head at depths
0/4096/32768, with fixed token histories and three warmed complete cycles
per width. Matched ROCm 7.2.3 and 10.1 cycle costs agree within 2%.
Full-head, full target-row, greedy and concurrent cost curves retain their
previous values.
The policy uses committed acceptance history, not measured request timings.
Target arithmetic, sampling filters and rejection correction are unchanged.

An initial candidate widened short resumed replies too early. One identical
12-token answer took 217 → 240 ms, with 12 rather than 10 proposals.
The final policy keeps the initial conservative width until acceptance is
observed, and retains the established costs within two blocks of the output
limit. The serving adapter passes the remaining budget through; the engine
still bounds each physical block. A focused cancellation/resume control now
passes all 84 request/phase checks with identical output, prompt/cache work
and draft counts. Warm 12-token decode is 211.02 → 211.41 ms; the later
eight-token reply is 196.35 → 196.31 ms.

The controller test covers head fallback, configured versus live occupancy,
budget bounds, native-context cost extrapolation and restored acceptance
history. Existing compact-proposal probability, rejection/residual, RNG and
stop checks pass on both compilers. The expanded main-branch vocabulary
eligibility test also passes.

The first short-tail candidate still widened constrained tool output using
the smaller-head estimates. In a 32K thinking/file-edit request, its longest
pause rose from 111.18 to 119.78 ms. Full target-row verification now keeps
the original costs. The focused final replay passes all 63 phases with
identical output and prompt/cache work: the longest pause is 112.87 ms,
and generation remains comparable at 68.84 tok/s. Exact seeded warm replay
preserves function names and argument bytes; generated call IDs are
request-specific.

The short-tail candidate passes the focused `state-edges`, `long-context` and `cache`
functional suites, with unchanged output and prompt/cache coverage. The
1,026 measured phases per compiler include tool constraints, literal tags,
sampled JSON, token limits, cancellation, RAM continuation and disk restart
with text, thinking, legacy tool history and images. Actual C2 lookup,
mixed cohorts, full-head fallback, stops and image/cancellation replay also
pass; the concurrent cost curve remains unchanged.

The six-request default-sampling screen retains 87 measured phases on each
compiler. All requests pass correctness and exact warm replay. The initial
short-tail candidate passes all timing phases; the final build preserves its
outputs and draft counts but flags the first cold code request and one warm
token gap. The focused reverse-order controls pass all 29 phases on each
compiler. Cold prefill is 168.76 → 169.65 ms on 7.2.3 and
167.37 → 167.27 ms on 10.1; the 10.1 warm maximum gap is
69.17 → 69.40 ms. Initial observations remain retained and inconclusive.
Complete extraction, executable URI tests against correct code and
three mutants, and 32K file-edit tool calls with thinking on/off pass their
task checks. Sampled completions may differ: on 10.1, the valid URI test
answer grows from 407 to 478 tokens, taking 5.22 → 6.32 seconds at
83.95 → 80.34 tok/s. This is not a claim of faster completion for every task.
The initial 7.2.3 completed-workflow phase comparisons pass. Initial
request-level timing flags remain in the evidence, including an unchanged
greedy schema request's maximum pause and cold disk restores.

Cloned, identical persisted records pass all focused restore phases on 7.2.3.
The first 10.1 clone comparison still flags a branched restore at
84.68 → 90.30 ms. Alternating controls then measure that same branch at
99.78 ms on the unchanged binary, 97.38 ms on the candidate and 127.97 ms
on the unchanged binary again. Another unchanged-binary restore varies
60.74 → 82.57 ms. This establishes baseline variability; it does not prove
the cause of the storage/copy delay. These disk timings remain inconclusive,
with all captures retained and the 5%/3 ms gate unchanged. The eight-token
greedy fixture does not execute the changed sampled policy.

After the full target-row guard, both compilers also pass a fresh replay of
`state-edges` and `long-context`, including constrained Chat/Responses,
literal tool tags, images, stop/cancellation and resumed seeded output.
All outputs and prompt/cache work match the controls across 289 measured
phases per compiler. Isolated timing flags remain inconclusive: 10.1's quoted
tool-argument stop has a 76.13 → 81.82 ms maximum pause; 7.2.3's short
zero-argument tool reply takes 180.32 → 190.16 ms, and its execution-mode
probe has a 77.59 → 88.58 ms maximum pause. These requests are greedy;
they retain their original policy and do not establish a sampled-policy regression.

Focused standard benchmark controls, one production capture per cell,
before → candidate (tok/s):

| Workload | ROCm 7.2.3 | ROCm 10.1 |
| --- | ---: | ---: |
| CLI AR d0 pp2048 | 1,873.03 → 1,870.44 | 1,856.39 → 1,870.11 |
| CLI AR d0 tg128 | 29.36 → 29.37 | 29.38 → 29.39 |
| HTTP MTP d32K mixed prefill | 1,608.77 → 1,613.37 | 1,600.39 → 1,601.93 |
| HTTP MTP d32K repetitive prefill | 1,611.12 → 1,584.58 | 1,593.06 → 1,591.93 |
| HTTP MTP d32K mixed generation | 40.40 → 40.49 | 42.11 → 42.11 |
| HTTP MTP d32K repetitive generation | 115.34 → 115.37 | 115.00 → 115.06 |

The HTTP rows use the canonical greedy pp2048/tg128 workload. Completion
hashes, prompt/cache counts and draft counts agree within each compiler.
The before binaries include main `41d93463` and the selector fix; the
candidate adds the scoped sampled-head policy. These spot controls do not
replace the standard benchmark tables or their reference numbers.
Frozen binary hashes and source patches identify each arm. The benchmark
wrapper's checkout stamp and environment fingerprint are not a replacement
for that build provenance: 10.1 uses container GCC 15.2 and Clang 24, while
7.2.3 uses pinned Nix GCC 15.3 and Clang 22.

The [migration state](ROCM.md) records the sampled screen and qualification
limits. Initial captures, rejected candidates, source patches,
binary hashes and request traces are retained under
`/tmp/gufo-rocm10-20261023` on the test host. Sampled text and completion length
can change with draft widths; completed-task latency and token throughput
are reported separately.

## LLVM selector scheduling, October 23, 2026

The compiler-dependent loop bound changes scheduling only. F32 queries, stored
F16 keys, each FMA, the reduction tree, top-k ordering and cache contents retain
their original contracts. Clang 22 compiles the original unrolled loop.

On ROCm 10.1 / Clang 24, the selector test passes its independent F64 formula,
exact F32 reduction, precision-boundary, tied-score, causal-tail and chunked
replay checks. A 38,760-token source-code prompt followed by 75 fixed decode
tokens produces 18,624,000 finite, byte-identical logits before and after the
change. The independent MTP audit also passes full-width normalization,
predictor attention, recursive carry, image positions, ragged batches and
exact candidates after differently chunked cache rebuilding.

Matched production controls against `cc5f405a` pass 533 request/phase
measurements on each toolchain: ROCm 7.2.3 / Clang 22 and ROCm 10.1 / Clang 24.
These cover greedy/default-sampled lookup, ordinary generation, C2, filter
boundaries, stops, cancellation, images and exact warm replay. Outputs and
prompt/cache work agree within each toolchain. The gate remains both 5% and
3 ms per request/phase.

After rebasing onto `ab122e44`, all 56 retained HTTP AR phase measurements
pass on each toolchain, including calibration requests and the d32K cached
continuation. Completion hashes and prompt/cache/draft counts agree.
The six-request sampled screen passes all 87 phases on 7.2.3 and 86 on 10.1.
The remaining 10.1 cold-code maximum-gap observation is 74.38 → 79.11 ms;
its focused reverse-order control passes all 29 phases, with the same gap
73.16 → 73.53 ms. The initial observation remains inconclusive, rather than
being replaced with the repeat. Both compilers pass complete extraction,
generated URI tests against correct code and three deliberate defects, and
32K file-edit tool calls with thinking on/off and exact intended file contents.

Compiler versions do not produce identical model logits or seeded text.
Both agree with the independent F32 CPU reference at 28 of 29 positions on
the short cache/cancellation prompt; both disagree at the same final position.
On the fixed long-history continuation, 67 of 75 top tokens agree across
compilers, with median/maximum KL divergence 0.01497/0.33537. Mean next-token
negative log likelihood over 74 fixed labels is 3.8152 on 7.2.3 and 3.8009 on
10.1. These small numerical checks and functional fixtures are regression
evidence, not a statistical language-quality result.

The 10.1 container uses `rocm/dev-ubuntu-26.04:10.1.0-full`, image digest
`sha256:4f5ed1bf6532a4b9920400401b0ad0f706af356dae61e40afefbfd6c8073f0ce`,
with the repository's ROCr queue-memory patch applied. The 7.2.3 controls use
the pinned Nix toolchain and patched ROCr. Loaded runtime paths were verified;
all model jobs run serially, without overlapping builds. Scripts, binaries,
hashes, full request logs, numerical dumps and initial observations remain in
`/tmp/gufo-rocm10-20261023` on the test host. No reference benchmark data change.

### Cross-compiler quantization check

A matched first-layer capture locates an initial difference in Q8 activation
rounding: embeddings, normalized inputs and stored scales agree, but 37 of
327,680 codes differ. The 7.2.3 compiler contracts the scaled product into
`roundf`'s fractional remainder; 10.1 materializes the product first.

An independent CPU calculation follows the reference quantizer's FP32
scale, reciprocal, product and round-away-from-zero steps. Among the 37
differing codes, 10.1 agrees at 32 and 7.2.3 at five. Across the entire capture,
14 codes differ from that reference on 10.1, versus 41 on 7.2.3; fast reciprocal
arithmetic also differs from strict CPU division. Using a separate FP64
calculation with the exact max/127 scale gives 24 versus 13 agreements among
the differing codes. Neither compiler is an exact CPU oracle.

Reproducing the old fused remainder on 10.1 restores the first mixing step
byte for byte, but is rejected: it is less faithful to the independent
quantizer and does not recover the sampled thinking speed. Its warm screen
measures code/prose/thinking at 62.55/44.60/44.91 tok/s, versus unchanged 10.1's
70.51/47.86/45.69. These free-running answers differ; the observations do not
isolate execution cost. No rounding or sampling change is retained. This
operator check does not establish overall language quality.

### Focused performance measurements

Focused production comparison against main `ab122e44`, with the selector-loop
change as the candidate. Each compiler uses its matching patched ROCr runtime;
ROCm 10.1 runs in a container. HTTP AR uses the canonical ordered d0/d32K
pp2048/tg128 history. Sampled C1 rows use model defaults, 128 output tokens and
an exact warm replay. Values below are tok/s, one capture per cell.

| Workload | 7.2.3 main | 7.2.3 candidate | 10.1 main | 10.1 candidate |
| --- | ---: | ---: | ---: | ---: |
| AR prefill, d0 | 1796.29 | 1794.72 | 1791.06 | 1798.38 |
| AR prefill, d32K | 1645.17 | 1650.79 | 1622.34 | 1647.64 |
| AR generation, d0 | 28.91 | 28.90 | 28.90 | 28.96 |
| AR generation, d32K | 28.84 | 28.88 | 28.92 | 28.93 |
| Default-sampled code, warm | 62.66 | 62.66 | 70.50 | 70.51 |
| Default-sampled prose, warm | 48.66 | 48.69 | 47.88 | 47.86 |
| Default-sampled thinking, warm | 54.02 | 54.04 | 45.72 | 45.69 |

The candidate preserves output and prompt/cache/draft counts within each
compiler. Across compilers, sampled text and draft work differ: the thinking
screen takes 38 draft rounds on 7.2.3 and 58 on 10.1. Its slower 10.1 result is
also present on unchanged main; this is not a claim that every sampled answer
is equally fast across compiler versions.

Fixed-history MTP diagnostics separate that variation from execution cost.
Across C1/C2 and widths 1–8, most initial 10.1 cycles are slightly faster.
C1 widths 7/8 initially flag at 72.04/71.52 ms for target verification versus
61.03/65.97 ms on 7.2.3. A reverse-order repeat of unchanged binaries gives
58.67/63.56 ms on 10.1 and 59.04/63.94 ms on 7.2.3. The initial flags remain
unqualified; they do not establish a persistent compiler regression. These
diagnostics replay fixed tokens and report the median of three warmed cycles,
not completed-answer throughput.

Complete default-sampled extraction, executable URI-test generation and
32K file-edit tool calls pass on both compilers, including edits with thinking
on/off and preservation of literal protocol tags. The complete benchmark tables and reference data remain unchanged.

## Exact sampled decode, October 21, 2026

The candidate retains target weights and forward arithmetic. Its compact
verification union contains the best 64 finite unpenalized logits and every
penalized token. An omitted token therefore cannot enter a post-penalty top-k
of at most 64. Existing FP64 penalty/filter/normalization and rejection
correction operate on the original F32 values and vocabulary IDs. Negative
penalties and repeat penalties below one are included in this proof.
Grammar activation, forced reasoning closure, unbounded/larger top-k and
excessive penalty history keep the full-row path.

The independent GPU/CPU candidate oracle covers real vocabulary size 248,320,
ragged sizes, 1–7 rows, ties/signed zero, nonfinite inputs and promoted penalty
tokens. Target probability bits, accepted/rejected tokens, residual correction
and RNG agree with full-row sampling. Shared core sampling and existing MTP
sampling tests pass. The independent predictor audit checks KV-only/full
catch-up at 1/8/9/32/33 rows, full-width normalization, attention and recursive
carry. Native C2/4/6/8 checks retain exact full logits, sampled RNG, failure and
cancellation isolation, mixed head eligibility and used-session restores.

Native sampling and cache suites pass default thinking/nonthinking, greedy,
unbounded sampling, EOS, short budgets, RAM and serialized replay. HTTP tests
pass bounded/unbounded top-k and min-keep boundaries, actual concurrent
drafting, stops, cancellation, images and exact warm replay. The selected API
suites pass structured output, reasoning budgets, sampling defaults, tools,
long context and cache/disk restart. Public extraction, executable URI tests
(including three rejected mutants) and 32K source-edit tool calls pass with
thinking on/off and literal protocol tags. A real Pi 0.87.0 continued session
passes editing, code creation and literal-tag fixtures: 12 HTTP requests,
10 successful tool calls, and actual executed assertions.

The stronger sampled-copy bypass changes proposal policy only after a
previous copy establishes its source and a full 24-token match remains.
Target rejection is still mandatory. The matched captures above retain
identical wording, but a proposal-policy change does not promise identical
seeded text across versions for every prompt. The target sampling distribution
and deterministic replay within a version are the contracts.

An image cancellation/resume timing flag reproduced twice: TTFT
3.62–3.78 → 9.45–9.46 ms. Admission profiling isolates 5.61 ms in image
context setup, where a first image descriptor invalidates captured text
graphs. Image-capable Flash-Next slots now prepare an empty stable descriptor
before any graph capture. The affected request measures 3.71 ms TTFT and
1462.23 ms decode, versus main's 3.78 ms and 1494.80 ms. The complete lookup
history retains every output and expected prefill/cache count; no PP/decode
or token-gap flag remains in that pair.

Initial comparisons use main `da5d5a2b`. Final production binaries compare
main `0c03748a` with candidate `1d3b8c68`, after rebasing onto the merged
small-prefill changes. They use the same Nix toolchain; builds and inference
never overlap. All observations,
including failed prototypes and harness-only failures, remain in
`/tmp/gufo-flash-decode-20261021` on the test host and
`/Users/fbozzo/projects/flash-decode-20261021` on the editing host. The benchmark
artifact's source field identifies its harness tree; the retained binary
paths and hashes identify each measured implementation. No llama.cpp values
or reference accuracy claims are changed.

Initial comparisons retain 938 lookup phase measurements (six queue/TTFT
flags), 80 workflow measurements (one 3.24 ms snapshot flag), and 242
concurrent benchmark measurements. The latter flags include the variable
mixed-C2 controller, unordered peer preparation/queue latencies and one
5.76 ms seven-token prefill difference. Outputs and prompt/cache work agree.
Timing uses the unchanged 5% and 3 ms gate per request/phase. Faster requests
never offset slower ones; remaining variable observations are not timing
passes. The two workflow pairs leave 78 of 80 phases passing, two snapshot
observations inconclusive and no confirmed failure. The initial image-resume
failure and its fix are retained separately.

The raw C4 queue comparison also flags the same body twice, but its recorded
admission rank changes from second on main to fourth on the candidate. Both
arms reach the first token at about 10–11 ms. This is unmatched scheduling
work, not evidence of slower inference; its queue timing stays unqualified.
Other variable C6/C8 queue and first-token observations remain unqualified.
Focused repeats, admission counters and every comparison are retained with
the raw reports.

Final native candidate checks pass again: independent target-candidate and
predictor oracles, batch isolation, sampling, and RAM/serialized cache replay.
The image-descriptor operator test compares captured fused attention against
an independent null-descriptor reference near 128K, including guard bytes,
idempotent preparation and empty-layout restore. The six selected API suites
pass again, including image/thinking/tool cancellation and disk restart. The
final real Pi conversation passes 13 requests and ten successful tool calls;
the generated files and actual Node/Python assertions were inspected.

The current-main default-sampled screen retains identical outputs and expected
cache work, with all 276 per-request/phase measurements passing. Its two
simultaneous requests are identified by their submission cohort, since network
arrival order can swap within a cohort; bodies, outputs, measured values and
the order between cohorts are unchanged. The original label-based comparison
is retained separately.

All six final canonical AR/MTP depth spots retain completion hashes and
prompt/cache/draft counts. Prefill rates differ by at most 0.21%. The raw
concurrent comparison retains 342 phase measurements including single-user
prefix preparation. Identical C8 peers exchange request indices and admission
roles: matching their recorded cache work and `resident_at_admission`, without
sorting by timings, removes the apparent preparation and queue slowdowns.
The separate admission comparison has 288 measurements, with only greedy C2
decode/duration flags remaining. Its first cold seven-token tails improve
from about 108.6/105.7 to 95.4/92.5 ms for mixed/repetitive input.

Greedy C2 remains unqualified: the final pair is 62.17 → 55.30 tok/s, while
the focused reverse-order control is 53.30 → 55.29. Main proposes 93 versus
39 drafts per response across its runs; the candidate proposes 56 in both.
Every completion still matches isolated AR. These different adaptive paths
do not establish a branch speedup or an attributable slowdown. The raw flags,
earlier controls and all retained observations remain available; no global
timing pass is claimed.

### Focused performance measurements

Focused production comparison against main `0c03748a`, candidate `1d3b8c68`,
with the same Nix toolchain, weights and ordered request histories. Model
defaults are used for sampled requests; concurrency is physical, and the C>1
rates sum the individual decode rates. These measurements qualify the decode
changes separately from the complete October 18 benchmark tables. They do not
refresh the reference data.

| Workload | Main tok/s | Candidate tok/s |
| --- | ---: | ---: |
| Copy existing code, default sampling, C1 | 109.59 | 112.30 |
| Generate new code, default sampling, C1 | 75.18 | 75.40 |
| Generate prose, default sampling, C1 | 39.74 | 39.76 |
| Thinking/code, default sampling, C1 | 51.93 | 52.05 |
| Greedy code control, C1 | 70.42 | 70.51 |
| Default-sampled copying, C2 aggregate | 132.93 | 135.45 |

These 128-token screens retain identical outputs and prompt/cache counts;
all 276 matched request/phase measurements pass the 5% and 3 ms gate. The
screens measure speed, not complete coding-task accuracy.

Earlier complete-workflow and mixed-cohort checks against main `da5d5a2b`,
before rebasing onto the merged small-prefill changes, remain separate:

| Workload | Main tok/s | Candidate tok/s |
| --- | ---: | ---: |
| Extraction with thinking, 726 tokens | 82.52 | 82.76 |
| Generate and execute URI tests, 441 tokens | 78.54 | 78.79 |
| File-edit tool call at 32K, thinking off | 76.42 | 77.01 |
| File-edit tool call at 32K, thinking on | 72.40 | 72.47 |
| Mixed greedy/default-sampled copying, C2 | 137.63 | 139.50 |
| Mixed greedy/default-sampled copying, C4 | 200.22 | 202.94 |
| Mixed greedy/default-sampled copying, C6 | 218.78 | 221.54 |
| Mixed greedy/default-sampled copying, C8 | 237.34 | 240.87 |

All outputs in these matched captures agree. The extraction, URI and file-edit
fixtures check their actual results. A real Pi conversation also passes, with ten
successful tool calls and executed assertions; its unseeded run is correctness
evidence, not a matched speed claim.

Canonical greedy pp2048/tg128 spots on current main and candidate use the
original single-user context capacity and cache preparation:

| Mode | Cached depth | Main PP | Candidate PP | Main TG | Candidate TG |
| --- | ---: | ---: | ---: | ---: | ---: |
| AR | 0 | 1793.17 | 1792.97 | 28.87 | 28.88 |
| AR | 32,768 | 1651.93 | 1650.85 | 28.74 | 28.77 |
| MTP mixed | 0 | 1724.11 | 1727.57 | 36.18 | 36.31 |
| MTP mixed | 32,768 | 1607.67 | 1605.39 | 40.56 | 40.51 |
| MTP repetitive | 0 | 1737.22 | 1733.71 | 118.61 | 120.25 |
| MTP repetitive | 32,768 | 1609.07 | 1609.65 | 113.61 | 115.07 |

Concurrent greedy TG: repetitive C2 166.15 → 170.96, repetitive C8
245.05 → 247.09, mixed C8 133.65 → 134.61 tok/s. All measured concurrent
completions match isolated AR controls.

Mixed C2 remains variable: the current pair is 62.17 → 55.30 tok/s and the
focused reverse-order control is 53.30 → 55.29. The pre-rebase pair was
62.17 → 55.36 and its reverse-order repeat was 53.30 → 64.03. Main proposes
93 versus 39 drafts per response across its current runs; the final candidate
proposes 56 in both. Both builds produce identical text. These observations do
**not** establish a C2 speedup or a confirmed slowdown. Individual cache,
queue and preparation timing flags remain visible in the
quality record above; no blanket
timing pass is claimed.

## Earlier qualification

**Component-cache admission, October 20:** production code `195f9374` avoids
repeated slot ranking, full checkpoint lookup results for presence checks and
a second cold reset. Focused cache/runner/index/RAM-and-disk CPU tests pass;
the cold-reset regression fails on main. Native Flash AR/MTP slot reuse and
cancellation/reset pass. Native MTP greedy/default-sampled RAM/serialized replay
remains exact.

All six HTTP profiles pass correctness: Flash AR/MTP and Qwen Q4/Q8 AR/DFlash2.
Each candidate profile retains 81 matched responses and expected cache/prefill
counts, plus 65 continuation requests covering long context, cancellation,
images and disk restart: 876 recorded requests in total. Speculative modes
perform actual drafts; AR performs none. Qwen compatibility uses the open #540
adapter at `7f822ff4` versus that adapter plus this fix (`982e54c6`), separately
from the PR's main baseline `e121a214`.

Timing uses the unchanged 5% and 3 ms gate, per request and phase, including
startup. Flash AR uses unchanged-main controls; Flash MTP and Qwen Q4/Q8 AR
also use focused alternating repeats. Qwen Q4/Q8 DFlash2 retains one matched
pair each, with flags unqualified. Every observation remains retained.
**No profile has a blanket timing pass.**

| Profile | Passed measurements | Inconclusive | Confirmed failures |
| --- | ---: | ---: | ---: |
| Flash AR | 1,187 | 16 | 0 |
| Flash MTP | 1,078 | 125 | 0 |
| Qwen Q4 AR | 1,177 | 27 | 0 |
| Qwen Q4 DFlash2 | 1,198 | 7 | 0 |
| Qwen Q8 AR | 1,178 | 26 | 0 |
| Qwen Q8 DFlash2 | 1,185 | 20 | 0 |

The AR cold-request queue median is 3.593 → 2.661 ms; MTP is 4.834 → 2.698 ms
(55 cold requests in each matched 81-request history). These distributions
are descriptive. Remaining prefill, token-gap, snapshot, queue and TTFT
flags stay unqualified; faster requests never offset slower ones. The
[PR comparison](https://github.com/gufo-org/gufo/pull/680)
retain current d0 rates; short public-corpus C1/C2/C4/C6/C8 checks additionally
retain exact outputs, actual drafts and individual phase timings. This benchmark
pair retains 80 queue and 28 TTFT flags as inconclusive; its other measured
phases stay within the gate.

Snapshot elapsed time includes delayed scheduler collection and overlaps
inference. A completed 133 MB capture call took 7.77 ms in the instrumented
identical cohort; collection measurements include 104.77 ms. Capture-call time
includes planning and transfer completion. A retry-sharing experiment saved
three copies but repeatedly added 7–9 ms to some decode phases; it was removed.

Production binaries, hashes, commands, all reports, failed experiments and
interrupted attempts remain in
`/home/mixer/gufo-qualification/issue678-2026-10-20` (`provenance.json`,
`retained-*-qualified.json`, `retained-speed-summary.json`). The benchmark's
source field identifies the harness tree; provenance maps the measured binaries
to their actual source revisions. A stopped Q4 DFlash2 benchmark has no completed
report and is retained separately from its completed `-resume1` run.

**Host-memory stalls, October 18:** `fec4759d` protects runtime-owned queue-save,
launch/copy and code-loading buffers from host compaction while preserving
application managed memory and ordinary fine-grained host allocations. Owned
HIP host buffers retain their coherence policy and gain best-effort long-term
pins. Cache backing grows in bounded pieces before actual prefill work, after
pending captures. Cold HC, Q8 and argmax functions resolve before readiness
without trial launches. Model arithmetic, sampling and checkpoint contents are
unchanged. The Nix package selects the patched ROCr runtime automatically.

Independent memory checks cover GPU/CPU visibility, managed fallback and
restricted pinning. Weight-upload bytes, bounds and failed-read cleanup pass.
All 903,984,064 vision-upload bytes are checked, and 29 vision stages containing
8,421,376 finite floats remain byte-identical to main. Independent projection,
MTP and bidirectional snapshot-copy checks pass, including pageable/unaligned
fallbacks, borrowed device sources, guards and graph replay.

Controls on main `bff0eec6` pass all 1,918 sampling/stop measurements, 345
measurements across the 24-request history through 56K, and 302 concurrent
measurements against the control with matching admission order. Original
request-index flags remain recorded. After rebasing onto `f948ddd3` and its
image-preprocessing cache, the production package passes cache correctness and
every compared prefill, decode and disk-restore phase. Eight image-history
requests with one/four 1080p screenshots and text controls retain exact outputs
and cache work; all 56 phase checks pass. Pi passes tools, editing and code
creation with thinking enabled, ten HTTP requests, nine successful tool calls
and executed tests.

The complete HTTP benchmark refresh retains all 24 depth completion hashes
and prompt/cache counts, and all 63 concurrent completions match fresh AR
controls. Initial repetitive-MTP PP samples at 16K/32K/64K were
1573/1541/1469 tok/s. A matched ordered repeat measures 1647/1624/1567,
versus current main's 1654/1634/1570. All nine observed requests pass 63
phase comparisons; server logs also confirm matching work and passing phases
for all sixteen requests including prefix preparation. Both candidate samples
contribute to the published mean and standard deviation. The initial dips'
cause remains unproven. Cold-file loading takes 16.42 seconds versus main's
15.95 seconds, within the timing gate.

One fully cached thinking replay retains a queue/TTFT flag of about 3.22 ms:
queue time is 7.70 → 10.93 ms, while an earlier unchanged-main control reaches
11.47 ms. This remains **inconclusive**, not a timing pass. The post-build probe
retains all twelve outputs/tool arguments and observes 98.5–101.4 ms maximum
token gaps, versus 208–283 ms in the incomplete fix. Its small 4 ms total KFD
eviction counter is retained; this is not a guarantee that every possible
operating-system or concurrency stall is eliminated.
[Detailed controls and failed observations](artifacts/host-memory-stalls-review.json).

**Issue #632, October 17:** compared with merged main `8497711e`, indexer plan
preparation moves the first hipBLASLt module load before readiness without
running trial GEMMs. The independent F64 projection check and prepared/lazy
F16/BF16 replay pass for 45 shapes, including ragged widths. Native AR/MTP
snapshot, borrowed-state, sampled/serialized replay and device-exhaustion
checks remain exact. Cache backing tests cover real long-term pins, allocations
larger than 1 GiB, GPU/CPU bytes, failed registration, lifetime and low-memlock
fallback; a separate process also passes with the io_uring syscalls denied.

The new explicit `cache-stalls` suite uses a public equivalent of the report's
24 ordered requests through 56K tokens. The supplied archive omits the inner
prompt script, so this is not an exact reporter replay. Final candidate
correctness and all 345 compared phase measurements pass. The standard d0
pp2048/tg128 control retains all 12 outputs and token/cache/draft counts, with no
timing flags. First-prompt PP drops from 257–260 to 145–146 ms; measured TG is
28.27 → 28.29 tok/s AR, 35.79 → 35.76 mixed MTP, and 116.95 → 116.88 repetitive
MTP. A real Pi conversation passes tools, editing and code generation with
thinking, 10 HTTP requests, 9 successful tool calls and executed tests.

Cache correctness also passes, including cancellation, image continuation and
disk restart. Timing is **not globally qualified**: after two final/main pairs
and an unchanged-main control, 843 of 860 measures pass and 17 remain
inconclusive. The queue outlier overlaps main's 9.30–13.70 ms range. One image
resume takes 181 versus 167 ms of decode in one pair and matches in the other;
individual thinking/token-gap and disk-snapshot variation remains recorded.
No timing margin was widened.

An earlier main run had a 2.83-second token gap during page migration and
reclaim. A matched control holding another 4 GiB retains all 24 outputs and
phases within the timing gate, but neither arm repeats that large gap. Long-term
pins prevent migration of the cache pages; these observations do not prove the
entire original pause came from their migration. Container syscall restrictions
can disable this extra protection; see [cache pinning](../../KV-CACHE.md).
The cold-prompt setup fix also works when pinning is unavailable.

**PR #613 review, October 17:** prompt lookup remains local to each Flash-Next
MTP session and now works in batched decoding. Greedy requests can bypass the
neural head; sampled requests retain a neural proposal first and require a
24-token match. Fewer than half the copied proposals being accepted disables
short resumption. Matches count only tokens generated since the request's first
speculative step, so shared chat wrappers cannot qualify a new answer.
Sampled copy sources stop at the request's starting context; greedy requests
can also copy repetition within their current answer. This avoids displacing
useful neural drafts with superficially similar parts of a new answer.
Fully reused prompts reset this state even without `Sync`. Snapshots retain
the request boundary and resume cursor; the payload compatibility version is 18.
Version-17 disk entries are not reused after upgrading, so their first request
needs a fresh prefill. New version-18 entries support normal restart reuse.

The prefill follow-up replaces the per-gather allocating n-gram map with
reusable scratch prepared before serving. It preserves token-to-row hashing, first
occurrences, I/O order, dequantization and copied values. The independent
n-gram tests pass, including 32,768-row, three-row and 32,769-row gathers
through the same storage, with reference BF16 values and untouched guards.
The repeated 8K MTP prefill flag is fixed: the same 2,042 input tokens after
8,189 cached tokens take 1,220 ms rather than 1,330 ms, matching main's
1,219 ms. The focused history retains every completion hash and cache/token
count. This fixes prefill allocation stalls; speculative output still arrives
in bursts, whose largest gaps are reported separately from total decode time.
At 12K repetition, the largest gap is 73.86 ms versus main's 63.58 ms, while
the same 128 output tokens take 1.218 seconds versus 2.634 seconds.
The full-width copy path is retained for throughput; that accepted 10 ms
burst-gap increase is not reported as a timing pass.
The final AR, mixed MTP and repetitive MTP sweeps cover all eight depths
from 0 to 128K. All 24 completion hashes and prompt/cache counts match main
`3389a585`; prefill, total decode and first-token phases pass the same gate.
The 12K burst-gap flag remains in the full sweep (64.02 → 70.19 ms).

The final C1/C2/C4/C6/C8 sweep retains all 63 AR-equivalent completions and
expected cache work. C8 ordinary/repetitive MTP reaches 126.87/245.11 versus
main's 124.94/196.31 tok/s. Five preparation flags compare different admission
positions: corresponding server admission, cache and batch roles agree across
all 60 C>1 preparations, including conservative log-rounding bounds.
The original request-index flags are retained.

C8 AR has a separate maximum-gap limitation: the initial pair measures about
72 → 103 ms, and a reverse-order repeat about 73 → 104 ms, with total
throughput within 0.3%. A matched pair recording individual HTTP events gives
72.74–72.87 ms on main and 72.36–72.52 ms on the candidate; its largest gap is
on the first generated step. Output and cache work remain identical.
This does not establish the cause of the earlier spikes: their attribution
remains inconclusive, and they are not reported as a timing pass.

Native tests with the same C8 model configuration compare isolated execution
with C2/C4/C6/C8 cohorts: tokens, full logits, RNG, accepted/proposed counts and
snapshot bytes agree exactly. They mix greedy and sampled requests,
vary budgets and peer order, and require accepted copies in both modes
(118 greedy and 43 sampled copies in the final run). Restoring into a used
session and rewinding into a copy also replay exactly. Direct, RAM and serialized
200-token continuations pass, including learned policies and both real sampling
presets. Regressions cover the first step, rejected-copy resumption, a shared
wrapper followed by a short answer, reindexing, boundary restoration, exclusion
of a sampled answer's own repetitions and eligibility on the next request.

The independent MTP oracle passes normalization, attention, recursive carry,
full Q8 head, ragged catch-up and independent batches. Point-mass frequency and
filter-boundary tests retain the target sampling distribution; target arithmetic
and the neural sampler are unchanged throughout the follow-up. Native EOS and
ignore-EOS checks pass. The final HTTP lookup test reaches every physical width
C2/C4/C6/C8 and accepts 3,172 copied tokens, including an all-greedy C8 mix of
copies and new text whose outputs match isolated execution. With the vision
projector loaded, greedy and default-sampled image requests cancel after 40 and
16 accepted copies respectively; both replay the exact uninterrupted response
from RAM with zero prefill.

A focused production run records 219 requests across sampling defaults, stops,
image inputs and cache histories. Sampling overrides, thinking, concurrent
seeded requests, Unicode stops, tools, PNG/JPEG/WebP and the Chat, Responses
and Messages image paths pass. Text, thinking, tool and image cancellation
histories also pass after a process restart with actual disk-cache hits and
exact replay. These are correctness checks, not matched timing comparisons.
The preceding batch, tool-agent and long-context checks remain applicable;
this was not an all-model or all-functional-suite run.

The expanded ordinary corpus uses real non-thinking defaults (temperature 0.7,
top-p 0.8, top-k 20, presence penalty 1.5), seed 31 and main `3389a585`.
All nine ordinary answers now match main exactly: aggregate decode is
46.57 → 46.60 tok/s on the final scratch build. Copying from the current
sampled answer had changed two
answers, slowed C++ generation by about 4% and made a debugging answer 5.5%
longer in wall time with an unsupported statistic. Restricting the sources
removes both divergences. All 14 corpus/coding/tool outputs retain exact text,
reasoning and tool arguments; prefill, total decode and first-token phases pass
the unchanged 5% and 3 ms gate. One JSON response's largest gap increases
130.96 → 150.03 ms. Replaying its complete preceding request history gives
154.84 ms on unchanged main and 150.77 ms on the candidate, with identical work
and no phase flags. The initial gap remains inconclusive.
Generated C++ passes an independent
deque oracle, wraparound and move-only checks; generated Python LRU code passes
independent replacement, eviction and value-preservation checks.

Three existing-file edits retain the exact intended content and improve
85.44 → 105.93 tok/s in aggregate (9.06 → 7.54 seconds for the requests).
Fresh Python LRU generation remains exact at 74.82 → 75.51 tok/s.
A separate tool-write case omits the final newline on both main and candidate;
its strict byte-preservation check fails equally and is not a qualified
copy-speed result. The JSON schedule's original checker incorrectly required
a particular goal field name absent from the prompt; saved outputs pass the
corrected checker, with the original failure reports retained.

The Pi and 115K measurements below precede the scratch-only follow-up; the
final row-value, corpus, image/cancellation and benchmark checks above qualify
that unchanged arithmetic and state handling.
Real Pi 0.87.0 runs read/edit/write/test tools across five continued tasks.
With thinking off, all 21 requests and 16 tool calls retain identical assistant
content, tool arguments and tool results. Agent-process time is 26.12 → 25.92
seconds; the original identical-code slowdown is gone. Precise child-exit
timestamps exclude the functional harness's 500 ms polling granularity.
A reverse-order repeat retains every observation: one prefill delay also
appears on unchanged main, while maximum-gap flags move between requests and
overlap passing observations. These phases remain inconclusive, not a global
timing pass.

With thinking enabled and about 25K source tokens, all five tasks pass
(17 tool calls): 107.36 → 104.96 seconds. File timestamps are fixed before each
task because an earlier `ls -la` result changed the prompts between arms;
that earlier comparison is retained but cannot attribute latency to the PR.
The corrected code-creation task is 33.45 → 33.37 seconds with identical work.
The bug-fix task produces 35 extra sampled tokens and takes 21.35 → 22.22
seconds; the following literal-protocol task needs one fewer request.
The whole-agent result includes those different sampled choices and is not an
isolated inference speedup. Every comparable request phase passes the gate.
Sessions and actual tool results are retained outside Git.

New public workflow fixtures substitute for the author's unavailable private
prompts. Extraction checks exact records; generated URI tests must pass correct
code and detect three deliberate defects. A frozen 115K-token source archive
precedes a tool-driven edit that must preserve every unrelated byte, including
literal thinking/tool tags. Default sampling is used with fixed seeds:

| Request | Main tok/s | Candidate tok/s | Quality/work |
| --- | ---: | ---: | --- |
| Extraction, thinking | 84.49 | 84.59 | Exact answer and reasoning, 707 tokens |
| URI test generation, no thinking | 82.99 | 83.47 | Exact tests, all three mutants detected, 435 tokens |
| 115K agent edit, no thinking | 63.26 | 73.69 | Exact file/tool arguments, 187 tokens; cached continuation |
| 115K agent edit, thinking | 67.50 | 75.18 | Exact file/tool arguments and reasoning, 219 tokens |

The final scratch build repeats extraction and URI generation with exact
outputs; all eight captured phases pass the same timing gate.
The first final 115K run flags one maximum token gap (120 → 137 ms);
prefill, total decode time, output and expected cache work pass.
A focused main/candidate repeat of the same warmup and cached edit passes all
32 measured phases: the maximum gaps are 120.07/120.18 ms and decode rates
63.37/75.14 tok/s. The original gap remains inconclusive; the table retains
the first observation rather than substituting the faster repeat.
Tool-call IDs are generated independently and excluded from content comparison.
Changing thinking mode changes the rendered prefix, so its full prefill is
expected. The original main run's erroneous cache-hit assertion is retained;
its saved outputs and cache work were independently revalidated after fixing
that assertion, without rerunning or replacing the measurements. A second seed
also passes extraction and URI quality checks on the preceding candidate.

Before the scratch follow-up, canonical prepared pp2048/tg128 repetition
reaches 115.77 versus 73.18 tok/s at C1.
At C6, aggregate ordinary/repetitive rates are 109.54/231.31 versus
108.74/177.98; all measured completion hashes and cache/prefill counts agree.
Five C6 preparation flags remain in the raw request-index comparison: the
identical requests were admitted in different orders. Server logs show the
same admission/cache roles and costs (the first cached seven-token prefill is
102.23 versus 102.13 ms). Comparing each corresponding admission role passes
without averaging or changing limits. Earlier first-token variability reversed
in the unchanged-main control and remains inconclusive, not a global timing
pass.

A separate C8 history runs copies, then new text, then four of each. Its mixed
copy step initially verified 44 rows and raised the longest pause from main's
157–163 ms to 200 ms. Bounding greedy copied suffixes in mixed cohorts reduces
that pause to 145 ms. Copying/new-text rates in the final mixed cohort are
28.08/18.58 versus main's 21.63/17.95 tok/s; all-copy cohorts retain full width
and reach 31.42 versus 25.29 tok/s. All 24 responses retain exact text and
expected cache work. Ordinary generation immediately after copies remains
18.00 → 17.56 tok/s per session, with its longest pause reduced from 215 to
162 ms. Small first-token flags remain inconclusive: main's 23–31 ms variation
changes which request indices flag across unchanged controls, while the
candidate takes about 29–31 ms. No decode or maximum-gap phase flags remain
against either final main control. This is not a global timing pass.
These focused observations do not replace the complete benchmark table or
establish statistical language quality.

llama.cpp `10b8b755` also verifies context proposals against the target, but its
ngram path samples the target directly and compares the token. Gufo retains
point-mass rejection sampling for copies alongside its neural p/q proposals.
Both preserve the target distribution; identical seeded streams or a speed
advantage over llama.cpp are not claimed. Target arithmetic and weights are
unchanged.

**PR #626 follow-up, October 17:** sparse attention bounds temporary K-fragment
scheduling while retaining every dot product and softmax operation. Healthy
MTP cache-slot reassignment retains bounded rollback scratch and its graphs;
failure and explicit reset still free it. Prepared Qwen vision weights use
coarse managed storage, removing the reproduced text-dispatch stalls after
sidecar upload. Image activations and encoder arithmetic are unchanged.
The focused numerical, cache and image checks are listed below.

The complete HTTP C1 sweep retains all 24 completion hashes and prompt/cache/
draft counts. Its 12 AR requests match current main `78248de2` and pass every
measured phase at 5% and 3 ms. The affected mixed-MTP history was replayed
through 64K; two unprofiled follow-ups reach 1551.64 and 1558.51 tok/s. The
second pins the initial run's exact cache budget and matches all 11 request
identities with no phase flags against that candidate control. That request
shows no reclaim, swap or compaction activity. The table uses the faster
follow-up; neither a specific driver cause nor worst-case latency is claimed.
Raw attempts and diagnostics remain outside Git.

The canonical repetitive d0 measurement is 73.80 tok/s with the full vision
sidecar ready. In the isolated allocation control it is 73.77, versus 73.35
on the exact published `32c4ece8` binary and 42.53 with the current raw device
uploads. Four image/text HTTP controls preserve outputs and cache/draft work
and pass the per-request phase gate. This focused evidence does not replace
the older functional timing limits or upstream accuracy qualifications below.

**PR #603's reduced draft vocabulary preserves the target distribution and
supports the real sampling defaults.** The final review keeps the established
length controller and AR backoff for every request; it removes the proposed
confidence/calibrated stopping policy. That policy lost learned state across
native sampled snapshots and caused repeatable mixed-session and short greedy
slowdowns. The reduced head is used for C1 sampled requests, including penalties
and constraints, and eligible greedy requests. Unplanned multi-session requests
and modified greedy sampling retain the full draft head.

October 15, 2026, against `f6ca085a`, with the same pinned Nix toolchain and
Unsloth UD-Q4_K_XL/shared Q8_0 MTP weights:
25,328,640 full-vocabulary target logits after a 28,092-token prompt and
102 fixed continuation steps are finite and byte-identical. Independent MTP
formula checks retain their 0.002 relative-RMS limit (worst 0.0007655).
Native session, snapshot, component-restore, probability-correction and mixed
head checks pass. New regressions cover actual default sampling, complete
sampler/RNG replay, used-session table restoration and the subset path with
penalties. All 12 independent native sampled replay cases pass; the original
PR failed all 12.

HTTP requests that omit sampling controls exercise both real presets. All
16 short and two 35K-context default completions pass the timing gate; 17/18
retain identical output and draft counts. Python/story generation improves
3–10% at short context and the deep Python cases improve 5–8%. Translation
defaults are mostly level. The standard C1 AR/MTP captures, including warmups,
retain exact output/work with no phase flags; greedy MTP gains 5–12% at
d0/d32K. Focused AR/MTP API, cancellation, long-context and RAM/disk cache
checks pass correctness. Pi completes four continued coding tasks with
17 default-sampling requests and 13 tool calls; actual edits and test results
were inspected.

This is **not a global timing or individual-answer quality pass**. API queue
flags include simultaneous requests admitted in a different order; the same
order reversal occurs within main's own captures. Small snapshot/cache-resume
flags remain inconclusive: the MTP image snapshot is 83.84 → 88.97 ms, while
unchanged main reaches 87.40 ms. The AR one-token cache-resume outlier
(51.31 → 73.80 ms) does not recur: the final pair is 51.29 → 54.06 ms,
and all 737 phases in that repeat pass. Raw review captures are retained
outside the repository. A non-default sampled
translation repeats about 9% slower with different wording and acceptance;
its translation is correct, but its extra grammar explanation mislabels a
declension. Changed proposals can consume RNG differently across builds.
These numerical, replay and functional checks establish regression evidence
against main, not upstream unquantized-model accuracy or a statistical
language-quality result. The Gufo benchmark refresh below retains all llama.cpp
data unchanged.

The cache follow-up in `fdc21849` changes storage and bookkeeping while
preserving model arithmetic and sampling. Large pinned segments populate with
at most four workers and disable transparent huge pages before GPU registration
(incorporating #605). Capture retains its prepared shared chunks across allocation
retries, avoids repeated history scans and impossible partial allocations, and
allocates larger payloads first. Shared KV rows and private states use opposite
ends of the same bounded pool; KV rows retain ascending addresses for contiguous
restores. Payload owners shrink from 128 to 16 bytes. Existing reader pins,
atomic validity publication and generation checks remain authoritative.

Byte-pressure retry admission preserves the reused frontier, forks, last copies
and current edit boundaries. A retry adding at most 64 tokens can reclaim covered
older history; longer resumed turns can reclaim only their own covered ancestors.
Cold side requests prefer shallow live slots, preserving deep continuations.
Native regressions reproduced and then fixed fragmented private-state allocation,
reverse KV placement, lost retry reuse and premature deep-slot eviction.

Aligned registered snapshot buffers use a vectorized integer copy through their
GPU aliases; other buffers retain HIP copies. All 150 independent size/offset/
storage cases pass, including ragged allocation tails and unregistering a source.
Real AR/MTP snapshots at 2047 and 8203 tokens match pageable copies byte-for-byte.
Restored full logits, sampled/greedy continuations, learned drafts, real images,
mutation/failure guards and copying during a peer's graph capture pass.

An already-warm target pass can need its first graph or a replacement after
releasing MTP rollback storage. The new path enqueues that pass once and records
its graph while the GPU executes. The graph is not launched a second time;
explicit reset still releases all rollback storage. This removed reproduced
8–10 ms MTP and 5 ms AR short-request regressions in preceding cache candidates.
The affected AR pressure prefix has 21 exact requests and 316 passing timing
phases; the flagged decode is 76.25 ms on main and 73.02 ms with this change.
Deferring graph recording to the following request was rejected because it
moved the delay there.

The final native session suite passes: 65 exact full-logit AR/MTP frontier
comparisons, eight bitwise fresh-load replays, 25 sampling configurations per
mode, eight 200-token cache replays, image attachment/replacement, cancellation,
rollback release and C2/4/6/8 isolation. The focused cache, backing, retention,
component-runner and snapshot tests pass without relaxed tolerances or missing
model skips. Pi on the final `fdc21849` production binary completes four
continued coding tasks with 17 HTTP requests and 13 paired tool calls/results,
using default sampling and low thinking. All 16 follow-up requests reuse cache;
their first-token times are 145–488 ms. Tasks take 7.0–13.1 seconds. These are
observed smoke timings, not a matched-main timing gate. Actual edits, expected
failing tests and successful repairs were inspected; no orphan tool results or
literal framing tags appear in assistant text. The preceding `e0bead8b` run's
eight tasks, 33 requests and 25 tool pairs remain in the retained evidence.

The final production depth sweep has 32 captured requests with exact outputs,
prompt/cache/draft counts and all 256 timing phases passing against post-#603
main. All 63 measured concurrent completions match fresh AR, and the complete
128-request concurrency history retains exact output. Concurrent preparation
can change which identical request supplies the shared prefix; cohort prefill
work is unchanged. Loading is 15.27 s, and the two memory rows are 88.75/90.73
GiB, matching post-#603 main within 0.01 GiB. The older published memory rows
came from an earlier source revision.

Completed MTP continuation and small-budget pressure captures have 219 requests,
including 208 complete responses with exact request/output/draft identities and
no lost reuse. The final AR pressure capture has 100 exact requests and no lost
reuse. Four timing flags concern one short decode: 79.36 ms here versus an older
73.06 ms control, while fresher unchanged main takes 79.06 ms. Both controls and
their concurrent admission histories are retained; this remains inconclusive.
The prepared C6/C8 MTP repeat confirms a small first-token cost: approximately
23 → 35 ms and 31 → 59 ms. Aggregate decode is 176.28 → 175.84 tok/s and
194.16 → 194.12 tok/s; C8 whole-request time increases about 0.5%. This startup
tradeoff was accepted for publication. It remains a timing-gate failure, not a
global timing pass. GPU traces locate the wait before session reset; changing
the copy method or queue limit did not resolve it, and neither experiment was
retained. All original timing flags and rejected candidates remain retained in
the [compact review record](artifacts/cache-checkpoint-review.json) and raw archives.
Flash is the only production consumer of this component cache; other text
models retain the legacy path.

**PR #596 repairs the complete functional inventory:** all 48 registered
Flash-Next suites pass in AR and MTP at `9985ff0a`, including all four deferred
cache workloads (122 expanded checks, 3,334 requests). The final allocation
follow-up retains the same harness and logical cache fixes and receives focused
continuation/cache revalidation. Existing numerical tolerances are unchanged;
CPU regressions, GPU continuation/snapshot checks and actual Pi tool/image
results are retained in [the qualification evidence](artifacts/functional-fixes.json).
This is functional regression evidence, not upstream model/conversion accuracy
or a clean global timing gate. The unreproduced five-minute MTP snapshot stall
and individual timing flags remain visible.

**PR #571's wave-owned prefill weights retain byte-exact logits:**
99,328,000 F32 values after a 34,159-token code/prose prompt and 400 fixed
continuation steps, batch 4096, capacity 65536, match the original PR control
`f06f32b9`. October 13, 2026 (Europe/Paris), kernel snapshot `a772fae6` atop
`33a91366`, with the same pinned toolchain. Independent FP64 routed checks,
unsaturated activations, odd outputs and guarded partial tiles pass. Fused SSM
convolution, output, state and replay remain exact, including 2049-token tails.
Deferred residual, F16 norm, packed Q8 bytes and independent host FP32 inject
checks pass unchanged. All 16 measured AR/MTP HTTP completions at d0/d32K
retain exact outputs and token/cache/draft counts, with no per-request
prefill/decode timing flag. The first unusually slow mixed-MTP control is
excluded as effect-size evidence.
These are regression checks. The older functional timing failures below remain
unresolved, and upstream model/conversion accuracy is not requalified.

**PR #518 retains main's full logits at 64K/128K:** 512 fixed code/prose token
positions, each checked at verification widths 1–8; all 1,017,118,720 logit
comparisons are byte-exact (KL/TV 0, top-1 agreement 100%). Prefill and 56
partial rollback/replay checks are exact too. October 10, 2026, candidate
`25cdb2a1` versus main `a41e88da`.
Unsloth UD-Q4_K_XL target, shared Q8_0 MTP;
[identities](artifacts/model-identities.json) · [review evidence](artifacts/decode-kernels-review.json).
These regression and consistency checks do not qualify original unquantized-model
accuracy or GGUF conversion.

**The functional timing gate is not clean.** All 182 AR/MTP request cases pass
correctness in both paired captures, with exact output and prompt/cache/prefill
counts. After an additional unchanged-main control, AR `legacy-tool-cancel`
single-token decode remains 44.63 ms versus 38.88–38.92 ms, and MTP
`disk-spacing` 289-token continuation prefill remains 557.21 ms versus
516.68–526.89 ms. Initial candidate samples were 159.66/951.43 ms.
Other queue/snapshot timing flags remain inconclusive; the initial 58.25 s
queue stall did not recur. The gate remains 5% and 3 ms per request/phase.
Concurrent setup and maximum-token-gap flags are also retained in the evidence.

| Check | Result |
| --- | --- |
| Vision weight storage, October 17 | Flash-Next and 27B each retain 29 finite, byte-identical encoder stages on a deterministic 1024² texture: 134,742,016 and 137,363,456 F32 values respectively. Prepared upload, repeated upload, resident-byte accounting and later encoding pass with both projectors. Four matched production image/text requests retain exact outputs and token/cache/draft counts, with no phase exceeding both 5% and 3 ms; cached image retry performs zero prefill. The original uncached repetitive text history recovers 73.77 tok/s with the full vision sidecar prepared, versus 73.35 on the exact published binary. Native storage parity does not resolve the existing image accuracy gap against the reference. |
| Sparse scheduling, October 17 | Against PR #626 head `c1088f89`, 17,134,080 finite full-vocabulary logits are byte-identical after 133,945 prompt tokens and 69 fixed continuation steps. The complete independent FP64 attention error report is identical, including ragged and 128K cases; all 12 saved real selection masks retain exact attention outputs. |
| Healthy MTP slot reassignment, October 17 | The new adapter regression fails before the fix and passes afterward in AR/MTP. Reused and fresh sessions agree on full logits, greedy/sampled tokens and draft counts; memory stays within the existing reservation, while cancellation and explicit reset release scratch. Guard lifetime and seeded native RAM/serialized cache replay pass. The focused HTTP `cache` and `long-context` suites pass with the Qwen sampling preset, 624 actual drafts and 458 accepted drafts, including text/thinking/tool/image cancellation and disk resume. These are correctness results, not a global timing pass. |
| Wave-owned prefill weights, October 13 | Routed WMMA, projection/convolution, MoE epilogue and HC mix checks pass with unchanged tolerances. Routed inputs end at their allocation boundary; independently reset output tails retain their guards. Fused SSM output/state/replay and full fixed-history logits are exact. HTTP C1 pp2048/tg128, two AR captures with two repetitions and two mixed/repetitive MTP captures: 16 exact measured completions, including actual drafts. All request phases pass the timing gate; the anomalous first mixed-MTP control is excluded. |
| Decode-kernel review, October 10 | All 13 operator tests and native snapshot/session/deep-execution checks pass. Independent MTP formulas: worst relative RMS 0.0007655 (limit 0.002), 41 reported comparisons over eight text/image states. Sampled replay, cancellation, C2/4/6/8 and RAM/disk restoration pass. [Evidence](artifacts/decode-kernels-review.json). |
| Focused HTTP regression, October 10 | `state-edges`, `tool-agent`, `long-context`, `cache`, Flash AR/MTP only. Two main/PR pairs plus unchanged-main controls; no output or expected cache/prefill-work changes. Timing failures remain visible above and in the [evidence](artifacts/decode-kernels-review.json). |
| Concurrency, October 10 | All 63 tg128 requests match fresh AR completions: 21 AR, 21 mixed MTP and 21 repetitive MTP at C1/2/4/6/8. Prepared prefixes are reused; mixed/repetitive MTP propose 882/2170 drafts and accept 517/1959. [Evidence](artifacts/decode-kernels-review.json). |
| Real-image regression checks | 1024×1024 encoding passes. Nine main/candidate HTTP image, retry and continuation responses have identical outputs and token/cache counts. AR/MTP C4 image batching, sampled replay, cancellation and RAM/disk restoration pass. The disk fixture reprocesses only its seven-token assistant suffix; identical RAM retries require zero prefill. [Evidence](artifacts/prefill-long-context.json). |
| Prefill kernels | Final build: 7,946,240 logits after 32,640 prompt tokens remain byte-identical to main (KL 0). The earlier deep-context gate reaches 259,938 tokens with exact logits. Independent GDN/projection/selector checks, ragged chunks and graph replay pass. [Evidence](artifacts/prefill-long-context.json). |
| Mapped input and MTP catch-up | Exact Q8 projections, full/tail predictor stages and candidates; sampled output, acceptance, RAM/serialized replay and C2/4/6/8 state/RNG checks pass. Default sampling and thinking-off penalties are covered at d0/32K/128K. [Evidence](artifacts/prefill-long-context.json). |
| HTTP depth sweep, October 10 | All 24 pp2048/tg128 AR and mixed/repetitive MTP completions through d128K match current main's output hashes and prompt/cache/draft counts. [Evidence](artifacts/decode-kernels-review.json). |
| Unused final-layer outputs | Full-vocabulary logits match unpruned execution byte-for-byte through 133,120 tokens, also with mixed code/Unicode text and image input. Mixed batching and sampled RAM/serialized snapshot replay preserve logits and RNG state. [Evidence](artifacts/prefill-final-rows.json). |
| Bounded selector scratch | After 133,824 prompt tokens, 15,892,480 logits are byte-identical to the previous build. Ragged chunks retain exact masks and the independent FP64 score check. [Evidence](artifacts/prefill-deep-context.json). |
| Unused normalization removal | 15,892,480 logits are byte-identical to main; ragged residual-only fusion matches normalized fusion and the separate epilogue/combine. [Evidence](artifacts/prefill-normalization.json). |
| MTP versus scalar CPU formulas, eight text/image states | Fusion/attention relative RMS <0.0008 (limit 0.002); full-width normalization, split projections, recursive carry and full Q8 head checked |
| Batched MTP/AR, C2/C4/C6/C8 | Logits, tokens, acceptance, RNG and every 1–8-token rollback prefix match isolated execution |
| Sampling | 25 AR/MTP configurations, including top-p zero; FP64 target filtering/CDF, p/q acceptance, residual correction, seeded replay and short token budgets pass |
| Greedy penalties | GPU selection matches CPU at every 1–7-row verification width, including FP32-sensitive ties; native tool/JSON constraints remain active. C2/4/6/8 outputs match the previous implementation; image/tool cancellation, RAM/disk restore and OpenAI SDK checks pass. [Evidence](artifacts/penalty-verification.json). |
| Prefill and cache | Full logits match across tested chunk boundaries, short tails and restored state through 4096 tokens |
| Seeded MTP cache rebuilding | Two seeds × 200 tokens replay exactly after different prefill splits, cache bypass and replacement. K/V-only prefill and compact catch-up preserve full-head candidates across 1/8/9/32/33-row chunks. [Evidence](artifacts/mtp-cache-replay.json). |
| Scalar versus bulk prefill, 2176 tokens | Same top-1; logit RMSE 0.18, not bit-identical |
| Serving | Cancellation, three-turn continuation, reasoning/tool history, concurrent image/text isolation and disk restart pass |
| Prompt checkpoints | In-pass and borrowed checkpoints restore exactly in AR/MTP, including rewinds, branches, resets and destruction; 14 functional cache jobs pass correctness. [Details](PROMPT-CHECKPOINTS.md) |
| Sparse attention | Independent FP64 operator error ≤2.83e-7 (limit 1e-6). At 32K/128K, 256 fixed-token code/prose rows: mean KL 5.82e-4 and 256/256 top-1 agreement with an FP64-attention diagnostic. [Evidence](artifacts/attention-tiles-review.json). |

The attention diagnostic retains the quantized weights and other native
operators. Regrouping FP32 sums can change long-context text across builds;
matched-prefill AR/MTP logits and sampled snapshot replay are exact at
32K/128K. Session tests also cover image/text restoration and C2/4/6/8.

Sampled MTP can consume different RNG draws from AR. Seeded replay requires
the same build, request budget, capacity and sampling configuration; live cost
timings only steer greedy decoding. Draft sampling uses the selected Q8 head's
top 64 logits, with proposals corrected against the full target distribution;
upstream draft-sampler equivalence is not claimed.
MTP cache projections now keep the same arithmetic across prompt splits.
Older Flash-Next disk checkpoints are rejected and rebuilt once.

## Vision

**One Gufo encoder comparison fails:** relative L2 **6.47%** versus official
Transformers BF16, above the **5%** limit, on a 1024×1024 synthetic texture.
Both use the same converted GGUF weights. Against FP32, Gufo and Transformers
BF16 differ by **7.83% / 8.10%**, respectively. This is numerical drift, not an
image-answer score; **llama.cpp was not tested**. Optimizations retain native
embedding bytes but do not resolve this gap. [Evidence](artifacts/vision-parity.json).

## Extended context (YaRN)

**With YaRN off, the extension is a no-op: full-model logits and served
continuations are byte-identical to the pre-change build.** Unsloth
UD-Q4_K_XL target, shared Q8_0 MTP; September 26–29, 2026. YaRN-off identity
and the `--context` path were rechecked on the final build on September 29.

| Check | Result |
| --- | --- |
| YaRN off vs. pre-change build | Full-model logit dumps byte-identical on both prefill paths (fused F16 GEMM and PrepareAttention), 1,321-token prompt; 16/16 served continuations byte-identical |
| Prefill/decode, YaRN off | Prefill within 2% of the pre-change build at 8k/32k/131k (two runs each); decode and MTP acceptance unchanged at 29k (132/211 drafts both) |
| Needle retrieval, YaRN on | 12/12 via both `/v1/chat/completions` and raw `/v1/completions`: factor 1.5625 at capacity 409,600 (126,820 / 297,246 / 389,499 prompt tokens); factor 2.5 at capacity 409,600 (297,247 / 389,500, chat) and at capacity 655,360 (297,246 / 585,454) |
| Cache identity, YaRN on | A 29,646-token session cached under factor 1.5625 is not restored after restarting under factor 2.5 (full prefill; stored as a separate entry); it is restored after restarting under 1.5625 (0.8 s from disk, no prefill) |

Needle retrieval passed at every tested factor/capacity pair. A
369,989-token long read named the final scene correctly and quoted the
excerpt's last words verbatim; one image placed after 297,964 text tokens
in the same request (factor 2.5) was described correctly.

Logit agreement between YaRN on and off is close on a shared 1,927-token
prompt (dense attention for every implementation, since it is under the
2,048-token indexer budget): top-1 flips and KL divergence against the CPU
oracle and llama.cpp are similar with YaRN on or off (top-1 flips 21–29 of
1,927 positions; KL p50 ~4e-5; KL p90 ~0.01). At depth (the last 64
positions of long prompts), gufo versus llama.cpp agreement is 61/64 top-1
both at 248,487 tokens with YaRN off (KL median 0.011) and at 297,484
tokens with YaRN 1.5625 (KL median 0.008); the divergence there is gufo's
sparse attention versus llama.cpp's dense attention, present with YaRN off
too.

Idle GTT grows with capacity under YaRN (about 27.5 KiB per context token,
MTP resident): 88.5 GiB at capacity 262,144, 92.4 GiB at 409,600, 98.8 GiB
at 655,360. Prefill of the part of a prompt beyond 262,144 visible tokens
runs on the per-token attention kernel at about 350–375 tokens/s; a
585,454-token prompt averaged 493 tokens/s overall, with decode at 36
tokens/s.

Gaps: short-text quality with YaRN on is not measured (Qwen's model card
warns static YaRN can hurt short texts); there is no Transformers reference
beyond the native context, so agreement above uses llama.cpp and the CPU
oracle; and the fused WMMA kernel is not used for chunks whose visible
context exceeds 262,144 tokens.

## Reproduce

Tests live in [`tests/models/qwen38_flash_next`](../../../tests/models/qwen38_flash_next).
Use `--batch-only`, `--sampling-only`, `--prefill-only` or `--cache-only` on the
session test; the snapshot test covers persistent image/text state.

`--execution-only 133120` runs the focused deep AR/MTP logit and snapshot-replay
check without the other session suites.

For independent MTP checks:

```sh
nix develop -c cmake --build --preset gpu-test \
  --target qwen38_flash_next_model_tests qwen38_flash_next_gpu_probe
nix develop -c build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_gpu_probe \
  --model "$MODEL" --mtp-model "$MTP" --batch 2048 --mtp-audit
```

The [vLLM](https://github.com/vllm-project/vllm/blob/751f6807d9cb3de50c27a5f27188c4fb04fe0e2b/vllm/models/qwen4_exp/amd/mtp.py)
and [SGLang](https://github.com/sgl-project/sglang/blob/993d1fccbaafe3e79d91567d2fc1d665cc94fa50/python/sglang/srt/models/qwen4_exp_mtp.py)
formulas supply independent predictor checks; pinned Transformers ignores MTP
weights. [Vision reproduction](../qwen3.8-27b/QUALITY.md#vision).

### F32 reference fixtures

[`tests/fixtures/qwen38_flash_next/f32_reference`](../../../tests/fixtures/qwen38_flash_next/f32_reference)
holds the CPU F32 reference's log-sum-exp and top-256 logits for 2,479 prefill
positions over four prompts (documentation, C++ source, chat, tool calls),
for the model shards and reference revision in its `manifest.json`. Score
a build in about a minute, without the CPU reference:

```sh
F=tests/fixtures/qwen38_flash_next/f32_reference
PROBE=build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_gpu_probe
for p in docs:2048 code:512 chat:512 tools:512; do
  "$PROBE" --model "$MODEL" --prompt-file "$F/prompts/${p%%:*}.txt" \
    --batch "${p##*:}" --context 4096 --logit-rows 256 --dump "dumps/${p%%:*}.bin"
done
nix develop -c python3 tests/models/qwen38_flash_next/f32_reference.py compare \
  --manifest "$F/manifest.json" --dumps dumps [--against main-dumps]
```

Each `--logit-rows` dump has a `.json` sidecar with its batch, rows per pass,
token count and a hash of the token ids; `compare` rejects dumps whose sidecar does not match the
manifest. `--against` adds paired KL and next-token NLL intervals (overall
and per prompt) against a second build. Regenerate the
fixtures when the shards or the reference change: run
`qwen38_flash_next_reference_probe --tokens ... --dump` alone (no GPU model
resident: 661 tokens took 858 s, against 46 s per token beside the GPU
model), then `f32_reference.py write`. `gpu_probe --nll` measures the
next-token NLL over a long prompt file in 2,048-token passes.

## Benchmark method

All Gufo tables were refreshed on October 18, 2026, with the clean production
Nix build `fec4759d` (ELF SHA-256 prefix `1d822fb97b4639c61`).
The pinned compiler and dependencies are unchanged; the package includes the
qualified ROCr host-memory patch. Every single-user row reproduces the
previous qualified completion hashes and prompt/cache counts.
Four prior alternating mixed
MTP C2 repeats gave 110 drafts (about 54 tok/s) on main twice and the PR once,
and 144 drafts (about 60 tok/s) on the PR once, all with identical text.
The checked platform profile is `performance`, with tuned
`accelerator-performance`. Temperature telemetry is retained throughout.
Model jobs run sequentially without overlapping builds or profiling.
Commands, binary/model identities, counts and
retained samples are in [artifacts](artifacts/bench.json) and
[model identities](artifacts/model-identities.json). Full raw requests, phase
timings, diagnostic profiles and repeated controls are retained outside Git.

The driver is `tools/bench/model-bench.py run --target gufo --fresh`.
One warmed sample per point, HTTP, greedy, thinking off, penalties disabled;
automatic cache RAM budget. Repetitive MTP through 64K also includes one
focused repeat, preserving the ordered history and the initial sweep's realized
17,213,143,040-byte RAM cache budget. Those rows report mean ± sample standard
deviation, retaining the initial slower samples. Both repeat arms start after
GPU temperature recovery; telemetry and the initial flags remain recorded.
Single-user uses ordered pp2048/tg128 histories at
every depth from 0 to 131072, with capacity 133760. MTP pp is the maximum across
the independently retained mixed/repetitive workloads. Diagnostic profiled
runs and abandoned candidates do not supply published cells. Focused repetitive
MTP, loading, cache and screenshot controls use current main `f948ddd3`.
Controls preserve prompts, settings and ordered cache histories;
per-request phase flags require both a 5% and a 3 ms increase.

C1/2/4/6/8 use the same d0 prompts. Every session prefills before measured tg128,
with at most four prompt-tail tokens reevaluated. Rates sum individual decode
rates. Fresh C1 AR controls provide completion hashes for both workloads.
Depth calibration depends on the ordered sweep; paired speed controls retain
that order. Deep AR/MTP HTTP cache frontiers differ by one token, so exact
snapshot replay is checked separately with identical prefill boundaries.

All llama.cpp data retain September 22–23 provenance, unchanged at the user's
request; no reference benchmark or quality refresh was run. AR uses `b11069`;
historical MTP uses `6fcaa16f`. Reference capacity is 35456 through 32K, 68224
at 64K and 133760 at 128K. A separate October 10 check of that MTP pin at
pp2048/tg128, C1/capacity 4096 found that fresh and prepared mixed-prose MTP
outputs matched each other but differed from its own AR output. Both generated
128 tokens and MTP proposed 152 drafts; repetitive MTP was not attempted after
that failure. The historical speed values are not newly qualified equivalence
measurements. [Evidence](artifacts/decode-kernels-review.json).

Loading measures cold target/sidecar files to HTTP readiness, verified with
`POSIX_FADV_DONTNEED` and `mincore` on every file, C1/MTP/capacity 262144;
runtime libraries may remain warm. Memory uses C1/AR/capacity 133121 and peak
global HIP allocation, including idle memory. See the
[benchmark workflow](../../../.agents/skills/benchmark-model/SKILL.md).
