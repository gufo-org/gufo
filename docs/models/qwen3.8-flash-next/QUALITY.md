# Qwen3.8 Flash-Next quality

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

## Benchmark method

Single-user and concurrency tables were refreshed on October 17, 2026, with
production Nix build `3b02d877` plus the reusable PLE gather scratch
(binary SHA-256 prefix `93ec6640be94adb9`). The artifacts retain the dirty
build identity, and the production file hashes identify the exact source.
Loading and memory use the
October 16 `fdc21849` capture (binary prefix `b1ab0e9795df6fba`).
The pinned toolchain is unchanged. Every single-user row reproduces main's
completion hashes and prompt/cache counts; prompt lookup changes draft counts.
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
automatic cache RAM budget. Single-user uses ordered pp2048/tg128 histories at
every depth from 0 to 131072, with capacity 133760. MTP pp is the maximum across
the independently retained mixed/repetitive workloads. Diagnostic profiled
runs and abandoned candidates do not supply published cells. Matched AR and
MTP controls for this refresh use main `3389a585`.
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
