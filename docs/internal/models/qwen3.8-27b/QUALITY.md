# Qwen3.8 27B quality

**Greedy DFlash2 matches isolated AR on all 160 measured mixed/repetitive
requests across Q4/Q8 and C1/2/4/6/8.** Targets are Unsloth UD-Q4_K_XL /
UD-Q8_K_XL; the draft is Q4_K_M. [Identities and dated evidence](artifacts/model-identities.json).
These are execution-consistency checks; original-target and GGUF-conversion
parity remain unqualified.

| Check | Q4_K_XL | Q8_K_XL |
| --- | ---: | ---: |
| AR/DFlash2 mixed completions, eight depths 0–128K | 8/8 match | 8/8 match |
| AR repetitive C2/C4/C6/C8 requests versus C1 | 20/20 match | 20/20 match |
| DFlash2 mixed requests versus isolated AR, C1–C8 | 59/59 match | 59/59 match |
| DFlash2 repetitive requests versus isolated AR, C1–C8 | 21/21 match | 21/21 match |
| Target verification, widths 2–8 and ragged cohorts | Full logits/features match scalar | Full logits/features match scalar |
| Serving and state | Cancellation, continuation, RAM/disk restore and seeded replay pass | Same; also eight 32K continuation forks |

## Draft integration on October 21

Rebased onto main `e060aeed` (#699), preserving Flash-Next's requirement for
an established accepted copy before skipping a sampled neural proposal.
The shared lookup's source predicate and resume requirement have a combined
regression test. After rebasing, the Flash-Next lookup tests and Qwen27B
DFlash2 lookup, private RNG, request reset and restore checks pass. The
affected Flash-Next engine and continuation adapter also compile.

The serving measurements below predate this rebase. These focused integration
checks do not replace the remaining latency qualification or a full sweep.

## October 21 coding and prose follow-up

The retained change affects only five-token IQ4_XS FFN projections. It keeps
both original lane partials and their reduction order; three unused output
columns are zero and never read or write extra token rows. Actual 17408/34816
by 5120 gate/up and 5120 by 17408 down projections match scalar decode
byte-for-byte, including output guards. The independent CPU-qualified IQ4_XS
scalar error is 2.38e-7 against the existing 1e-6 limit. Cold read-only mapped
weights exceeding 128 MiB and three input scales also match exactly.

Native C2–C8 checks pass exact layers, logits, selector probabilities, private
RNG/controller state, mixed widths/positions and persistent replay. The final
HTTP default-sampling history through `sampling_concurrent_thinkingFalse`
passes all 467 timing checks against `77c82ecb`, without output, work or
coverage changes. Scalar AR, wide prefill, sampling policy and lookup state
have no new implementation changes.

The sampled coding/prose requests match the previous branch's complete text,
reasoning, tool/finish fields and draft/cache counts. Independently checked
generated interval code passes its own tests, 300 half-integer coverage cases,
input-preservation and invalid-input checks. Generated retry code passes
exact call/delay counts, capped delays, successful return, original last-error
identity, nonretryable errors and invalid configuration. Both main and the
candidate needed a larger output budget for the complete thinking example:
the retained 4096-token-budget run finishes at 3377 identical tokens.
The earlier 1536/3072-token truncations are retained, not counted as completed
task passes. [The functional checker](../../../../tests/functional/generation_quality.py)
requires completed responses.

The prose example is a speed and output-consistency control. Both original
main and the previous branch make numerical mistakes in their queueing
examples; this is not a factual-quality pass. The kernel change preserves
the previous branch's text and does not alter the target distribution.

One coding maximum-token-pause comparison is **inconclusive**:
138.43 → 155.87 ms, despite unchanged total decode time. An earlier two-kernel
candidate also reached 157.92 ms. A later retained-candidate run measured
137.69 ms; the focused identical-history replay measured candidate 137.72 ms
versus previous-branch 137.93 ms, with every request/phase check passing.
The outlier's cause is not established. The earlier cache/snapshot timing
limitations below remain unresolved; no stall-free claim is made.

Standard mixed d0/32K pp2048/tg128 controls preserve original main's complete
output and prompt/cache/draft counts. Q8 serving and full benchmark sweeps
were not repeated. Reports, commands, inputs and rejected experiments remain
under `/tmp/gufo-qwen27b-dflash2-20261021` on the development host.

## October 21 prefill and sampling review

Main `0c03748a` and `perf/qwen27b-dflash2-speed` use the same release toolchain,
Q4 target and Q4 DFlash2 draft. All six measured C1 AR/DFlash2 d0/32K
completions and prompt/cache counts match exactly. The candidate does not
change weights, cache precision, attention arithmetic or the target sampling
distribution. Draft probabilities change when numeric penalties or prompt
lookup apply; ordinary p/q acceptance and residual correction remain intact.

The native target prefill checks pass on Q4/Q8. Q4's cropped final output
tile matches forced full-layer capture for full-vocabulary logits and
continued cache state, including 2,048 rows at d0 and 1,025 rows at d8K.
The attention operator retains its independent FP64 checks; skipped query
rows remain untouched and the retained output/LSE bytes match full execution
across workspace head groups. Sampling passes 216 policies, including
speculative rejection/residual correction. New candidate-order tests compare
against both the original full GPU sort and independent CPU stable ordering
for vocabulary sizes 1,023/1,024/1,025/32,769/248,320, top counts
1/20/32/33/64, ties, signed zero and nonfinite inputs.

The second-stage checks cover four changes:

- Native replay of the exact 34,822-token repetitive history matches AR,
  including every proposal/acceptance round after memory restore, serialized
  restore and independent draft re-priming. Its lower neural acceptance is
  reproducible without a stale draft cache; this is evidence for that history.
- DFlash2 reuses Flash-Next's token-history index with private request and
  resume state. Copies report point-mass q and consume no draft RNG. Native
  memory/persistent/component restoration, old snapshot formats, request reset
  and invalid restored boundaries pass. HTTP C1/C2/C4 default sampling,
  greedy, mixed cohorts, stop, image cancellation and replay pass with 1,596
  actually accepted lookup tokens. Flash-Next's index tests still pass.
  Greedy requests allow a shorter initial match only when existing private
  neural acceptance falls below its neutral estimate; sampled requests retain
  24 tokens. This decision uses snapshotted state, without timing or new RNG.
  Its native test checks adaptive versus fixed-policy gates and exact
  pre-copy replay after memory and persistent restore. An independent source
  fixture contains real message, thinking and tool tokens: a continuation
  crossing a message boundary must select an earlier source, while thinking
  and tool tokens remain copyable. Flash-Next's default source selection is
  unchanged and tested separately.
- Eight independent CPU oracles qualify conditional draft probabilities with
  repeat-window expiry and positive/negative frequency/presence penalties.
  Native C2–C8 controls retain each request's exact layers, logits, selector
  probabilities, RNG and restored state. The original 216 sampling policies
  and target verification tests pass. Additional top-k/top-p/min-p filtering
  was rejected: it added a verification round to a short sampled request.
- Q4 four-token verification preserves every scalar output byte on real
  gate/up/down shapes, with three/eight-token controls. The independent CPU-qualified
  scalar relative error is 8.18e-8 against a 1e-6 limit. Cold mapped-weight
  controls cover three input scales; other width/format schedules are retained.

Selected Q4 HTTP `sampling-defaults`, `long-context`, `state-edges` and `cache`
checks pass correctness, including thinking, cancellation and RAM/disk restart.
After removing the sampled filtering and the overly permissive greedy lookup
gate, the focused long-context rerun passes all 124 timings with identical
output and work. Final default sampling has no output/history or coverage
changes from its matched first-stage control. Q8 serving and the full depth
sweep were not repeated.

The later matched lookup control exposed and fixed a real slowdown. Copying
the prompt's message-end tag and assistant header repeatedly wasted
verification: a default-sampled copy needed 24 rounds instead of main's 17.
Disabling lookup alone restored all 110 accepted proposals; disabling draft
penalties did not help. Selecting an earlier source without message framing
restores 17 rounds and 110/110 acceptance while skipping neural draft work.
C1/C2/C4 copies, mixed greedy peers, image copying and stop/resume all retain
main's exact output, request history and cache work. Their final comparison
passes all prefill/decode timings: 413 total pass, with five queue timings
inconclusive. The rejected implementation's 47 timing flags are retained.

Timing remains **inconclusive** for some concurrent/cache observations.
Final default sampling passes 1,401 timings and leaves 11 inconclusive around
admission, snapshot work, token gaps and differing physical cohort widths.
One width-one request has a 347.68 ms snapshot and 317.08 ms token gap, without
slower decode. A focused main/candidate replay retaining its preceding history
passes all 467 timings, with width two and token gaps around 158–163 ms. It
does not reproduce or qualify the original width-one spike. Earlier unchanged
main also reached a 219.75 ms gap at width one; that is not evidence for the
larger 317 ms observation's cause. Admission pauses remain about 189–206 ms
on both. The original first-stage comparison also left two disk-restore flags,
which were not repeated and remain unqualified. No margin was widened; these
results are not a blanket latency pass.

Public workflow substitutes pass exact record extraction and generated URI
tests, including three injected bugs. The file-edit fixture fails on both
main and candidate solely because the identical tool-generated contents omit
the required final newline. Literal thinking/tool tags are preserved. The
fixture remains strict; its subsequent edit/thinking loop is not qualified.

Raw profiles, binaries and all observations are retained outside Git under
`/tmp/gufo-qwen27b-dflash2-20261021` on the test host, with reports copied to
the editing host's `qwen27b-dflash2-20261021` directory. The principal records
are `bench-main-q4`, `bench-v3-q4`, `bench-v7-final-q4`, `bench-v8-q4`,
`acceptance-d32767.json`, `long-control-v3`, `long-control-v7`,
`sampling-v8-final`, both `snapshot-control-*` directories, `lookup-v8-q4`,
`lookup-main-control-q4`, `lookup-timing-pair`, `workflows-main` and
`workflows-v8-final`.
Superseded experiments and every timing flag remain retained. The focused
[performance table](BENCHMARKS.md#october-21-focused-prefill-and-sampling-comparison)
keeps the dated main/candidate results; the older full sweep and llama.cpp
measurements are unchanged. Feature capture/storage work in #688 is separate.

## October 19 wave64 verification review

The wave64 verification attention (#660) reproduces the scalar decode bits.
Single-row outputs from main `23fd87dc` and the candidate are byte-identical
in the production 24/4-head shape (fp32 and fp16 caches, gated and plain,
starts 0–65,536), and verification rows 2–8 match those single rows. Every
Q4/Q8 DFlash2 single-user completion from 0 to 128K matches the earlier
October 19 rows; all concurrency completions match Gufo AR. The Q4 DFlash2
functional `batch`, `long-context`, `conversation`, `state-edges`,
`cache-concurrency` and `cache` suites pass against main with no output
change; 2,239 of 2,246 timings pass and 7 are inconclusive.

## October 19 recurrent-state review

The refreshed DFlash2 concurrency corpus matches fresh, isolated AR on all
160 requests. Main `e7730942` and PR `d5e90965` also match all 54
teacher-forced full-vocabulary logit rows, plus the native prefill and
continuation fingerprints. Verification widths, ragged cohorts through C8,
rollback, both recurrent-buffer parities and 2,080-row diverging prefill pass.
Q4/Q8 AR and DFlash2 HTTP checks cover sampling, thinking, tools, cancellation,
images and RAM/disk continuation.

The matched Q4/Q8 mixed depth sweeps also preserve every completion and draft
count from 0 to 128K. PP/TG passes the per-request gate at all sixteen points.
Current main already shows the lower PP rates relative to the September table;
the candidate stays close to main and improves decode at every measured depth.

The local snapshot cleanup preserves serialized bytes with differently filled
destination buffers; Q4/Q8 draft restore and HTTP disk-restart checks pass.
Matched performance controls retain every timing flag. Remaining overlapping
snapshot timings and admission pauses are inconclusive, not a blanket latency
pass. See the [review record](artifacts/recurrent-fork-review.json).
The earlier eight-depth AR/DFlash2 and quantization qualification retains its
original date; these new checks establish consistency with current main.

## Quantization comparison

September 24, 2026: short teacher-forced comparison against **Gufo's BF16 target**, with identical
prefixes: three prose/code fixtures, nine logit rows each, full 248,320-token
vocabulary, temperature 1 without filtering. This measures quantization and
native arithmetic together; BF16 here is not an independent upstream oracle.
Results and reproducible identities are in [the compact record](artifacts/quantization-quality.json).

| Target versus BF16 | Mean KL divergence (nats) | Mean total variation | Top-1 agreement |
| --- | ---: | ---: | ---: |
| Q4_K_XL | 0.001650 | 0.01566 | 27/27 |
| Q8_K_XL | 0.00006537 | 0.003128 | 27/27 |

## DFlash2 and sampling

The pinned [original DFlash2 operators](https://github.com/z-lab/dflash/tree/07ebd93db9f472af339b644bb70221ad8428328a)
were checked with independently decoded GGUF weights: worst stage relative
RMSE **5.83e-6**, full-logit maximum error **7.72e-5**, proposal total variation
**1.60e-5** on a 24-token prefix/seven proposals. Limits are 1e-4, 1e-3 and 1e-4.
This qualifies draft execution, not weight conversion.

AR and verification apply penalties → temperature → top-k → top-p → min-p.
Tests cover 216 GPU sampling cases and 23 served strategies, including p/q
acceptance, full-vocabulary residual correction, nonfinite inputs and replay.
**Sampled DFlash2 need not match AR's same-seed sequence**; within-mode seeded
replay requires the same configuration. GPU/CPU distributions have numerical
tolerances, not bit-identical upstream sampler parity.

## Vision

Earlier Q4/Q8 AR and DFlash2 checks cover multi-image, sampled/greedy C4,
cancellation, generated-history continuation and disk restart. The current
component migration retains a strict Q4 cold-prefill versus decoded-history
failure; see [the current status](#component-serving-migration). Q8 also requires exact
cold/live/restored continuation logits. Encoder controls use official
Transformers operators over the **same converted BF16 GGUF**; they cannot detect
conversion errors. Flash-Next has a separate [vision limit](../qwen3.8-flash-next/QUALITY.md#vision).
Q4/Q8 image snapshots restored after cancellation inside the image encoder or
text prefill match all target logits exactly; token-limit endings and stripped
tool reasoning retain the compatible conversation prefix.

## Autoregressive component checkpoints

October 10, 2026: the model-local component adapter passes Q4/Q8 checks in
both FP16 and FP32 attention KV layouts. Every restored component byte and
full continuation logit row matches uninterrupted execution. The checks cover
earlier-boundary edits, pending generation tokens, leased-row preservation,
committed recurrent replay, failed/discarded transfers and completion lifetime.
Image regressions restore before the first image and between two images while
retaining the matching request's future images and complete RoPE layout;
those focused image checks use production FP16 KV.

Production FP16 probes at 32K and 100K also match every restored byte and
sixteen continuation logit rows/token decisions after transfer to another slot.
These are Gufo execution-consistency checks. They do not extend upstream or
weight-conversion parity, qualify alternate BF16 recurrence, or qualify the
common checkpoint allocator. The [adapter inventory and commands](CONTINUATION-ADAPTER.md)
and [card 15 results](https://github.com/gufo-org/gufo/blob/0fb50ce436044a9687c78635e7afbc2bb65fc6f4/docs/cache-redesign/cards/15-qwen27b-ar-adapter.md)
record the scope and retained evidence.

## Long-context FP32 attention qualification (2026-10-13)

A new capacity probe exposed an invalid FP32 prefill launch at 16K: its
context-sized score array plus reduction scratch used 66,560 bytes of LDS,
above gfx1151's 64 KiB limit. The bounded online fallback passes the existing
attention controls and new three-row causal checks around the LDS boundary,
32K and 100K in both KV representations. The worst absolute FP64-oracle error
is 5.04e-7 (limit 1e-5); long rows match scalar online attention exactly, and
BF16 output bits match independent rounding. The pinned `64639c28` Q4 FP32
32K full-model probe also passes every restored component byte and sixteen
continued full-logit/token checks. After the telemetry-only repair, the pinned
`e008ffc7` binary passes Q4/Q8 FP16 and Q4 FP32 at both 32K and 100K,
and Q8 FP32 at 32K. Every restored component byte and sixteen continued
full-logit/token checks match. The user stopped the Q8 FP32 100K probe and
skipped further long-capacity probes for both KV precisions. That case has no
qualification result. Matched-main timing flags remain open. All original aborts and diagnostics remain retained in
`/home/mixer/gufo-qualification/qwen27b-component-2026-10-13`.

## Reproduce

Run only the affected [model suite](../../../../tools/qwen27b/check.py):

```sh
nix develop -c python3 tools/qwen27b/check.py fast
nix develop -c python3 tools/qwen27b/check.py reference \
  --model "$MODEL" --reference-model "$BF16_REFERENCE"
nix develop -c python3 tools/qwen27b/check.py serving \
  --model "$MODEL" --dflash-model "$DRAFT"
```

[Draft oracle](../../../../tools/qwen27b/dflash_reference.py),
[vision oracle](../../../../tools/qwen27b/vision_reference.py) and
[vision serving checks](../../../../tools/qwen27b/vision_check.py) retain source
pins and compare operators, full logits, RNG, image/state isolation and replay.
Full logits stay outside Git. A missing-weight skip is not a pass.

## Meaning of exact

Matching Gufo AR/speculative tokens or completion hashes tests consistency.
Matching llama.cpp hashes also does not establish original-model accuracy;
quantized implementations can disagree. Keep the reference, weights and history
fixed before interpreting any numerical difference.

## Benchmark method

All Gufo rows refreshed October 19, 2026 on `11813e51` (#660, rebased on
`23fd87dc`); llama.cpp retains September 23 measurements. One warmed sample per
point, greedy, thinking off, `performance` platform profile.
Single-user rows use pp2048/tg128 and cached depths 0–128K. Each mode creates
its own eight-token prefix reply; histories can differ across engines.
Concurrency uses the older short-prompt corpus, not matching pp2048 prompts;
its C1 is not directly comparable to single-user d0. AR uses repetitive text;
DFlash2 uses mixed/repetitive cases. Rates sum individual decode rates.
Loading uses cold files (`POSIX_FADV_DONTNEED`, zero resident pages checked
with `mincore`), C1/DFlash2/capacity 262144; memory uses C1/AR peak HIP
allocation. Commands, hashes and counts remain in [artifacts](artifacts/bench.json)
and the [benchmark workflow](../../../../.agents/skills/benchmark-model/SKILL.md).

## Component serving migration

October 20 refresh: main `1b0a5a84` is merged (`8ed4da99`), with fixes
`4d802349`, `e8713aba` and `94fee48e`. The last stops a prompt that ends
exactly at a partial-prefill frontier from decoding a placeholder token; it
recomputes from a shallower boundary, and Q4 AR/DFlash2 suites do the same work. Q4/Q8 native adapter, DFlash2 state, serving-cache,
target numerical and attention operator checks pass. Restored component bytes
and full logits match across FP16/FP32 KV, edits, rollback, images, sampled
frontiers, cancellation, disk compatibility, concurrent forks and cleanup.

All selected HTTP profiles pass every stage (23 stages, 290 requests): Q4 AR
and DFlash2 twice, Q8 DFlash2, and Flash-Next AR twice and MTP once on both
main and the candidate, because the PR moves the shared snapshot copy helper
and touches the shared verifier, backend and scheduler. Q4 DFlash2 observes
3,423 proposed and 2,016 accepted draft tokens. Main fails only `cache-edits`,
which its legacy 27B cache does not support.

Speculative rollback (`RestoreState`) previously guarded every attention row
although it only swaps recurrent buffers. Each rejected draft therefore copied
borrowed rows in the foreground, and disk-only checkpoints were retired before
publication. After the fix, every over-RAM DFlash2 checkpoint publishes and
concurrent cold DFlash2 decode returns to main's 53 ms per token from 60–64 ms.

High-context and first-request probes (main/PR/PR/main) pass with equal work.
Standard pp2048/tg128 stays within 1.6% of main. Deep-peer AR and DFlash2
pass 142K crash recovery, twelve short peers reusing the system boundary and a
second restart. Pi AR and DFlash2 pass tools, edit and bugfix workflows; a first
AR run failed only because the model wrote a trailing newline.
Over-RAM passes through `E1`. `E2` released its publication wait at the
36.6 s cost bound, 6 s before the deeper checkpoint, which was queued behind
two older jobs, was published. That timing-dependent case remains unqualified.

Memory: the component cache commits about 2.27 GB of backing before readiness
in every mode (`GrowAhead`), as Flash does on main. Each DFlash2 checkpoint
also carries a fixed 84.7 MB of draft ring and feature state. Main's legacy
snapshots live in device (GTT) memory and the component cache keeps them in
host RAM, so compare system totals: the peak MemAvailable drop over a full run
is 50.3 GiB versus main's 52.4 GiB (Q4 AR) and 55.0 versus 57.4 GiB (Q4
DFlash2).

Accepted timing costs, repeated in both observations: a 3–6 ms admission floor
in `queue_ms` (component acquire costs about 2.8 ms; Flash shows the same floor
on main) and higher `cache_snapshot_ms` where main captured nothing or less
in `cache-concurrency`. Optimization is tracked in
[#678](https://github.com/gufo-org/gufo/issues/678). Disk requests queue less than on main. See the
[dated review record](artifacts/component-cache-review.json) for commands,
binaries and every observation. The earlier cold-prefill versus
decoded-history vision difference remains unqualified; the wide prefill route
restored by `dabc85cf` is retained.
