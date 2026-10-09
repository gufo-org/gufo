# 07 · Retention policy port

**Milestone:** Common package · **Depends on:** 05, 06 · **Size:** M ·
**Affects:** nothing at runtime · **Status:** done

## Goal

Port today's checkpoint selection and eviction ranks into the new package
unchanged, and log every admission and removal so retention can be replayed.

## Scope

- Capture candidates, as today:
  - stable boundary and complete prompt;
  - up to four 2,048-token grid points;
  - learned divergence points with at least 512 tokens of improvement;
  - skip grid points within 128 tokens of the prompt end.
- Removal ranks, as today: retry 0, history 1, covered continuation 2, branch
  point or last copy 3. LRU within a rank. Record limit 128.
- Adapter pass-plan query, so optional boundaries align with planned prefill
  passes. Exact required boundaries are never rounded.
- An event for every admission, refusal and removal, with rank, reason and
  unique bytes freed.

## One unavoidable difference

Admission charges unique bytes instead of each checkpoint's full payload. The
same budget therefore admits more checkpoints than today. This is intended, but
it changes behavior, so card 19 reports it separately from storage effects.

## Not in this PR

Any new policy: rank changes, a cost-aware shorter restore, density tuning.
Each needs its own card with measurements.

## Test first

Port the policy cases from the current cache's unit tests and replay them
against the new package with the fake adapter.

## Step baseline

Replay the RFC's archived W1–W4 request traces through the package with the
fake adapter using real component sizes. Record checkpoints admitted, refused
and evicted, and unique retained bytes over time.

## Implementation record

[`retention.hpp`](../../../src/cache/retention.hpp) and
[`retention.cpp`](../../../src/cache/retention.cpp) add a caller-serialized policy
owning at most 128 checkpoint records and their prefix-index entries. Retention
roles (retry, history, continuation, branch point) remain distinct from capture
provenance (prompt, generated, grid, learned). Removal ranks are computed from
exact token prefixes, compatibility and supplemental input identities. The port
preserves inferred branching, learned points surviving their original branch,
deeper learned points superseding shallower ones, exact replacement preserving
learned status, edited-tail replacement at record capacity, source protection,
advancing a family before evicting another family's last copy, and LRU ties.
Retry/history admission cannot remove a last useful continuation.

Capture admission uses the existing ledger reservations, including actual
shared-chunk ownership and separately charged metadata. It never sums full
checkpoint payloads as independent charges. `ExecutionHistory::NewPayloadBytes`
provides a preflight count of new private state, private tails and newly owned
chunks. The caller supplies that count before admission; a payload exceeding
the total/RAM budget refuses before eviction. Actual resource exhaustion is a
`ResourceExhausted` (a `bad_alloc` subclass), permitting rank-eligible removal
and a bounded retry at the same execution boundary. Ordinary allocation,
injected and transfer failures refuse retention without evicting for record
capacity. Reservations and partially captured handles unwind before failure
reporting. This package remains disconnected from production serving.

`Adapter::PlanPrefill` is a pure host query returning exact, increasing planned
pass ends. Required reused/stable/complete-prompt captures keep their exact
positions. Grid selection retains the existing four-point distribution, the
2,048-token minimum improvement from a reused frontier and the strict >128-token
end gap. A grid may use a preceding pass end within 128 tokens if it still meets
those constraints; otherwise it keeps its original grid point. Learned points
need >=512 tokens of improvement and >64 tokens before prompt end, and coalesce
with an already planned capture within the existing 64-token slack. Capture
results report whether each boundary would split the queried plan. Existing
retained boundaries are skipped. An empty prompt has no capture candidates.

Each admission, refusal, removal and LRU touch emits a bounded typed event with
an instance-local sequence, checkpoint ID, boundary, retention role, computed
rank, reason, last-use clock, retained record count, current ledger bytes and
unique retained payload bytes. Removal records the actual reduction in ledger
bytes, including metadata; shared/external reader and persistence owners can
prevent payload reclamation. Logical bytes released by dropping policy ownership
are not reported as spendable physical bytes. Events contain no prompt tokens.
Replaying their IDs/actions reconstructs every retained set and LRU update;
recomputing ranks independently additionally requires the original token trace.
The event sink must be nonthrowing and cannot reenter the policy. Shutdown logs
removals, and destroys policy-owned index entries before releasing its metadata.

[`retention_test.cpp`](../../../tests/cache/retention_test.cpp) ports the relevant
policy cases from the current cache, exercises real ledger capacity and shared
chunks, and verifies the hard 128-record limit, input identity isolation,
cancellation, failed allocation/copy, impossible admission, exact replacement,
reader pins after eviction, and event-log replay. Real captures with the delayed
fake adapter settle all copies before the next append mutates recurrent state.
Capture factories must reserve before allocation and finish every transfer before
returning; `Admit` is host-synchronous and does not run the next prefill pass.

## Results

Step baseline on Linux 7.2.9 x86-64, AMD Ryzen AI MAX+ 395, based on main
`7035ebbf`, using flake.lock-pinned GCC 15.3.0, CMake 4.3.4 and Python 3.14.6.
The RFC's original E2 W1–W4 token arrays and request metadata were recovered from
`a32fc43bcb9ec66264f166b5162d93a9223557d5`; all 133 archived file checksums passed
against the manifest retained in `b9ca6b4e18abe50d4d446f5460823b7bcb4cdb33`.

[`retention_trace.py`](../../../tests/cache/retention_trace.py) exports those
arrays using the standard library. [`retention_replay.cpp`](../../../tests/cache/retention_replay.cpp)
replays them through the actual package, using the fake adapter's pass plan and
size-only captured owners. The measured aggregate payload shapes are Flash-Next
MTP **119,328,358 fixed bytes + 27,460 bytes/token**, and 27B DFlash2
**243,700,000 fixed bytes + 65,536 bytes/token**, matching the RFC's constants.
These are rounded measured aggregate layouts represented as one append component
and one private component, not complete numerical model inventories. The RAM
budget comes from each workload's archived `e8.json`; token lengths use the
archived retokenization, without scaling to reported production token counts.

| Trace | Requests | Admitted | Refused | Evicted | Peak unique payload bytes | Final records | Required / optional predicted splits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fn/w1 | 36 | 76 | 9 | 41 | 9186913708 | 35 | 36 / 13 |
| fn/w2 | 36 | 76 | 0 | 25 | 9216642132 | 51 | 36 / 1 |
| fn/w3 | 60 | 112 | 30 | 71 | 9111036350 | 41 | 60 / 18 |
| fn/w4 | 33 | 97 | 0 | 5 | 8869009478 | 42 | 33 / 13 |
| q27/w1 | 36 | 79 | 6 | 35 | 22926332864 | 44 | 36 / 13 |
| q27/w2 | 36 | 77 | 0 | 16 | 23196579552 | 61 | 36 / 2 |
| q27/w3 | 60 | 123 | 18 | 70 | 23338764288 | 53 | 60 / 18 |
| q27/w4 | 33 | 97 | 0 | 0 | 20686749440 | 41 | 33 / 11 |

This is a sequential checkpoint-only policy baseline. It uses real prefix lookup
and inherited execution history, but excludes live frontiers, decoded output,
concurrent scheduling, disk restoration/persistence and model execution. It clears
retention at archived server-lifetime changes. Shutdown/restart removals are
logged but excluded from the eviction column. Required/optional split counts
are predictions against the queried prefill plan, not executed GPU passes.
Its outcomes must not be compared directly with the RFC's end-to-end simulation
or production counters; unique-byte admission changes are qualified separately
at switch-over in card 19. The CSV logs include unique retained bytes after every
event; a separate replay verified IDs, sequence order, touches, retained counts
and empty final teardown for all eight traces.

The host-synchronous capture microbenchmark
[`retention_bench.cpp`](../../../tests/cache/retention_bench.cpp) measured 32-byte
fake-adapter private capture submission plus completion wait, with 100 warmups
and 1,000 measured checkpoints per case:

| Fake CPU case | ns/checkpoint |
| --- | ---: |
| No queued unrelated work | 112.84 |
| 10,000 hash updates queued on the capture stream | 8,715.90 |
| Same work queued on an independent stream | 63.66 |

The shared FIFO makes the host wait drain preceding work; the independent
stream does not. These are single-run host-fixture observations, not GPU stalls,
transfer bandwidth or an inference speed comparison. The production host stall
and shared HIP-stream coupling seen in #445 remain unmeasured until concrete
model adapters are connected. The contract and fake fixture cannot establish
those GPU costs.

### Validation and replay commands

CPU configuration/build/test commands used a `pkgs.mkShell` with the repository's
flake.lock-pinned nixpkgs providing GCC, CMake, Ninja, pkg-config, Python,
clang-tools, ICU, curl, libpng, libjpeg, libwebp and OpenSSL. Production library
configuration used the production package's `inputsFrom` dependency shell.

```sh
cmake --preset cpu-test -DGUFO_BUILD_TOOLS=ON
cmake --build --preset cpu-test --target cache_retention_test cache_retention_replay cache_retention_bench --parallel 4
ctest --preset cpu-test -R '^cache_(retention|slot|checkpoint|ledger|adapter|adapter_lifecycle|prefix_index|boundary)_test$' --output-on-failure
cmake --preset cpu-sanitizer
cmake --build --preset cpu-sanitizer --target cache_retention_test cache_slot_test cache_checkpoint_test cache_ledger_test cache_adapter_test cache_adapter_lifecycle_test cache_prefix_index_test --parallel 4
ctest --preset cpu-sanitizer -R '^cache_(retention|slot|checkpoint|ledger|adapter|adapter_lifecycle|prefix_index|boundary)_test$' --output-on-failure
cmake --build --preset pr --parallel 4
cmake --preset release -DGUFO_BUILD_TOOLS=OFF
cmake --build --preset release --target gufo_cache --parallel 4
nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py
```

After following the RFC's archive recovery/checksum instructions, run for each
`fn`/`q27` and `w1`/`w2`/`w3`/`w4` pair:

```sh
python3 tests/cache/retention_trace.py "$CACHE_EXP_DIR/results" fn w1 > /tmp/gufo-card07-fn-w1.tokens
build/cpu-test/tests/cache/cache_retention_replay /tmp/gufo-card07-fn-w1.csv < /tmp/gufo-card07-fn-w1.tokens
build/cpu-test/tests/cache/cache_retention_bench
```

All eight focused CPU and ASan/UBSan tests, all 49 hosted PR checks, the
production release cache library, shared formatting and diff whitespace checks
passed. Reports and generated token/CSV/summary artifacts remain in ignored
`build/` directories or `/tmp/gufo-card07-*`. No GPU/model qualification is
claimed because serving behavior is unchanged.

## Done when

- [x] Ported cases pass.
- [x] A recorded event log is enough to replay the retention decisions.

## Review focus

- Parity with the current cache's policy: anything missed?
- Is the pass-plan query enough to avoid splitting prefill passes for
  optional checkpoints?
- In-pass captures must finish before the next chunk mutates their sources:
  the guard is host-synchronous and transfer sources must remain unchanged.
  Measure the resulting host stall per checkpoint and shared-stream coupling,
  as seen in [#445](https://github.com/gufo-org/gufo/pull/445), alongside any
  additional prefill pass splits.

## References

Material to consider during implementation. Before reusing code from a
project, check its license and add it to `THIRD_PARTY_NOTICES.md` when
required (see the README).

- [Marconi: Prefix Caching for the Era of Hybrid LLMs](https://arxiv.org/abs/2411.19379) (MLSys 2025): admitting recurrent states by reuse likelihood and FLOP-aware eviction. For later policy cards: this card ports today's policy unchanged.
- [KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows](https://arxiv.org/abs/2507.07400) (NeurIPS 2025): eviction beyond LRU using the expected next use of an agent's prefix.
- [TraceLab: Characterizing Coding Agent Workloads for LLM Serving](https://arxiv.org/abs/2606.30560): misses after long idle gaps between user turns; step-type driven eviction.
- [llama.cpp server slots and context checkpoints](https://github.com/ggml-org/llama.cpp/blob/41abbfd59/tools/server/server-context.cpp): checkpoint spacing and bounded checkpoint lists.

## RFC

[Checkpoint selection and eviction](../RFC.md#checkpoint-selection-and-eviction)

## Review notes

### Independent review, round 1

A fresh-context reviewer reported two confirmed P2 findings in
[PR #503](https://github.com/gufo-org/gufo/pull/503).
Admission now shares the prefix index's resident-coherence predicate, checking
its complete registered inventory, layouts and exact private-state boundary
before insertion or replacement. Regressions for missing components, changed
layout/row bytes/chunk geometry and private state before the boundary verify
that the existing exact checkpoint and ledger bytes remain unchanged.

The replay previously retained every `PrefixLookup` candidate handle throughout
admission. Those unintended reader pins prevented victims' storage reclamation,
distorting the retention baseline. It now releases lookup handles after selecting
and planning; the selected source remains protected by its record ID. All eight
traces were rerun and their logs replayed successfully. The corrected table above
supersedes this original, invalid harness evidence from `7de22156`:

| Trace | Requests | Admitted | Refused | Evicted | Peak unique payload bytes | Final records | Required / optional predicted splits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fn/w1 | 36 | 81 | 5 | 80 | 9116310526 | 1 | 36 / 14 |
| fn/w2 | 36 | 76 | 0 | 25 | 9216642132 | 51 | 36 / 1 |
| fn/w3 | 60 | 116 | 26 | 75 | 9111036350 | 41 | 60 / 18 |
| fn/w4 | 33 | 97 | 0 | 6 | 8822587420 | 42 | 33 / 13 |
| q27/w1 | 36 | 83 | 3 | 53 | 22911687872 | 30 | 36 / 14 |
| q27/w2 | 36 | 77 | 0 | 17 | 23196579552 | 60 | 36 / 2 |
| q27/w3 | 60 | 125 | 16 | 72 | 23344317312 | 53 | 60 / 18 |
| q27/w4 | 33 | 97 | 0 | 0 | 20686749440 | 41 | 33 / 11 |

The original summary and CSV logs remain in `/tmp/gufo-card07-initial-replay/`;
corrected files are `/tmp/gufo-card07-replay-summary.json` and
`/tmp/gufo-card07-{fn,q27}-{w1,w2,w3,w4}.csv`. This is a correction of CPU replay
ownership, not an inference performance result. The initial focused clang-tidy
check also identified missing deleted special members on the event sink; these
are fixed, and focused tidy passes.

### Independent review, round 2

The second fresh-context reviewer confirmed both round 1 fixes and independently
replayed all eight corrected CSV logs, including retained IDs, touches, counts,
sequence resets, payload peaks and empty teardown. It identified a remaining
parity error: `preserve_source` is a byte-admission flag in today's cache, while
full-record publication may still advance a redundant source before evicting
another family's last copy. The port had incorrectly applied that flag at record
pressure too.

Record-pressure source advancement now follows `ContinuationCache::Commit`
regardless of the flag. The sole-record continuation replacement also matches
that path. Byte-pressure admission continues protecting the incomplete source
when requested. The new regression failed before the fix and passes afterward
for both flag values, with separate coverage of byte-pressure protection.
Focused retention CPU, ASan/UBSan, tidy, release-library, formatting and
documentation checks pass. The trace baseline never reaches its 128-record
limit, so this record-pressure fix does not alter the corrected table.

### Independent review, round 3

The third fresh-context reviewer verified the earlier fixes and replay evidence,
and confirmed an optional-admission parity gap. At a full record pool, today's
`ReserveSnapshot` checks that an eligible non-source record exists before its
exact-replacement search. The new policy had exempted exact matches, allowing
retry/history captures to bypass that guard. An exact history could downgrade
an inferred continuation branch and lose its reusable boundary under pressure.

The original optional-role guard now runs before exact-match handling. New
regressions failed on the previous revision and pass after the fix: retry and
history exact replacement with one/two full records refuse before capture; an
inferred branch remains protected afterward; an exact learned point's refused
retry reports the original retry role/rank; and a permitted history replacement
with another eligible record still preserves learned status. Both the third
review's old/new scratch reproductions and the branch-loss example are retained
under `/tmp/gufo-pr503-r3-*`. The fixed policy is validated with the focused
ordinary/sanitizer test, production library rebuild and the final hosted suite.
The archived baseline stays below its 128-record limit and is unaffected.
