# 01 · Functional coverage for real cache workloads

**Milestone:** Preparation · **Depends on:** — · **Size:** L (combined scenario
families) · **Affects:** tests only · **Status:** in progress

## Goal

Grow functional coverage of the ways clients actually use the cache, so the
switch-over (card 19) can show the new cache handles them. Any scenario that
improves coverage is welcome, whether today's cache passes or fails it.

## Already covered

Keep these; extend them where a pattern below fits better than a new suite.

| Suite or tool | What it covers |
| --- | --- |
| `cache-growth` | Growing conversation, unchanged retry, Messages thinking replay, dropped reasoning |
| `cache-edits` | Edited history |
| `cache-depth` | Deep borrowed frontiers across branches and slot reuse |
| `cache-rotation` | Switching conversations keeps useful history; snapshot budget |
| `cache-shared-prefix` | New conversations reuse a shared system prompt |
| `cache-concurrency` | Concurrent requests sharing a prefix prefill it once |
| `cache-bridge` | Bridge rewriting the user message in history, with short unrelated requests in between |
| `cache`, `cache_disk_spacing.py`, `continuation.py --restore` | Restart persistence, cancellation and replay, an image appended after a cached turn |
| `system-injection` | Mid-conversation system messages (Qwen hoists them into the leading system turn) |
| `agent_long.py`, `pi_agent.py`, `opencode_agent.py` | Real agents past 100k tokens, with compaction and pruning **disabled** |

## Usage patterns to add

| # | Pattern | Seen in | The scenario checks |
| --- | --- | --- | --- |
| 1 | **Compaction near the context limit.** History is replaced by a summary; system prompt and tools stay. | Claude Code, Codex, OpenCode. Rare but large drops (> 64k tokens) in TraceLab. | Reuse covers system prompt and tools only. Under pressure, the abandoned history's checkpoints go before live conversations' ones. |
| 2 | **Tool-output pruning.** Old tool results are replaced by a placeholder deep in history. | OpenCode prune, Claude Code tool-output clearing. llama.cpp reports full re-processing on Gated DeltaNet models. | Restore the last coherent checkpoint before the first pruned result; prefill only from there. |
| 3 | **Micro-reductions at user turns.** A few hundred tokens removed near the end. | Codex (TraceLab). | Reuse up to a checkpoint before the change, not a cold prefill. |
| 4 | **Human-paced gaps.** A conversation idles while others run, then resumes. | TraceLab: most user-turn misses follow > 5 min idle. | A resumes from RAM or disk after other conversations filled slots and budget. Simulate with pressure, not wall time. |
| 5 | **One large tool result.** A single append of 10k+ tokens, then a retry or edit right after it. | TraceLab: rare 10k+ appends carry over 70% of prefill. | The retry or edit does not prefill the large append again. |
| 6 | **Volatile tokens near the start.** A per-request header or timestamp in the system prompt. | Claude Code's attribution header defeats llama.cpp reuse. | No false hit, and one-use entries do not push useful checkpoints out. |
| 7 | **UI side requests.** Title, tag and follow-up generation after every reply, with a prompt that embeds the chat. | Open WebUI background tasks. | The main conversation's next turn still reuses its history. |
| 8 | **Claude Code over `/v1/messages`** with tools and streaming. | gufo #469. | A long tool loop reuses history turn after turn. |
| 9 | **Subagent fan-out, then the parent resumes.** | Claude Code Task tool; RFC W2. | Children share the parent prefix; the parent's deep checkpoint survives in RAM or on disk. |
| 10 | **Model swap restart.** The server is stopped and started between turns, as llama-swap does; also an abrupt kill while a write is queued. | llama-swap setups; RFC W4. | The next turn restores from disk; after a kill, only published checkpoints hit. |
| 11 | **A short RAM hit beside a deeper disk checkpoint.** | RFC E3 (W2). | The deeper usable checkpoint is selected. |
| 12 | **A checkpoint larger than the staging budget.** | RFC. | It persists and restores after a restart. |

Several of these fail on today's cache (2, 11, 12 at least); others pass. Both
kinds are useful.

## Rules for expectations

- Express expected work through boundaries ("prefill ≤ gap to the nearest
  retained boundary"), not exact token counts tied to today's grid policy.
- Assert through what the server reports today (cached tokens, prompt
  progress). Card 19 adds richer per-request fields.
- Compare answers with uncached controls, as the existing suites do.
- Suites that need disk or restarts are selected explicitly, not part of `all`.

## Activation

The new model workloads remain opt-in until card 19. Both functional runners
exclude `cache-compaction`, `cache-transforms`, `cache-pressure` and
`cache-messages-loop` from `--suite all`; the disk lifecycle runner is separate.
Their known legacy-cache failures cannot fail routine functional runs. The
passing CPU harness/gate checks remain enabled. Explicit baseline/development
runs retain strict failures. Card 19 enables the model workloads in routine
qualification after the new cache satisfies their contracts, with separate
settings for RAM pressure and disk/restart cases.

## Not in this PR

Changes to the current cache to make failing scenarios pass.

## Step baseline

For each new scenario, run it on current `main` when it lands and record per
request: prefilled and restored tokens, TTFT, and pass or fail. That run is
the comparison point for card 19.

### Combined implementation

- Compaction mechanics: `cache-compaction` preserves system/tools while replacing
  a near-limit tool history with a summary; checks retries and continued growth
  against uncached controls. See the [results](#results).
- History edits: `cache-transforms` covers tool-output clearing, micro-reductions
  and a 10K+ tool result followed by retry/edit (patterns 2, 3, 5).
- Retention: `cache-pressure` covers compaction abandonment, idle pressure,
  volatile headers, UI tasks and parent/child fan-out (1 pressure, 4, 6, 7, 9).
- Messages: `cache-messages-loop` replays twelve actual streamed tool cycles,
  growing past 10K tokens, with uncached Chat controls (8).
- Disk: `cache_lifecycle.py` covers graceful restart, an abrupt kill during a
  deterministically held unpublished write, deeper disk state beside short RAM
  state, and a checkpoint exceeding staging (10–12).
- These families land in one PR. Generated reports, logs and disk snapshots stay
  outside Git; per-request results and provenance are recorded in Markdown.

The [results](#results) record all families in AR and MTP, including the strict
legacy-cache work failures.

## Results

All twelve patterns were exercised in Flash-Next AR and MTP, with 440 requests
including uncached controls. Some new tests fail on the current cache:

- `cache-pressure`: useful conversation checkpoints are lost under competing
  traffic; AR also retains the abandoned branch beyond the expected boundary.
- `tiered`: a short RAM hit hides the deeper published disk checkpoint.
- `oversized`: a checkpoint larger than staging is not published or restored.

These are strict cache-work failures, not expected passes. Every completed warm
output matched its uncached control. Compaction mechanics, history edits,
streamed Messages tool loops, graceful restart and the queued-write kill passed
in both modes. The 75 focused CPU test cases passed. The legacy cache is
unchanged; these contracts must pass at card 19's switch-over.

### Coverage and commands

| Family | Pattern | Entry point | Settings |
| --- | --- | --- | --- |
| Compaction | 1 mechanics | `run.py --suite cache-compaction` | context 32768, sessions 1 |
| History transformations | 2 pruning, 3 micro-reduction, 5 large append/retry/edit | `run.py --suite cache-transforms` | context 32768, sessions 1, roomy RAM |
| Retention | 1 abandonment, 4 idle pressure, 6 volatile headers, 7 UI side requests, 9 parent/children | `run.py --suite cache-pressure` | context 32768, sessions 2, RAM 2147483648 |
| Messages | 8 streamed tool loop | `run.py --suite cache-messages-loop` | context 32768; also measured after the pressure history |
| Disk lifecycle | 10 graceful restart/queued-write kill, 11 deeper disk beside short RAM, 12 oversized staging | `cache_lifecycle.py` | context 32768, sessions 1, disk 8 GiB |

The [compaction baseline](#compaction-baseline) records its original
32 request results. The [functional README](../../../tests/functional/README.md)
documents all suite selections and the standalone disk runner. No suite is
added to `all`; disk/restart cases cannot affect an ordinary invocation.

Normal runner command, with a fresh output directory and selected suites:

```sh
nix develop -c python3 tests/functional/run.py \
  --record-baseline --output /tmp/card01 --sampling-preset qwen38 \
  --suite cache-transforms -- \
  /path/to/main/gufo serve llm \
  --model /path/to/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --mtp-model /path/to/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --speculative off --sessions 1 --context 32768 --think off
```

For pressure, select `--suite cache-pressure`, use sessions 2 and add
`--cache-ram-bytes 2147483648`. Messages may run immediately afterwards to
check the growing tool loop on a busy cache. Change `off` to `mtp` for MTP;
keep the same sidecar in AR to verify that override. Disk cases use the
README's standalone command, and `--case` selects individual histories.

### Numerical and work contracts

All warm histories precede uncached controls. Complete assistant text, tool
arguments/types, finish reason and generated-token counts must match controls.
Messages calls are actual model-authored streamed calls; each result replays
the observed call ID. Its twelve cycles grow beyond 10K prompt tokens.
Deterministic tool outputs and synthetic client transformations are fixtures;
the suite does not invoke the Claude Code executable or a real UI/subagent.

Tool-bearing history edits allow the existing 96-token framing/tokenizer
boundary allowance; they do not depend on a checkpoint grid. A micro-reduction
must remove 100–600 measured tokens. The large tool append must exceed 10K
measured tokens. Retry-copy pressure is tested separately from these edits.
The pressure workload fills at least twice the configured RAM budget, then
checks live histories before probing an abandoned branch.

The disk runner drains graceful writes before measuring restart. Tiered lookup
must reach the deepest usable published boundary despite a short RAM hit.
Oversized staging observes a checkpoint larger than its 1 MiB staging area.
The test-only preload gate holds a private-cache file before `fsync` returns;
the crash case verifies queued bytes, waits for the HTTP completion log and
SIGKILLs only that child. Restart can restore only logged published boundaries.
Each case deletes its disk directory in `finally`, including failed cases.

CPU checks cover delayed controls, precise edit shapes, lost frontiers, false
header hits, abandoned retention, actual tool-result pairing, publication
boundaries, failed-case cleanup and the preload gate's scope/release/kill.
Synthetic CPU responses do not qualify model numerics; live controls do that here.

### Provenance and limits

The production binary was built on clean main `d221a01c74052737a04148ef86a47e1db3e270b7`
with `nix build`; SHA-256
`de0b42cbd49d8dacf853513983760c39c50e5931c432926ab7293db0d0230047`.
The PR is based on `e9b0f0b3`, whose only additional change is DS4 IQ2 kernels;
the measured Flash-Next paths, serving/cache code and toolchain are unchanged.

Hardware: AMD Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151, 125 GiB unified memory,
Linux 7.2.9, ROCm 7.2.3. The harness uses pinned Python 3.14.6, OpenAI SDK 2.41.1
and the repository's flake.lock (SHA-256
`faabad820ec09caab7351bf48910cec895a38cf1766e098e3065aea86f405ed6`).
Models are the local Flash-Next UD-Q4_K_XL four shards and shared Q8_0 MTP
sidecar; their documented source identities are in
[model-identities.json](../../models/qwen3.8-flash-next/artifacts/model-identities.json).
No model files were downloaded or rehashed for this test-only change.

This is a step baseline with one observation per request, not a performance
comparison or new-cache qualification. No averages or timing tolerances are
used to hide individual outcomes. Other model families, real client compaction
policies, >64K compaction drops and physical power-loss recovery are outside
these measurements. Messages streaming exposes usage but omits full phase
timings; server logs provide TTFT and draft work, and buffered controls provide
full phase timings. All original reports/logs stay outside Git, including
superseded harness attempts. No generated artifacts are included in the PR.

### Measured results

Each workload records warm requests before its uncached controls (`_cold`).
`Reuse` is the reported cached/restored token count; `Work` checks the retained
boundary. Every completed control below matched text, typed tool arguments,
finish reason and completion-token count exactly. Work failures remain failures.
TTFT is in milliseconds and includes one observation for each request.

| Family / mode | Work result | Requests |
| --- | --- | ---: |
| History transformations · AR | passed | 28 |
| History transformations · MTP | passed | 28 |
| Pressure · AR | failed | 106 |
| Pressure · MTP | failed | 98 |
| Messages tool loop · AR | passed | 48 |
| Messages tool loop · MTP | passed | 48 |
| Disk lifecycle · AR / restart | passed | 6 |
| Disk lifecycle · AR / tiered | failed | 8 |
| Disk lifecycle · AR / oversized | failed | 6 |
| Disk lifecycle · AR / crash | passed | 6 |
| Disk lifecycle · MTP / restart | passed | 6 |
| Disk lifecycle · MTP / tiered | failed | 8 |
| Disk lifecycle · MTP / oversized | failed | 6 |
| Disk lifecycle · MTP / crash | passed | 6 |

#### History transformations · AR

Status: **passed**. 28 requests. Local evidence: `/tmp/gufo-card01-final-transforms-ar`.
Harness SHA-256: `9983991da46c81aa70a2f261c1c562b1b24555ea5446c0cd136c4efa7a4d1816`. Loaded mode: `off`;
drafts proposed/accepted: 0/0 across this server run.

| Request | Prompt | Prefill | Reuse | TTFT ms | Work |
| --- | ---: | ---: | ---: | ---: | --- |
| `transform_prune_seed` | 4517 | 4517 | 0 | 3147.856 | passed |
| `transform_prune_append` | 15768 | 11249 | 4519 | 7315.431 | passed |
| `transform_prune_retry` | 15768 | 0 | 15768 | 2.747 | passed |
| `transform_prune_deep` | 15790 | 20 | 15770 | 144.852 | passed |
| `transform_prune_edited` | 4600 | 81 | 4519 | 225.569 | passed |
| `transform_micro_seed` | 4516 | 4516 | 0 | 3120.838 | passed |
| `transform_micro_append` | 15767 | 11249 | 4518 | 7464.488 | passed |
| `transform_micro_retry` | 15767 | 0 | 15767 | 2.976 | passed |
| `transform_micro_deep` | 16461 | 692 | 15769 | 721.147 | passed |
| `transform_micro_edited` | 16237 | 468 | 15769 | 556.384 | passed |
| `transform_large_seed` | 4516 | 4516 | 0 | 3099.192 | passed |
| `transform_large_append` | 15767 | 11249 | 4518 | 7230.725 | passed |
| `transform_large_retry` | 15767 | 0 | 15767 | 2.962 | passed |
| `transform_large_edited` | 15768 | 18 | 15750 | 128.547 | passed |
| `transform_prune_seed_cold` | 4517 | 4517 | 0 | 2924.151 | passed |
| `transform_prune_append_cold` | 15768 | 15768 | 0 | 9985.156 | passed |
| `transform_prune_retry_cold` | 15768 | 15768 | 0 | 9889.657 | passed |
| `transform_prune_deep_cold` | 15790 | 15790 | 0 | 9839.675 | passed |
| `transform_prune_edited_cold` | 4600 | 4600 | 0 | 3031.740 | passed |
| `transform_micro_seed_cold` | 4516 | 4516 | 0 | 2958.086 | passed |
| `transform_micro_append_cold` | 15767 | 15767 | 0 | 10189.468 | passed |
| `transform_micro_retry_cold` | 15767 | 15767 | 0 | 9966.652 | passed |
| `transform_micro_deep_cold` | 16461 | 16461 | 0 | 10470.766 | passed |
| `transform_micro_edited_cold` | 16237 | 16237 | 0 | 10322.030 | passed |
| `transform_large_seed_cold` | 4516 | 4516 | 0 | 3019.056 | passed |
| `transform_large_append_cold` | 15767 | 15767 | 0 | 10173.651 | passed |
| `transform_large_retry_cold` | 15767 | 15767 | 0 | 10069.532 | passed |
| `transform_large_edited_cold` | 15768 | 15768 | 0 | 10234.020 | passed |

#### History transformations · MTP

Status: **passed**. 28 requests. Local evidence: `/tmp/gufo-card01-transforms-mtp`.
Harness SHA-256: `9983991da46c81aa70a2f261c1c562b1b24555ea5446c0cd136c4efa7a4d1816`. Loaded mode: `mtp`;
drafts proposed/accepted: 84/30 across this server run.

| Request | Prompt | Prefill | Reuse | TTFT ms | Work |
| --- | ---: | ---: | ---: | ---: | --- |
| `transform_prune_seed` | 4517 | 4517 | 0 | 3223.434 | passed |
| `transform_prune_append` | 15768 | 11249 | 4519 | 7445.726 | passed |
| `transform_prune_retry` | 15768 | 0 | 15768 | 2.825 | passed |
| `transform_prune_deep` | 15790 | 20 | 15770 | 151.848 | passed |
| `transform_prune_edited` | 4600 | 81 | 4519 | 237.931 | passed |
| `transform_micro_seed` | 4516 | 4516 | 0 | 3181.666 | passed |
| `transform_micro_append` | 15767 | 11249 | 4518 | 7539.726 | passed |
| `transform_micro_retry` | 15767 | 0 | 15767 | 2.789 | passed |
| `transform_micro_deep` | 16461 | 692 | 15769 | 761.819 | passed |
| `transform_micro_edited` | 16237 | 468 | 15769 | 573.772 | passed |
| `transform_large_seed` | 4516 | 4516 | 0 | 3213.126 | passed |
| `transform_large_append` | 15767 | 11249 | 4518 | 7676.021 | passed |
| `transform_large_retry` | 15767 | 0 | 15767 | 2.461 | passed |
| `transform_large_edited` | 15768 | 18 | 15750 | 130.008 | passed |
| `transform_prune_seed_cold` | 4517 | 4517 | 0 | 3073.027 | passed |
| `transform_prune_append_cold` | 15768 | 15768 | 0 | 10456.988 | passed |
| `transform_prune_retry_cold` | 15768 | 15768 | 0 | 10413.908 | passed |
| `transform_prune_deep_cold` | 15790 | 15790 | 0 | 10399.763 | passed |
| `transform_prune_edited_cold` | 4600 | 4600 | 0 | 3191.550 | passed |
| `transform_micro_seed_cold` | 4516 | 4516 | 0 | 3103.430 | passed |
| `transform_micro_append_cold` | 15767 | 15767 | 0 | 10591.682 | passed |
| `transform_micro_retry_cold` | 15767 | 15767 | 0 | 10561.120 | passed |
| `transform_micro_deep_cold` | 16461 | 16461 | 0 | 10859.638 | passed |
| `transform_micro_edited_cold` | 16237 | 16237 | 0 | 10705.748 | passed |
| `transform_large_seed_cold` | 4516 | 4516 | 0 | 3157.956 | passed |
| `transform_large_append_cold` | 15767 | 15767 | 0 | 10691.322 | passed |
| `transform_large_retry_cold` | 15767 | 15767 | 0 | 10564.702 | passed |
| `transform_large_edited_cold` | 15768 | 15768 | 0 | 10668.312 | passed |

#### Pressure · AR

Status: **failed**. 106 requests. Local evidence: `/tmp/gufo-card01-all-fn-ar`.
Harness SHA-256: `618481ffb00a6562706a3ffe60106ce73dc75d5d6de8680b9a985c288fd78b90`. Loaded mode: `off`;
drafts proposed/accepted: 0/0 across this server run.

Failed work expectations:

- `pressure_compacted: reused=0, boundary=[1, None], prefilled=2286`
- `pressure_abandoned_probe: reused=11230, boundary=[0, 2382], prefilled=7`
- `pressure_after_volatile: reused=0, boundary=[11242, None], prefilled=11279`
- `pressure_child_GAMMA: reused=0, boundary=[11284, None], prefilled=11327`
- `pressure_parent_resume: reused=0, boundary=[11284, None], prefilled=11324`

Pressure filled 4323028308 snapshot bytes in 23 requests
against 2147483648 RAM bytes.

| Request | Prompt | Prefill | Reuse | TTFT ms | Work |
| --- | ---: | ---: | ---: | ---: | --- |
| `pressure_fill_00` | 150 | 150 | 0 | 301.921 | passed |
| `pressure_fill_01` | 150 | 150 | 0 | 320.523 | passed |
| `pressure_fill_02` | 150 | 150 | 0 | 302.806 | passed |
| `pressure_fill_03` | 150 | 150 | 0 | 303.538 | passed |
| `pressure_fill_04` | 150 | 150 | 0 | 305.787 | passed |
| `pressure_fill_05` | 150 | 150 | 0 | 305.147 | passed |
| `pressure_fill_06` | 150 | 150 | 0 | 301.716 | passed |
| `pressure_fill_07` | 150 | 150 | 0 | 302.223 | passed |
| `pressure_fill_08` | 150 | 150 | 0 | 302.581 | passed |
| `pressure_fill_09` | 150 | 150 | 0 | 305.028 | passed |
| `pressure_fill_10` | 151 | 151 | 0 | 305.803 | passed |
| `pressure_fill_11` | 151 | 151 | 0 | 303.429 | passed |
| `pressure_fill_12` | 151 | 151 | 0 | 303.961 | passed |
| `pressure_fill_13` | 151 | 151 | 0 | 303.498 | passed |
| `pressure_fill_14` | 151 | 151 | 0 | 307.505 | passed |
| `pressure_fill_15` | 151 | 151 | 0 | 305.318 | passed |
| `pressure_fill_16` | 151 | 151 | 0 | 304.402 | passed |
| `pressure_fill_17` | 151 | 151 | 0 | 307.011 | passed |
| `pressure_fill_18` | 151 | 151 | 0 | 307.177 | passed |
| `pressure_fill_19` | 151 | 151 | 0 | 306.459 | passed |
| `pressure_fill_20` | 151 | 151 | 0 | 307.839 | passed |
| `pressure_fill_21` | 151 | 151 | 0 | 305.435 | passed |
| `pressure_fill_22` | 151 | 151 | 0 | 306.741 | passed |
| `pressure_abandoned` | 11237 | 11237 | 0 | 7263.420 | passed |
| `pressure_live_ALPHA` | 11237 | 11237 | 0 | 7244.669 | passed |
| `pressure_live_BETA` | 11239 | 11239 | 0 | 7307.090 | passed |
| `pressure_compacted` | 2286 | 2286 | 0 | 1626.010 | failed |
| `pressure_idle_side_0` | 94 | 94 | 0 | 259.413 | passed |
| `pressure_idle_side_1` | 94 | 94 | 0 | 245.692 | passed |
| `pressure_idle_side_2` | 94 | 94 | 0 | 244.290 | passed |
| `pressure_idle_side_3` | 94 | 94 | 0 | 244.472 | passed |
| `pressure_idle_side_4` | 94 | 94 | 0 | 244.366 | passed |
| `pressure_idle_side_5` | 94 | 94 | 0 | 244.526 | passed |
| `pressure_idle_resume_ALPHA` | 11258 | 28 | 11230 | 148.656 | passed |
| `pressure_idle_resume_BETA` | 11261 | 29 | 11232 | 149.612 | passed |
| `pressure_abandoned_probe` | 11237 | 7 | 11230 | 114.607 | failed |
| `pressure_volatile_0` | 490 | 490 | 0 | 528.235 | passed |
| `pressure_volatile_1` | 491 | 491 | 0 | 510.835 | passed |
| `pressure_volatile_2` | 490 | 490 | 0 | 517.242 | passed |
| `pressure_volatile_3` | 491 | 491 | 0 | 515.081 | passed |
| `pressure_volatile_4` | 490 | 490 | 0 | 510.871 | passed |
| `pressure_volatile_5` | 491 | 491 | 0 | 531.301 | passed |
| `pressure_volatile_6` | 490 | 490 | 0 | 517.100 | passed |
| `pressure_volatile_7` | 491 | 491 | 0 | 524.527 | passed |
| `pressure_after_volatile` | 11279 | 11279 | 0 | 7304.455 | failed |
| `pressure_ui_title` | 11337 | 11337 | 0 | 7353.158 | passed |
| `pressure_ui_tags` | 11338 | 11338 | 0 | 7353.872 | passed |
| `pressure_ui_followup` | 11339 | 11339 | 0 | 7397.021 | passed |
| `pressure_after_ui` | 11300 | 28 | 11272 | 166.214 | passed |
| `pressure_child_ALPHA` | 11325 | 23 | 11302 | 168.660 | passed |
| `pressure_child_BETA` | 11327 | 25 | 11302 | 357.353 | passed |
| `pressure_child_GAMMA` | 11327 | 11327 | 0 | 7551.465 | failed |
| `pressure_parent_resume` | 11324 | 11324 | 0 | 7327.346 | failed |
| `pressure_fill_00_cold` | 150 | 150 | 0 | 322.060 | passed |
| `pressure_fill_01_cold` | 150 | 150 | 0 | 319.804 | passed |
| `pressure_fill_02_cold` | 150 | 150 | 0 | 305.645 | passed |
| `pressure_fill_03_cold` | 150 | 150 | 0 | 306.756 | passed |
| `pressure_fill_04_cold` | 150 | 150 | 0 | 306.390 | passed |
| `pressure_fill_05_cold` | 150 | 150 | 0 | 311.119 | passed |
| `pressure_fill_06_cold` | 150 | 150 | 0 | 305.041 | passed |
| `pressure_fill_07_cold` | 150 | 150 | 0 | 306.848 | passed |
| `pressure_fill_08_cold` | 150 | 150 | 0 | 308.764 | passed |
| `pressure_fill_09_cold` | 150 | 150 | 0 | 304.788 | passed |
| `pressure_fill_10_cold` | 151 | 151 | 0 | 308.748 | passed |
| `pressure_fill_11_cold` | 151 | 151 | 0 | 308.245 | passed |
| `pressure_fill_12_cold` | 151 | 151 | 0 | 307.873 | passed |
| `pressure_fill_13_cold` | 151 | 151 | 0 | 308.800 | passed |
| `pressure_fill_14_cold` | 151 | 151 | 0 | 307.310 | passed |
| `pressure_fill_15_cold` | 151 | 151 | 0 | 309.004 | passed |
| `pressure_fill_16_cold` | 151 | 151 | 0 | 308.142 | passed |
| `pressure_fill_17_cold` | 151 | 151 | 0 | 307.551 | passed |
| `pressure_fill_18_cold` | 151 | 151 | 0 | 309.776 | passed |
| `pressure_fill_19_cold` | 151 | 151 | 0 | 306.566 | passed |
| `pressure_fill_20_cold` | 151 | 151 | 0 | 307.742 | passed |
| `pressure_fill_21_cold` | 151 | 151 | 0 | 306.428 | passed |
| `pressure_fill_22_cold` | 151 | 151 | 0 | 306.855 | passed |
| `pressure_abandoned_cold` | 11237 | 11237 | 0 | 7314.035 | passed |
| `pressure_live_ALPHA_cold` | 11237 | 11237 | 0 | 7329.169 | passed |
| `pressure_live_BETA_cold` | 11239 | 11239 | 0 | 7385.940 | passed |
| `pressure_compacted_cold` | 2286 | 2286 | 0 | 1643.265 | passed |
| `pressure_idle_side_0_cold` | 94 | 94 | 0 | 252.339 | passed |
| `pressure_idle_side_1_cold` | 94 | 94 | 0 | 252.789 | passed |
| `pressure_idle_side_2_cold` | 94 | 94 | 0 | 244.049 | passed |
| `pressure_idle_side_3_cold` | 94 | 94 | 0 | 247.464 | passed |
| `pressure_idle_side_4_cold` | 94 | 94 | 0 | 245.398 | passed |
| `pressure_idle_side_5_cold` | 94 | 94 | 0 | 245.482 | passed |
| `pressure_idle_resume_ALPHA_cold` | 11258 | 11258 | 0 | 7425.431 | passed |
| `pressure_idle_resume_BETA_cold` | 11261 | 11261 | 0 | 7437.905 | passed |
| `pressure_abandoned_probe_cold` | 11237 | 11237 | 0 | 7355.507 | passed |
| `pressure_volatile_0_cold` | 490 | 490 | 0 | 528.563 | passed |
| `pressure_volatile_1_cold` | 491 | 491 | 0 | 530.107 | passed |
| `pressure_volatile_2_cold` | 490 | 490 | 0 | 524.986 | passed |
| `pressure_volatile_3_cold` | 491 | 491 | 0 | 517.585 | passed |
| `pressure_volatile_4_cold` | 490 | 490 | 0 | 523.175 | passed |
| `pressure_volatile_5_cold` | 491 | 491 | 0 | 518.067 | passed |
| `pressure_volatile_6_cold` | 490 | 490 | 0 | 523.693 | passed |
| `pressure_volatile_7_cold` | 491 | 491 | 0 | 530.536 | passed |
| `pressure_after_volatile_cold` | 11279 | 11279 | 0 | 7366.911 | passed |
| `pressure_ui_title_cold` | 11337 | 11337 | 0 | 7405.484 | passed |
| `pressure_ui_tags_cold` | 11338 | 11338 | 0 | 7428.075 | passed |
| `pressure_ui_followup_cold` | 11339 | 11339 | 0 | 7408.387 | passed |
| `pressure_after_ui_cold` | 11300 | 11300 | 0 | 7371.723 | passed |
| `pressure_child_ALPHA_cold` | 11325 | 11325 | 0 | 7511.269 | passed |
| `pressure_child_BETA_cold` | 11327 | 11327 | 0 | 7364.718 | passed |
| `pressure_child_GAMMA_cold` | 11327 | 11327 | 0 | 7388.860 | passed |
| `pressure_parent_resume_cold` | 11324 | 11324 | 0 | 7430.556 | passed |

#### Pressure · MTP

Status: **failed**. 98 requests. Local evidence: `/tmp/gufo-card01-pressure-messages-mtp`.
Harness SHA-256: `9983991da46c81aa70a2f261c1c562b1b24555ea5446c0cd136c4efa7a4d1816`. Loaded mode: `mtp`;
drafts proposed/accepted: 884/676 across this server run.

Failed work expectations:

- `pressure_compacted: reused=0, boundary=[1, None], prefilled=2286`
- `pressure_after_volatile: reused=0, boundary=[11242, None], prefilled=11279`
- `pressure_child_GAMMA: reused=0, boundary=[11284, None], prefilled=11327`
- `pressure_parent_resume: reused=0, boundary=[11284, None], prefilled=11324`

Pressure filled 4347658168 snapshot bytes in 19 requests
against 2147483648 RAM bytes.

| Request | Prompt | Prefill | Reuse | TTFT ms | Work |
| --- | ---: | ---: | ---: | ---: | --- |
| `pressure_fill_00` | 150 | 150 | 0 | 462.310 | passed |
| `pressure_fill_01` | 150 | 150 | 0 | 307.175 | passed |
| `pressure_fill_02` | 150 | 150 | 0 | 310.537 | passed |
| `pressure_fill_03` | 150 | 150 | 0 | 310.044 | passed |
| `pressure_fill_04` | 150 | 150 | 0 | 316.823 | passed |
| `pressure_fill_05` | 150 | 150 | 0 | 316.888 | passed |
| `pressure_fill_06` | 150 | 150 | 0 | 313.682 | passed |
| `pressure_fill_07` | 150 | 150 | 0 | 313.882 | passed |
| `pressure_fill_08` | 150 | 150 | 0 | 312.930 | passed |
| `pressure_fill_09` | 150 | 150 | 0 | 314.428 | passed |
| `pressure_fill_10` | 151 | 151 | 0 | 319.082 | passed |
| `pressure_fill_11` | 151 | 151 | 0 | 313.039 | passed |
| `pressure_fill_12` | 151 | 151 | 0 | 313.722 | passed |
| `pressure_fill_13` | 151 | 151 | 0 | 314.888 | passed |
| `pressure_fill_14` | 151 | 151 | 0 | 313.609 | passed |
| `pressure_fill_15` | 151 | 151 | 0 | 317.492 | passed |
| `pressure_fill_16` | 151 | 151 | 0 | 314.735 | passed |
| `pressure_fill_17` | 151 | 151 | 0 | 315.270 | passed |
| `pressure_fill_18` | 151 | 151 | 0 | 318.220 | passed |
| `pressure_abandoned` | 11237 | 11237 | 0 | 7804.876 | passed |
| `pressure_live_ALPHA` | 11237 | 11237 | 0 | 7786.640 | passed |
| `pressure_live_BETA` | 11239 | 11239 | 0 | 7763.389 | passed |
| `pressure_compacted` | 2286 | 2286 | 0 | 1733.824 | failed |
| `pressure_idle_side_0` | 94 | 94 | 0 | 265.566 | passed |
| `pressure_idle_side_1` | 94 | 94 | 0 | 254.377 | passed |
| `pressure_idle_side_2` | 94 | 94 | 0 | 251.089 | passed |
| `pressure_idle_side_3` | 94 | 94 | 0 | 255.071 | passed |
| `pressure_idle_side_4` | 94 | 94 | 0 | 252.494 | passed |
| `pressure_idle_side_5` | 94 | 94 | 0 | 253.508 | passed |
| `pressure_idle_resume_ALPHA` | 11258 | 28 | 11230 | 153.377 | passed |
| `pressure_idle_resume_BETA` | 11261 | 29 | 11232 | 151.623 | passed |
| `pressure_abandoned_probe` | 11237 | 11237 | 0 | 8158.453 | passed |
| `pressure_volatile_0` | 490 | 490 | 0 | 551.460 | passed |
| `pressure_volatile_1` | 491 | 491 | 0 | 549.980 | passed |
| `pressure_volatile_2` | 490 | 490 | 0 | 542.013 | passed |
| `pressure_volatile_3` | 491 | 491 | 0 | 540.417 | passed |
| `pressure_volatile_4` | 490 | 490 | 0 | 549.852 | passed |
| `pressure_volatile_5` | 491 | 491 | 0 | 543.809 | passed |
| `pressure_volatile_6` | 490 | 490 | 0 | 541.502 | passed |
| `pressure_volatile_7` | 491 | 491 | 0 | 546.434 | passed |
| `pressure_after_volatile` | 11279 | 11279 | 0 | 7773.351 | failed |
| `pressure_ui_title` | 11337 | 11337 | 0 | 7859.826 | passed |
| `pressure_ui_tags` | 11338 | 11338 | 0 | 7906.533 | passed |
| `pressure_ui_followup` | 11339 | 11339 | 0 | 7830.793 | passed |
| `pressure_after_ui` | 11300 | 28 | 11272 | 163.573 | passed |
| `pressure_child_ALPHA` | 11325 | 23 | 11302 | 319.350 | passed |
| `pressure_child_BETA` | 11327 | 25 | 11302 | 319.071 | passed |
| `pressure_child_GAMMA` | 11327 | 11327 | 0 | 7824.155 | failed |
| `pressure_parent_resume` | 11324 | 11324 | 0 | 7845.273 | failed |
| `pressure_fill_00_cold` | 150 | 150 | 0 | 324.384 | passed |
| `pressure_fill_01_cold` | 150 | 150 | 0 | 336.604 | passed |
| `pressure_fill_02_cold` | 150 | 150 | 0 | 314.530 | passed |
| `pressure_fill_03_cold` | 150 | 150 | 0 | 314.636 | passed |
| `pressure_fill_04_cold` | 150 | 150 | 0 | 318.021 | passed |
| `pressure_fill_05_cold` | 150 | 150 | 0 | 326.800 | passed |
| `pressure_fill_06_cold` | 150 | 150 | 0 | 322.520 | passed |
| `pressure_fill_07_cold` | 150 | 150 | 0 | 317.686 | passed |
| `pressure_fill_08_cold` | 150 | 150 | 0 | 315.397 | passed |
| `pressure_fill_09_cold` | 150 | 150 | 0 | 318.539 | passed |
| `pressure_fill_10_cold` | 151 | 151 | 0 | 324.469 | passed |
| `pressure_fill_11_cold` | 151 | 151 | 0 | 323.481 | passed |
| `pressure_fill_12_cold` | 151 | 151 | 0 | 319.153 | passed |
| `pressure_fill_13_cold` | 151 | 151 | 0 | 316.331 | passed |
| `pressure_fill_14_cold` | 151 | 151 | 0 | 323.241 | passed |
| `pressure_fill_15_cold` | 151 | 151 | 0 | 320.642 | passed |
| `pressure_fill_16_cold` | 151 | 151 | 0 | 324.917 | passed |
| `pressure_fill_17_cold` | 151 | 151 | 0 | 319.767 | passed |
| `pressure_fill_18_cold` | 151 | 151 | 0 | 318.864 | passed |
| `pressure_abandoned_cold` | 11237 | 11237 | 0 | 7860.549 | passed |
| `pressure_live_ALPHA_cold` | 11237 | 11237 | 0 | 7884.994 | passed |
| `pressure_live_BETA_cold` | 11239 | 11239 | 0 | 7877.029 | passed |
| `pressure_compacted_cold` | 2286 | 2286 | 0 | 1784.957 | passed |
| `pressure_idle_side_0_cold` | 94 | 94 | 0 | 267.964 | passed |
| `pressure_idle_side_1_cold` | 94 | 94 | 0 | 257.480 | passed |
| `pressure_idle_side_2_cold` | 94 | 94 | 0 | 254.793 | passed |
| `pressure_idle_side_3_cold` | 94 | 94 | 0 | 252.971 | passed |
| `pressure_idle_side_4_cold` | 94 | 94 | 0 | 255.849 | passed |
| `pressure_idle_side_5_cold` | 94 | 94 | 0 | 254.224 | passed |
| `pressure_idle_resume_ALPHA_cold` | 11258 | 11258 | 0 | 7894.341 | passed |
| `pressure_idle_resume_BETA_cold` | 11261 | 11261 | 0 | 7906.227 | passed |
| `pressure_abandoned_probe_cold` | 11237 | 11237 | 0 | 7882.841 | passed |
| `pressure_volatile_0_cold` | 490 | 490 | 0 | 560.420 | passed |
| `pressure_volatile_1_cold` | 491 | 491 | 0 | 550.801 | passed |
| `pressure_volatile_2_cold` | 490 | 490 | 0 | 555.545 | passed |
| `pressure_volatile_3_cold` | 491 | 491 | 0 | 544.534 | passed |
| `pressure_volatile_4_cold` | 490 | 490 | 0 | 560.341 | passed |
| `pressure_volatile_5_cold` | 491 | 491 | 0 | 571.187 | passed |
| `pressure_volatile_6_cold` | 490 | 490 | 0 | 544.647 | passed |
| `pressure_volatile_7_cold` | 491 | 491 | 0 | 564.140 | passed |
| `pressure_after_volatile_cold` | 11279 | 11279 | 0 | 7868.493 | passed |
| `pressure_ui_title_cold` | 11337 | 11337 | 0 | 7938.530 | passed |
| `pressure_ui_tags_cold` | 11338 | 11338 | 0 | 8074.784 | passed |
| `pressure_ui_followup_cold` | 11339 | 11339 | 0 | 7980.453 | passed |
| `pressure_after_ui_cold` | 11300 | 11300 | 0 | 7943.460 | passed |
| `pressure_child_ALPHA_cold` | 11325 | 11325 | 0 | 7959.814 | passed |
| `pressure_child_BETA_cold` | 11327 | 11327 | 0 | 7923.452 | passed |
| `pressure_child_GAMMA_cold` | 11327 | 11327 | 0 | 8108.353 | passed |
| `pressure_parent_resume_cold` | 11324 | 11324 | 0 | 7869.124 | passed |

#### Messages tool loop · AR

Status: **passed**. 48 requests. Local evidence: `/tmp/gufo-card01-all-fn-ar`.
Harness SHA-256: `618481ffb00a6562706a3ffe60106ce73dc75d5d6de8680b9a985c288fd78b90`. Loaded mode: `off`;
drafts proposed/accepted: 0/0 across this server run.

Twelve tool cycles ended at 13786 prompt tokens.

| Request | Prompt | Prefill | Reuse | TTFT ms | Work |
| --- | ---: | ---: | ---: | ---: | --- |
| `messages_loop_call_00` | 2113 | 2113 | 0 | 1458.800 | passed |
| `messages_loop_answer_00` | 3066 | 925 | 2141 | 820.300 | passed |
| `messages_loop_call_01` | 3087 | 19 | 3068 | 137.600 | passed |
| `messages_loop_answer_01` | 4040 | 925 | 3115 | 833.100 | passed |
| `messages_loop_call_02` | 4061 | 19 | 4042 | 134.400 | passed |
| `messages_loop_answer_02` | 5014 | 925 | 4089 | 829.500 | passed |
| `messages_loop_call_03` | 5035 | 19 | 5016 | 136.500 | passed |
| `messages_loop_answer_03` | 5988 | 925 | 5063 | 833.700 | passed |
| `messages_loop_call_04` | 6009 | 19 | 5990 | 136.600 | passed |
| `messages_loop_answer_04` | 6962 | 925 | 6037 | 834.400 | passed |
| `messages_loop_call_05` | 6983 | 19 | 6964 | 136.700 | passed |
| `messages_loop_answer_05` | 7936 | 925 | 7011 | 842.400 | passed |
| `messages_loop_call_06` | 7957 | 19 | 7938 | 139.400 | passed |
| `messages_loop_answer_06` | 8910 | 925 | 7985 | 839.300 | passed |
| `messages_loop_call_07` | 8931 | 19 | 8912 | 140.100 | passed |
| `messages_loop_answer_07` | 9884 | 925 | 8959 | 845.000 | passed |
| `messages_loop_call_08` | 9905 | 19 | 9886 | 140.000 | passed |
| `messages_loop_answer_08` | 10858 | 925 | 9933 | 845.500 | passed |
| `messages_loop_call_09` | 10879 | 19 | 10860 | 140.200 | passed |
| `messages_loop_answer_09` | 11832 | 925 | 10907 | 844.300 | passed |
| `messages_loop_call_10` | 11854 | 20 | 11834 | 139.700 | passed |
| `messages_loop_answer_10` | 12809 | 926 | 11883 | 846.300 | passed |
| `messages_loop_call_11` | 12831 | 20 | 12811 | 143.700 | passed |
| `messages_loop_answer_11` | 13786 | 926 | 12860 | 848.300 | passed |
| `messages_loop_call_00_cold` | 2113 | 2113 | 0 | 1406.321 | passed |
| `messages_loop_answer_00_cold` | 3066 | 3066 | 0 | 2269.108 | passed |
| `messages_loop_call_01_cold` | 3087 | 3087 | 0 | 2214.817 | passed |
| `messages_loop_answer_01_cold` | 4040 | 4040 | 0 | 2735.354 | passed |
| `messages_loop_call_02_cold` | 4061 | 4061 | 0 | 2698.576 | passed |
| `messages_loop_answer_02_cold` | 5014 | 5014 | 0 | 3566.615 | passed |
| `messages_loop_call_03_cold` | 5035 | 5035 | 0 | 3595.964 | passed |
| `messages_loop_answer_03_cold` | 5988 | 5988 | 0 | 4060.393 | passed |
| `messages_loop_call_04_cold` | 6009 | 6009 | 0 | 4063.974 | passed |
| `messages_loop_answer_04_cold` | 6962 | 6962 | 0 | 4930.565 | passed |
| `messages_loop_call_05_cold` | 6983 | 6983 | 0 | 4892.062 | passed |
| `messages_loop_answer_05_cold` | 7936 | 7936 | 0 | 5424.235 | passed |
| `messages_loop_call_06_cold` | 7957 | 7957 | 0 | 5422.565 | passed |
| `messages_loop_answer_06_cold` | 8910 | 8910 | 0 | 6231.445 | passed |
| `messages_loop_call_07_cold` | 8931 | 8931 | 0 | 6435.526 | passed |
| `messages_loop_answer_07_cold` | 9884 | 9884 | 0 | 6765.937 | passed |
| `messages_loop_call_08_cold` | 9905 | 9905 | 0 | 6785.505 | passed |
| `messages_loop_answer_08_cold` | 10858 | 10858 | 0 | 7411.465 | passed |
| `messages_loop_call_09_cold` | 10879 | 10879 | 0 | 7416.700 | passed |
| `messages_loop_answer_09_cold` | 11832 | 11832 | 0 | 7895.022 | passed |
| `messages_loop_call_10_cold` | 11854 | 11854 | 0 | 7922.329 | passed |
| `messages_loop_answer_10_cold` | 12809 | 12809 | 0 | 8666.295 | passed |
| `messages_loop_call_11_cold` | 12831 | 12831 | 0 | 8774.090 | passed |
| `messages_loop_answer_11_cold` | 13786 | 13786 | 0 | 9083.927 | passed |

#### Messages tool loop · MTP

Status: **passed**. 48 requests. Local evidence: `/tmp/gufo-card01-pressure-messages-mtp`.
Harness SHA-256: `9983991da46c81aa70a2f261c1c562b1b24555ea5446c0cd136c4efa7a4d1816`. Loaded mode: `mtp`;
drafts proposed/accepted: 884/676 across this server run.

Twelve tool cycles ended at 13786 prompt tokens.

| Request | Prompt | Prefill | Reuse | TTFT ms | Work |
| --- | ---: | ---: | ---: | ---: | --- |
| `messages_loop_call_00` | 2113 | 2113 | 0 | 1554.500 | passed |
| `messages_loop_answer_00` | 3066 | 925 | 2141 | 874.900 | passed |
| `messages_loop_call_01` | 3087 | 19 | 3068 | 140.500 | passed |
| `messages_loop_answer_01` | 4040 | 925 | 3115 | 876.600 | passed |
| `messages_loop_call_02` | 4061 | 19 | 4042 | 136.900 | passed |
| `messages_loop_answer_02` | 5014 | 925 | 4089 | 912.800 | passed |
| `messages_loop_call_03` | 5035 | 19 | 5016 | 139.100 | passed |
| `messages_loop_answer_03` | 5988 | 925 | 5063 | 908.100 | passed |
| `messages_loop_call_04` | 6009 | 19 | 5990 | 139.200 | passed |
| `messages_loop_answer_04` | 6962 | 925 | 6037 | 900.700 | passed |
| `messages_loop_call_05` | 6983 | 19 | 6964 | 140.700 | passed |
| `messages_loop_answer_05` | 7936 | 925 | 7011 | 905.200 | passed |
| `messages_loop_call_06` | 7957 | 19 | 7938 | 141.000 | passed |
| `messages_loop_answer_06` | 8910 | 925 | 7985 | 930.500 | passed |
| `messages_loop_call_07` | 8931 | 19 | 8912 | 142.700 | passed |
| `messages_loop_answer_07` | 9884 | 925 | 8959 | 890.000 | passed |
| `messages_loop_call_08` | 9905 | 19 | 9886 | 144.500 | passed |
| `messages_loop_answer_08` | 10858 | 925 | 9933 | 905.200 | passed |
| `messages_loop_call_09` | 10879 | 19 | 10860 | 143.500 | passed |
| `messages_loop_answer_09` | 11832 | 925 | 10907 | 914.800 | passed |
| `messages_loop_call_10` | 11854 | 20 | 11834 | 149.500 | passed |
| `messages_loop_answer_10` | 12809 | 926 | 11883 | 924.900 | passed |
| `messages_loop_call_11` | 12831 | 20 | 12811 | 147.000 | passed |
| `messages_loop_answer_11` | 13786 | 926 | 12860 | 907.800 | passed |
| `messages_loop_call_00_cold` | 2113 | 2113 | 0 | 1499.803 | passed |
| `messages_loop_answer_00_cold` | 3066 | 3066 | 0 | 2382.473 | passed |
| `messages_loop_call_01_cold` | 3087 | 3087 | 0 | 2335.538 | passed |
| `messages_loop_answer_01_cold` | 4040 | 4040 | 0 | 2735.020 | passed |
| `messages_loop_call_02_cold` | 4061 | 4061 | 0 | 2898.526 | passed |
| `messages_loop_answer_02_cold` | 5014 | 5014 | 0 | 3844.541 | passed |
| `messages_loop_call_03_cold` | 5035 | 5035 | 0 | 3849.487 | passed |
| `messages_loop_answer_03_cold` | 5988 | 5988 | 0 | 4404.066 | passed |
| `messages_loop_call_04_cold` | 6009 | 6009 | 0 | 4372.765 | passed |
| `messages_loop_answer_04_cold` | 6962 | 6962 | 0 | 5226.484 | passed |
| `messages_loop_call_05_cold` | 6983 | 6983 | 0 | 5203.429 | passed |
| `messages_loop_answer_05_cold` | 7936 | 7936 | 0 | 5733.860 | passed |
| `messages_loop_call_06_cold` | 7957 | 7957 | 0 | 5762.561 | passed |
| `messages_loop_answer_06_cold` | 8910 | 8910 | 0 | 6735.375 | passed |
| `messages_loop_call_07_cold` | 8931 | 8931 | 0 | 6700.124 | passed |
| `messages_loop_answer_07_cold` | 9884 | 9884 | 0 | 7230.666 | passed |
| `messages_loop_call_08_cold` | 9905 | 9905 | 0 | 7180.025 | passed |
| `messages_loop_answer_08_cold` | 10858 | 10858 | 0 | 8046.872 | passed |
| `messages_loop_call_09_cold` | 10879 | 10879 | 0 | 7862.031 | passed |
| `messages_loop_answer_09_cold` | 11832 | 11832 | 0 | 8547.175 | passed |
| `messages_loop_call_10_cold` | 11854 | 11854 | 0 | 8483.910 | passed |
| `messages_loop_answer_10_cold` | 12809 | 12809 | 0 | 9210.435 | passed |
| `messages_loop_call_11_cold` | 12831 | 12831 | 0 | 9126.452 | passed |
| `messages_loop_answer_11_cold` | 13786 | 13786 | 0 | 9779.415 | passed |

#### Disk lifecycle · AR

Local evidence: `/tmp/gufo-card01-lifecycle-fn-ar`. Harness SHA-256:
`86c19aca2bacbfea25204e5072314aaf01e499d514a9250575ad67d3d2cab9db`. Fault library SHA-256:
`f0f985ad02de5839f732f5edebbc5c9a38f2ca580b0562eae017c83bf6e11fa4`.

##### restart · passed

Published boundaries: `[2270, 13491]`.

initial: loaded `off`, drafts proposed/accepted 0/0; startup 15997.305 ms.

| Request | Prompt | Prefill | Reuse | TTFT ms | Work |
| --- | ---: | ---: | ---: | ---: | --- |
| `seed` | 2277 | 2277 | 0 | 1800.081 | passed |
| `append` | 13498 | 11219 | 2279 | 7060.295 | passed |

restarted: loaded `off`, drafts proposed/accepted 0/0; startup 16284.280 ms.

| Request | Prompt | Prefill | Reuse | TTFT ms | Work |
| --- | ---: | ---: | ---: | ---: | --- |
| `restore` | 13498 | 7 | 13491 | 621.788 | passed |
| `seed_cold` | 2277 | 2277 | 0 | 1651.071 | passed |
| `append_cold` | 13498 | 13498 | 0 | 8548.847 | passed |
| `restore_cold` | 13498 | 13498 | 0 | 8532.493 | passed |

##### tiered · failed

Published boundaries: `[2272, 13493]`.

Failed work expectations:

- `restore: cached=2281, boundary=[13493, None], disk=False, prefilled=11219`

initial: loaded `off`, drafts proposed/accepted 0/0; startup 16100.888 ms.

| Request | Prompt | Prefill | Reuse | TTFT ms | Work |
| --- | ---: | ---: | ---: | ---: | --- |
| `seed` | 2279 | 2279 | 0 | 1820.524 | passed |
| `append` | 13500 | 11219 | 2281 | 7251.435 | passed |

restarted: loaded `off`, drafts proposed/accepted 0/0; startup 16373.608 ms.

| Request | Prompt | Prefill | Reuse | TTFT ms | Work |
| --- | ---: | ---: | ---: | ---: | --- |
| `short_ram` | 2279 | 7 | 2272 | 395.772 | passed |
| `restore` | 13500 | 11219 | 2281 | 7240.684 | failed |
| `seed_cold` | 2279 | 2279 | 0 | 1764.223 | passed |
| `append_cold` | 13500 | 13500 | 0 | 8606.597 | passed |
| `short_ram_cold` | 2279 | 2279 | 0 | 1568.740 | passed |
| `restore_cold` | 13500 | 13500 | 0 | 8604.782 | passed |

##### oversized · failed

Observed checkpoint: 461284596 bytes; staging: 1048576 bytes.

Failed work expectations:

- `oversized_publication: checkpoint larger than staging was not published`
- `restore: cached=0, boundary=[13484, None], disk=False, prefilled=13500`

initial: loaded `off`, drafts proposed/accepted 0/0; startup 15770.604 ms.

| Request | Prompt | Prefill | Reuse | TTFT ms | Work |
| --- | ---: | ---: | ---: | ---: | --- |
| `seed` | 2279 | 2279 | 0 | 1842.967 | passed |
| `append` | 13500 | 11219 | 2281 | 7603.099 | passed |

restarted: loaded `off`, drafts proposed/accepted 0/0; startup 15770.534 ms.

| Request | Prompt | Prefill | Reuse | TTFT ms | Work |
| --- | ---: | ---: | ---: | ---: | --- |
| `restore` | 13500 | 13500 | 0 | 8836.083 | failed |
| `seed_cold` | 2279 | 2279 | 0 | 1566.519 | passed |
| `append_cold` | 13500 | 13500 | 0 | 8648.418 | passed |
| `restore_cold` | 13500 | 13500 | 0 | 8707.755 | passed |

##### crash · passed

Write held at fsync; queued 1099192308 bytes; child killed with signal 9.
Published boundaries at kill: `[2271]`.

initial: loaded `off`, drafts proposed/accepted 0/0; startup 16180.902 ms.

| Request | Prompt | Prefill | Reuse | TTFT ms | Work |
| --- | ---: | ---: | ---: | ---: | --- |
| `seed` | 2278 | 2278 | 0 | 1845.061 | passed |
| `queued_append` | 13499 | 11219 | 2280 | 7298.925 | passed |

restarted: loaded `off`, drafts proposed/accepted 0/0; startup 16181.880 ms.

| Request | Prompt | Prefill | Reuse | TTFT ms | Work |
| --- | ---: | ---: | ---: | ---: | --- |
| `restore_published` | 13499 | 11228 | 2271 | 7642.795 | passed |
| `seed_cold` | 2278 | 2278 | 0 | 1757.275 | passed |
| `queued_append_cold` | 13499 | 13499 | 0 | 8750.769 | passed |
| `restore_published_cold` | 13499 | 13499 | 0 | 8710.075 | passed |

#### Disk lifecycle · MTP

Local evidence: `/tmp/gufo-card01-lifecycle-fn-mtp`. Harness SHA-256:
`86c19aca2bacbfea25204e5072314aaf01e499d514a9250575ad67d3d2cab9db`. Fault library SHA-256:
`f0f985ad02de5839f732f5edebbc5c9a38f2ca580b0562eae017c83bf6e11fa4`.

##### restart · passed

Published boundaries: `[2270, 13491]`.

initial: loaded `mtp`, drafts proposed/accepted 6/2; startup 16096.939 ms.

| Request | Prompt | Prefill | Reuse | TTFT ms | Work |
| --- | ---: | ---: | ---: | ---: | --- |
| `seed` | 2277 | 2277 | 0 | 1959.256 | passed |
| `append` | 13498 | 11219 | 2279 | 7870.804 | passed |

restarted: loaded `mtp`, drafts proposed/accepted 12/4; startup 17085.516 ms.

| Request | Prompt | Prefill | Reuse | TTFT ms | Work |
| --- | ---: | ---: | ---: | ---: | --- |
| `restore` | 13498 | 7 | 13491 | 680.227 | passed |
| `seed_cold` | 2277 | 2277 | 0 | 1777.582 | passed |
| `append_cold` | 13498 | 13498 | 0 | 9383.119 | passed |
| `restore_cold` | 13498 | 13498 | 0 | 9412.464 | passed |

##### tiered · failed

Published boundaries: `[2272, 13493]`.

Failed work expectations:

- `restore: cached=2281, boundary=[13493, None], disk=False, prefilled=11219`

initial: loaded `mtp`, drafts proposed/accepted 6/2; startup 16472.054 ms.

| Request | Prompt | Prefill | Reuse | TTFT ms | Work |
| --- | ---: | ---: | ---: | ---: | --- |
| `seed` | 2279 | 2279 | 0 | 2032.247 | passed |
| `append` | 13500 | 11219 | 2281 | 7796.014 | passed |

restarted: loaded `mtp`, drafts proposed/accepted 18/6; startup 17080.407 ms.

| Request | Prompt | Prefill | Reuse | TTFT ms | Work |
| --- | ---: | ---: | ---: | ---: | --- |
| `short_ram` | 2279 | 7 | 2272 | 423.972 | passed |
| `restore` | 13500 | 11219 | 2281 | 7896.446 | failed |
| `seed_cold` | 2279 | 2279 | 0 | 1894.724 | passed |
| `append_cold` | 13500 | 13500 | 0 | 9244.046 | passed |
| `short_ram_cold` | 2279 | 2279 | 0 | 1683.953 | passed |
| `restore_cold` | 13500 | 13500 | 0 | 9312.787 | passed |

##### oversized · failed

Observed checkpoint: 490082715 bytes; staging: 1048576 bytes.

Failed work expectations:

- `oversized_publication: checkpoint larger than staging was not published`
- `restore: cached=0, boundary=[13484, None], disk=False, prefilled=13500`

initial: loaded `mtp`, drafts proposed/accepted 6/2; startup 16682.322 ms.

| Request | Prompt | Prefill | Reuse | TTFT ms | Work |
| --- | ---: | ---: | ---: | ---: | --- |
| `seed` | 2279 | 2279 | 0 | 1944.186 | passed |
| `append` | 13500 | 11219 | 2281 | 7842.131 | passed |

restarted: loaded `mtp`, drafts proposed/accepted 12/4; startup 16610.684 ms.

| Request | Prompt | Prefill | Reuse | TTFT ms | Work |
| --- | ---: | ---: | ---: | ---: | --- |
| `restore` | 13500 | 13500 | 0 | 9411.835 | failed |
| `seed_cold` | 2279 | 2279 | 0 | 1641.504 | passed |
| `append_cold` | 13500 | 13500 | 0 | 9281.857 | passed |
| `restore_cold` | 13500 | 13500 | 0 | 9286.057 | passed |

##### crash · passed

Write held at fsync; queued 1161704169 bytes; child killed with signal 9.
Published boundaries at kill: `[2271]`.

initial: loaded `mtp`, drafts proposed/accepted 6/2; startup 16575.800 ms.

| Request | Prompt | Prefill | Reuse | TTFT ms | Work |
| --- | ---: | ---: | ---: | ---: | --- |
| `seed` | 2278 | 2278 | 0 | 1919.785 | passed |
| `queued_append` | 13499 | 11219 | 2280 | 7940.685 | passed |

restarted: loaded `mtp`, drafts proposed/accepted 12/4; startup 16779.300 ms.

| Request | Prompt | Prefill | Reuse | TTFT ms | Work |
| --- | ---: | ---: | ---: | ---: | --- |
| `restore_published` | 13499 | 11228 | 2271 | 8019.438 | passed |
| `seed_cold` | 2278 | 2278 | 0 | 1849.966 | passed |
| `queued_append_cold` | 13499 | 13499 | 0 | 9276.996 | passed |
| `restore_published_cold` | 13499 | 13499 | 0 | 9211.297 | passed |

#### Run history and verification

The first AR combined run selected transformations, pressure, then Messages
on one server with a 2 GiB RAM budget. Its transformation results were
superseded: retry-copy pressure exposed a 17-token template opening, and the
initial 16-token edit allowance was too narrow. The final edit suite uses the
existing 96-token tool framing allowance and a separate roomy-budget server.
The pressure and Messages functions were unchanged; their original AR results
above are retained. MTP pressure then Messages ran on a fresh 2 GiB server.
These different histories and harness identities are explicit; the tables
are individual baselines and do not assert an AR/MTP timing comparison.

All raw attempts are retained outside Git. Graceful/crash disk directories
were deleted after each case; every spawned server was reaped.

Focused CTest passed all three targets (75 Python test cases):
`functional_runner_test`, `cache_workloads_test`, `cache_disk_faults_test`.
The shared C++ format check, dependency check, documentation link check and
`git diff --check` passed. No cache implementation changes or generated
artifacts are part of this PR.

### Compaction baseline

This records the compaction scenario family in card 01's combined test-only PR. It adds
`cache-compaction` to the functional runner, selected explicitly and excluded
from `all`. The cache implementation is unchanged.

#### Contract

The fixture preserves one system prompt and a declared `read_archive` tool
while replacing the rest of a near-limit conversation with a summary. A small
tool result calibrates the append length using the loaded model's reported
tokens. The large prompt must reach 75–95% of the configured context; the
summary must drop at least half that capacity.

A short warm branch before the large append measures a reusable coherent
system/tools checkpoint. Compaction must reuse at least that boundary, and
must not reuse beyond the seed prompt into abandoned history. The summary
changes the answer from ALPHA to BETA. Both unchanged retries must avoid
prefill, except for the existing documented assistant-opening allowance when
the server log proves a retry-copy refusal under pressure. The next turn must
advance reuse to the compacted conversation.

All eight warm requests precede the eight uncached controls. Every warm answer,
finish reason and generated-token count must exactly match its control.
Reported cache/prefill accounting and per-request metrics are mandatory. The fast harness
tests reject misses, excessive reuse, frozen continuation boundaries, stale
answers and unexpected retry work.

#### Reproduce

Build clean current main with `nix build`. Run once per mode, using the same
production binary and separate output directories:

```sh
nix develop -c python3 tests/functional/run.py \
  --record-baseline --output /tmp/compaction-ar --sampling-preset qwen38 \
  --suite cache-compaction -- \
  /path/to/main/result/bin/gufo serve llm \
  --model /path/to/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --mtp-model /path/to/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --speculative off --sessions 1 --context 32768 --think off
```

Use `--speculative mtp` and a fresh output directory for MTP. Keep the sidecar
in both commands so the AR run verifies the explicit override. No disk cache
is enabled by this suite.

#### Scope

This baseline uses deterministic client-authored tool calls/results and a
summary, with actual seed/continued assistant replies. It does not exercise a
real client's compaction policy, summary generation, streaming progress, a >64K drop, abandonment
priority under pressure, restart/disk behavior, or other model families.
Other scenario families are recorded in the [measured results](#measured-results). These measurements
are a current-main step baseline, not new-cache qualification or a timing
comparison; there is one sample per request and no speedup claim.

Assistant outputs, usage, request/output hashes, reports and server logs
stay in the listed local artifact directories. Raw HTTP bodies are not retained.
Per-request work and TTFT are recorded below; full raw measurements stay outside Git.

#### Recorded results · 2026-10-09

Both runs passed on clean main `d221a01c`, built with `nix build`, on AMD
Strix Halo gfx1151 (125 GiB unified memory), Linux 7.2.9 and ROCm 7.2.3.
The harness used pinned Python 3.14.6 and OpenAI SDK 2.41.1. The target was
Flash-Next UD-Q4_K_XL (four shards), with the shared Q8_0 MTP sidecar.
The binary SHA-256 is `de0b42cbd49d8dacf853513983760c39c50e5931c432926ab7293db0d0230047`.
The harness SHA-256 is `4820f6294d0fcae05a406e78c5fae1473dd886cf136a639260d5da42d3798cfc`;
the flake.lock SHA-256 is `faabad820ec09caab7351bf48910cec895a38cf1766e098e3065aea86f405ed6`.
Model sources and documented identities are in
[model-identities.json](../../models/qwen3.8-flash-next/artifacts/model-identities.json).
No generated test output is committed.

The prompt dropped from **27,542 to 2,302 tokens** (25,240 removed). Both
modes restored the measured 2,048-token shared checkpoint and prefilled only
254 summary tokens. Both unchanged retries prefilled zero tokens. Continuation
reused 2,304 of 2,327 tokens. All eight warm outputs matched their uncached
controls exactly. AR executed zero drafts; MTP proposed 48 and accepted 18.

All request results below passed. TTFT is the server measurement in ms;
reuse is the API’s `cached_tokens`, including live-state hits. Prefill, restore,
decode, queue and wall timings are retained separately in JSON.

| Mode | Request | Prompt | Reused | Prefilled | TTFT (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| AR | `seed` | 2277 | 0 | 2277 | 1685.971 |
| AR | `shared_probe` | 2277 | 2048 | 229 | 327.133 |
| AR | `sample` | 5912 | 2277 | 3635 | 2289.571 |
| AR | `near_limit` | 27542 | 2277 | 25265 | 16004.001 |
| AR | `large_retry` | 27542 | 27542 | 0 | 2.863 |
| AR | `summary` | 2302 | 2048 | 254 | 400.804 |
| AR | `summary_retry` | 2302 | 2302 | 0 | 2.178 |
| AR | `continue` | 2327 | 2304 | 23 | 158.519 |
| AR | `seed_cold` | 2277 | 0 | 2277 | 1598.554 |
| AR | `shared_probe_cold` | 2277 | 0 | 2277 | 1438.540 |
| AR | `sample_cold` | 5912 | 0 | 5912 | 3714.870 |
| AR | `near_limit_cold` | 27542 | 0 | 27542 | 17038.756 |
| AR | `large_retry_cold` | 27542 | 0 | 27542 | 17167.037 |
| AR | `summary_cold` | 2302 | 0 | 2302 | 1487.616 |
| AR | `summary_retry_cold` | 2302 | 0 | 2302 | 1467.575 |
| AR | `continue_cold` | 2327 | 0 | 2327 | 1620.112 |
| MTP | `seed` | 2277 | 0 | 2277 | 1751.093 |
| MTP | `shared_probe` | 2277 | 2048 | 229 | 335.480 |
| MTP | `sample` | 5912 | 2277 | 3635 | 2398.382 |
| MTP | `near_limit` | 27542 | 2277 | 25265 | 16784.072 |
| MTP | `large_retry` | 27542 | 27542 | 0 | 3.415 |
| MTP | `summary` | 2302 | 2048 | 254 | 402.933 |
| MTP | `summary_retry` | 2302 | 2302 | 0 | 2.037 |
| MTP | `continue` | 2327 | 2304 | 23 | 156.848 |
| MTP | `seed_cold` | 2277 | 0 | 2277 | 1505.902 |
| MTP | `shared_probe_cold` | 2277 | 0 | 2277 | 1489.474 |
| MTP | `sample_cold` | 5912 | 0 | 5912 | 3913.012 |
| MTP | `near_limit_cold` | 27542 | 0 | 27542 | 18216.332 |
| MTP | `large_retry_cold` | 27542 | 0 | 27542 | 18053.213 |
| MTP | `summary_cold` | 2302 | 0 | 2302 | 1532.032 |
| MTP | `summary_retry_cold` | 2302 | 0 | 2302 | 1527.583 |
| MTP | `continue_cold` | 2327 | 0 | 2327 | 1555.754 |

Raw artifacts: `/tmp/gufo-card01-compaction-fn-ar-03` and
`/tmp/gufo-card01-compaction-fn-mtp`. Earlier attempts remain retained: the
first lacked the SDK; the second exposed an overly strict harness expectation
that required almost the full system prefix despite a coherent earlier
checkpoint. Their reports and distinct harness hashes remain outside Git as
superseded evidence, not cache failures under the final contract.

Initial compaction-only validation: all 63 functional-runner unit tests passed with the pinned
Python environment; `python3 tools/ci/check-docs.py` and `git diff --check`
passed. That initial step left production paths and C++ sources unchanged.

## Done when

- [x] Each added scenario runs on current `main`, with its result recorded.
- [x] New suites are documented in `tests/functional/README.md`.

## Review focus

- Which patterns matter most for our users? Prioritize follow-up depth by that.
- Patterns missing from the list.

## References

Material to consider during implementation. Before reusing code from a
project, check its license and add it to `THIRD_PARTY_NOTICES.md` when
required (see the README).

- [TraceLab: Characterizing Coding Agent Workloads for LLM Serving](https://arxiv.org/abs/2606.30560)
  (Claude Code and Codex traces: idle gaps, append sizes, compaction).
- [Claude Code, llama.cpp, and the hidden prompt cache killer](https://mykolaaleksandrov.dev/posts/2026/06/claude-code-llamacpp-prompt-cache-fix/)
  (attribution header changing the start of the prompt).
- [llama.cpp #24714](https://github.com/ggml-org/llama.cpp/issues/24714) and
  [#21831](https://github.com/ggml-org/llama.cpp/issues/21831) (full
  re-processing on hybrid models).
- [Open WebUI #17123](https://github.com/open-webui/open-webui/issues/17123)
  (title and follow-up generation after every reply).
- [Marconi: Prefix Caching for the Era of Hybrid LLMs](https://arxiv.org/abs/2411.19379)
  (branch-point checkpoints and admission for recurrent state).
- [TraceLab dataset and analysis code](https://github.com/uw-syfi/TraceLab): real Claude Code and Codex sessions to derive scenarios from.
- [Agentic AI Workload Characteristics](https://arxiv.org/abs/2605.26297): turn structure and tool-use phases of agent workloads.

## RFC

[TDD and behavioral contracts first](../RFC.md#tdd-and-behavioral-contracts-first) ·
[Client history transformations](../RFC.md#client-history-transformations)

## Review notes
