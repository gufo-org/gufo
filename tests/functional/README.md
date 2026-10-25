# Model API functional tests

The runners start isolated local servers and check real HTTP responses. Manual
suites use production binaries and local weights. Devlab CI downloads only its
selected models from pinned revisions.

## Embedding, ASR and TTS service checks

`model_services.py` starts an isolated loopback production server, checks readiness
and advertised model identity, runs real HTTP inference, and shuts down its owned
process. These suites are explicit and separate from the text runner's `--suite all`.
They do not download weights or voice recordings. Missing inputs fail the run.
Use `nix develop`, or install `tests/functional/requirements.txt` in a virtualenv.
Build the production binary with `cmake --preset release` and
`cmake --build --preset release --parallel 4` inside the pinned development shell.

```sh
nix develop -c python3 tests/functional/model_services.py embedding \
  --binary build/release/gufo --model models/embeddinggemma-2 \
  --output artifacts/model-services/embedding-1

nix develop -c python3 tests/functional/model_services.py asr \
  --binary build/release/gufo --model models/audio/Qwen3-ASR-1.7B \
  --output artifacts/model-services/asr-1

nix develop -c python3 tests/functional/model_services.py tts \
  --binary build/release/gufo --model models/audio/Qwen3-TTS-12Hz-1.7B-Base \
  --variant base \
  --asr-binary /path/to/pinned/gufo --asr-model models/audio/Qwen3-ASR-1.7B \
  --output artifacts/model-services/tts-base-1
```

The default audio root is the packaged `fixtures/audio/` English WAV and reviewed
transcript, authorized by the maintainer for publication. Override `--audio-root`
only with a folder containing that English pair. ASR generates silence in memory
for its second concurrent input. Base TTS registers the English recording as the
`narrator` ICL clone. Reports retain the fixture hash and expected transcript.
For the other checkpoints, select `--variant custom` with the CustomVoice model
(default speaker `ryan`, override with `--voice`) or `--variant design` with the
VoiceDesign model (fixed English voice instructions). Neither requires a clone.

| Suite | Required checks |
| --- | --- |
| Embedding | Existing text and image HTTP suites: independent reference vectors, dimensions, normalization, encodings, batch ordering, concurrent requests, image formats, invalid inputs and recovery |
| ASR | English transcript WER ≤ 0.15, exact greedy replay, JSON/text/verbose formats, complete SSE matching buffered output, concurrent speech/silence, rejected options, malformed WAV recovery, stream cancellation and subsequent valid inference |
| TTS | Voice discovery, valid non-silent 24-kHz mono PCM16 audio, two fixed texts with ASR WER ≤ 0.25, exact same-seed warm replay, WAV/PCM/SSE equality, different concurrent texts/seeds, rejected inputs, stream cancellation and subsequent valid inference |

Audio cancellation must appear against the cancelled request's ID in the server
log; a client closing a response alone does not satisfy the assertion. SSE must
include its terminal event and a complete payload. Tests never retry requests.
The audio checks bound output duration to 0.3–20 seconds and require RMS > 0.001;
TTS requests use checkpoint sampling, fixed seeds and a 180-codec-step cap.

TTS intelligibility requires an explicitly selected ASR evaluator binary. Pin its
revision and checkpoint independently when changing ASR or TTS runtime behavior;
using the same candidate as its own evaluator does not qualify those changes.
The current ASR malformed-WAV path returns `500 transcription_failed` rather than
400. The suite records this known limitation and checks its exact error and
subsequent recovery; it does not claim correct client-error classification.

Every invocation requires a new output directory. `report.json` retains status,
per-case results, request timings and headers (including available server phase
measurements), binary/harness hashes, source revision, weight file metadata and
available download revision metadata. Response bodies, generated WAVs, evaluator
transcripts and server logs stay beside it. Embedding subreports retain individual
requests. Exceptions leave a failed report and return a nonzero exit code.
Timings are observations; this runner does not perform matched baseline timing
qualification. WebSocket streaming, long recordings, broad language/voice quality
and speaker-similarity scoring remain separate coverage.

Devlab speed comparisons use a pinned main revision and candidate, matching the
runner, toolchain, weights, settings and request histories. `service_phases.py`
uses server-emitted request timings and work counters: ASR `Server-Timing`
phases, TTS synthesis time and codec/audio work, and embedding request duration
and token/input counts. Client latency and first-output observations remain
separate. Each request and phase retains every paired repeat at the existing
5% and 3-ms thresholds; mismatched work stays inconclusive.

Repeat with fresh output directories to check stability. Preserve failed attempts
alongside successful runs. CPU checks for metric and incomplete-stream refusal:

```sh
nix develop -c python3 -m unittest discover -s tests/functional -p test_service_http.py
```

Qwen 27B deep-history recovery uses separate, explicitly selected workloads:

```sh
python3 tests/functional/deep_peers_qwen.py \
  --binary /path/to/production/gufo --model /path/to/qwen27b.gguf \
  --mode ar --source-revision BUILD_SHA --expect-component --out /tmp/qwen-deep
python3 tests/functional/over_ram_qwen.py \
  --binary /path/to/production/gufo --model /path/to/qwen27b.gguf \
  --mode dflash2 --draft /path/to/dflash2.gguf \
  --source-revision BUILD_SHA --expect-component --out /tmp/qwen-over-ram
```

Each starts its own loopback server, retains responses, loaded mode, actual draft
counts and process/host memory samples, and removes only its owned disk cache
after retaining an inventory. Use a new output directory for every invocation.
The deep workload verifies 140K+ tokens, observes a published checkpoint before
SIGKILL, checks newest compatible checkpoint selection after restarts and peers,
and compares recovery with `cache_prompt: false` at temperature zero. The over-RAM
workload verifies a completed cancellation frontier survives short peers and
requires a rewritten final message to wait for successful deeper publication.
That edited fork must select the observed published frontier; it cannot require
a previous live boundary that has no restorable checkpoint.
It fails if that publication race was not exercised. These are correctness
workloads outside hosted model CI; retained phase times do not qualify speed.
CPU evidence-refusal checks run with
`python3 -m unittest discover -s tests/functional -p test_qwen_recovery.py`;
they do not establish model correctness.

Qwen 27B cancellation inside a prefill pass has a focused latency regression
check. It disconnects both a cold prompt and a partially completed prompt,
serves a short cached peer, then verifies retained work and resumed output:

```sh
nix develop -c python3 tests/functional/qwen_prefill_cancel.py \
  --output /tmp/qwen-prefill-cancel -- \
  /path/to/production/gufo serve llm --model /path/to/qwen27b.gguf \
  --speculative dflash2 --dflash-model /path/to/draft.gguf
```

Use `--speculative off` for AR. The test owns its loopback server, records each
request's phase timings and rejects a multi-second prefill blocking the short
peer. Apply the usual matched 5% and 3 ms timing gates to its retained results.
Add `--deep-peers` before `--` and serve with `--sessions 4` to fill the other
slots with duplicate deep conversations. This checks that a short request
after cancellation can reuse a redundant slot while another live copy retains
the generated frontier. It also rejects the roughly 280 ms restore stall.

Flash-Next MTP and Qwen27B DFlash2 prompt lookup have a focused runner. It
requires actual accepted copies, compares isolated and C2/C4/C6/C8 requests under greedy and the model's
default sampling, mixes greedy copies with new text at C8, and checks stops
followed by full replay. It also checks top-k 64/65, disabled top-k and a
65-token minimum support across warm replay and stops, covering compact
sampled verification and its complete-vocabulary fallbacks:

```sh
nix develop -c python3 tests/functional/prompt_lookup.py \
  --output /tmp/prompt-lookup -- \
  /path/to/production/gufo serve llm --model /path/to/model.gguf \
  --speculative mtp --mtp-model /path/to/mtp.gguf
```

For Qwen27B, substitute `--speculative dflash2 --dflash-model /path/to/draft.gguf`.
Use `--max-concurrency 4` before `--` for a focused C2/C4 check; the default
covers C2/C4/C6/C8.

The runner owns its sessions and the loopback listener. It retains individual
request timings and server logs, and stays outside hosted CI and `--suite all`.
The native Flash-Next session test additionally checks full logits, RNG and
snapshot restoration into previously used sessions.
Pass `--mmproj /path/to/mmproj.gguf` with the server command to add image-backed
greedy and sampled copies. These cases require accepted lookup tokens before
client cancellation, confirm the server observed cancellation, and replay the
same output from the cached image prompt without prefill.

`generation_quality.py` validates reviewed coding responses from the sampled
Qwen27B generation controls. It checks generated interval tests plus independent
coverage/input-preservation cases, and retry call counts, delay caps and error behavior.
It requires completed output, so a truncated thinking response cannot pass:

```sh
nix develop -c python3 tests/functional/generation_quality.py \
  /tmp/generation/results.json \
  --thinking-results /tmp/completed-thinking/results.json
```

The reports contain `code` and `code-think` entries from `chat_result`, including
text and finish reason. This executes the reviewed generated Python on the
development host. The requests and measurement scope are recorded in the
[Qwen27B quality notes](../../docs/models/qwen3.8-27b/QUALITY.md#october-21-coding-and-prose-follow-up).

`prompt_lookup_workflows.py` checks extraction, generated URI tests and edits
after a long source archive against an already running Flash-Next MTP server:

```sh
nix develop -c python3 tests/functional/prompt_lookup_workflows.py \
  --base-url http://127.0.0.1:8080 --model MODEL \
  --output /tmp/lookup-workflows --depth 32768 --seed 31
```

Select `--case extract`, `--case uri-tests` or `--case agent-edit` for focused
runs; set the server's context above the requested depth plus 2,048 tokens.
Extraction must return the exact records. Generated Python tests run
against correct code and must detect three deliberate defects. Edits must
preserve every unrelated byte, including literal protocol tags. The edit checks
cached continuation and switching thinking mode, which changes the prefix and
requires prefill. Use `--archive /tmp/lookup-workflows/archive.txt` to retain
the same source input when comparing revisions. These are public substitutes
for PR #613's private cases; they do not reproduce the author's exact prompts.
Run the existing Pi `edit` and `creation` cases for real tool execution, too.

`embeddings.py` checks EmbeddingGemma 2 through the OpenAI SDK against a running `gufo serve embedding`.  It covers the fixture probes against sentence-transformers float32 vectors, input order, base64 and float encodings, Matryoshka `dimensions`, concurrent identical requests, and the error shapes.  It is standalone and outside `--suite all`:

```sh
nix develop -c python3 tests/functional/embeddings.py \
  --base-url http://127.0.0.1:8080/v1 --model embeddinggemma-2
```

Flash-Next startup identity checks use a separate runner with an initially empty
artifact-hash cache, as on a fresh container. RAM-only serving must skip full-file
fingerprints and still reuse its prompt. Disk serving must fingerprint the model
and restore the same reply after a restart:

```sh
nix develop -c python3 tests/functional/cache_startup.py \
  --output /tmp/cache-startup-mtp -- \
  /path/to/production/gufo serve llm --model /path/to/model.gguf \
  --speculative mtp --mtp-model /path/to/mtp.gguf --context 8192
```

Run again with `--speculative off` for AR. `--case ram` or `--case disk` selects
only that lifecycle. For Qwen 27B, add `--model-kind qwen27b` before `--`,
use its target weights, and select `--speculative off` or
`--speculative dflash2 --dflash-model /path/to/draft.gguf` in the server command.
Commands, startup times, request timings, responses and logs
remain in the output directory. This check stays outside hosted model CI.

The cluster workflow uses [impact.json](impact.json) to select suites from changed
paths. Inspect a plan locally without running Gufo:

```sh
python3 tests/functional/impact.py --model flash-next-q4 \
  src/models/qwen38_flash_next/continuation_adapter.cpp
```

`--repo PATH --base SHA --head SHA` reads a Git diff instead. Renames retain both
old and new ownership. Rules with the highest explicit priority apply; equal
priority rules contribute a union. Unknown paths expand to all compatible
coverage. Suite metadata must cover every name in `run.py`; dependencies are
deliberately conservative, so even narrow source rules can select many suites.

`--tier pr` plans the per-PR set: one or two suites per behavior (`discovery`,
`state-edges`, `messages-count-tokens`, `responses`, `stops`, `conversation`,
`image-inputs`, `structured`, `tools`, `tool-streaming`, `sampling-defaults`,
`batch` and `cache`). Every other suite is marked `"tier": "release"` and moves
to `release_only`; the deep cache histories, `cache-stalls`, `long-context` and
`reasoning-budget` still run on a PR when a changed path matches their
`escalate` patterns: cache code, the suite's own module, or Flash-Next model
code for depth checks. `run.py
--tier pr` also trims transport matrices that a case does not depend on:
`tool-reasoning` keeps the streamed mode, whose partial parses end in the same
final parse a buffered reply uses, and `tool-native-schemas` rotates endpoint,
stream and tool choice across cases. `--tier full` (the default for manual runs)
keeps every suite and combination.

The literal-closer cases from #786 (a lone `</invoke>` and a closer before or
between calls) use the deterministic HTTP fixtures in
`tests/cli/openai_chat_test.cpp`, with buffered/streamed and whole/bytewise output.
Qwen27 Q4/Q8 can omit the requested line or call a tool instead, so these parser
checks must not depend on a model following a request to copy tool markup.

`Devlab functional checks` runs on every push to a non-draft PR, like `CI`
(`pull_request`), with the repository's fork approval policy. A newer push cancels
the PR's earlier run. PRs use the `pr` tier; release-please PRs and dispatches
(unless they choose `tier: pr`) use `full`. Maintain/Admin users can dispatch a
PR number. Authorization pins its head and GitHub's merge of it into main. Like
`CI`, the GPU job builds and tests that merge commit with the suites it contains,
so tests added on main never run against a PR binary that predates them. The
merge's main parent is the diff base and the timing baseline. The GPU controller
recreates the hosted plan from the merge commit.

`devlab_plan.py` selects owners per path using reviewed priority rules in
`devlab.json`, then uses `impact.py` for each selected LLM's affected suites.
Selections from separate paths are combined:

| Changes | Automatic model coverage |
| --- | --- |
| Shared LLM behavior, cache, sampling or Qwen framing/vision | Flash-Next Q4/MTP |
| Private Qwen27 implementation | Qwen27 Q4/DFlash2 |
| Private DeepSeek implementation | DeepSeek V4/DSpark (disabled, see below) |
| Shared LLM and private DeepSeek code | Flash-Next and DeepSeek (disabled, see below) |
| Embedding, ASR or TTS code | That category's representative |
| Shared infrastructure or unmapped paths | One representative per supported category |
| Documentation, image generation or H3-only paths | No GPU job in this workflow |

Models listed under `disabled` in `devlab.json`, with a reason, are left out of
the plan and named in the report. DeepSeek V4 is disabled for now: its 81 GiB
of weights and its suite set make the run too long to finish within a PR.
Remove its entry to restore it.

The embedding and ASR representatives are EmbeddingGemma 2 and Qwen3-ASR. TTS
uses Base cloning in the PR tier; the full tier exercises Base, CustomVoice and
VoiceDesign. Manual `ar` adds the selected LLMs' AR profiles. The existing manual
`benchmark` option adds Flash-Next and its main/candidate AR depth sweep.
Qwen27 Q8 remains an explicit manual functional qualification outside this bounded
CI selection. Numerical and agent-replay obligations outside the selected suites
remain separate; representative coverage is not all-model qualification.

An ephemeral GPU runner needs the documented native build dependencies, GPU
access and enough temporary disk space for its selected checkpoints. It does not
need mounted models or audio. `devlab_models.py` downloads only selected model
assets and necessary drafts/projectors/evaluator weights from fixed Hugging Face
revisions while both production binaries build. Optional `$GUFO_CI_CACHE`
(default `$RUNNER_TOOL_CACHE/gufo`) reuse is an optimization, not a prerequisite.
The authorized controller checkout contains the English voice fixture. TTS uses
the main Gufo binary and a pinned ASR checkpoint as its intelligibility
evaluator. Missing downloads or fixtures fail visibly.

`devlab_run.py` executes models sequentially on the same GPU. LLM configuration
groups use fresh server/cache histories; service suites use `model_services.py`.
Correctness runs on the candidate only. A model failure does not hide other
models or their speed observations. Each LLM's `phases.py` arms exercise startup,
short, long and follow-up requests; `service_phases.py` measures bounded service
histories. Both alternate main/candidate servers for two paired repeats,
with GPU cool-down. Every request and phase retains all measurements; a slowdown
requires both >5% and >3 ms in every repeat. Different output/work counters mark
relevant comparisons inconclusive. The plan selects timing gates from production
changes; harness-only changes retain timing observations without gating unrelated
noise. Startup phases gate only loader/configuration/build changes. Missing timing
evidence is always incomplete. Display medians do not determine verdicts.

The hosted report job updates one PR comment with the same correctness and phase
layout for every selected model, selection reasons, known limitations and links
to the run/artifacts. Missing selected groups, missing timing arms or mismatched
revisions cannot pass. Correctness failures and consistently slower server phases
set the advisory status to `failure`; incomplete, noisy or unavailable evidence
sets `error`. The ordinary PR check remains advisory; automatic release merging
requires it. The optional AR depth benchmark remains observational.

The PR write token stays on hosted runners. Fork runs upload their text report;
`devlab-comment.yml` publishes it from `workflow_run` without executing PR code.
It also maintains the running placeholder. Artifacts retain the plan, asset map,
partial summaries, request bodies, generated audio, server logs and all timing
arms. The client dependencies are pinned in `requirements.txt`; harness imports
are checked before execution. Model correctness stays outside hosted CPU CI.

Planner, report and timing-contract checks run without weights or a GPU:

```sh
python3 -m unittest discover -s tests/functional -p test_impact.py
python3 -m unittest discover -s tests/functional -p test_report.py
python3 -m unittest discover -s tests/functional -p test_devlab.py
```

The startup-loading runner polls health, readiness and generation from process
start. Every answer during the load must be 503 `Loading model`, `/health`
included, and the real server must answer all three after the listener handoff:

```sh
nix develop -c python3 tests/functional/startup_loading.py \
  --output /tmp/startup-loading -- \
  /path/to/production/gufo serve llm --model /path/to/model.gguf \
  --cache-disk /tmp/startup-loading-disk
```

It fails if no loading answer was observed, so an instant load is not a pass.
`observations.json` keeps every probe with its time; this check stays outside
hosted model CI and `--suite all`.

Device-loss process tests use a separate opt-in runner because each lost-device
case must terminate its server. Build its CMake-owned preload helper, then run
against a production binary and one explicitly selected model/mode:

```sh
nix develop -c cmake --preset cpu-test
nix develop -c cmake --build --preset cpu-test --target device_loss_faults
python3 tests/functional/device_loss.py \
  --fault-library build/cpu-test/libdevice_loss_faults.so \
  --output /tmp/device-loss-check -- \
  /path/to/production/gufo serve llm --model /path/to/model.gguf \
  --sessions 2 --context 4096 --think off --speculative off
```

It checks recoverable HIP errors, a pending probe's five-second timeout,
buffered/streaming error contracts, idle loss without any HTTP traffic, arrival
while an idle probe stays pending, cleanup avoidance and exit status 75.
Buffered failures return JSON 503. Streams commit HTTP 200 at admission, so a
failure during prefill or after the first token emits a terminal SSE error. A
blocked SSE writer must still trigger the ten-second forced-exit watchdog,
without a health request or peer disconnect. Repeat `--case` for focused checks.
Reports retain commands, loaded mode, actual warm-generation drafts, raw
responses and logs, including failures. These tests inject errors only inside
the child process; they do not establish behavior after a physical GPU reset.
The helper is excluded from normal builds and the runner stays outside hosted
CI and `--suite all`. The existing scheduler/HTTP CPU tests cover sticky loss,
queued peers, skipped invalidation and listener observation during a blocked
write without loading weights.

```sh
nix develop -c python3 tests/functional/run.py \
  --record-baseline --output /tmp/api-baseline --sampling-preset qwen38 \
  --suite tools --suite sampling-defaults --suite batch -- \
  /path/to/baseline/gufo serve llm --model /path/to/model.gguf \
  --mmproj /path/to/mmproj.gguf --sessions 4
```

Then use the candidate binary, a new output directory and
`--baseline /tmp/api-baseline` instead of `--record-baseline`. Keep the model,
server options and suite order identical, with no competing GPU/build work.
Use clean main and the rebased PR with the same production toolchain. Compare
only affected suites. Reuse matching baseline evidence; do not repeat the whole
correctness matrix on main. A baseline must match the revision, harness,
toolchain, weights, settings and preceding cache history.

Use `deepseek4` for DeepSeek. Pass the model's normal speculative options for
DFlash2, MTP or DSpark. Server sampling arguments become the expected defaults;
`--mmproj` enables image cases. Keep informational server logging enabled.

Each invocation tests **one model and one mode**. Model-specific changes need
only the affected target/modes. For shared text changes, run the affected suites
once per row/mode below; no full benchmark sweep is needed.

For correctness capture of slow cold prompts, `--request-timeout SECONDS` sets
the SDK deadline (default 120); `--suite-timeout SECONDS` separately bounds the
whole suite (default 900). The request deadline is retained in both reports and
must match for baseline comparison. Longer deadlines do not change assertions
or timing margins; retain the original timeout failure when investigating it.

| Target | `--speculative` modes | Sidecar option |
| --- | --- | --- |
| Qwen27B Q4_K_XL | `off`, `dflash2` | `--dflash-model` |
| Qwen27B Q8_K_XL | `off`, `dflash2` | `--dflash-model` |
| Flash-Next Q4_K_XL | `off`, `mtp` | `--mtp-model` |
| DeepSeek Flash 0731 | `off`, `dspark` | `--dspark-model` |

Keep the sidecar path in the `off` command to test the explicit override.
`report.json` checks the loaded mode and records observed draft counts; `off`
must perform zero drafts. `state-edges` requires actual draft execution when
enabled, so a loaded-but-unused sidecar cannot qualify that mode. Use the normal
draft limit for this suite. Audio and image/video generation have separate tests.

Card 01's new cache workloads stay outside routine functional runs until the
new cache lands in card 19. Both `run.py --suite all` and
`openai_sdk.py --suite all` exclude `cache-compaction`, `cache-transforms`,
`cache-pressure` and `cache-messages-loop`. The disk lifecycle runner is
standalone and is never invoked by `all`. Existing functional suites and the
passing CPU workload/gate checks remain active. Explicitly selecting a new
workload still runs its strict assertions and may fail on the legacy cache;
use that selection only to record a baseline or develop the new cache.
Card 19 enables these workloads in routine qualification with their required
RAM, disk and restart settings once their contracts pass.

| Suite | Checks |
| --- | --- |
| `discovery` | Health, readiness, model ID/context and advertised text/image inputs; no generation |
| `responses` | SDK buffered, streaming and async Responses |
| `stops` | Text, Unicode, reasoning and tool stops; peer isolation |
| `conversation` | Thinking/efforts, images, cancellation and RAM reuse |
| `image-inputs` | PNG, JPEG and WebP uploads in Chat and Responses; URL spellings, bad uploads and recovery; Messages image blocks in a user turn and a tool result |
| `tool-images` | Chat/Responses function and custom-call image outputs; text/image order, image-only output, actual colors, streamed retries, changed pixels, continuation reuse, role/URL errors and unchanged text-only controls |
| `image-count` | 17+ images in one message and across turns; Chat/Responses, sampled thinking/JSON, concurrent colors, limits, cancellation and RAM/disk replay |
| `tools` | Required/named/auto, schemas, literal arguments and tool history |
| `auto-tools` | Focused subset for optional tool calls |
| `tool-edges` | Referenced argument types, literal CR, unusual keys, named Responses metadata, foreign tool markers in prose and parallel calls (no DeepSeek text after the call block) |
| `tool-reasoning` | Quoted tags, exact literal arguments, early stops, disabled tools, envelope framing, completed tool-result continuations and warm replay of contaminated history; Chat/Responses |
| `tool-streaming` | Incremental argument arrival and exact assembly in Chat/Responses/Messages; stable IDs, literal framing, thinking, stopped/limited calls and successful retries |
| `reasoning-separator` | No leading separator newlines after reasoning in Chat/Responses, plain/tools/JSON; exact streamed/buffered text, warm retry, continuation and thinking-off paragraph breaks |
| `reasoning-budget` | llama.cpp-compatible budgets: Responses/Chat `reasoning_budget_tokens` (buffered and streamed), the older `thinking_budget_tokens` spelling, and Messages `budget_tokens` cap the ASCII fixture at N tokens while preserving its greedy reasoning prefix; adaptive and `-1` retain the server default, invalid/oversized values fail, sampled requests obey the cap, and replayed capped thinking preserves cache reuse |
| `tool-agent` | Ordinary nested agent schemas, edit/read/finish turns, no protocol switch, limits, stops/retry, images and sampled peers |
| `tool-agent-loop` | Bounded autonomous read/edit/verify loop; each turn checks cache reuse and detects repeated actions |
| `tool-history` | Legacy names, result pairing, current-tool constraints, images, cached retry, stops/limits and sampled peers |
| `messages-tools` | Messages `tool_use` call, `tool_result` replay with cache reuse, buffered and streamed events, and an uncached Chat control of the same prompt |
| `messages-count-tokens` | Counts match complete Messages prompt usage with system/history, Unicode, all tool choices and thinking; cold/warm cache counts, no inference side effects, over-context sizing and request errors |
| `messages-image-counts` | Image counts match buffered/streamed Messages usage, including tool results, warm caches, PNG/JPEG/WebP, multiple images, Unicode and over-context sizing; no inference side effects; text-only models reject image counts and generation. Requires `--expected-input-modalities text,image` with a projector, or `text` for capability rejection checks |
| `tool-untyped` | Open/typed tools, refs and finite values: framing, arguments, streaming, turns, limits, stops/retry and sampled peers |
| `tool-mixed` | JSON-only neighbors, annotated refs, extra keys, URI and nullable arguments across Chat/Responses; images, stops/retry and sampled peers; a union neighbor keeps native calls, so a replayed reasoning/call turn is reused in full |
| `tool-native-schemas` | opencode's tool set beside each schema family that used to force a JSON envelope (pattern, oneOf, allOf, not, open objects), auto and required, strict: native calls, no prompt instruction, typed arguments and full reuse of the generated call; Chat/Responses |
| `tool-choice-cache` | Required/auto/omitted transitions preserve the complete generated call, unchanged-choice controls, prompt-token equality and zero-prefill retries |
| `tool-parameter-order` | Wide native schemas retain intended arguments across declaration orders, all three APIs, streaming, sampling and thinking; includes the 10/11/12-field controls from #641 |
| `tool-native-types` | Focused subset: enum/const/inferred string types, literal delimiters and exact continuation reuse after new literal tool markers; Chat/Responses, text/images |
| `tool-schema-edges` | Wildcard JSON types, conditional fields, impossible schemas, nested metadata and required-call timing; both APIs, cache, stops and sampled peers |
| `state-edges` | Actual AR/draft execution, tiny thinking budgets, zero-argument tools, schema changes, stops (including inside quoted calls), image retry and failed-request recovery |
| `structured`, `structured-limits` | Request JSON schemas, SDK parsing, limits and stops |
| `sampling-defaults`, `sampling-ranges` | CLI/request overrides, partial/null settings and range validation |
| `batch` | Independent requests across Chat, Responses and Completions; batched short greedy replies match serial ones; sessions 1–8 |
| `progress` | Opt-in progress on all text endpoints; output/sampling equality, limits, stops, images, batching and cancel/resume |
| `stream-start` | Plain streams on all text endpoints send headers before a cold prefill completes; a stream queued behind every session sends them after the five-second bound |
| `prefill-scheduling` | Short arrival during a cold prefill: work-aligned arrival, independent per-request timings, unchanged output and no repeated prefill. Requires sessions ≥2 and context ≥16384; not in `all` |
| `long-context` | Longer multi-turn recall, endpoint switching, sampled JSON and cancellation |
| `metrics` | Live slots, Prometheus cache/time/draft counters, uncached work, endpoint totals, queueing and cancellation |
| `cache` | Interrupted text/thinking/tool/image histories, ordinary and legacy tool names, RAM and disk restart; disk checkpoint spacing for a growing conversation and a branch restored after restart |
| `cache-edits` | Reuse earlier work after editing the latest message, shortening an older tool result, or editing an earlier user message and dropping later turns; compare with uncached responses |
| `cache-growth` | Keep cache reuse advancing over several turns when the client omits reasoning; check reasoning replay and thinking-off controls, including Messages thinking blocks, and compare with uncached responses |
| `cache-depth` | Histories beyond 16K, concurrent branches from rewritten replies, a side conversation, resume, unchanged retries and exact uncached controls; use context 32768 and also check sessions 1 |
| `cache-stalls` | Flash-Next AR/MTP: sequential coding/planning prompts with 6.4K/25.6K/55.6K/56K padding and three repeats each, thinking on and tg128. Checks prefix reuse and exact fully cached replay; retains each request's PP/TG and maximum-gap timing gate. Public equivalents of #632, whose prompts were not attached. Use sessions 2/context 262144 to match its capacity; minimum context 65536. Explicit selection only, outside `all` |
| `cache-rotation` | Check cache RAM limits and keep history across conversations and small side requests; compare answers with uncached controls |
| `cache-shared-prefix` | New conversations under one system prompt, one after another, with long and short tasks: from the third on they restore the whole shared prefix; compare answers with uncached controls |
| `cache-bridge` | A chat bridge sends each user message with metadata its history copy drops, under a full RAM budget with small unrelated requests between turns: from the third turn on, reuse reaches the user turn two back; compare answers with uncached controls. Not in `all` |
| `cache-compaction` | A tool history grows near the context limit, then a summary replaces it while system/tools stay; check shared-boundary reuse, unchanged retries, continued growth and exact uncached answers. Requires context ≥ 16384. Not in `all` |
| `cache-transforms` | Clear an old tool result, remove a few hundred late user tokens, or retry/edit after a 10K+ result; restore coherent pre-edit boundaries and compare uncached answers. Requires context ≥ 32768. Not in `all` |
| `cache-pressure` | Fill the RAM budget, compact an abandoned history, resume idle conversations, send volatile headers/UI tasks, then fan out children and resume their parent. Requires a small explicit RAM budget. Not in `all` |
| `cache-messages-loop` | Twelve streamed `/v1/messages` tool-call/result cycles with actual calls and growing history; each request checks reuse and has an uncached Chat control. Requires context ≥ 16384. Not in `all` |
| `system-injection` | System/developer messages after the conversation start, in Chat and Responses: accepted, followed by the model and equal to uncached responses; the hoisted turn's reuse is recorded and the next turn must reuse it in full |
| `cache-concurrency` | Concurrent identical prompts, shared-system fan-out with short and long tasks, short or no shared prefixes, a retained conversation beside a newcomer, and a cancelled leader; check waits, prefill work and uncached answers |

For `discovery` (also included in `all`), pass `--expected-input-modalities text` or `text,image` before
the server command. Projectors can load automatically beside the weights, so
the expectation is explicit rather than inferred from `--mmproj`.

For `image-inputs` and `image-count`, pass the model's `--mmproj`. Both use small
fixed images and are included in `all` only with that option. `image-count`
restarts the server for disk replay; its first four cases are 1/16-image timing
controls usable on older main with `--through-case image-count:image_count_control_16_True`.

`tool-images` also requires `--mmproj`. For matched existing-behavior timing
controls on main, stop at
`--through-case tool-images:tool_images_control_complete`; full candidate runs
continue into the newly supported image-bearing tool outputs. A text-only
model must not qualify the image suite by skipping its assertions.

Repeat `--suite` to select affected tests; `--suite all` explicitly runs all. For long
contexts, use server `--context 32768`; actual prompt depth is recorded. `cache`
uses its own 8 GiB disk budget and 1 GiB staging area inside the output directory;
the runner removes that disk cache when the run ends, keeping reports and logs.
Flash-Next allocates that staging as pinned host memory at startup, beyond its
RAM budget, although its default 1 MiB transfer buffer suffices. The runner
keeps 1 GiB so results stay comparable with retained runs and other models.
For timing controls on revisions predating progress, use `--allow-missing-progress`
with `--record-baseline`. Candidate qualification always requires progress events.
Model runs stay outside hosted CI; CI checks the runner and measurement logic.
The metrics suite reconciles verification-round counts with request timings and
terminal logs, including cancelled requests; AR must report zero rounds and
speculative modes must execute actual rounds.
For metrics changes, run `--suite metrics` with AR and the affected speculative
mode. It checks all three text endpoints and reconciles cancelled work with the
terminal logs. It also checks both slot endpoints, active request identities and
progress, queued-request exclusion, prompt privacy, idle cleanup, and the
in-flight KV ratio. Concurrent shared-prefix requests also check that parked
followers reserve slots and keep newer arrivals queued within `--sessions`.
Scrapes are not recorded as generation requests.

Tool framing is removed when it directly echoes an accepted call or a client
`<invoke name="X">` envelope names a declared tool. Other raw XML, standalone
closers and spelled vocabulary tokens are literal content. A full Qwen call naming
a declared tool inside an unfinished fence or inline backtick span uses the
legacy fallback only if its arguments satisfy the schema. Completed fences and inline
spans keep markers literal; inline spans can cross nonblank lines, but a closing
backtick must arrive before the next blank line. Streaming holds a possible call
inside an open span until the final parse resolves it. These text-only rules
cannot distinguish literal XML naming a declared tool from a failed envelope
attempt, or an unfinished code example from a real call. Actual EOS handling
uses token IDs in the backend.

Server-authored JSON tool and response-format instructions are template framing;
client message content remains literal text, including spelled tool delimiters.
With tools declared, trailing whitespace can be held until the next decoded
piece, so a streamed delta ending in whitespace may arrive one token later.
Cancellation fixtures that stop after a fixed number of deltas can therefore
replay different assistant text from main. Compare those timings using matched
interrupted histories, not a direct baseline for the divergent session; this
also applies to later disk-spacing and cancellation requests in that history.

For repeated screenshot preprocessing (#627), `image_history.py` measures full
RAM-cache retries with 1, 2 and 4 synthetic 1080p screenshots, similar-token text
controls and 128px image controls. It uses only the Python standard library:

```sh
python3 tests/functional/image_history.py --output /tmp/image-history-main -- \
  /path/to/production/gufo serve llm --model /path/to/model.gguf \
  --mmproj /path/to/mmproj.gguf --sessions 1 --context 32768 \
  --cache-ram-bytes 8589934592 --think off
```

Repeat with the candidate, identical model/mode/options and a fresh output path.
The harness owns host, port and debug logging. It retains every request,
response, per-request queue time, process RSS and preprocessing-cache byte count;
retries require identical output, full prompt reuse and zero prefill. Add
`--turns 1 --kind 1080p --repeats 7` for a focused control. Also add
`--fresh-encoding` to change PNG metadata on each retry while preserving pixels,
tokens and continuation identity: this measures preprocessing misses without
GPU prefill. Add `--image /path/to/1920x1080.png` to check a supplied PNG,
including the digest cost of a larger encoded payload. The native
`qwen_vision_cache_test --capacity-probe` separately
checks the default byte ceiling with 22 resized 1080p images.

For real coding-agent regressions, run `pi_agent.py` against a local server with
`--base-url`, `--model`, `--pi /path/to/pi-0.87.0`, `--server-log`, and a fresh
`--output` directory. Repeat `--case` to select affected tasks;
`--case literal-protocol --passes 1` exercises a real write/read/verify loop
containing literal ChatML vocabulary spellings.
`--case wide-tool --passes 1` loads a fixture extension with a 12-field `todo`
schema and verifies two actual updates and their tool results, covering #641.
Both Chat and Responses require exactly the requested fields, including when
Pi omits `strict`; unrequested default values also fail the test.
It replays #368's five tasks, verifies the generated code independently, and
retains Pi sessions, HTTP/SSE and per-request timings. It executes generated
commands in disposable fixtures using isolated Pi configuration. Use `--passes 1`
for a focused check; the default five passes matches the reported debug workload.
`--conversation --context-file FILE` additionally tests retained long history.

For Claude Code sessions over `/v1/messages`, run `claude_code_agent.py` with
`--base-url`, `--model`, `--claude /path/to/claude`, `--server-log` and a fresh
`--output` directory outside any project that has a `CLAUDE.md` (the CLI would
read it). It runs the real CLI non-interactively with its `Read`, `Write`,
`Edit` and `Bash` tools allowed, an isolated home and configuration, and every
model alias mapped to `--model`, through the same recording proxy. The
`simple`, `tools`, `edit` and `literal-protocol` tasks match Pi's and are
verified the same way: files and generated code directly, the CLI's
stream-json transcript for tool use and framing leaks, and the server log for
completed requests and cache reuse on every later agent turn. Claude Code ends
each request with a `system` message whose copy differs in the next request,
so later turns reuse the prompt up to that message, not the whole previous
prompt.

For image tool results, explicitly select `--case image-read --passes 1` with
a vision-capable model. The task opens a fixture through Pi's real `read`
tool and checks both the returned image pixels and the model's color answer.
Run with `--api openai-responses` and `--api openai-completions` to check both
client transports; retain the wire requests to inspect the image placement.
This image task is excluded from the default text-task list.

For a conversation that **actually grows past 200K tokens through tool results**,
run `agent_long.py --agent pi|opencode --executable PATH --base-url URL
--model NAME --output DIR --min-context 201000 --complex-tools --stress-turns 20`
against a server at its supported context limit (262144 for Flash-Next).
For Pi Responses, add `--api openai-responses --server-log SERVER_LOG`.
It disables compaction and uses one session. The complex tools exercise nested
unions, references, arrays, nullable fields and literal XML/JSON in transactional
updates, with independent state/digest checks. Both Pi and OpenCode execute the
same tools. Every request checks framing, arguments and that cached tokens equal
the previous prompt plus generated tokens; transcripts
and phase timings are retained. No synthetic system-padding counts as growth.
If Pi omits a reasoning-only reply, the report identifies that history change
and verifies reuse through the measured pre-generation boundary instead.
Use `--resume PREVIOUS_OUTPUT` to continue either client's retained session,
including after a server restart. To test cancellation, add `--interrupt tool`
or `--interrupt thinking` to a resumed run. The harness sends SIGINT only after
observing that phase in a live stream and records the interruption separately.
Then resume that output directory without `--interrupt`. The harness preserves
the unfinished task and its files, and requires successful tool execution and
state/digest checks without repeating committed changes. An interrupted run
alone is not a pass. Resume prompts distinguish completed calls for this task
from earlier transactions; the checks still require the file read and catalog
query, even when the model could infer their contents from earlier tasks.
Use `--interrupt-after-tools 5` with `--complex-tools --interrupt tool` to
interrupt the receipt after the transaction has committed. Resuming must
preserve the committed revision and finish the receipt without another apply.

`opencode_agent.py` runs real opencode (`--opencode PATH`, default on `PATH`)
against a local server with `--base-url`, `--model` and a fresh `--output`.
Each task uses isolated opencode configuration and a small stdio MCP server
whose tools carry every schema family `tool-native-schemas` covers, so every
turn declares them beside opencode's own tools. It checks that framing never
reaches content or replayed history, that MCP arguments keep their JSON types,
that tasks complete, and that each next agent request reuses the previous one.

Cache checks use real assistant replies and run cold controls after the warm
history, so the controls cannot hide a missed checkpoint. `cache-edits` checks
latest-message edits, shortened tool results and rewinds. `cache-growth` checks
omitted, preserved and explicitly discarded reasoning, plus thinking off.
Unchanged retries in `cache-growth`, `cache-depth`, `long-context`, reasoning-separator,
tool-history, `tool-choice-cache`, schema edges, state edges, progress and metrics
must prefill nothing, unless the runner's server log shows that memory pressure
refused the retry copy; the retry may then prefill only the assistant opening
after the stable boundary. In agent histories the mutable suffix includes the
final user turn; `tool-history` bounds it with an independent count-tokens
request rather than extra slack, and `tool-choice-cache` does the same per case.
Generic capture failures do not qualify for this exception.
Run `nix develop -c python3 tests/functional/test_long_context.py` to check
that only a capacity refusal in the preceding request's log window permits
the framing suffix; lost context, changed output and unrelated refusals fail.
Run `nix develop -c python3 tests/functional/test_tool_choice.py` to check that
the tool-choice retry may prefill its final user turn only when the server
logged a capacity refusal during the continued request.
Component disk-spacing checks correlate drained publication logs with the
checkpoint IDs captured by their own requests, excluding delayed writes from
earlier suites even when their token counts match.
`cache-rotation` visits four conversations and eight small side requests; use
`--sessions 1` to verify retention is independent of execution slots. It also
checks the startup RAM cap; byte/record pressure is covered by CPU tests.
`cache-bridge` needs RAM pressure: run it in its own invocation with a budget
that holds at most 16 of its conversation checkpoints, such as
`--cache-ram-bytes 2147483648` for Flash-Next at the default context.
Background requests fill the budget first; a larger budget fails as unqualified.
`cache-concurrency` sends each group at once. With `--sessions 2` or more,
requests sharing a long prefix must wait for one prefill and then prefill only
their own tail; groups sharing little or nothing must not wait. With
`--sessions 1` no request may wait.

`cache-compaction` measures a small tool result to size a larger append for the
loaded tokenizer. The measured history must reach 75–95% of the configured
context, and compaction must drop at least half that capacity. Run explicitly
with `--suite cache-compaction` and server `--context 32768 --sessions 1`.
It preserves the declared tools and system text, replaces all other turns with
a summary, checks reuse at or beyond a coherent system/tools checkpoint measured
by a short warm branch before the large append, then retries
and continues the compacted conversation. Cold controls run after every warm
request. The seed is a genuine model reply; tool calls/results and the summary
are deterministic client-authored fixtures. This covers compaction mechanics,
not a real client's summary generation or retention under RAM/disk pressure.
The [card 01 results](../../docs/cache-redesign/cards/01-functional-coverage.md#results)
record all scenario families, including failures on the legacy cache.

Run `cache-transforms` with a roomy RAM budget in its own invocation. Its
retried prompts end with a user turn after a tool result, and Qwen templates
place the stable boundary before such a turn, so under pressure an unchanged
retry prefills that turn as well as the assistant opening. The suite therefore
checks zero-prefill retries without adding pressure.
`cache-pressure` instead runs explicitly with `--cache-ram-bytes 2147483648`,
`--context 32768`, and `--sessions 2` (also supports one session, with sequential
children). It verifies that background requests published at least twice the
configured budget before the measured histories. UI prompts embed actual chat
history, and volatile headers change the first system tokens. Warm histories
and all children finish before cold controls start.

`cache-messages-loop` streams actual model-authored `read_archive` calls and
replays their IDs/typed arguments with deterministic tool results. Compaction
is not performed in this loop. Its final prompt must exceed 10K tokens.
Messages streams expose token usage but not full server phase timings;
request TTFT and draft work are joined from the server log, while the buffered
Chat controls expose full phase timings. Every streamed call/result is checked
against the same uncached rendered prompt, after the entire warm loop.

Disk workloads use a separate runner so their restarts and publication gate
cannot affect other suites:

```sh
nix develop -c cmake --preset cpu-test
nix develop -c cmake --build --preset cpu-test --target cache_disk_faults
nix develop -c python3 tests/functional/cache_lifecycle.py \
  --output /tmp/cache-lifecycle --fault-library build/cpu-test/libcache_disk_faults.so -- \
  /path/to/production/gufo serve llm --model /path/to/model.gguf \
  --context 32768 --sessions 1 --think off --speculative off
```

Repeat `--case restart|tiered|oversized|crash|side-traffic` to select cases. `restart` models
llama-swap stopping and starting a server between turns. `tiered` first warms a
short RAM prefix after restart, then requests a history with a deeper published
disk checkpoint. `oversized` uses 1 MiB staging beneath an observed larger
checkpoint. `crash` uses the CMake-built, child-only preload helper to hold a
private-cache file at `fsync`, verifies queued bytes and the completed HTTP
request, then SIGKILLs that child. Restart may restore only previously published
boundaries. The helper is disarmed for initial publication and is covered by
CPU process tests. Every case owns an 8 GiB disk budget, checks free space and
deletes its cache on completion/failure. Reports and logs remain outside Git.
For the 100K+ case, use `--seed-lines 7300 --minimum-published-tokens 100000`
with context 133760, selecting `oversized` and `crash`. Actual prompt usage,
publication and disk restoration must meet that depth; line count alone is
not evidence. `--publication-timeout 300` allows the large checkpoint to finish
before the deliberate crash without changing request timing gates. The
oversized case waits for the deep publication before graceful shutdown and
still uses 1 MiB staging. Keep the same explicit RAM limit in both modes.
Use `--staging-bytes` to match the staging budget of a reported restart,
tiered or crash workload (default: 2 GiB). This value is recorded in the
report; the oversized case always keeps its 1 MiB staging allocation.
For checkpoints larger than committed RAM, use `--require-over-ram` with
`--case oversized` or `--case side-traffic`. For example, Flash-Next at context 262144 with
`--seed-lines 12500 --minimum-published-tokens 174000` and
`--cache-ram-bytes 4593364992` exercises a 175K seed and a 186K continuation.
The assertion compares the published payload with the server's actual
post-allocation RAM budget; a queued job alone does not qualify the case.
`side-traffic` keeps the system prompt at at most 256 background lines and
places the remaining seed in the user history. After publishing the deep append
and restarting, it alternates four short conversations sharing that system with
one further deep turn, for three rounds. Every deep turn must retain at least
the last published boundary; each new deep publication must advance before the
next round. Cold output controls run after all warm traffic. This covers the
short-peer/deep-history pattern reported in #542; the system-only oversized
fixture cannot establish that behavior.
Known cache-work failures exit nonzero and retain controls/evidence; they are
not skipped, marked as expected passes, or fixed in this test-only card.

Publication-wait regression #617 uses the same process-local fault library,
filtered to hold a manifest after its payload is written. It checks both an
expired recomputation budget and publication before that budget expires:

```sh
nix develop -c python3 tests/functional/publication_wait.py \
  --output /tmp/publication-wait --fault-library build/cpu-test/libcache_disk_faults.so -- \
  /path/to/production/gufo serve llm --model /path/to/flash-next.gguf \
  --mtp-model /path/to/mtp.gguf --speculative mtp --sessions 1 --context 65536
```

The runner uses a fresh 8 GiB disk tier and a 1 GiB RAM budget unless explicitly
set, requires an actual over-RAM publication and disk restore, and retains all
HTTP requests/responses and timings. It removes its own cache after each case.
Use `--case bound --expect cap` only for a matched pre-fix main timing control;
candidate correctness uses the default `bounded` expectation. The bound case
schedules release after 130 seconds and releases earlier if the waiter finishes;
the child-only helper has a longer safety timeout. The bound waiter re-prefills
from the deepest checkpoint it can restore. When RAM is tight, that can be a
shallow seed on disk, so the waiter must match the cold answer and reasoning
mode, not the full reasoning text. The retry must still match cold exactly. The
publication case waits for the scheduler's actual park before starting its
250 ms release timer, then checks the matching request/checkpoint release before
its deadline and a disk restore. These checks stay outside hosted CI and the
routine API suites.

PR #677's staging-reclamation regression reuses this runner with `--case reclaim`
before `--`. It defaults to a 256 MiB RAM cache, requires a retained checkpoint
to release RAM during the deep request, then verifies publication and restoration
of the exact held 50K+ token checkpoint. It also checks the loaded speculative
mode, actual draft work, fewer than 64 prefill tokens and output against a cold
control. A run that never reclaims RAM fails rather than passing on disk use
alone. For example:

```sh
nix develop -c python3 tests/functional/publication_wait.py \
  --case reclaim --output /tmp/publication-reclaim \
  --fault-library build/cpu-test/libcache_disk_faults.so -- \
  /path/to/production/gufo serve llm --model /path/to/flash-next.gguf \
  --mtp-model /path/to/mtp.gguf --speculative mtp --sessions 1 --context 65536
```

Unchanged retries must reproduce the complete output with zero prefill. After a
restart, disk restores may re-prefill less than one 2048-token disk step;
greedy and zero-prefill restores must still reproduce their output. Edited
histories must retain a useful earlier prefix and match their cold answer;
free-form reasoning may vary with prefill chunk shapes. Use the recorded
requests and phase timings to investigate failures, not a full model sweep.

Continuation report rows use `status: "passed"` for successful validation.
`exact` records whether the resumed and follow-up assistant messages both match
their reference hashes; `exact_required` records whether that equality is
required. A sampled disk restore with partial re-prefill can pass with
`exact: false`; greedy and zero-prefill restores still fail on a mismatch.

Every request checks its applicable response format, expected output and timings.
Missing measurements fail. `comparison.json` reports per-request prefill, decode,
queue, restore and wall times, plus server startup/restart. Missing cases,
changed output/token counts or unexpected prefill/cache work fail immediately.
Timing margins remain **both 5% and 3 ms**. One overrun is **inconclusive**, not
proof of regression. Exit codes: **0 pass, 1 fail, 2 inconclusive/unqualified**.

Use the code diff to identify affected paths. Investigate their timing flags;
rerun only affected histories, alternating main and PR. Keep unrelated timing
variance visible without expanding into another full matrix.
`--through-case long-context:long_cancel_replay` replays preceding selected
suites/cases and stops before the next request. Keep the original suite arguments
and server/cache settings. Disk-restore investigations use the `cache` suite.
Combine the original pair and focused follow-ups without discarding results:

```sh
python3 tests/functional/compare.py \
  --pair /tmp/main /tmp/pr --pair /tmp/main-repeat /tmp/pr-repeat \
  --output /tmp/timing-evidence.json
```

Add `--control /tmp/main-repeat /tmp/main-control` for a fresh unchanged-main
control with the same focused history.
Every request/phase is judged separately: all observed candidate times within
margin pass; repeated separation from stable main fails; overlapping or variable
timings stay inconclusive. Sample values and flag counts remain visible for
stall investigation. Faster requests never offset slower ones. Fix regressions;
inconclusive timings remain unqualified. Never widen margins to pass.

Reports and logs survive failures. These checks complement the standard speed
benchmark and numerical quality tests; they do not establish upstream model
parity. Sampled DSpark may differ across concurrency levels; fixed-path replay
and greedy equality remain checked. API behavior and references are in
[SERVER.md](../../docs/SERVER.md).

`embedding_images.py` checks typed text/image `/v1/embeddings` requests against
the official float32 image fixtures, including mixed batches, multiple images,
the Frigate trailing newline, base64 dimensions, concurrency, PNG/WebP decoding,
invalid image destinations and recovery. It requires Pillow and numpy and stays
outside `--suite all`:

```sh
nix develop -c python3 tests/functional/embedding_images.py \
  --base-url http://127.0.0.1:8080/v1 --output /tmp/image-embeddings.json
```
