# Text API functional tests

The runner starts an isolated local server and checks real responses with the
OpenAI SDK. Use production binaries and local weights; no models are downloaded.

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

| Suite | Checks |
| --- | --- |
| `discovery` | Health, readiness, model ID/context and advertised text/image inputs; no generation |
| `responses` | SDK buffered, streaming and async Responses |
| `stops` | Text, Unicode, reasoning and tool stops; peer isolation |
| `conversation` | Thinking/efforts, images, cancellation and RAM reuse |
| `image-inputs` | PNG, JPEG and WebP uploads in Chat and Responses; URL spellings, bad uploads and recovery |
| `image-count` | 17+ images in one message and across turns; Chat/Responses, sampled thinking/JSON, concurrent colors, limits, cancellation and RAM/disk replay |
| `tools` | Required/named/auto, schemas, literal arguments and tool history |
| `auto-tools` | Focused subset for optional tool calls |
| `tool-edges` | Referenced argument types, literal CR, unusual keys, named Responses metadata, foreign tool markers in prose and parallel calls (no DeepSeek text after the call block) |
| `tool-reasoning` | Quoted tags, exact literal arguments, early stops, disabled tools, envelope framing, completed tool-result continuations and warm replay of contaminated history; Chat/Responses |
| `reasoning-separator` | No leading separator newlines after reasoning in Chat/Responses, plain/tools/JSON; exact streamed/buffered text, warm retry, continuation and thinking-off paragraph breaks |
| `tool-agent` | Ordinary nested agent schemas, edit/read/finish turns, no protocol switch, limits, stops/retry, images and sampled peers |
| `tool-agent-loop` | Bounded autonomous read/edit/verify loop; each turn checks cache reuse and detects repeated actions |
| `tool-history` | Legacy names, result pairing, current-tool constraints, images, cached retry, stops/limits and sampled peers |
| `messages-tools` | Messages `tool_use` call, `tool_result` replay with cache reuse, buffered and streamed events, and an uncached Chat control of the same prompt |
| `tool-untyped` | Open/typed tools, refs and finite values: framing, arguments, streaming, turns, limits, stops/retry and sampled peers |
| `tool-mixed` | JSON-only neighbors, annotated refs, extra keys, URI and nullable arguments across Chat/Responses; images, stops/retry and sampled peers; a union neighbor keeps native calls, so a replayed reasoning/call turn is reused in full |
| `tool-native-schemas` | opencode's tool set beside each schema family that used to force a JSON envelope (pattern, oneOf, allOf, not, open objects), auto and required, strict: native calls, no prompt instruction, typed arguments and full reuse of the generated call; Chat/Responses |
| `tool-native-types` | Focused subset: enum/const/inferred string types, literal delimiters and exact continuation reuse after new literal tool markers; Chat/Responses, text/images |
| `tool-schema-edges` | Wildcard JSON types, conditional fields, impossible schemas, nested metadata and required-call timing; both APIs, cache, stops and sampled peers |
| `state-edges` | Actual AR/draft execution, tiny thinking budgets, zero-argument tools, schema changes, stops (including inside quoted calls), image retry and failed-request recovery |
| `structured`, `structured-limits` | Request JSON schemas, SDK parsing, limits and stops |
| `sampling-defaults`, `sampling-ranges` | CLI/request overrides, partial/null settings and range validation |
| `batch` | Independent requests across Chat, Responses and Completions; sessions 1–8 |
| `progress` | Opt-in progress on all text endpoints; output/sampling equality, limits, stops, images, batching and cancel/resume |
| `stream-start` | Plain streams on all text endpoints send headers before a cold prefill completes; a stream queued behind every session sends them after the five-second bound |
| `prefill-scheduling` | Short arrival during a cold prefill: work-aligned arrival, independent per-request timings, unchanged output and no repeated prefill. Requires sessions ≥2 and context ≥16384; not in `all` |
| `long-context` | Longer multi-turn recall, endpoint switching, sampled JSON and cancellation |
| `metrics` | Live slots, Prometheus cache/time/draft counters, uncached work, endpoint totals, queueing and cancellation |
| `cache` | Interrupted text/thinking/tool/image histories, ordinary and legacy tool names, RAM and disk restart; disk checkpoint spacing for a growing conversation and a branch restored after restart |
| `cache-edits` | Reuse earlier work after editing the latest message, shortening an older tool result, or editing an earlier user message and dropping later turns; compare with uncached responses |
| `cache-growth` | Keep cache reuse advancing over several turns when the client omits reasoning; check reasoning replay and thinking-off controls, including Messages thinking blocks, and compare with uncached responses |
| `cache-depth` | Histories beyond 16K, concurrent branches from rewritten replies, a side conversation, resume, unchanged retries and exact uncached controls; use context 32768 and also check sessions 1 |
| `cache-rotation` | Check cache RAM limits and keep history across conversations and small side requests; compare answers with uncached controls |
| `cache-shared-prefix` | New conversations under one system prompt, one after another, with long and short tasks: from the third on they restore the whole shared prefix; compare answers with uncached controls |
| `cache-bridge` | A chat bridge sends each user message with metadata its history copy drops, under a full RAM budget with small unrelated requests between turns: from the third turn on, reuse reaches the user turn two back; compare answers with uncached controls. Not in `all` |
| `system-injection` | System/developer messages after the conversation start, in Chat and Responses: accepted, followed by the model and equal to uncached responses; the hoisted turn's reuse is recorded and the next turn must reuse it in full |
| `cache-concurrency` | Concurrent identical prompts, shared-system fan-out with short and long tasks, short or no shared prefixes, a retained conversation beside a newcomer, and a cancelled leader; check waits, prefill work and uncached answers |

For `discovery` (also included in `all`), pass `--expected-input-modalities text` or `text,image` before
the server command. Projectors can load automatically beside the weights, so
the expectation is explicit rather than inferred from `--mmproj`.

For `image-inputs` and `image-count`, pass the model's `--mmproj`. Both use small
fixed images and are included in `all` only with that option. `image-count`
restarts the server for disk replay; its first four cases are 1/16-image timing
controls usable on older main with `--through-case image-count:image_count_control_16_True`.

Repeat `--suite` to select affected tests; `--suite all` explicitly runs all. For long
contexts, use server `--context 32768`; actual prompt depth is recorded. `cache`
uses its own 8 GiB disk budget and 1 GiB staging area inside the output directory;
the runner removes that disk cache when the run ends, keeping reports and logs.
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

For real coding-agent regressions, run `pi_agent.py` against a local server with
`--base-url`, `--model`, `--pi /path/to/pi-0.87.0`, `--server-log`, and a fresh
`--output` directory. Repeat `--case` to select affected tasks;
`--case literal-protocol --passes 1` exercises a real write/read/verify loop
containing literal ChatML vocabulary spellings.
It replays #368's five tasks, verifies the generated code independently, and
retains Pi sessions, HTTP/SSE and per-request timings. It executes generated
commands in disposable fixtures using isolated Pi configuration. Use `--passes 1`
for a focused check; the default five passes matches the reported debug workload.
`--conversation --context-file FILE` additionally tests retained long history.

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
Unchanged retries in `cache-growth` and `cache-depth` must prefill nothing,
unless the runner's server log shows that memory pressure refused the retry
copy; the retry may then prefill only the assistant opening after the stable
boundary.
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
