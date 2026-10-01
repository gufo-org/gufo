# Text API functional tests

The runner starts an isolated local server and checks real responses with the
OpenAI SDK. Use production binaries and local weights; no models are downloaded.

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
| `tools` | Required/named/auto, schemas, literal arguments and tool history |
| `auto-tools` | Focused subset for optional tool calls |
| `tool-edges` | Referenced argument types, literal CR, unusual keys and named Responses metadata |
| `tool-reasoning` | Quoted tool tags stay in reasoning; edit arguments stay intact across Chat, Responses, streaming and early stops |
| `state-edges` | Actual AR/draft execution, tiny thinking budgets, zero-argument tools, schema changes, stops, image retry and failed-request recovery |
| `structured`, `structured-limits` | Request JSON schemas, SDK parsing, limits and stops |
| `sampling-defaults`, `sampling-ranges` | CLI/request overrides, partial/null settings and range validation |
| `batch` | Independent requests across Chat, Responses and Completions; sessions 1–8 |
| `progress` | Opt-in progress on all text endpoints; output/sampling equality, limits, stops, images, batching and cancel/resume |
| `long-context` | Longer multi-turn recall, endpoint switching, sampled JSON and cancellation |
| `metrics` | Live Prometheus counters, uncached work, endpoint totals, queueing and cancellation |
| `cache` | Interrupted text/thinking/tool/image histories, RAM and disk restart |

For `discovery` (also included in `all`), pass `--expected-input-modalities text` or `text,image` before
the server command. Projectors can load automatically beside the weights, so
the expectation is explicit rather than inferred from `--mmproj`.

For `image-inputs`, pass the model's `--mmproj` in the server command. It uses
small fixed images and is included in `all` only when `--mmproj` is supplied.

Repeat `--suite` to select affected tests; `--suite all` explicitly runs all. For long
contexts, use server `--context 32768`; actual prompt depth is recorded. `cache`
uses its own 8 GiB disk budget and 1 GiB staging area inside the output directory.
For timing controls on revisions predating progress, use `--allow-missing-progress`
with `--record-baseline`. Candidate qualification always requires progress events.
Model runs stay outside hosted CI; CI checks the runner and measurement logic.
For metrics changes, run `--suite metrics` with AR and the affected speculative
mode. It checks all three text endpoints and reconciles cancelled work with the
terminal logs. Scrapes are not recorded as generation requests.

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
