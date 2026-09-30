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
The first run records a reference; the second qualifies the change.

Use `deepseek4` for DeepSeek. Pass the model's normal speculative options for
DFlash2, MTP or DSpark. Server sampling arguments become the expected defaults;
`--mmproj` enables image cases. Keep informational server logging enabled.

| Suite | Checks |
| --- | --- |
| `responses` | SDK buffered, streaming and async Responses |
| `stops` | Text, Unicode, reasoning and tool stops; peer isolation |
| `conversation` | Thinking/efforts, images, cancellation and RAM reuse |
| `tools` | Required/named/auto, schemas, literal arguments and tool history |
| `auto-tools` | Focused subset for optional tool calls |
| `structured`, `structured-limits` | Request JSON schemas, SDK parsing, limits and stops |
| `sampling-defaults`, `sampling-ranges` | CLI/request overrides, partial/null settings and range validation |
| `batch` | Independent requests across Chat, Responses and Completions; sessions 1–8 |
| `long-context` | Longer multi-turn recall, endpoint switching, sampled JSON and cancellation |
| `cache` | Interrupted text/thinking/tool/image histories, RAM and disk restart |

Repeat `--suite` to select affected tests; omitting it runs everything. For long
contexts, use server `--context 32768`; actual prompt depth is recorded. `cache`
uses its own 8 GiB disk budget and 1 GiB staging area inside the output directory.
Model runs stay outside hosted CI; CI checks the runner and measurement logic.

Every request checks its applicable response format, expected output and timings.
Missing measurements fail. `comparison.json` reports per-request prefill, decode,
queue, restore and wall times, plus server startup/restart. Missing cases,
changed output/token counts, or slowdowns exceeding **both 5% and 3 ms** fail
qualification (`--max-slowdown`, `--noise-ms`). Investigate flagged steps; repeat
only when needed to separate a regression from timing noise. **Each step must
pass independently; faster steps never compensate for slower ones.**

Reports and logs survive failures. These checks complement the standard speed
benchmark and numerical quality tests; they do not establish upstream model
parity. Sampled DSpark may differ across concurrency levels; fixed-path replay
and greedy equality remain checked. API behavior and references are in
[SERVER.md](../../docs/SERVER.md).
