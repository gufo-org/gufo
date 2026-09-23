# Benchmark methodology

Build with `nix build` and measure binaries under `result/bin`. Per-model
results and qualification gaps live in:

- [DeepSeek V4 Flash](models/deepseek-v4-flash/BENCHMARKS.md)
- [Qwen3.8 27B](models/qwen3.8-27b/BENCHMARKS.md)
- [Qwen3.8 Flash-Next](models/qwen3.8-flash-next/BENCHMARKS.md)
- [Qwen3-TTS](models/qwen3-tts/BENCHMARKS.md)
- [Qwen3-ASR](models/qwen3-asr/BENCHMARKS.md)
- [MiniMax H3](models/minimax-h3/BENCHMARKS.md)

## Model benchmark driver

`tools/bench/model-bench.py` measures the tables of a model's `BENCHMARKS.md`
over HTTP for Gufo and for the reference project of its category (llama.cpp
for Qwen; antirez/ds4 for DeepSeek), writes per-table artifacts, and renders the tables
and SVG charts between the `<!-- bench:<id> -->` markers. Workloads and table
layouts are declared in `docs/models/<model>/artifacts/bench.json`; the
`benchmark-model` skill in `.agents/skills` describes the procedure.

Reference runtimes are explicitly selected Nix packages, excluded from Gufo,
the normal development shell and hosted checks:

| Reference | Package | Executable |
| --- | --- | --- |
| Qwen AR / DFlash2 | `.#llama-cpp-reference` | `llama-server` |
| Flash-Next MTP | `.#llama-cpp-mtp-reference` | `llama-server-mtp` |
| DeepSeek AR / DSpark | `.#ds4-reference` | `ds4-server`, `ds4-bench` |

For example, enter the development shell and add only the needed baseline:

```sh
nix develop
nix shell .#llama-cpp-reference .#llama-cpp-mtp-reference
python3 tools/bench/model-bench.py --model qwen3.8-flash-next --gguf "$MODEL" \
  --mtp "$MTP" run --target reference --table single-mtp
```

DeepSeek's pinned HTTP server lacks separate pp/tg durations. Its native
`ds4-bench` supports single-session AR and DSpark with per-frontier CSV timing;
it does not implement concurrency. The pinned ROCm server also disables DSpark
in native batching. See the
[DS4 measurement status](models/deepseek-v4-flash/EVALUATION.md#benchmark-method)
before running its reference sweep.

New or refreshed text-model cards use the same pp2048 prose/copying prompts
for single-user d0 and concurrency, with tg128. The driver shares their prompt
generator and records prompt hashes. C1 should cross-check d0; investigate
input, cache, controller and timing differences before attributing a gap to
batching. Short repeated-word corpora remain useful quality probes, but their
rates cannot replace these matched workloads.

For decode comparisons, set `prefill_first: true` in the concurrency table.
Prepare every session with the full prompt and one output token, wait for all
preparations, then time identical tg128 requests with prefix reuse. The driver
pins llama.cpp slots and rejects replay exceeding four prompt tokens. This keeps
peers' long prefills outside llama.cpp's elapsed generation timer; Gufo reports
active decode time. Preserve a separate fresh AR completion-hash control.

## Direct and serving measurements

`gufo bench` measures the model path. Keep model artifact, prompt length,
generation length, context depth, speculative mode, and sampling controls
fixed when comparing runs. `pp2048` means 2,048 new prompt tokens; context
depth is separate from the amount of new prefill work.

`tools/serving/gufo-serving-bench.py` measures HTTP serving with synchronized
requests. It distinguishes server-stage prefill/decode throughput, scheduler
and client TTFT, token ITL, whole-request throughput, and aggregate concurrency
throughput. C1 serving and a direct single-user benchmark are comparable only
when prompt, cache state, sampling, speculation, and timed scope match.

Model tables report the **sum of individual request decode rates** in each
concurrent group, averaged across measured groups. Prefill and queue time
remain in the latency diagnostics. Missing decode timings are **TODO**;
whole-request throughput is not a substitute.

`--endpoint-profile openai` supports other OpenAI-compatible servers. Missing
server-stage metrics remain null. `--reference-report` compares completion
hashes against an existing C1 reference; category filters select a focused
subset. Qualify acceptance with a fresh server so prompt-cache hits do not
hide work. See [serving benchmark commands](PERFORMANCE.md) for invocation.

`tools/bench/speculative-corpus.py` compares direct autoregressive and
speculative token traces. Request correctness, acceptance, and end-to-end
speed are separate requirements.

## Quality method (matched-token, per position)

Use the independent references and model-owned limits described in
[testing](TESTING.md). Tokenize once and feed teacher and candidate the same
predetermined history. Compare full next-token distributions before appending
the next evaluation token. Free-running text cannot isolate numerical error
after the first differing choice.

```text
p_t = softmax(teacher_logits)
p_c = softmax(candidate_logits)
KL  = sum_v p_t[v] * (log p_t[v] - log p_c[v])
NLL = -log(p_c[target_token])
```

Record non-finite counts, normalized-logit error, KL, teacher-token
NLL/perplexity, top-1 agreement, and top-k overlap as applicable. Aggregates
are over scored positions. A changed tokenizer, template, suite, or history
changes the comparison identity.

Capability evaluation is separate: [gufo eval](EVAL.md) grades free-running
answers through the OpenAI-compatible server.

## Focused kernel measurements

Build development tools with `cmake --preset gpu-test` and
`cmake --build --preset gpu-test --target gufo-kernel-bench tune_hipblaslt`.
They are omitted from the production package.

`gufo-kernel-bench` exercises production HIP entry points without loading a
model. Correctness sentinels run outside the timed interval:

```sh
./build/gpu-test/gufo-kernel-bench --help
./build/gpu-test/gufo-kernel-bench \
  --case decode-attention --context 4096,8192,12288,16384 \
  --warmup 5 --repetitions 30 --json
```

Standalone microbenchmarks under `tools/bench` and model-specific tool folders
report component correctness alongside timing. Their numbers do not replace
end-to-end model measurements. Collect profiler traces separately and use
[the performance guide](PERFORMANCE.md) to identify the limiting operation.

## Reporting and artifact retention

Record model/engine revisions, artifact hashes, hardware/software fingerprint,
input identity, context, output count, sampling, concurrency, cache state,
oracle identity, and the exact timed scope. A numerical mismatch or device
failure is not a passing result. Add repetitions when timing noise or a
suspected regression requires them.

Checked-in JSON retains fixtures, model provenance, controller calibration,
and bounded result summaries. Serving summaries preserve aggregate metrics,
per-case completion hashes, and the original report digest. Raw samples,
profiler traces, generated tuning databases, and full-logit captures stay in
ignored `artifacts/`, except for independent reference data needed by a
maintained correctness test.

Serving reports exclude endpoint hosts, prompts, generated text, model paths,
raw token IDs, and timestamps. The canonical machine fingerprint excludes
hostnames, usernames, secrets, timestamps, and private paths.
`gufo diagnose --validate-artifact` validates its supported diagnostic schema;
each model/tool owns the validation of its own result schema.

Downloaded tokenizer files and `.direnv` shell state are local artifacts.
Count tracked source, including HIP translation units, with:

```sh
nix run nixpkgs#cloc -- --vcs=git --force-lang=C++,hip
```
