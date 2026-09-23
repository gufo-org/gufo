---
name: benchmark-model
description: Measure every retained number in docs/models/<model>/BENCHMARKS.md for one Gufo model on Strix Halo gfx1151 over HTTP, run the same workload against the reference project's OpenAI-compatible server (llama.cpp, audio.cpp, ...), and report the gain per cell.
metadata:
  origin: gufo
---

# Benchmark a model

Fill or refresh `docs/models/<model>/BENCHMARKS.md` for one model. Which
tables exist and which external project is the comparison baseline depend on
the model category below. Read the model's `README.md`, `BENCHMARKS.md` and
`EVALUATION.md` first, then [docs/BENCHMARKS.md](../../../docs/BENCHMARKS.md)
for methodology. Follow the user's machine, time and Git instructions.

## Start

Ask the user before measuring anything:

1. Which model (`docs/models/<model>`) to benchmark.
2. Where the model files are: target GGUF or safetensors directory per
   variant, draft/MTP/DSpark support files, mmproj, audio fixtures. Model
   files live wherever the user keeps them; nothing in Git records paths.
3. Full benchmark (every table, every row, both targets) or only the rows
   with `TODO` cells in `BENCHMARKS.md`. A TODO-only refresh is valid only
   when the retained rows of that table came from this driver with the same
   workload; a table whose retained cells come from another method (`gufo
   bench`, an older prompt) must be refreshed whole, because one Gain column
   cannot mix two methods.

Then confirm the time budget. Measured on Strix Halo with Qwen3.8 27B:
a single-user depth table (0–32K, six depths) takes 4–7 min for Gufo and
about 10 min for llama.cpp; the 64K and 128K rows add roughly as much again
because the prefix itself must be prefilled; memory 1–2 min per target; each
concurrency table about 5 min per target and mode; loading about 1 min per
variant. One variant with both targets and both modes is about 1 h 20 min;
Q4 + Q8 took 2 h 41 min. When the budget cannot hold everything, measure in
this order and say what was left: Q4 before Q8, single-user AR, single-user
speculative, concurrency, memory, loading.

Driver commands run longer than a tool call may; start each one in the
background with its output redirected to a log, then block on that log with
an `until grep -q ... ; do sleep 30; done` loop (repeat the loop when it
times out) — a turn that ends while the driver runs does not resume by
itself. Progress lines are flushed as they happen. Before starting, check
whether the host lets you drop the page cache (`sudo -n true` or `doas -n
true`); without it the loading table is skipped, so say so up front instead
of discovering it in the run.

## Measurement model

Everything is measured **over HTTP on both sides** so the prompt, timed scope
and transport are identical: Gufo through `gufo serve`, the reference through
its own OpenAI-compatible server. `gufo bench` and the standalone CLIs are
kernel-iteration tools; their numbers are not published in the comparison
tables.

The driver is `tools/bench/model-bench.py`. `docs/models/<model>/artifacts/bench.json`
declares the category, variant ids, sweep grid, workloads and reference server
flags; model file paths are passed on the command line. The driver launches
the right server itself on a private port, runs one target at a time, writes
one JSON per table (and per mode for concurrency tables) into
`docs/models/<model>/artifacts/`, and renders the Markdown tables between
`<!-- bench:<table> -->` markers in `BENCHMARKS.md`:

```sh
nix build   # production Gufo binary at result/bin/gufo
FILES="--gguf q4=/path/to/target-q4.gguf --draft q4=/path/to/draft-q4.gguf --mmproj /path/to/mmproj.gguf"
nix develop -c python3 tools/bench/model-bench.py --model <model> tables            # ids and existing artifacts
nix develop -c python3 tools/bench/model-bench.py --model <model> $FILES run --target gufo --table single-ar-q4,single-dflash2-q4
nix develop -c python3 tools/bench/model-bench.py --model <model> $FILES run --target reference --table single-ar-q4,single-dflash2-q4
nix develop -c python3 tools/bench/model-bench.py --model <model> render           # or render --check
```

`run` flags: `--table <id>[,<id>]` selects tables (without it every table
runs in `bench.json` order; loading and image-encoder are skipped with a
message when they cannot run); `--todo` measures only rows that have a
`TODO` in a cell owned by the target (row-granular: one `TODO` acceptance
cell re-measures that row's pp and tg too); `--fresh` discards the rows of
an existing artifact instead of merging into them — use it for a full
refresh so an interrupted run cannot leave mixed-date rows; `--repetitions N`
overrides every table's repetition count (mean ± sd above 1);
When a reference server is killed mid-table (SIGKILL is the kernel OOM
killer on this host), the driver marks that row and every deeper depth or
larger concurrency `unavailable` in the artifact and `render` shows `n/a`
for them, since they are a host limit rather than missing work.
The thinking workload enables reasoning per request
(`chat_template_kwargs.enable_thinking`) rather than through a server flag,
so the cached prefix renders the same way; on Gufo its depth rows are
blocked by gufo-org/gufo#248 (the prefix cache misses when thinking is on),
so only d0 is measurable today.
`--depths 0,4096` restricts single-user tables to those depths and
`--mode ar` / `--mode <spec>` restricts to one mode (use it to skip a
reference speculative mode the reference cannot load);
`--drop-caches "<privileged command>"` gives the loading table its
page-cache drop (`sync; echo 3 > /proc/sys/vm/drop_caches`) on hosts without
passwordless sudo/doas; `--config` and `--artifacts-dir` point at an
alternative `bench.json` and output directory for experiments. A depth that
fails is reported, left as it was, and listed in the artifact notes; the
other depths are still stored. A partial
re-run under changed flags updates only the rows it measured; every row
records the exact server command it ran under. Server logs go to the
ignored `artifacts/model-bench/`. `render` keeps hand-entered Gufo cells that
no artifact covers and prints which rows those are; when a table was skipped
in a refresh, blank those cells to `TODO` by hand rather than leave an old
number next to a fresh reference. Reference cells always come from
artifacts. It draws one SVG chart per
table with data into
`docs/models/<model>/artifacts/charts/` (matplotlib, deterministic output),
placed as an image line right after the table. Tables stay the source of
truth for agents; charts are an addition for readers. `--no-charts` skips them.

Starting a new model: copy `assets/llm-bench-template.json` to
`docs/models/<model>/artifacts/bench.json` and
`assets/llm-benchmarks-template.md` to the card, replace the placeholders,
delete the tables the model does not have, and render. The template mirrors
the Qwen3.8-Flash-Next card, including which prose belongs next to each
table.

`bench.json` (`schema: gufo-model-bench/1`) holds: `category`; `model.id`;
`files` (the file roles the CLI must supply, with a description each);
`variants` (id, label, required roles; single-variant models use `default`
and unsuffixed table ids); `sampling`; `speculative` (mode, label, Gufo
arguments with `{role}` placeholders, and which reference mode it is compared
against); `tables` keyed by table id with that table's workload, and
`gufo.serve` / `reference.server` + `args` for the two servers. Readiness is
`GET /ready` on Gufo and `GET /health` on llama-server.

Reference servers (llama.cpp, audio.cpp, stable-diffusion.cpp) come from this
repository's `flake.nix`; do not use ad-hoc installs.

Request-level timing comes from the response, not the client clock, whenever
the server provides it: Gufo and llama-server both return llama.cpp-compatible
`timings` (`prompt_n` = newly processed tokens, `cache_n` = reused tokens,
`prompt_ms`, `predicted_n`, `predicted_ms`); Gufo adds `usage.gufo` stage
metrics. Servers without timings get client-measured whole-request wall time,
and the table says so.

## Rules for every category

1. Production binary: `nix build`, `./result/bin/gufo`. The driver records
   the Git revision, machine fingerprint, reference tool version, measurement
   date and every server command line (file paths reduced to basenames) in
   each artifact. You record by hand, in `artifacts/model-identities.json`
   or the card's prose: the Gufo binary SHA-256 prefix, the model files'
   repository/revision/quantization/hashes, and the reference backend
   (ROCm/Vulkan). Hosts, prompts, generated text and private paths never
   enter Git.
2. Same input, same artifact, same timed scope on both sides: same GGUF or
   safetensors, same prompt or audio file, same output length, same
   seed/temperature, same context depth, same concurrency, fresh server per
   point, no prompt-cache hits unless the cell says so. If the reference cannot
   match (different quantization, no equivalent speculative mode, no batching),
   measure the closest configuration and state the mismatch under the table.
3. Warm once, then time. One warmed sample per point is acceptable for a
   sweep; add repetitions when two runs differ by more than about 2% or a
   regression is suspected, and report mean ± sd. A sample whose generated
   token count is below the requested output length (context overflow, early
   stop) is not a result; the driver rejects it and names the table's
   `context` to raise. Run model jobs sequentially; nothing else on the GPU.
4. Before publishing a speed refresh of unchanged code, run the model's fast
   correctness suite when it has one (Qwen3.8 27B: `nix develop -c python3
   tools/qwen27b/check.py fast`; Qwen3.8-Flash-Next and DeepSeek V4 Flash:
   none today — say so in the card and rely on the concurrency `Exact`
   checks); the full `EVALUATION.md` suites are for changed kernels or
   models. Greedy speculative output must match AR token
   IDs. Concurrency runs compare every completion hash against the Gufo AR
   C1 reference and retain the counts in quality artifacts; single-user tables
   check token counts only. A mismatch or device fault is not a result.
5. A cell without a current qualified measurement is `TODO`, never a stale
   number. Keep dates per table; do not sum stage medians into a headline.
6. Retained JSON goes to `docs/models/<model>/artifacts/`; raw samples,
   traces, audio, images and videos stay in the ignored top-level `artifacts/`.

## Comparison columns

Every headline table carries the reference project next to Gufo:

```
| ... | Gufo <metric> | <ref> <metric> | Gain | ... |
```

- `Gain` is a percentage, positive when Gufo is better:
  - throughput (tok/s, audio s per wall s): `(Gufo / ref − 1) × 100`
  - latency, RTF, wall time, memory: `(ref / Gufo − 1) × 100`
- No difference column: both raw values are visible, and the extra column
  makes wide tables unreadable. When a cell is `mean ± sd`, Gain uses means.
- Name the reference in the header (`llama.cpp pp`, `audio.cpp RTF`).
- `TODO` on either side leaves `Gain` as `TODO`.
- A reference cell the host cannot produce for hardware reasons — the
  reference is OOM-killed at the required context, the model does not fit —
  is `n/a`, not `TODO`. A missing software feature (the reference cannot
  load a sidecar yet) stays `TODO` with the upstream reference noted, since
  a newer pin can fill it. Declare `n/a` cells in `docs/models/<model>/artifacts/unavailable.json`
  (`{"<table id>": {"<row label>" | "*": "<reason>"}}`; the suffix
  `-speculative` on a concurrency table id targets only its speculative
  reference column) and state the reason under the table. `render` prints
  `n/a` there and in the matching Gain cell.
- Keep AR comparisons in the dedicated AR table. If the reference lacks a
  speculative mode, leave that mode's cells TODO; do not launch another AR
  sweep for each speculative workload. Existing combined-mode cards label
  their fallback explicitly as `Gain vs <ref> AR`.
- One table per quantization and per mode; never pack `pp / tg` pairs or two
  quantizations into one cell.

The driver package is `tools/gufo/model_bench/` (`cli.py`, `llm.py`,
`render.py`, `charts.py`); `tools/bench/model-bench.py` is a shim.

Rendered tables sit between `<!-- bench:<table-id> -->` and
`<!-- /bench -->` markers so `render` can replace them in place; the prose
around them (dates, acceptance notes, caveats) is hand-maintained.

## LLM (GGUF text and vision models)

Models: `deepseek-v4-flash`, `qwen3.8-27b`, `qwen3.8-flash-next`.
Reference: **llama.cpp** `llama-server`, same GGUF, ROCm build,
`-ngl 999 -fa on --cache-reuse 0 --cache-ram 0 --jinja --reasoning off`,
`-np` equal to the largest concurrency, `-c` covering the deepest sweep
point plus 2048 + 128. `--cache-ram 0` matters: the driver sends
`cache_prompt=false`, so llama-server's 8 GiB RAM prompt cache is never
used, and on a unified-memory host it is the difference between a deep
sweep running and the kernel OOM-killing the server. When a reference row
still dies, check `free` and the kernel log before recording `n/a` — a run
that finishes only by evicting and re-reading its mapped weights is not a
speed measurement either.
Gufo: `gufo serve llm --think off --max-pending-per-client 8`, greedy, seed
1; the driver sets `--sessions 1` for single-user tables and `--sessions C`
(llama.cpp `-np C -c 4096·C`) for concurrency tables, and attaches `--mmproj`
only for the image-encoder table. The sweep parameters are identical across
the three models so the documents stay comparable.

Tables, in order (table ids in parentheses; append `-<quant>` when the
model has several quantizations, e.g. `single-ar-q4`):

1. **Loading** (`loading`). Cold-file-cache launch to readiness (`/ready` on
   Gufo, `/health` on llama-server)
   (drop caches, start the server, poll), then Gufo snapshot size and restore
   time for the documented prompt. Reference: `llama-server` readiness with
   the same GGUF. `Target | Gufo ready | llama.cpp ready | Gain`.
2. **Single user, autoregressive** (`single-ar`). pp2048 / tg128, C1, depth
   `0,4096,8192,12288,16384,32768`; add `65536,131072` when the model's
   context allows. Depth `d` is a cached conversation prefix: the driver sends
   a user turn of about `d` tokens answered with an 8-token generated reply,
   then the measured request continues that conversation with a new user
   turn of about 2048 tokens that ends with a request for a long continuation
   (so greedy decoding does not stop at EOS before 128 output tokens) and 128
   output tokens; Gufo reuses its prompt snapshot and llama-server its prompt
   cache (`cache_prompt=true`). Neither server
   exposes a tokenizer endpoint, so the driver calibrates the synthetic text
   (`prefix.generator`, seeded) against each server's reported
   `prompt_n`/`cache_n` and accepts a point only when `cache_n` is within
   `depth_tolerance` (at least 32 tokens) of `d` and `prompt_n` within it of
   2048; the actual counts are stored per sample. The table's `context` must
   be the deepest depth + 2048 + 128 plus a calibration margin (512 in the
   shipped configurations). The first turn of a fresh conversation (d0) is a
   prompt-cache miss on Gufo and decodes measurably slower than a cached
   continuation; d0 is therefore a different regime from d4096+, not noise.
   `Depth | Gufo pp | llama.cpp pp | Gain | Gufo tg | llama.cpp tg | Gain`.
3. **Single user, speculative** (`single-<spec>` and
   `single-<spec>-repetition`, DFlash2 / DSpark / MTP as the model supports).
   Same grid plus accepted draft tokens per step from each server's
   usage/timings. Two workloads, because speculative decoding depends on the
   output: the measured turn of `single-<spec>` asks for a detailed summary
   and a story (generic prose, `workload: prose`); `single-<spec>-repetition`
   asks the model to repeat the passage word for word (fully predictable
   output, `workload: repetition`, the single-user analogue of the
   `repetition` corpus). The AR depth table uses the prose task; keep one AR workload for each
   fixed depth and batch configuration instead of repeating it by text category.
   llama.cpp runs the same draft file through `--spec-type draft-dflash`,
   `draft-mtp` or `draft-dspark` (`speculative.reference.args` in
   `bench.json`, otherwise llama.cpp's defaults; tune them only when the
   reference project documents better values, and record the change). When
   the pinned llama.cpp cannot load the sidecar (Qwen3.8-Flash-Next MTP with
   b11069: `check_tensor_dims: tensor 'token_embd.weight' not found`),
   `speculative.reference.server` in `bench.json` can name a parallel build
   for the speculative cells only — `flake.nix` exposes `llama-server-mtp`
   from the open ggml-org/llama.cpp#28243 branch — and the card must say the
   speculative reference comes from that branch. Without such a build,
   record the error once, run the reference with `--mode ar`, and leave its
   speculative cells `TODO`.
   `Depth | Gufo pp | llama.cpp pp | Gain | Gufo tg | llama.cpp tg | Gain | Gufo accepted/step | llama.cpp accepted/step`.
   `accepted/step` is the mean number of accepted draft tokens per
   verification step, `draft_n_accepted / (predicted_n − draft_n_accepted)`
   (each step also yields one target token, so tokens per step is this plus
   one). It is proportional to the speculative speedup and independent of
   how many tokens were proposed, unlike an acceptance rate, which a
   controller that drafts fewer tokens inflates; Gufo's adaptive controller
   makes the rate meaningless as a comparison.
   When a model's `bench.json` has no `speculative.reference`, the reference
   column falls back to llama.cpp AR and is labelled `Gain vs llama.cpp AR`.
4. **Multiple users**. Measure AR once in `multi-ar` (`modes: ["ar"]`),
   using `repetition_word` and 128 output tokens to keep the batch full.
   `multi-mixed` and `multi-repetition` use only the model's speculative mode
   in `modes`, so they never schedule or report another AR performance sweep.
   Each table compares Gufo with the reference in that same mode:
   `Users | Gufo <mode> | llama.cpp <mode> | Gain`.
   Existing combined-mode cards remain supported until migrated.
   `C = 1,2,4,6,8`, context capacity 4096, fresh server per point,
   `cache_prompt=false`. Workloads come from
   `docs/models/qwen3.8-27b/artifacts/speculative-corpus.json`; mixed cases
   may end before the output cap, so retain actual token counts.
   Metric: sum of individual request decode rates, averaged across cohorts.
   Save isolated Gufo C1 AR completion hashes once per workload in
   `<table>-gufo-ar.json`; these are reusable quality references, not another
   performance table. Refresh them when target arithmetic, weights, tokenizer
   or request settings change. The driver rejects missing speculative case
   references before loading a model. Keep cross-engine agreement and batch
   consistency in `EVALUATION.md`; agreement is not an accuracy score.
   Reject cache-assisted corpus measurements when `cache_prompt=false`.
5. **Memory** (`memory`). Peak device-global HIP memory in use
   (`hipMemGetInfo` total − free, the counter Gufo's loader logs as
   `gpu_device_used_mib`, sampled every 250 ms by the driver through the
   `libamdhip64` the Gufo binary links) while pp2048+tg128 and a 16K-prefix
   pp4096+tg128 run, both servers autoregressive at the same context
   capacity, no projector loaded. The idle value before the server starts is
   stored per row. llama-server preallocates its KV cache, so its footprint
   does not grow with the prefix; say so under the table.
   `Workload | Gufo GiB | llama.cpp GiB | Gain`.
6. **Image encoder** (`image-encoder`, vision models only). Prefill time
   (`prompt_ms`) of a chat request with one 256×256 or 1024×1024 gradient PNG
   and a one-line text turn, `max_tokens` 1, no prompt cache, one warm-up and
   three timed samples, both servers loading the same BF16 projector. The
   time covers the projector encode plus the prefill of the image and text
   tokens; the encode alone is not separable over HTTP.
   `RGB image | Merged tokens | Gufo ms | llama.cpp ms | Gain`.

Concurrency performance artifacts are `gufo-serving-bench` corpus reports:
`multi-ar[-<quant>]-gufo-ar.json` and `multi-ar[-<quant>]-reference.json` for AR;
`<table>-gufo-<spec>.json` and `<table>-reference-<spec>.json` for speculative runs.
Speculative tables also retain `<table>-gufo-ar.json` as a C1 quality reference;
the other tables use the compact `model-bench-table` schema with one entry per
row and the actual `cache_n`/`prompt_n` counts.

## ASR (audio to text)

Model: `qwen3-asr`. Reference: **audio.cpp** server (`qwen3_asr` model spec,
ROCm), `/v1/audio/transcriptions`, same WAV, greedy. Both transcripts are
compared against the official reference transcript.

1. **Loading.** Cold-file-cache readiness; first-request and warm latency for
   a short clip.
2. **Single request.** The documented fixed recording (15.05 s English), one
   warmup, three timed requests, median. Rows: request time, RTF (request /
   input duration, lower is better), audio seconds per wall second; Gufo-only
   rows for audio-encoder time and text prefill+generation from `usage.gufo`.
   Columns: `Metric | Gufo | audio.cpp | Gain`.
3. **Concurrent HTTP.** C = 1,2,4,6,8, same recording: per-request latency
   median / p95 and aggregate audio seconds per wall second, both servers.
4. **Long-form.** One recording over ten minutes: wall time, RTF, transcript
   diff between the two implementations and against the reference.
5. **Profile** (Gufo only). Dispatches, GPU work, GPU-busy share, projection
   share from `tools/prof/prof.py --stages qwen-asr`, run separately.

## TTS (text and reference audio to speech)

Model: `qwen3-tts`. Reference: **audio.cpp** server (`qwen3_tts` model spec,
ROCm), `/v1/audio/speech`, same text, seed, sampling and variant
(CustomVoice / VoiceDesign / Base ICL); variants audio.cpp does not expose
are `n/a`.

1. **Loading.** Cold-file-cache readiness per variant; first-request and warm
   latency for a short sentence.
2. **Single request.** The documented 37-word paragraph, 64 codec frames
   (5.12 s), one warmup, three timed, median. Per variant:
   `Variant | Gufo WAV | audio.cpp WAV | Gain | Gufo RTF | audio.cpp RTF | Gain | First PCM | Complete PCM`.
   RTF = request time / generated audio duration. Gufo streamed and buffered
   PCM must be identical; seeded runs must repeat exactly. First-audio latency
   is measured on the streaming endpoint of each server when it has one.
3. **Transports** (Gufo only). PCM HTTP, SSE, WebSocket first-audio and
   complete times; byte-identical PCM across transports.
4. **Concurrency.** C = 1,2,4,6,8 completion times on both servers;
   execution is serialized, report the queue effect honestly.
5. **Profile** (Gufo only). Dispatches, GPU work, GPU busy, projection share
   per variant from `tools/prof/prof.py --stages qwen-tts`.

Natural-EOS duration and intelligibility belong to `EVALUATION.md`.

## Image generation and editing

Model: `qwen-image-2.1`. Reference: **stable-diffusion.cpp** server on ROCm
(`/v1/images/generations`) with the same checkpoint when it supports the
model; otherwise an OpenAI-compatible wrapper around the official Diffusers
pipeline on ROCm, labelled `Diffusers (ROCm)`. Same prompt, seed, size,
steps, scheduler and guidance.

1. **Loading.** Metadata readiness, first generation including component
   loading, and the same request warm.
2. **Generation and edit, C1.** 1024×1024, the default 40 steps, seed 42.
   Rows: generation; edit with one reference image (edit only where the
   reference supports it). Columns:
   `Mode | Steps | Prompt (s) | Denoising (s) | VAE (s) | Gufo total | <ref> total | Gain`;
   stage columns are Gufo `usage.gufo` values. Add 512×512 / 20 steps as a
   second control when the reference is slow.
3. **GPU split** (Gufo only). Native projections, fused feed-forward, fused
   attention, convolution and idle share from a separate request-only profile.
4. **Concurrency.** Requests are queued; report C2 completion times and state
   that no batching is implemented.

A two-step smoke check is a development control, never the headline. Image
quality is measured in `EVALUATION.md`.

## Video / audiovisual generation

Model: `minimax-h3`. Reference: **audio.cpp** server (`minimax_h3` model
spec) if it executes on ROCm; verify first and fall back to a wrapper around
the official Diffusers pipeline, labelled. Full generation requires the user's
explicit approval and `--allow-full-generation`.

1. **Loading.** Metadata readiness; first prompt-encoder layers time; peak
   retained device memory.
2. **Presets.** Per preset (`exact 512x512`, `exact-1344x768`, `fast`,
   `aggressive`): single-request wall, `first_preview_ms`, peak resident
   memory, delivery-quality note. Columns:
   `Preset | Gufo wall | <ref> wall | Gain | Gufo peak GiB | <ref> peak GiB | Gain`.
3. **Phase split** (Gufo only). Prompt encoding, AdaLN precompute, denoising,
   VisualVAE, AudioVAE, mux, from `tools/gufo/h3_profile.py`.
4. **Component inventory.** Tensor bytes per component; never summed as if
   resident together.
5. **Queue.** One worker; report C2 queue latency and say there is no
   parallel execution.

## Finish

Render the tables, check that every Gufo/reference pair used the same
workload identity, and rewrite the prose around each table: dates, method,
server versions and flags, artifact file names, caveats such as a skipped
loading table, and the reproduction block with the exact `model-bench.py`
commands (paths as placeholders). Remove statements the new numbers
contradict. List the remaining `TODO` cells with the reason (tool missing,
reference unsupported, time budget). Update `EXPERIMENTS.md` only when a measurement changes a
retained decision. Summarize per table: Gufo, reference, best and worst gain,
and every cell where completion hashes or transcripts did not match.
