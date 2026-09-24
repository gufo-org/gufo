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

Establish the model, file paths, requested tables/rows and time budget from the
conversation before asking for missing information. Existing authorization and
paths remain valid. Prefer one warmed sample per point, run only the requested
sweeps, and repeat only to investigate a concrete discrepancy. Prefix setup at
64K/128K dominates runtime; avoid rebuilding it for unrelated checks.

Start long runs in the background with output redirected to a log. Inspect
progress and process status at short intervals while continuing independent
work; do not leave a run unreviewed. Before starting, establish
how loading will get cold model files. Use a privileged page-cache drop, or
`POSIX_FADV_DONTNEED` on every target/sidecar file and verify zero resident
pages with `mincore`. Record the method; cold model files do not imply cold
runtime libraries. Skip loading if neither method is available.

## Measurement model

Use **HTTP on both sides** by default so the prompt, timed scope
and transport are identical: Gufo through `gufo serve`, the reference through
its own OpenAI-compatible server. `gufo bench` and the standalone CLIs are
kernel-iteration tools unless the user authorizes a native reference benchmark.
For DeepSeek, the user permits `ds4-bench` single-session AR/DSpark comparisons.
Match prompt tokens, prefill interval, decode accounting and cache frontier first;
label the native timing scope. Never silently compare it with whole-request HTTP
wall time. `ds4-bench` does not implement C>1.

The driver is `tools/bench/model-bench.py`. `docs/models/<model>/artifacts/bench.json`
declares the category, variant ids, sweep grid, workloads and reference server
flags; model file paths are passed on the command line. The driver launches
the right server itself on a private port, runs one target at a time, writes
one JSON per table (and per mode for concurrency tables) into
`docs/models/<model>/artifacts/`, and renders the Markdown tables between
`<!-- bench:<table> -->` markers in `BENCHMARKS.md`:

```sh
nix build   # production Gufo binary at result/bin/gufo
FILES="--gguf q4=/path/to/target-q4.gguf --draft q4=/path/to/draft-q4.gguf"
nix develop -c python3 tools/bench/model-bench.py --model <model> tables            # ids and existing artifacts
nix develop -c python3 tools/bench/model-bench.py --model <model> $FILES run --target gufo --table single-ar-q4,single-dflash2-q4
nix develop -c nix shell .#llama-cpp-reference -c python3 tools/bench/model-bench.py --model <model> $FILES run --target reference --table single-ar-q4,single-dflash2-q4
nix develop -c python3 tools/bench/model-bench.py --model <model> render           # or render --check
```

`run` flags: `--table <id>[,<id>]` selects tables (without it every table
runs in `bench.json` order; loading and image-encoder are skipped with a
message when they cannot run); `--todo` measures only rows that have a
`TODO` in a cell owned by the target (row-granular: a missing tg cell
re-measures that workload's pp and tg too); `--fresh` discards the rows of
an existing artifact instead of merging into them — use it for a full
refresh so an interrupted run cannot leave mixed-date rows; `--repetitions N`
overrides every table's repetition count (mean ± sd above 1);
When a reference server is killed mid-table (SIGKILL is the kernel OOM
killer on this host), the driver marks that row and every deeper depth or
larger concurrency `unavailable` in the artifact and `render` shows `N/A`
for them, since they are a host limit rather than missing work.
`--depths 0,4096` restricts single-user tables to those depths and
`--mode ar` / `--mode <spec>` restricts to one mode (use it to skip a
reference speculative mode the reference cannot load);
`--drop-caches "<command>"` supplies a checked cache-reset command before
each loading launch; `--config` and `--artifacts-dir` point at an
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

Select reference runtimes explicitly: `nix shell .#llama-cpp-reference` for
Qwen AR/DFlash2, `.#llama-cpp-mtp-reference` for Flash-Next MTP, or
`.#ds4-reference` for DeepSeek AR/DSpark. Combine the two llama packages when
refreshing both modes. None is part of Gufo, the default dev shell or hosted
checks. Preserve the pinned recipes under `.devops/nix`; do not use ad-hoc builds.
Antirez readiness is `/v1/models`. Its HTTP payload lacks stage timings;
`tools/ds4/server_metrics.py` reads the pinned server's existing prompt-done
and final-decode timers, verifies token counts and retains the per-request
cohort durations. Do not substitute client wall time or infer client IDs from
interleaved log lines. Qualify prepared-prefix reuse before publishing C>1.

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
   seed/temperature, same context depth and same concurrency. Start a fresh
   server per depth sweep or concurrency point; allow only the prefix reuse
   declared by the workload. If a missing reference feature prevents a close
   match, mark those reference cells `N/A` and explain why.
3. Warm once, then time. Prepared decode cohorts use one output token to
   finish all prompts before timing tg128. Other corpus warmups use only the
   first concurrent group, capped at 16 output tokens. Single-user
   runs share tokenizer calibration across modes of the same target variant.
   One warmed sample per point is acceptable for a
   sweep; add repetitions when two runs differ by more than about 2% or a
   regression is suspected, and report mean ± sd. A sample whose generated
   token count is below the requested output length (context overflow, early
   stop) is not a result; the driver rejects it and names the table's
   `context` to raise. Run model jobs sequentially; nothing else on the GPU.
4. Before publishing a speed refresh of unchanged code, run the model's fast
   correctness suite when it has one (Qwen3.8 27B: `nix develop -c python3
   tools/qwen27b/check.py fast`; DeepSeek: `nix develop -c tools/ds4/check.py fast`;
   Flash-Next has no dedicated fast model suite). Contract tests and concurrency
   hashes do not replace independent model qualification; the full
   `EVALUATION.md` suites are for changed kernels or models. Greedy speculative output must match AR token
   IDs in the model checks. HTTP runs retain completion hashes: concurrency
   compares them against the Gufo AR C1 reference, while matching single-user
   AR/speculative depth rows allow the same text-consistency check. These hashes
   do not prove original-model accuracy. A mismatch or device fault is not a result.
   Qualify the reference's speculative path against its own AR on the matched
   pp2048 control before a full sweep, including prepared C1 reuse. If that
   equivalence fails, retain compact evidence and mark the comparison `N/A`.
5. A supported cell without a current qualified measurement is `TODO`, never a
   stale number; an unsupported comparison is `N/A`. Keep measurement dates;
   do not sum stage medians into a headline.
6. Retained JSON goes to `docs/models/<model>/artifacts/`; raw samples,
   traces, audio, images and videos stay in the ignored top-level `artifacts/`.

## Results-card format

Use six sections in this order: single-user AR, single-user speculative,
multiple-user AR, multiple-user speculative, loading time, memory occupation.
Keep only interpretation-critical notes beside the numbers. Exclude thinking
sweeps and image-plus-text prefill from the default text-model card. Keep
`EVALUATION.md` concise: key quality metrics/limits, essential reproduction
commands and measurement provenance; link detailed evidence under `artifacts/`.
Identify the actual implementation and precision behind each quality comparison.

Keep the results card numerical and concise. Put model, quantization and mode
in the first column header as well as the row dimension (depth/users/workload).
Throughput headers must say `tok/s`; loading and memory retain seconds and GiB.
Add `---` between each Q4 table/figure and its Q8 counterpart.
Keep acceptance statistics in artifacts, not Markdown columns or charts. Group
mixed/repetitive speculative results into one table and figure per quantization
for single-user depth sweeps and for concurrency. Single-user tables share pp
columns and show separate tg/gain columns per text type; choose the highest
measured pp per engine/depth across the workloads and state this beside the table.
Retain independent workload artifacts and TODO cells. Place **Loading time**
immediately before **Memory occupation**. Put commands and detailed methodology in
`EVALUATION.md`; keep only interpretation-critical notes beside the tables.

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
  is `N/A`. Missing reference features that prevent a close comparison are also
  `N/A`, with a concise reason. `TODO` means a supported measurement is pending. Declare `N/A` cells in `docs/models/<model>/artifacts/unavailable.json`
  (`{"<table id>": {"<row label>" | "*": "<reason>"}}`; the suffix
  `-speculative` on a concurrency table id targets only its speculative
  reference column) and state the reason under the table. `render` prints
  `N/A` there and in the matching Gain cell.
- Keep AR comparisons in the dedicated AR table. If the reference lacks a
  speculative mode, mark that mode's cells N/A; do not launch another AR
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
Qwen reference: **llama.cpp** `llama-server`, same GGUF, ROCm build,
`-ngl 999 -fa on --cache-reuse 0 --cache-ram 0 --jinja --reasoning off`,
`-np C` and total `-c` covering every session's context. `--cache-ram 0`
disables llama-server's separate host snapshot cache; live slot prefix reuse
remains available. On a unified-memory host, avoiding that extra allocation
can decide whether a deep sweep fits. When a reference row
still dies, check `free` and the kernel log before recording `N/A` — a run
that finishes only by evicting and re-reading its mapped weights is not a
speed measurement either.
Gufo: `gufo serve llm --think off --max-pending-per-client 8`, greedy, seed
1; the driver sets `--sessions 1` for single-user tables and `--sessions C`
(llama.cpp `-np C -c 4096·C`) for concurrency tables, and attaches `--mmproj`
only for the image-encoder table. The sweep parameters are identical across
the three models so the documents stay comparable.

DeepSeek reference: **antirez/ds4**, pinned by `.#ds4-reference`, same target
and DSpark GGUF. The server uses `--rocm --ctx <per-session-capacity>`;
AR C>1 adds `--batched-session C`. At pin `0aaea5a2`, any `--batched-session`
value disables DSpark on ROCm, including 1. Omit it for C1 DSpark; keep C>1
DSpark reference cells N/A until a build supports them. DSpark adds
`--dspark --mtp-model <support>`. Disable
thinking in each HTTP request. Native `ds4-bench` supports the sidecar but
uses one session; its `ctx_tokens` is the post-prefill frontier, not cached
depth, and `prefill_tokens` is the increment from the preceding frontier.
Reject short/EOS-truncated rows and retain full `gen_tps`, not steady-only rates.

Tables (append `-<quant>` for several quantizations, e.g. `single-ar-q4`);
place loading immediately before memory in the rendered card:

1. **Single user, autoregressive** (`single-ar`). pp2048 / tg128, C1, depth
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
   shipped configurations). d0 starts fresh; deeper points reuse a prefix.
   Check actual cache/prefill counts before accepting a measurement.
   `Depth | Gufo pp | llama.cpp pp | Gain | Gufo tg | llama.cpp tg | Gain`.
2. **Single user, speculative** (`single-<spec>`,
   DFlash2 / DSpark / MTP as the model supports).
   Same depth grid. Retain accepted draft tokens per step in the
   artifacts, without displaying them in the results card. Two workloads, because speculative decoding depends on the
   output: the mixed workload asks for a detailed summary
   and a story (generic prose, `workload: prose`); the repetitive workload
   asks the model to copy the passage word for word (`workload: repetition`).
   The AR depth table uses the prose task; keep one AR workload for each
   fixed depth and batch configuration instead of repeating it by text category.
   The table's `workloads` map retains `single-<spec>` and
   `single-<spec>-repetition` artifact identities. The runner measures each
   workload separately; `--todo` selects its missing tg cells or missing shared
   pp. The renderer selects the highest pp per engine/depth, preserving that
   measurement's standard deviation, and recalculates pp gain from the maxima.
   llama.cpp runs the same draft file through `--spec-type draft-dflash`,
   `draft-mtp` (`speculative.reference.args` in
   `bench.json`, otherwise llama.cpp's defaults; tune them only when the
   reference project documents better values, and record the change). When
   the pinned llama.cpp cannot load the sidecar (Qwen3.8-Flash-Next MTP with
   b11069: `check_tensor_dims: tensor 'token_embd.weight' not found`),
   `speculative.reference.server` in `bench.json` can name a parallel build
   for the speculative cells only — `flake.nix` exposes `llama-server-mtp`
   from the pinned ggml-org/llama.cpp#28243 branch — and the card must say the
   speculative reference comes from that branch. Without such a build,
   record the error once and mark its speculative cells `N/A`.
   `Model / depth | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive`.
   Retain proposed/accepted draft counts in artifacts, never in card columns
   or figures. Do not replace a missing speculative reference with another AR
   sweep; mark its cells `N/A` and explain the missing support.
3. **Multiple users, autoregressive**, followed by **4. Multiple users,
   speculative**. Use the **same pp2048 prompts as single-user d0**,
   tg128 and `C = 1,2,4,6,8`; context capacity 4096 per user is sufficient.
   `multi-ar` uses `prompt_tokens: 2048`, `workload: prose`, `modes: ["ar"]`.
   `multi-<spec>` inherits `prompt_tokens: 2048` and speculative-only `modes`;
   its mixed/repetitive workloads use `workload: prose` / `repetition` and
   `corpus_layout: homogeneous`. Every user receives the matching d0 prompt.
   The driver shares `turn_prompt` with the depth sweep, records its hash and
   calibration, and rejects wrong prompt counts or short output.

   C1 is a direct check against the single-user d0 row. Compare token counts,
   completion hashes, draft counts and server flags before interpreting timing
   differences. **Do not compare a short repeated-word corpus with a pp2048
   passage-copying task under the same labels.** Legacy short-prompt tables
   require a complete workload refresh, not relabeling old measurements.

   Start a fresh server per concurrency. Set `prefill_first: true` and
   `cache_prompt: true`: prepare **every session** with the exact prompt and
   one output token, wait for all preparations, then release the identical
   prompts together for one measured tg128 cohort. Pin llama.cpp requests to
   distinct `id_slot` values in both phases. Validate that each measured
   request reuses the entire prompt (at most four final tokens may be reevaluated)
   and generates all 128 tokens. Retain preparation evidence separately.
   Antirez DS4 uses a zero-output preparation request because generating a
   token advances its live cache past the prompt. Gufo DS4 first builds a C1
   prompt checkpoint, then prepares the concurrent sessions from it to keep
   prefill arithmetic consistent. Record both choices in the artifacts.
   llama.cpp's recurrent checkpoint is four tokens before the prompt frontier;
   other models may only need the final token for logits. This avoids mixing
   Gufo's active decode clock with llama.cpp's elapsed
   generation clock while peers perform long prefills. Do not silently reuse
   old cold-cohort results under this method. Check prepared C1 output/drafts
   against fresh single-user d0 and verify decode overlap in server timings.
   Report the sum of individual request decode rates, averaged across cohorts.
   Group the two
   speculative workloads into one table and figure per quantization; no extra
   AR performance columns or sweeps. Keep useful controller statistics in JSON.

   Save Gufo C1 AR completion hashes once per workload in
   `<table>-gufo-ar.json`: the mixed reference can come from the dedicated AR
   C1 measurement; repetitive needs only one AR quality request. Refresh these
   references when prompts, arithmetic, weights, tokenizer or settings change.
   To generate a missing reference, use a temporary `--config` containing that
   workload's `multi-*` table with `modes: ["ar"]`, `concurrency: [1]`,
   `prefill_first: false` and `cache_prompt: false`; keep its prompt recipe.
   Run that table on Gufo and retain only its C1 completion hashes and provenance.
   `--mode ar` alone filters configured modes; it does not add an AR run to a
   speculative-only table.
   Keep a fresh AR hash control independent of the prepared cache path.
   Gufo speculative output must match its AR reference before publishing.
   Cross-engine agreement belongs in `EVALUATION.md`, not as an accuracy score.
5. **Loading time** (`loading`). Cold model files to HTTP readiness (`/ready`
   on Gufo, `/health` on llama-server), at the same C1 context capacity and
   speculative mode. `Model / target | Gufo ready (s) | llama.cpp ready (s) | Gain`.
6. **Memory** (`memory`). Peak device-global HIP memory in use
   (`hipMemGetInfo` total − free, the counter Gufo's loader logs as
   `gpu_device_used_mib`, sampled every 250 ms by the driver through the
   `libamdhip64` the Gufo binary links) while pp2048+tg128 and a 16K-prefix
   pp4096+tg128 run, both servers autoregressive at the same context
   capacity, no projector loaded. The idle value before the server starts is
   stored per row. llama-server preallocates its KV cache, so its footprint
   does not grow with the prefix; say so under the table.
   `Workload | Gufo GiB | llama.cpp GiB | Gain`.
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
are `N/A`.

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
workload identity, and keep only concise interpretation notes beside them. Put dates, method,
server flags, artifact provenance and reproduction commands in `EVALUATION.md`.
Remove statements the new numbers contradict. Explain any remaining `TODO`
measurements and `N/A` comparisons separately. Update `EXPERIMENTS.md` only when a measurement changes a
retained decision. Summarize per table: Gufo, reference, best and worst gain,
and every cell where completion hashes or transcripts did not match.
