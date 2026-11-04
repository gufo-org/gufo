# Gemma 4 as a Gufo extension: migration plan

Status: planned 2026-11-04. Covers all three Gemma 4 checkpoints (31B, 31B QAT,
26B-A4B), their `gemma4-assistant` drafters and the vision sidecar.

## Goal

Move the Gemma 4 engine out of the `feat/gemma` fork branch into an independent
source extension, `franzmoca/gemma4-gufo-extension`, built against an unmodified,
pinned upstream Gufo. Structure follows
[`gufo-org/lfm3-gufo-extension`](https://github.com/gufo-org/lfm3-gufo-extension)
at `fb488aebef5abc3fefcbb21ce0d53eebe8cc3e27` and
[docs/extensions.md](../../extensions.md) on upstream `main`.

Context: [gufo-org/gufo#426](https://github.com/gufo-org/gufo/pull/426). The
maintainers asked to target `main`, pin both the Gufo and extension revisions,
and contribute missing generic hooks as focused `ext-api` PRs. Each PR carries
its own change record. Gemma's framing, grammar and defaults stay in the
extension. Open cards: [#977](https://github.com/gufo-org/gufo/issues/977) (tool
grammar/parsing), [#978](https://github.com/gufo-org/gufo/issues/978) (reasoning
delimiters), [#979](https://github.com/gufo-org/gufo/issues/979) (provider options),
[#980](https://github.com/gufo-org/gufo/issues/980) (sampling defaults), with
#979 sharing option plumbing with [#974](https://github.com/gufo-org/gufo/issues/974).

## Decisions

| Topic | Decision |
| --- | --- |
| Repository | `franzmoca/gemma4-gufo-extension` |
| Upstream baseline | `origin/main` at `d2b01f24` (v0.28.1-47, extension API series 0.1); re-pin once the PRs merge |
| Flash-Next / Qwen code Gemma links today | Copy the few kernels and helpers needed into the extension; drop the link to private family/model libraries |
| Upstream PRs | Prepared locally, **not opened until phase 6 passes**, so contract gaps found during the port are fixed in the PRs first |

## Branches

All in this repository (`franzmoca/gufo` fork):

- `upstream-main`: tracks `origin/main`, starting at `d2b01f24`. Never carries local
  commits.
- One branch per prepared upstream PR, each based on `upstream-main` (for example
  `ext/reasoning-markup`, `ext/tool-dialect`, `ext/provider-options`,
  `ext/sampling-defaults`, `ext/prompt-spans`, `core/cache-window-rows`, ...).
- `gemma-main`: `upstream-main` with every PR branch merged. The extension builds
  against this tree during development; when PR branches change, `gemma-main`
  is rebuilt by re-merging them.
- `feat/gemma`: frozen as the parity baseline; fork `main` keeps it merged until
  switch-over (phase 7).

The extension repository records the exact `gemma-main` revision it was validated
against; after the PRs merge upstream it pins an upstream revision instead.

## Inventory of `feat/gemma` outside `src/models/gemma4`

About 25k lines differ in `src/` from v0.28.1. `src/models/gemma4` (~12k) moves
nearly unchanged; the work is in the ~3.5k lines of shared files.

| Code on `feat/gemma` | Destination |
| --- | --- |
| `src/models/gemma4/**`, `tests/models/gemma4/**`, `tools/gemma4/**`, `tools/bench/gemma4_*.hip`, `docs/models/gemma-4-*` | Extension repository, includes rewritten to `<gufo/ext/...>` |
| `Gemma4TextRunner`, `Gemma4TextRunnerState`, `Gemma4ImageContext` in `src/cli/serve/inference_backend.cpp` | Extension `runner.cpp` on the public `ext::TextModelRunner` |
| `src/cli/prompt/gemma4_prompt.cpp`, Gemma parts of `src/cli/bench/bench.cpp` | Extension `drivers.cpp` (prompt, chat, bench drivers) |
| `OutputMarkup` / `Markup()` for reasoning delimiters, optional thought prefix | `ext-api` PR for #978 |
| `JsonConstraint::ToolFormat::kGemma4`, its grammar and the Gemma call parsing in `openai_chat` | `ext-api` PR for #977; grammar and parser code in the extension |
| `gemma4` sampling preset in `text_sampling_defaults.hpp`, `WithoutTopP` | `ext-api` PR for #980 |
| `--image-tokens`, `--draft-calibration` | `ext-api` PR for #979 |
| `TextPromptContext::PrefillStop` | New `ext-api` PR (see [Prefill hooks](#prefill-hooks)) |
| `TextModelRunner::SetPromptLookahead` | Dropped: the runner implements upstream's in-pass checkpoint instead (see [Prefill hooks](#prefill-hooks)) |
| MTP verification rounds in the metrics | Check against the public `TextDecodeStep::draft_*` counters; extension-side if they suffice |
| `gufo_qwen38_flash_next` link: `mtp_sampling.hpp`, top-64 proposal kernel, `draft_vocab.hpp`; Gemma's additions to Flash-Next `kernels.hip.cpp` | Copied into the extension with provenance notes |
| Qwen `hip/ops/gemm.hpp`, `hip/ops/token.hpp`, Gemma's additions to `prefill_quant_gemm.hip`, `quant_ops.hpp`, `gemm_route.hpp` | Copied into the extension; the Qwen libraries return to upstream state |
| `src/core/hip/*` (managed alloc, weight upload, utils), GGUF reader, SHA-256, sampling, prompt lookup | Public headers (`gufo/ext/experimental/*`, `gguf_reader.hpp`, `sha256.hpp`, `sampling.hpp`, `prompt_lookup.hpp`) |
| `src/cli/serve/logging.hpp` include | Replace with a public facility or drop |
| Cache: sliding-window row components, in-place restore, batched transfers (`src/cache/*`, `component_text_cache`) | Core PRs (model-neutral; DeepSeek V4.1 uses them too) |
| Capture join to the first decode batch (`TextRunnerPool::Request::AwaitSnapshot`, scheduler) | Optional core PR after switch-over (phase 8); not needed by the extension |
| Prompt-lookup agreement rule (`core/speculative/prompt_lookup.hpp`) | Core PR |
| 2 MiB pinned pages (`core/platform/pages*`) | Core PR, needs Qwen measurements first |
| Antialiased bicubic resize moved into `src/core/image` | Core PR, or copy into the extension if upstream prefers |
| JSON arrays-without-items fix (`json_constraint.cpp`) | Core PR |
| Q4_0/Q5_0 in `ggml_dequant`/`ggml_gemm` | Recheck against upstream #930 (all GGML block formats); core PR only for what is still missing |
| Functional `gemma4` preset, `templates.py`, Gemma expectations in `tests/functional/*.py` | Extension `tests/functional/` (as LFM3 does); generic harness fixes as a core PR |
| `model_bench` per-variant tables, self-reference, depth sampling | Core tooling PR |
| `tools/ci/kernel-resources.json` Gemma allowances, `check-docs.py` Gemma cards | Find how extensions declare kernel resources; otherwise an `ext-api` PR |

## Prefill hooks

`feat/gemma` adds three prefill/scheduling hooks to the server. Only
`PrefillStop` needs a new public contract:

- `PrefillStop`: required; new `ext-api` PR.
- `SetPromptLookahead`: replaced by upstream's existing in-pass checkpoint.
- `AwaitSnapshot`: an internal `TextRunnerPool::Request` method used by the
  scheduler's capture join, not part of the runner contract. Gemma runs
  correctly without it; it becomes an optional, model-neutral core PR at the
  end (phase 8).

### `TextPromptContext::PrefillStop(position, up)`: indivisible inputs

Gemma image soft tokens attend to each other bidirectionally: the rows of one image
must be evaluated in the same forward, together with its closing token. If
the pool ends a prefill step inside an image, its first rows are computed
without seeing the rest and the KV is wrong. The pool picks step boundaries
itself, from the token budget, RAM-cache checkpoints, the stable snapshot
boundary and shared-prefix boundaries, so the runner cannot simply refuse them.

The hook lets the model-owned prompt context move a candidate stop to a legal
position: back to the image's first row, or forward past its closing token
when the step starts at the image (that step then exceeds its budget). The pool
applies it to checkpoints, the snapshot boundary and each step's end. Without
it, images placed near a cache boundary or a prefill-budget edge produce
incorrect output or a checkpoint inside an image that cannot be restored.
The fix "Prefill never stops inside an image" (2026-10-08) relies on it.

The runner already handles its own token budget: `Prefill` shortens a step
before an image, or extends it past the budget to take the whole image. It
cannot fix positions the pool chooses: RAM-cache grid checkpoints (every 2,048
tokens), the component cache's planned captures, shared-prefix boundaries (two
different images sit behind identical placeholder tokens) and the step frontier.
The runner only sees `prompt.first(frontier)`; if the frontier falls inside an
image, no later step can finish that image either.

Upstream models do not need the hook: in Qwen 27B and Flash-Next the language
model is causal over image tokens. Their image embeddings are injected at any
position range (`vision_input_.Inject(..., start_pos, batch_size)`), even one
row at a time in batched decode and MTP, so every pool boundary is legal.
llama.cpp has the same constraint for Gemma: it aborts unless its micro-batch
holds the whole image.

Upstream's `TextPromptContext` carries only cache identity. The default
(`return position`) changes nothing for other models. For the PR, declaring the
spans as data may be preferable to a virtual: a sorted list of indivisible
`[first, end)` spans, with the pool applying the same snapping rule.

### `TextModelRunner::SetPromptLookahead`: replaced by in-pass checkpoints

The server stops prefill at the stable boundary to snapshot it. The chat
template's generation prompt (3–7 tokens) after that boundary then ran as its
own decode-width forward, reading every weight again: 35 ms on 26B and 112 ms on
31B per request. On `feat/gemma`, `SetPromptLookahead` hands the runner the
tokens past the frontier. `Session::Sync` evaluates them in the same forward,
keeps the session at the boundary for the snapshot, and the next step adopts
the rows. Measured on 26B UD-Q4_K_XL, HTTP pp2048: +6.3% at depth 0, +10.2% at
8K, +2.2% at 128K ([EXPERIMENTS](../gemma-4-26b-a4b/EXPERIMENTS.md),
"Generation prompt in the prompt's last forward").

The problem is not Gemma-specific, and upstream already solves it for Flash-Next
(#445). A runner declaring `in_pass_checkpoint` receives
`PrefillThrough(state, prompt, offset, max_input_tokens, snapshot_position,
&captured, capture)`: it prefills past the snapshot boundary and captures the
snapshot inside the pass. For Gemma this is straightforward: rows below the
boundary survive the rest of the pass (the ring keeps 2,064 rows beyond the
window), so the checkpoint needs only the boundary row's logits and hidden
state ([CONTINUATION-ADAPTER](CONTINUATION-ADAPTER.md), "In-pass checkpoints
generalize today's prompt lookahead").

The extension therefore implements `PrefillCheckpointBytes` and
`PrefillThrough`, sets `in_pass_checkpoint`, and drops `SetPromptLookahead`. No
`ext-api` PR is needed for this. To check during the port:

- **Disk cache.** Upstream skips in-pass checkpoints when the disk store is
  enabled (`!impl_->disk_store` in `TextRunnerPool`). The `gemma` server runs
  with disk cache on, so it would lose the gain. Find why the restriction exists.
  If it can be lifted, prepare a small generic core PR (`core/in-pass-disk-store`)
  that benefits Flash-Next too, with Flash-Next measurements.
- **Coverage.** In-pass covers the snapshot boundary only. The lookahead also
  ran ahead across grid checkpoints and shared-prefix boundaries, but the measured
  gain came from the generation prompt. Phase 6 compares the two directly.

## Phases

### 0. Baseline and repositories

- Create the `upstream-main` and `gemma-main` branches and the extension
  repository. The extension uses the LFM3 layout: `extension.json`,
  `extension.cmake`, `probe.cpp`, `provider.cpp`, `runner.cpp`, `drivers.cpp`,
  `hip/`, `tests/`, `docs/`, `flake.nix`.
- Capture the parity baseline on `feat/gemma` with the current toolchain. Keep
  every result.
  - Functional suites for 31B UD-Q4_K_XL, 31B QAT and 26B UD-Q4_K_XL, MTP and AR,
    including `long-context` and `cache`.
  - Decode and prefill KL against the FP32 reference.
  - The standard speed cells of each card.
  - A `pi_agent.py` replay.

### 1. Core PR branches (no `ext-api`)

- Cache window rows and in-place restore, prompt-lookup agreement, JSON arrays
  fix, bicubic resize, functional harness and `model_bench` tooling.
- If the disk-store restriction on in-pass checkpoints can be lifted,
  `core/in-pass-disk-store` (see [Prefill hooks](#prefill-hooks)).
- Check Q4_0/Q5_0 against #930 and the 2 MiB pages against Qwen.
- Mostly taken from the fork's `shared` branch, rebased on `upstream-main`.

### 2. `ext-api` PR branches

Order:

1. #978 reasoning
2. #980 sampling defaults
3. #977 tool grammar and parsing
4. #979 provider options
5. Indivisible prompt spans (`PrefillStop`)

Each branch adds an external fake-provider fixture that uses a dialect, preset or
option absent from core. It also adds a change record and runs
`tools/ci/check-extension-api.py --write`, the cards' focused tests and the
format check. Merge every branch into `gemma-main`.

### 3. Extension skeleton

- Manifest: a `gemma4` text provider for `gguf:gemma4`. The drafter is a
  sidecar of the target, not a separate provider.
- Targets:
  - CPU: config, tokenizer, chat template, tool syntax, probe.
  - HIP: kernels, engine, vision, the copied Flash-Next/Qwen kernels.
  - Text: runner, drivers, provider.
  - Test targets.
- Keep the per-file `-fno-fast-math -ffp-contract=off` on the projection
  kernels; verification rows only match single-token decode bit for bit with
  it.
- CPU-only registration keeps probes with null factories.
- Exit: the extension builds against `gemma-main`, passes the include-isolation
  checks, and `gufo diagnose --section extensions` lists the provider.

### 4. Runner, continuation adapter, vision, drivers

- Port the Gemma runner from `inference_backend.cpp` onto `ext::TextModelRunner`.
- Port the continuation adapter onto the public cache/continuation headers.
- Replace `SetPromptLookahead` with `in_pass_checkpoint`:
  `PrefillCheckpointBytes` and `PrefillThrough`.
- Vision goes through `supports_vision` and the prompt context.
- Port the prompt, chat and bench drivers.

### 5. Tests and tools

- Move the CTest targets to `gufo_extension_target(... role test)`. Functional
  expectations go to `tests/functional/`, replay and reference tools to `tools/`.
- Reference fixtures stay independent of the candidate.

### 6. Parity validation

Against the phase 0 baseline:

- Greedy outputs bit-identical.
- Decode KL unchanged.
- Affected functional suites pass, including `cache` and `long-context`, with
  loaded mode and actual drafts verified.
- `pi_agent.py` replay passes.
- In-pass checkpoints match the lookahead's gain: pp2048 at depth 0, 8K and
  128K on 26B and 31B, with RAM and disk cache. They also match the snapshot
  contents and first logits of the lookahead path.

Timings:

- Per request and phase at 5% / 10 ms, alternating baseline and candidate runs.
- One full card sweep at the end.

Fix gaps in the PR branches, not with workarounds in the extension.

### 7. Publish and switch over

- Open the core and `ext-api` PRs upstream.
- Move the model cards into the extension repository. Its README lists the
  maintainer, supported Gufo revisions, platforms, quant formats and the Nix
  pin.
- The `gemma` server script builds pinned Gufo plus the extension.
- Fork `main` stops merging `feat/gemma`.
- Update gufo-org/gufo#426 to point to the extension and close it as superseded.
- After the PRs merge, re-pin the extension to an upstream revision and raise
  `api_version` to the first revision with all the contracts it uses.

### 8. Optional: capture join

After switch-over, prepare the model-neutral capture join as a core PR: the
scheduler waits up to 10 ms (`AwaitSnapshot`) for a peer whose prompt capture is
finishing, and lets it join the batch's first decode. Upstream acceptance needs
measurements on an upstream model (Qwen 27B or Flash-Next, concurrent requests);
none are recorded yet.

## Open questions

- How extensions declare kernel scratch/VGPR allowances (`kernel-resources.json`).
- Whether the MTP verification metrics need more than the public `TextDecodeStep`
  counters.
- Whether upstream wants the bicubic resize in core or per model.
