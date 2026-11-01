# RFC: model extensions

Status: draft for maintainer review. Written 2026-10-30 against main
`5d66de8c`; revised after nine reviews the same day. Nothing in this document
is implemented. Inventories describe the inspected revision; counts are
reproducible with the commands in the [evidence appendix](#evidence-appendix),
and estimates are labelled as such.

This is the single review document for turning Gufo models into build-time
extensions. It records the current coupling, the alternatives considered,
the recommended design, its stability policy and a delivery plan made of
small, testable PRs.

Contents:

- [Context](#context) and [problem statement](#problem-statement).
- [Goals](#goals-and-non-goals) and [glossary](#glossary).
- [Current system](#how-the-system-is-designed-now), with a coupling inventory.
- [Evaluated solutions](#evaluated-solutions), including prior art and a
  comparison table.
- [Recommended design](#recommended-design-build-time-source-extensions).
- [Interface stability policy](#interface-stability-policy).
- [Distribution](#distribution) and [quality tiers](#quality-tiers-and-conformance).
- [Success criteria](#success-criteria-and-evaluation) and [risks](#risks).
- [Implementation plan](#implementation-plan) and [rollout](#compatibility-and-rollout).
- [Decisions](#open-questions), [evidence](#evidence-appendix) and
  [references](#references).

## Context

Gufo serves a growing set of models on AMD Strix Halo (`gfx1151`): text
generation (DeepSeek V4 Flash, Qwen3.8 27B, Qwen3.8 Flash-Next), embeddings
(EmbeddingGemma 2), speech recognition (Qwen3-ASR), speech synthesis
(Qwen3-TTS), image generation (Qwen-Image-2.1) and audiovisual generation
(MiniMax H3). Each model ships hand-optimized kernels, its own numerical
contracts and, for text models, speculative decoding and continuation-cache
integration.

Every model and quantization added today raises the maintenance load of the
whole project: a new model edits the CLI, the server, core enums, the build,
CI and the test registries, and every later change to those shared layers
must be checked against every model. Maintainers want a new model to arrive
as a self-contained unit that registers itself, so that integrating it
requires the minimum number of PRs to the core, ideally none beyond a
registration.

Each Gufo process serves one model, together with its own draft, MTP head or
vision sidecar when those are enabled. Official artifacts are Nix outputs,
OCI images built by the Toolboxes repository and a Windows ZIP
([release process](../RELEASING.md)).

## Problem statement

Model-specific knowledge is spread across shared layers. The consequences:

- Adding a model requires edits to `src/cli/` (serve, prompt, bench,
  kernel-bench, transcribe, video), core enums in `src/core/`, the root
  `CMakeLists.txt`, `cmake/Checks.cmake`, `CMakePresets.json`, GitHub
  workflows, Nix files and the functional-test registries. Reviews of a new
  model therefore also review the core.
- Shared files grow with every model. `inference_backend.cpp` (4,070 lines)
  contains every text model's runner; `openai_chat.cpp` (4,191 lines) scans
  per-model tool-call markers.
- The model/core boundary is implicit and moves often. Core serving features
  change the per-model runner interface regularly.
- Out-of-tree experimentation, such as a contributor's fork carrying a new
  architecture, cannot be combined with upstream without merge conflicts in
  the same shared files.
- Nothing prevents a model from depending on another model's internals or
  on the CLI: four models include Qwen sources, DeepSeek includes server
  logging, and nine model test files include `src/cli/` headers.

A plugin mechanism alone does not reduce kernel-tuning work per model and
quantization. That cost is addressed separately (see
[non-goals](#goals-and-non-goals) and [host services](#host-services)); this
RFC removes the integration and coupling cost and creates the seam that lets
shared kernels be reused.

## Goals and non-goals

Goals:

- A model is an extension: one directory with its sources, kernels, tests,
  documentation and manifests. Building and testing an in-tree model
  **within an existing capability** touches no file outside its own
  directories: no CLI, core, CMake, preset, workflow, Nix or registry edits,
  except new flake inputs, new modules in the shared `tools/gufo/` Python
  package and new Windows dependencies in `vcpkg.json`. A new extension
  absent from `extensions.lock` counts as unshipped by default.
  Shipping it in official artifacts additionally takes a one-line edit to
  the pinned extension list. Committed generated files (the models index,
  third-party notices, the merged functional registries) are regenerated
  for both testing and shipping; model-specific functional
  suites that
  [AGENTS.md](../../AGENTS.md) keeps in `tests/functional/` remain there.
- An out-of-tree extension builds into Gufo with zero changes to the Gufo
  repository, through CMake and Nix.
- Registration is explicit, generated by the build, and debuggable. No
  reliance on static-initialization side effects.
- The boundary is at step level: one call performs a whole prefill pass,
  decode step, decode batch, embedding batch or synthesis chunk. There are no
  per-layer or per-kernel calls in the serving interfaces, so the boundary
  costs no measurable performance.
- A versioned interface whose evolution is mostly additive, with a written
  stability policy and measurable criteria for declaring it stable.
- Official artifacts keep working as today. Third parties can publish their
  own images with a documented recipe.
- The core keeps owning HTTP/OpenAI semantics, cross-request scheduling for
  text, the
  continuation cache policy and budgets, diagnostics and the functional
  harness.

Non-goals for the initial implementation:

- Runtime loading of binary plugins (`.so`/`.dll`). Kept as a later,
  separately evaluated layer on the same interface.
- A stable ABI. Only source (API) compatibility is promised.
- New capability kinds without core changes. Capabilities are core-owned;
  adding one (for example realtime speech-to-speech) is a core change.
- A generic graph or IR model format that removes model code.
- Reducing per-model kernel tuning. The design must not block a shared
  kernel library, but building one is separate work.
- Hosting several unrelated models in one process.
- Changing any model's numerical behavior or cache identity. Migration is a
  refactor and must be byte-identical where the lossless-precision gate
  applies.

## Glossary

| Term | Meaning |
| --- | --- |
| Extension | A unit of model code, with manifests, built into Gufo and registered through the generated registry. |
| In-tree extension | An extension under `src/models/<name>/` in the Gufo repository. |
| Out-of-tree extension | An extension in another repository, added with `GUFO_EXTENSION_DIRS` or the Nix `extensions` argument. |
| Metadata manifest | `extension.json`: declarative facts (name, versions, capabilities, keys, options, conditions, packages, tests) readable by CMake, Nix and Python without building. |
| Build manifest | `extension.cmake`: the build logic (targets, sources, HIP options) for the extension. |
| Registry | The build-generated table that maps artifact keys to extension providers. |
| Provider | One loadable model kind registered by an extension, for one capability. |
| Capability | A core-owned family of behavior with its own interface: text generation, embeddings, speech recognition, speech synthesis, image generation, video generation. |
| Artifact key | A matchable identity of a primary artifact, such as `gguf:qwen4exp` or a safetensors `architectures[0]`. |
| Probe | A provider's pure, HIP-free check that confirms it can load an artifact matched by key. |
| Artifact set | The files a provider loads: primary file, shards, sidecars (such as `--mmproj`) and draft artifacts. |
| Family library | A shared library of code used by several extensions of one model family, such as Qwen control tokens. |
| Host services | Core facilities an extension may call: allocation, staging, snapshot backing, transfers, cache contracts, sampling and constraint state, logging, tracing, kernel plans. |
| Extension API | The versioned C++ headers under `include/gufo/ext/`, the CMake functions and the metadata schema extensions use. |
| Conformance suite | Capability-level functional tests every extension must pass. |
| Tier | The support level of an extension: core or community. |

## How the system is designed now

### Build and registration

Models are built in three different ways:

- **Separate static libraries:** DeepSeek V4 Flash (only under
  `ENGINE_ENABLE_HIP`), Qwen3.8 Flash-Next, Qwen3-TTS, Qwen3-ASR,
  EmbeddingGemma 2 and Qwen-Image-2.1 build their own `gufo_<name>` targets.
  Flash-Next builds its library and CPU tests without HIP too; the hosted
  CPU `check-pr` runs some of them.
- **Compiled into `gufo_core`:** the whole Qwen 27B model (tokenizer, chat
  template, forward pass, HIP kernels, vision encoder, continuation adapter)
  is added with `target_sources(gufo_core PRIVATE ...)` from
  `src/models/qwen/CMakeLists.txt`. MiniMax H3's sources are listed in the
  root `CMakeLists.txt` and also compiled into `gufo_core`, as are
  `src/eval/{dataset,extract,regrade}.cpp`, `src/testing/compare` and, under
  HIP,
  `src/core/speculative/speculative_verifier.cpp`, whose header includes the
  Qwen executor, generator, DFlash policy and tokenizer.
- **Inside the HTTP library:** `src/models/deepseek_v4_flash/tool_calls.cpp`
  is compiled into `gufo_http`.

`gufo_core` therefore compiles and exposes Qwen 27B and H3 to every
consumer. The repository root is a public include directory of almost
every target (`gufo_core`, `gufo_platform`, `gufo_cache`, `gufo_hip_host`,
`gufo_logging`, `gufo_diagnostics`, `gufo_kernel_bench_support` and the
model libraries); every source includes by root-relative
path (`#include "src/models/<name>/..."`), so nothing restricts which headers
a model includes. CMake cannot remove a public include directory for only
some consumers.

There is no registry. The CLI selects a model either by subcommand
(`gufo serve llm|embedding|tts|asr|image|video`) or, within `llm`, by reading
`general.architecture` from the GGUF and branching: `deepseek4`, `qwen4exp`,
otherwise Qwen 27B as the fallback for any other architecture. Within Qwen,
vision support and sampling presets are chosen by tensor shape
(`qwen35.embedding_length == 5120`; 64 layers, hidden size 5120 and vocabulary
248,320). `prompt` and `bench` repeat their own architecture checks and load
models directly, without the server's runner layer. `kernel-bench` calls
individual Qwen kernels. `eval` is an HTTP client and loads no model; its
only model coupling is the DeepSeek fixture path compiled in through
`GUFO_EVAL_SOURCE_DATA`.

### The text-generation boundary today

The real per-model boundary for text is `TextModelRunner`
(`src/cli/serve/text_model_runner.hpp`), not `TextGenerationBackend`.
`TextGenerationBackend` is the request-level interface the HTTP layer uses,
and it has one implementation, `InferenceBackend`. Below it,
`TextGenerationScheduler`, `TextRunnerPool` and `ComponentTextCache` drive
per-model runners through `TextModelRunner`'s 46 virtual methods:

- Prefill: `Prefill`, `PrefillThrough` with in-pass checkpoint callbacks.
- Decode: `SelectNext`, `Advance`, `DecodeStep`, `DecodeBatch`,
  `AdvanceBatch`, which take core-owned `SamplerState` (grammar constraint,
  reasoning budget, stop handling, logprobs).
- State: `Snapshot`, `RestoreOrFork`, persistent-snapshot serialization,
  streaming and restore.
- Resources and planning: `ResourceClaim`, `SupportedPlans`, `PollDevice`,
  `BuildConstraintVocabulary`.
- `TextRunnerCapabilities`, which tells the scheduler whether incremental
  prefill, snapshots, forks, batched multi-token decode (with a width
  limit), in-pass checkpoints, partial prefill retention and prefill
  rollback are available.

The implementations `QwenTextRunner`, `DeepSeekTextRunner` and
`QwenFlashNextTextRunner` all live in `inference_backend.cpp`, on top of a
shared `HipTextModelRunner`, which only does device probing (`PollDevice`,
`DeviceUsable`). Next to it, the template `CreateHipComponentCacheResources`
holds the component cache's pinned-host backing (segment sizing, commit
growth, host-reserve pressure retries, a 2 GiB pre-readiness reserve) for
Qwen 27B and Flash-Next. It calls a method that is not part of
`cache::Adapter`, `BackingPlan(budget)`, which each model's adapter
implements with its own `ContinuationBackingClass` struct: an unlisted
model/core contract. DeepSeek's native snapshots and all three runners'
`ResourceClaim` use the same host budget functions
(`HostSnapshotBudgetBytes`, `HostSnapshotCeilingBytes`).

Each model also builds its cache compatibility identity there, as one
model-specific serialization in which core-owned lines (schema, artifact and
draft fingerprints, context tokens, decode concurrency, draft limits) are
interleaved with model lines. A second, process-local identity
(`model_instance=<pointer>`) keys RAM reuse within one process.

The header is not only the runner contract. It also holds RAM and disk cache
options, memory-budget functions and `TextRunnerPool`, and it depends on
`ChatRequest`, `TextGenerationBackend::InitialOutputState`,
`ContinuationState`, `cache::ResourceLedger` and `ComponentCacheResources`.

The runner contract also depends on serve-layer types:
`TextRunnerState` and `TextRunnerSnapshot` derive from `ContinuationState`
and `ContinuationSnapshot` (`src/cli/serve/continuation_cache.hpp`), and the
header includes `text_publication_wait.hpp`, `TextGenerationBackend` and
`sampling::JsonConstraint::ToolFormat`. Its out-of-line defaults
(`text_model_runner.cpp`), the continuation cache, the scheduler and
`inference_backend.cpp` are compiled directly into the `gufo` executable,
and model tests compile those sources into their own binaries; there is no
serving library.

The contract moves often. In the 30 days before 2026-10-30, the
`TextModelRunner` class body changed in 12 commits, 11 of them substantive
(the whole header in 28, including cache options and pool changes that did
not affect runners), and `text_generation_backend.hpp` in 22. Commits that
changed the runner's virtual methods include #524, #573, #445, #821 and
#683.

### Continuation state per model

Prompt reuse reaches models through two paths:

- **Component cache adapter:** Qwen 27B and Flash-Next implement
  `cache::Adapter` (`src/cache/adapter.hpp`): components, compatibility
  identity, slots, capture/restore and prefill planning.
- **Native runner snapshots:** DeepSeek V4 Flash has no `cache::Adapter`. It
  provides `snapshot`, `fork` and `prefix_reuse` through
  `DeepSeekTextRunnerSnapshot` and a persistence descriptor in its runner.

The [continuation cache RFC](../cache-redesign/RFC.md)'s all-model cutover is
still open, so both paths are current product behavior.

### Coupling inventory

References to each model outside `src/models/`, under `src/cli`, `src/core`,
`src/cache`, `src/eval` and `src/testing` (matching lines from the
[inventory command](#inventory-commands); the count includes comments and
depends on the pattern, so it indicates where coupling lives rather than
exact effort). The root `CMakeLists.txt` mentions models on 225 of its 1,366
lines.

| Model | Files outside its directory that reference it |
| --- | --- |
| DeepSeek V4 Flash | `serve/inference_backend.cpp` (34), `bench/bench.cpp` (18), `serve/openai_chat.cpp` (15), `prompt/prompt.cpp` (14), `serve/inference_backend.hpp` (2), `eval/eval.cpp` (1) |
| Qwen (27B and shared) | `serve/inference_backend.cpp` (27), `prompt/prompt.cpp` (13), `bench/kernel_bench_main.cpp` (6), `core/speculative/speculative_verifier.hpp` (4), `bench/bench.cpp` (3), `core/quant/ggml_dequant.hpp` (2, comments), `serve/text_generation_backend.hpp` (2), `serve/inference_backend.hpp` (1), `core/speculative/draft_backend.hpp` (1), `serve/http_server.cpp` (1), `serve/openai_chat.cpp` (1), `core/quant/ggml_gemm.hpp` (1, comment) |
| Qwen3.8 Flash-Next | `serve/inference_backend.cpp` (16), `prompt/prompt.cpp` (8), `bench/bench.cpp` (4), `bench/bench.hpp` (2), `serve/inference_backend.hpp` (2) |
| Qwen3-TTS | `serve/tts_service.cpp` (23), `serve/tts_service.hpp` (13), `serve/audio_tts_api.cpp` (12), `serve/tts_websocket.cpp` (5), `serve/serve.cpp` (2) |
| Qwen3-ASR | `serve/asr_service.cpp` (13), `transcribe/transcribe.cpp` (9), `serve/asr_service.hpp` (7), `serve/audio_asr_api.cpp` (5), `serve/asr_websocket.cpp` (5) |
| EmbeddingGemma 2 | `serve/embedding_service.cpp` (16), `serve/embedding_service.hpp` (6), `serve/embeddings_api.cpp` (2), `serve/serve.cpp` (1) |
| MiniMax H3 | `serve/video_jobs.cpp` (12), `video/video.cpp` (11), `serve/video_jobs.hpp` (6), `serve/video_api.cpp` (3), `video/video.hpp` (2) |
| Qwen-Image-2.1 | `serve/image_api.hpp` (5), `serve/image_api.cpp` (1), `serve/serve.cpp` (1) |

Paths are under `src/cli/` unless they start with `core/`.

Model knowledge that is not a model-name reference but must still move:

| Location | What it encodes |
| --- | --- |
| `src/core/json_constraint.hpp` / `.cpp` | `ToolFormat { kJson, kQwen, kDeepSeek }`; the Qwen XML and DeepSeek DSML tool grammars are interleaved with schema-driven value grammars through the schema compiler. |
| `src/cli/serve/openai_chat.cpp` | Streaming tool-call marker scanning per dialect, a JSON fallback that recognizes every dialect's markers plus client envelope closers, per-dialect reasoning-separator rules, and parsers using core helpers (`SchemaAccepts`, `PythonLiteralsToJson`, `QuoteTracker`). |
| `src/core/text_sampling_defaults.hpp` | `TextModelPreset { kQwen38, kDeepSeekV4Flash }`, selected by architecture string and layer shape. |
| `src/core/model_config.hpp`, `src/core/gguf_reader.cpp` | `ModelConfig` and `GgufReader::ExtractModelConfig` are shaped for Qwen hybrids (`ssm.*`, `full_attention_interval`, `nextn_predict_layers`, fallback architecture `"qwen"`); `architecture` defaults to `"qwen35"`. |
| `src/cli/serve/inference_backend.hpp` | `TextSpeculativeBackend { kDisabled, kDFlash, kDSpark, kMtp }`: valid `--speculative` values and their validation are model-specific. |
| `src/cli/serve/text_generation_backend.hpp`, `src/core/speculative/` | Server and speculative types come from `src/models/qwen/` (`ChatMessage`, `ChatTool`, tokenizer, Qwen executor and DFlash policy). |
| `src/cli/serve/serve.cpp`, `bench/bench.cpp` | Per-command option definitions (`--speculative`, `--enable-yarn`, draft paths), help text, `--mmproj` sidecar discovery, GPU queue profiles chosen per subcommand before any HIP call. |
| `src/cli/serve/http_server.cpp` | `/v1/models` listing: `owned_by` strings (text models report `gufo`) and H3's five preset model IDs with `root` and `capability`. |
| `tools/ci/` | `kernel-resources.json` scratch allowances keyed by model kernel symbols, vendored model paths in `check-format.py`, a model list in `check-docs.py`, DS4 in `check-dependencies.py`. |
| `src/cli/main.cpp` | Help text naming models. |
| `cmake/Checks.cmake`, `CMakePresets.json` | Hand-maintained `check-pr` test lists (including Windows) and per-model test presets. |
| `.github/workflows/` | Explicit Windows build target lists and devlab workflows. |
| `.devops/nix/package.nix`, `mk-serve.nix`, `flake.nix` | Evaluation-time source-filter exceptions for model fixtures, per-model module arguments (`dflashModel`, `dsparkModel`, `mtpModel`), and per-model references. `package.nix` passes its own CMake flags rather than using presets. |
| `tests/functional/devlab.json`, `impact.json` | Two schemas with different profile sets: model profiles, shared assets (also across models, such as the ASR asset in the TTS round trip), representatives, disabled entries, core-path impact rules that enumerate every model's profiles, and rules mapping model-owned core paths (such as `src/cli/serve/embedding*`, `asr*`) to models. |
| `tests/functional/*.py` | 21 of 87 files contain model-specific logic; AGENTS.md keeps them in `tests/functional/`. |
| `tools/` | Per-model tools (`tools/qwen27b`, `tools/qwen-flash`, `tools/ds4`, `tools/h3`, `tools/models/*`), model Python tests in `tests/tools/`, per-model flake checks, devshell Python dependencies and reference packages in `flake.nix` and profiler knowledge. |
| Root `CMakeLists.txt` install rules, `THIRD_PARTY_NOTICES.md`, `licenses/`, `docs/models/README.md` | H3 manifest, DeepSeek eval fixture, `LICENSE.ds4`, notices and the model index. |

Cross-model and model-to-CLI dependencies:

| Extension-to-be | Depends on |
| --- | --- |
| Flash-Next | Qwen tokenizer and chat template (its cache identity hashes `QwenChatTemplate::OfficialTemplateSha256()`), Qwen kernels (`prefill_quant_gemm.hpp`) and the Qwen vision stack. |
| Qwen3-TTS | `qwen/control_tokens.hpp` and Qwen HIP ops (`qwen/hip/ops.hpp`). |
| Qwen-Image-2.1 | Qwen tokenizer and control tokens. |
| MiniMax H3 | `qwen/control_tokens.hpp` only. |
| DeepSeek V4 Flash | `src/cli/serve/logging.hpp` (three files): the only production `src/cli/` include in a model directory. |
| Model tests | Nine test files across eight test directories include `src/cli/` headers (for example `qwen3_asr/tests/audio_api_test.cpp`, `tests/models/embedding_gemma2/embeddings_api_test.cpp`). |

Core headers used by model directories (by include count, from the
[inventory command](#inventory-commands)):

| Model | Core headers |
| --- | --- |
| Qwen 27B | `gguf_reader` (26), `hip/detail/*` dispatch telemetry and GEMV dispatcher (20), `hip/hip_utils` (17), `quant/ggml_dequant` (7), `model_config` (5), `sampling` (4), `image` (3), `speculative/draft_backend`, `hip/snapshot_transfer`, `hip/managed_alloc`, `hip/host_allocations`, `crypto/sha256`, `speculative/prompt_lookup`, `reasoning`, `quant/ggml_gemm`, `platform/prefault`, `platform/pages`, `mapped_prefetch`, `hip/weight_upload`, `hip/transfer_pool`, `hip/snapshot_copy`, `hip/heap_placement` |
| Flash-Next | `hip/managed_alloc` (5), `gguf_reader` (5), `sampling` (3), `session_mode` (2), `quant/ggml_dequant` (2), `speculative/prompt_lookup`, `platform/pages`, `platform/mapped_file`, `platform/fs`, `hip/weight_upload`, `hip/transfer_pool`, `hip/snapshot_transfer`, `hip/snapshot_copy`, `hip/host_allocations` |
| DeepSeek V4 Flash | `platform/mapped_file` (3), `sampling` (2), `session_mode`, `reasoning`, `platform/publish`, `platform/os_info`, `platform/fs`, `mapped_prefetch`, `json`, `hip/weight_upload`, `hip/snapshot_transfer`, `hip/managed_alloc`, `gguf_reader`, `crypto/sha256` |
| EmbeddingGemma 2 | `json` (6), `hip/hip_utils` (2), `platform/mapped_file`, `platform/library`, `image` |
| Qwen3-ASR | `json` (3), `hip/weight_upload` (2), `utf8`, `platform/prefault`, `platform/mapped_file`, `mapped_prefetch`, `gguf_reader`, `cancellable_gate` |
| Qwen3-TTS | `json` (3), `mapped_prefetch` (2), `hip/weight_upload` (2), `platform/prefault`, `platform/mapped_file`, `gguf_reader` |
| Qwen-Image-2.1 | `platform/mapped_file`, `mapped_prefetch`, `json`, `image`, `hip/weight_upload`, `gguf_reader`, `crypto/sha256` |
| MiniMax H3 | `platform/mapped_file` (6), `platform/os_info` (4), `mapped_prefetch` |

### Existing seams to build on

- **`TextModelRunner`** already separates the scheduler and cache from
  per-model execution. Its runner contract is the starting point for the
  text extension interface, after the header is split and the
  implementations move out of `inference_backend.cpp`.
- **`cache::Adapter`** is a model-provided interface owned by a common
  package. The cache RFC already states that "plug in means an explicit
  interface and registration; dynamic library loading is not required". This
  RFC applies that principle to the whole model.
- **Service classes** (`EmbeddingService`, `TtsService`, `AsrService`,
  `ImageService`, `VideoJobService`) isolate the HTTP surface from the model
  for non-text capabilities, although each is written against one concrete
  model's types (for example `EmbeddingService` options use
  `embedding_gemma2::VideoOptions`). EmbeddingGemma 2's batch scheduler lives
  in its model directory.
- **`devlab.json` and `impact.json`** already describe models, assets and
  impact rules as data, consumed by the harness and by
  `.github/workflows/devlab-*.yml` and `windows-build.yml`.

## Evaluated solutions

### Build-time source extensions — recommended

An extension is source code with manifests. CMake discovers in-tree
extensions under `src/models/*/` and out-of-tree ones from
`GUFO_EXTENSION_DIRS`, builds them against the extension API, and generates a
registry translation unit that calls each extension's entry point. Nix
exposes the same thing as `gufo.override { extensions = [ ... ]; }`, reading
the same declarative metadata.

Compatibility is a source (API) promise: an extension is always compiled with
the same compiler, ROCm and flags as the host, and is statically linked with
one copy of `gufo_core`. There is no ABI to freeze, no duplicated singletons,
no DLL/CRT problems on Windows, and the boundary costs nothing compared with
today. The cost is that combining an official artifact with an arbitrary
third-party extension requires a rebuild.

The detailed design follows in
[Recommended design](#recommended-design-build-time-source-extensions).

### Runtime binary plugins

Extensions as shared libraries loaded with `dlopen`/`LoadLibrary` from a
plugin directory. This is the only option where an official artifact accepts
a third-party model without a rebuild.

Requirements this would add:

- A C ABI, or an exactly pinned toolchain checked at load time, because the
  C++ ABI (layouts, `std::` types, exceptions) is not stable across compilers
  and standard libraries. Prior art that succeeds (ONNX Runtime's versioned
  `OrtApi` function table, Triton's `TRITONBACKEND_*` API with major/minor
  version checks, ggml's dynamically loaded backends) uses a C interface.
- `gufo_core` as a shared library. If the executable and a plugin both
  linked it statically, one process would contain two transfer pools, two
  staging budgets and two pinned-memory accountings; serve and the cache
  would use one copy while the model used the other, and neither copy would
  see the other's allocations.
- Load-time verification of ROCm and HIP runtime versions, and Windows
  handling of CRT heaps across DLL boundaries.

Performance would not change if the boundary stays at step level: one
indirect call costs nanoseconds against millisecond decode steps, and Gufo
does not use LTO today, so no cross-module inlining is lost. The cost is
engineering and stability, not speed. Freezing an ABI before the API has been
exercised by eight models, while the runner contract still changes weekly,
would freeze its mistakes.

Recommendation: defer. Add it later as a C shim over the same C++ interface
when there is concrete demand (see [rollout](#compatibility-and-rollout)).

### Separate worker process

The core process owns HTTP, scheduling and cache; each model runs in a worker
process speaking a Gufo IPC protocol, sharing device memory through
`hipIpcGetMemHandle`. Ollama's runner subprocesses are an example of this
split.

It gives language independence and crash isolation. It also moves the
continuation cache, which manipulates model device memory under strict
ownership rules, across a process boundary; turns step-level scheduler calls
such as `DecodeBatch` into IPC round trips; doubles process lifecycle and
failure handling; and depends on HIP IPC support on Windows. The cost exceeds
the benefit for a one-model-per-process server.

### Data-only extensions

An extension as configuration (architecture parameters, tensor naming, op
graph) interpreted by shared kernels, as in ggml-style engines. This removes
code per model only for architectures that are combinations of existing
operations; every model Gufo ships today has bespoke kernels or state
(DeltaNet, compressed attention pools, MTP heads, diffusion schedulers).
It is not a replacement. A future "configurable decoder" extension could be
built on top of the source-extension mechanism for families that fit.

### Keep the current design

No migration cost. All current costs remain and grow with each model; the
implicit boundary keeps making shared changes risky. This is the baseline.

### Prior art

| Project | Mechanism | Lesson for Gufo |
| --- | --- | --- |
| vLLM | Out-of-tree models register through the `vllm.general_plugins` Python entry point calling `ModelRegistry.register_model`; plugin functions must be re-entrant; `VLLM_PLUGINS` filters which load. The documented registration call is guaranteed stable. | Registration by architecture name is enough as the integration point. Keep the registration surface tiny and promise its stability explicitly. |
| SGLang | `SGLANG_EXTERNAL_MODEL_PACKAGE` names a package whose modules expose `EntryClass`; also in-code `ModelRegistry` registration. | A single, discoverable entry symbol per extension. |
| Modular MAX | `max serve --custom-architectures path:package` imports a package exposing `ARCHITECTURES`, matched against the checkpoint's `architectures[0]`; a custom architecture overrides a built-in one of the same name. | Match on artifact identity, and make override behavior explicit. Gufo chooses an explicit error and selection over implicit precedence. |
| Triton Inference Server | Backends are shared libraries behind the `TRITONBACKEND_*` C API; minor versions add optional functions, majors must match. | If binary plugins come later: C API, major/minor negotiation, a coarse model-instance/execute boundary. |
| ONNX Runtime | `OrtApiBase::GetApi(version)` returns a versioned function table; structures carry a version field. | The reference shape for a future ABI layer: append-only, version-requested tables. |
| llama.cpp / ggml | Backends can be built as dynamically loaded libraries (`GGML_BACKEND_DL`, `ggml_backend_load_all`); models remain in-tree. Embedders hit "no backends are loaded" when they forget the loader. | Dynamic loading adds integration failure modes; explicit, generated registration avoids them. |
| Godot | C++ modules are compiled into the engine (requiring rebuilt export templates); GDExtension later added runtime-loaded native libraries, forward-compatible only (4.2 extensions work on 4.3, not the reverse). | The recommended path: build-time modules first, a binary interface once the API has matured. |
| Zephyr RTOS | Out-of-tree modules are discovered by a `zephyr/module.yml` or `zephyr/CMakeLists.txt` marker and passed with `EXTRA_ZEPHYR_MODULES`. | A declarative metadata file plus CMake build logic per module is proven at scale. |
| Linux kernel | `stable-api-nonsense.rst`: no stable in-kernel API; whoever changes an interface fixes all in-tree users. | Core-tier models stay in-tree so interface changes are fixed with them; out-of-tree extensions get an explicitly versioned surface. |

### Comparison table

Green means a good fit for the row; yellow means conditional or extra work;
red means the option leaves the requirement unresolved.

| Requirement | Source extensions, recommended | Binary plugins | Worker process | Data-only | Current |
| --- | --- | --- | --- | --- | --- |
| New in-tree model without core edits | 🟢 Directory plus manifests | 🟢 Same | 🟢 Same | 🟡 Only for fitting architectures | 🔴 Edits CLI, core, CMake, CI |
| Out-of-tree model with no Gufo PR | 🟢 `GUFO_EXTENSION_DIRS` / Nix | 🟢 Drop-in library | 🟢 Separate binary | 🟡 Config only | 🔴 Fork required |
| Official artifact plus third-party model, no rebuild | 🔴 Rebuild required | 🟢 Drop-in | 🟢 Drop-in | 🟢 Drop-in | 🔴 Fork required |
| Performance | 🟢 Identical to today | 🟢 With step-level boundary | 🟡 IPC per scheduler step, cross-process cache | 🔴 Loses bespoke kernels | 🟢 Baseline |
| Compatibility promise | 🟢 API only | 🔴 ABI, C shim, toolchain checks | 🟡 Wire protocol | 🟡 Schema | 🟢 None needed |
| Single `gufo_core` state | 🟢 One static link | 🟡 Requires shared core | 🟡 Split across processes | 🟢 One process | 🟢 One process |
| Continuation cache integration | 🟢 In-process adapter or snapshots | 🟢 In-process | 🔴 Cross-process device memory | 🟡 Generic only | 🟢 Existing |
| Windows | 🟢 Same build | 🟡 DLL/CRT boundary | 🟡 HIP IPC support | 🟢 Same build | 🟢 Same build |
| Incremental delivery | 🟢 Model by model | 🟡 After ABI design | 🟡 After protocol design | 🔴 Rewrite | 🟢 Nothing to deliver |

## Recommended design: build-time source extensions

### Layout

```text
include/gufo/ext/            public extension API (versioned headers)
  version.hpp                GUFO_EXTENSION_API_VERSION_{MAJOR,MINOR,REVISION}
  registry.hpp               Registry, ProviderInfo
  artifact.hpp               ArtifactKey, ArtifactSet (primary, shards, sidecars, drafts)
  options.hpp                OptionSpec, ParsedOptions
  host.hpp                   Host services handle
  text_generation.hpp        capability interfaces, one header each
  embeddings.hpp
  speech_recognition.hpp
  speech_synthesis.hpp
  image_generation.hpp
  video_generation.hpp
  platform/                  re-exported platform leaves (mapped files, pages, fs, ...)
  experimental/              no stability guarantee (initially: tool syntax and
                             grammar builder, kernels and HIP utilities,
                             transfer primitives, most runner extensions)

cmake/GufoExtension.cmake    gufo_extension(), gufo_extension_hip_sources()
schemas/extension.schema.json

src/models/<name>/           an in-tree extension
  extension.json             metadata manifest
  extension.cmake            build manifest
  extension.cpp              entry point: registers providers
  ...                        model sources, kernels, cache integration
  tests/                     unit/operator tests, registered by the manifest
  functional/                devlab and impact registry fragments
  tools/                     model-specific tools (optional)

src/families/<family>/       family libraries, each with its own manifests
  qwen/control_tokens/ ...

<build>/generated/extension_registry.cpp      one per executable
<build>/generated/<name>_constants.hpp       per-extension constants from metadata
```

Documentation can stay under `docs/models/<name>/` and tests under
`tests/models/<name>/` during migration, provided they are registered by the
manifests; physical moves are an
[open question](#to-resolve-before-dependent-implementation).

### Include scheme

- Extensions include the extension API as `<gufo/ext/...>` and their own
  files relative to their extension root, which the build adds as a private
  include directory. In-tree extensions keep root-relative includes of their
  own files during migration.
- Core headers that models use today (see the
  [inventory](#coupling-inventory)) are classified one by one: re-exported
  under `include/gufo/ext/` (stable, `platform/` or `experimental/`), wrapped
  by a host service, or made private to the core. Publishing these headers
  is its own plan step, before the pilot.
- The root `CMakeLists.txt` also adds the repository root tree-wide
  (`include_directories`), force-includes `libstdcxx_noinline.hpp` and sets
  Windows compile definitions per directory; out-of-tree directories added
  with `add_subdirectory` inherit these, so `gufo_extension()` and the step
  11 allowlist check account for inherited directory properties. The
  `GUFO_PLATFORM_ONLY` early return skips discovery and registry
  generation.
- Re-exporting means **forwarding headers**: `include/gufo/ext/...` headers
  include the real `src/...` headers, which stay where they are (the
  platform leaves stay in `src/core/platform/`, as AGENTS.md requires). The
  repository root therefore remains a public include directory, and the
  isolation check is the permanent enforcement mechanism: extension code may
  include only `gufo/ext/...`, its own files and its declared `depends`.
  `gufo_extension()` adds no further public include directories.
- Because forwarded headers stay in `src/`, the API surface is defined by a
  generated list (`include/gufo/ext/SURFACE`): the transitive include
  closure of the forwarding headers, such as `json_constraint.hpp` reached
  through `sampling.hpp`. The API-change check, version bumps and the 1.0
  churn measurement watch those paths, including `src/cache/adapter.hpp`.

### Target graph

Extensions need libraries to link against, and the registry must not create
a cycle. The targets are:

```text
gufo_platform, gufo_logging, gufo_hip_host     platform leaves, logging, HIP host
  ^
gufo_cache, gufo_cache_hip                      cache contracts, staging, transfers,
  ^                                             component cache backing
gufo_core                                       HIP utilities, GGUF, JSON, sampling,
  ^                                             constraints
gufo_ext_runtime                                extension API types, registry, defaults,
  ^        ^                                    neutral continuation bases, host services,
  |        |                                    host memory-budget queries
  |        |
  |        extensions                           gufo_<name>, family libraries
  |        ^
gufo_serving, gufo_http, gufo_llm_cli           scheduler, pool, component text cache,
gufo_diagnostics                                InferenceBackend, HTTP, prompt/bench,
  ^                                             diagnose
gufo_test_support                               HTTP and serving fixtures for model API
  ^                                             tests (test executables only)
gufo, gufo-kernel-bench, test executables       each compiles its own generated registry
```

`gufo_logging` keeps its name and position; logging code only moves out of
`src/cli/serve/`.

Extensions depend on `gufo_ext_runtime` and the libraries below it, never on
`gufo_serving` or `gufo_http`. Serving, HTTP, CLI and diagnostics libraries
depend on `gufo_ext_runtime` and see extensions only through the registry
interface. The generated registry is compiled into each executable,
including `gufo-kernel-bench`, so no library depends on extensions; test
executables get a registry for only the extensions they test.
`gufo_diagnostics` receives the `Registry&` as an argument instead of
referencing an executable-defined symbol.

Interim edges during migration: until steps 8 and 10 finish, `gufo_http`
links the non-text model libraries, `gufo_serving` contains the text
runners, and `gufo_llm_cli` and `gufo-kernel-bench` link model libraries
directly. Between steps 5 and 6, `gufo_serving` does not exist yet: the
`gufo` executable, `gufo_http`, `gufo_llm_cli` and the three test binaries
that compile `inference_backend.cpp` link `qwen_text` and the Qwen 27B
library directly, because server types still come from them. These edges are listed in the isolation report and removed model by
model; the final graph holds only after step 10.

### Metadata and build manifests

The **metadata manifest** carries everything CI, Nix, Python and the
registry generator need without compiling or configuring:

```json
{
  "schema": 1,
  "name": "qwen38_flash_next",
  "api_version": "0.1",
  "version": "0.4.0",
  "entry": "gufo_ext_qwen38_flash_next_register",
  "depends": ["qwen_text", "qwen_kernels", "qwen_vision"],
  "providers": [{
    "id": "qwen38-flash-next",
    "capability": "text_generation",
    "keys": ["gguf:qwen4exp"],
    "speculative": ["mtp"]
  }],
  "options": [
    {"name": "--mtp-model", "type": "path", "commands": ["serve llm", "bench", "prompt"]}
  ],
  "targets": {
    "gufo_qwen38_flash_next": {"requires": []},
    "gufo_qwen38_flash_next_hip": {"requires": ["hip"]}
  },
  "platforms": ["linux", "windows"],
  "packages": [{"pkg_config": "libavformat", "nix": "ffmpeg", "windows": "ffmpeg[avformat]"}],
  "compile_definitions": {"GUFO_EXAMPLE_DATA": "install_data:example.json"},
  "runtime_libraries": ["gufo-video-libraries.txt"],
  "install_data": [],
  "licenses": [],
  "production_paths": {"include": ["**/*.cpp", "**/*.h", "**/*.hpp", "**/*.hip", "extension.*"],
                       "exclude": ["tests/", "functional/", "tools/", "fixtures/"]},
  "vendored_paths": [],
  "kernel_resources": "kernel-resources.json",
  "docs": {"title": "Qwen3.8 Flash-Next", "inputs": "Sharded Q4 GGUF; text/images",
           "guide": "docs/models/qwen3.8-flash-next/README.md"},
  "tests": [
    {"target": "qwen38_flash_next_config_test", "ctest": "qwen38_flash_next\\.config",
     "check_pr": true, "exclude_os": []},
    {"target": "qwen38_flash_next_ngram_test", "ctest": "qwen38_flash_next\\.ngram",
     "check_pr": false, "windows_build": true}
  ],
  "test_labels": ["qwen38_flash_next"],
  "fixtures": "tests/fixtures/",
  "bundled_tests": [],
  "functional": "functional/",
  "tools": "tools/",
  "python_checks": [{"name": "example-quality", "tests": "tests/tools/", "sources": ["tools/"]}]
}
```

`packages` maps each external dependency per ecosystem (pkg-config module,
Nix attribute, vcpkg port and features), and the `pkg_check_modules` calls
for model dependencies (libav, sndfile, soxr) move from the root build into
the extensions that need them; CMake cache variables with Nix store paths
come from `extension.nix`, and `runtime_libraries` replace today's core-owned settings such as H3's ffmpeg
paths, EmbeddingGemma's libav paths and the library list the Windows
workflow bundles. `compile_definitions` replaces model data paths passed on
the `gufo` executable today (`GUFO_H3_SOURCE_MANIFEST`,
`GUFO_H3_INSTALLED_SOURCE_MANIFEST`); eval's paths stay core-owned.
`licenses` and `install_data` drive the Nix install check.
The example shows the shape, not Flash-Next's actual dependencies. A
dependency that needs provisioning not expressible this way is a core
change. Family libraries use the same schema with `"kind": "library"` and no
`entry` or providers; the entry point lives in a target with no build
conditions, and providers whose HIP target is not built register as
excluded. Test targets are created only under `BUILD_TESTING`.
`production_paths` uses recursive globs relative to the extension root, with excludes; `install_data`, `licenses` and `runtime_libraries` paths are included implicitly.
Configure-time validation tolerates `fixtures`, `functional` and `tools`
being absent, because production sources filter them out. Test registration
lives in the manifests, so editing a test source does not change the
production derivation, but adding or renaming a test does.
`python_checks` and `extension.nix` replace the per-model flake checks
(`h3-manifest`, `h3-quality`, `h3-ml-quality`), the per-model devshell
Python dependencies (librosa, LPIPS, torchvision) and reference packages
(`ds4-reference`, `llama-cpp-mtp-reference`); model
Python tests such as `tests/tools/h3_*` move with their extension. Flake
check names are public flake outputs, so generated checks keep today's
names.

`production_paths` limits what enters production Nix sources, so editing an
extension's test sources, functional fragments or tools does not change the
production derivation. `vendored_paths`, `kernel_resources` and `docs`
replace the model knowledge in `tools/ci/` and the models index.
`test_labels` replace per-model test presets (`ctest --preset gpu-full -L
<label>`), because `CMakePresets.json` (presets version 3) cannot include
generated files. `fixtures` and `bundled_tests` describe model fixtures
(today under `tests/fixtures/<model>`) and post-package tests that the
Windows workflow copies into the bundle and runs, such as EmbeddingGemma's
video processor test.

Build conditions apply per target, so CPU libraries and CPU tests build
without HIP while HIP providers are skipped. The **build manifest**
(`extension.cmake`) defines the targets named in the metadata.
`gufo_extension_hip_sources()` applies the host's HIP flags to an extension's
kernels; extensions add flags only through it, per file or per target,
and may define several targets (such as Flash-Next's `_mmq` sublibrary
with its own HIP standard) (for example `-O2`,
`-mwavefrontsize64`, `-mllvm=-amdgpu-sched-strategy=iterative-ilp` or
`GGML_HIP_NO_VMM`). The metadata schema and the CMake functions are part of
the extension API and follow the same
[stability policy](#interface-stability-policy); toolchain-specific flag
values are exempt from it.

Today's flags depend on where each target is created in the root
`CMakeLists.txt`: DeepSeek, Flash-Next, TTS, ASR and EmbeddingGemma are
added before the warning and sanitizer flags, so their objects are not
sanitized in `cpu-sanitizer`, and test assertions come from a
directory-scoped `-UNDEBUG` that forces some models to register tests
through deferred functions. `gufo_extension()` instead applies warnings,
sanitizers and assertion settings per target role (library or test), so
discovery order does not change compilation; the first migration records
which models' sanitizer or assertion state changes as a result.

The root build:

1. Collects `src/models/*/` and `src/families/*/*/` manifests and every
   directory in `GUFO_EXTENSION_DIRS`.
2. Filters with `GUFO_EXTENSIONS`. The default is every discovered in-tree
   extension, which is what `.#default`, every preset, devlab Linux and
   devlab Windows build, so a new in-tree model is tested without further
   edits. Only the invocations that produce shipped artifacts set
   `-DGUFO_EXTENSIONS_FROM_LOCK=ON` and read the pinned list from one file,
   `extensions.lock`: the `.#release` package, the release-tag path of the
   reusable Windows workflow, and the rolling Toolboxes `edge` images (lock
   mode is exposed as a package argument, so Toolboxes can select it). The
   AGENTS.md build wording is updated with the step 3 amendment. Two hosted
   checks keep the modes aligned: a CPU configure and build with
   `-DGUFO_EXTENSIONS_FROM_LOCK=ON`, and a report of in-tree extensions not
   in the lock (unshipped by default). A third check requires that unshipped
   extensions share no artifact key, option name or `/v1/models` ID with
   shipped ones, so a lock-mode build's behavior is a strict subset of the
   default build's, and the release-tier Devlab run on the default build
   covers it. `.#default` and the NixOS serve module differ from release
   artifacts only by those extensions.
3. Validates at configure time, in pure CMake (`string(JSON ...)`, available
   at the 3.21 minimum, with no Python dependency for production builds):
   required fields, unique names, present `depends`, compatible API
   versions, satisfiable target conditions, and option declarations that
   agree on type for the same name and command, and that no extension
   declares a core-owned option (such as `--context`, `--speculative` or
   `--provider`). Full JSON-schema validation
   of `extension.schema.json` runs in a hosted Python check.
4. Generates `extension_registry.cpp`, which calls each entry point in a
   fixed order and records the keys of extensions that were discovered but
   excluded, so the CLI can report "built without extension X":

   ```cpp
   // Generated. Do not edit.
   void gufo_ext_qwen38_flash_next_register(gufo::ext::Registry&);

   void gufo::ext::RegisterBuiltExtensions(Registry& registry) {
     registry.BeginExtension({"qwen38_flash_next", "0.4.0",
                              "src/models/qwen38_flash_next"});
     ::gufo_ext_qwen38_flash_next_register(registry);
     // ...
     registry.AddExcluded({"deepseek_v4_flash", {"gguf:deepseek4"},
                           "requires hip"});
   }
   ```

   Generated calls avoid static-initializer registration, which static
   libraries drop silently when no symbol is referenced, and which runs in an
   unspecified order.
5. Generates one `<name>_constants.hpp` per extension (provider IDs, keys,
   option names), so registration code uses the declared values instead of
   repeating them. Editing one manifest re-runs configure and relinks
   executables, but recompiles only that extension's sources.

CI workflows, `cmake/Checks.cmake`, test presets and the docs index read the
metadata manifests from the source tree, not from build output. Nix reads
them at evaluation time with `builtins.fromJSON` for the source filter and
installed data. What JSON cannot express (package overrides such as
`torchvisionRocm` and `lpips.override`, reference packages, check
derivations, CMake flags with store paths) goes in an optional
`extension.nix` fragment, a function `pkgs -> { buildInputs, cmakeFlags,
devshellPython, checks, references }` imported by the flake. New flake
inputs (such as `audio-cpp` for reference runtimes) cannot come from an
extension; they remain a core `flake.nix` and `flake.lock` edit.

### Registration and selection

The entry point registers providers using the generated constants:

```cpp
void gufo_ext_qwen38_flash_next_register(gufo::ext::Registry& registry) {
  using namespace gufo::ext::generated::qwen38_flash_next;
  registry.AddTextGeneration({
      .info = kProviderQwen38FlashNext,       // id, keys
      .probe = &ProbeFlashNext,               // pure, HIP-free
      .resolve = &ResolveFlashNextArtifacts,  // shards, --mmproj, MTP draft
      .options = FlashNextOptions(),
      .create = &CreateFlashNext,             // returns unique_ptr<TextModel>
  });
}
```

Selection uses keys plus a mandatory probe:

1. The core reads the primary artifact's identity without HIP, as one of
   the key kinds `gguf:<general.architecture>`,
   `safetensors:<config.json architectures[0]>`,
   `diffusers:<model_index.json _class_name>` (Qwen-Image) or
   `manifest:<source manifest id>` (H3), and
   collects the providers for that key and capability.
2. Each candidate's probe confirms or rejects the artifact using metadata
   only, for example tensor shapes. Several extensions may claim the same key
   (two models on one GGUF architecture with different kernels); probes
   decide between them.
3. Exactly one confirmation is required. Zero confirmations report the
   candidates and, for keys of excluded extensions, "built without extension
   X". Two or more is an error naming the providers, resolved with
   `--provider <id>` (a provider ID from the metadata) or by excluding an
   extension with `GUFO_EXTENSIONS`.

Today's Qwen 27B fallback ("any architecture that is not DeepSeek or
Flash-Next") becomes an explicit key, `gguf:qwen35`, with a probe for the
supported shapes. Artifacts the fallback accepted by accident fail with a
clear message instead.

GPU queue profiles stay per capability, chosen from the subcommand before any
file I/O or HIP call, as today. A provider-level override is added only if a
model needs one, and must then be static metadata.

The provider's `resolve` step turns the primary path and options into an
`ArtifactSet`: sharded GGUF files, sidecars discovered next to the model
(today's `--mmproj` behavior), draft artifacts (MTP, DFlash, DSpark), or
pinned source manifests (H3). Fingerprinting stays as it is today: computed
lazily, only when the disk cache needs an identity, over the same members as
today. Diagnostics report the set's paths and sizes.

The serve subcommands (`llm`, `embedding`, `tts`, `asr`, `image`, `video`)
map to capabilities and stay as they are. `prompt`, `bench` and `transcribe`
select providers the same way.

### Capability interfaces

Capabilities are core-owned. Each has one versioned interface, shaped from
what the current service classes and the `TextModelRunner` contract already
need, and a defined split of scheduling responsibility:

| Capability | Model-side interface starts from | Scheduling owned by |
| --- | --- | --- |
| Text generation | The `TextModelRunner` contract and `TextRunnerCapabilities` | Core: `TextGenerationScheduler`, `TextRunnerPool`, component cache. |
| Embeddings | `EmbeddingService`'s model calls, with model-neutral options | Extension batch scheduler, as EmbeddingGemma 2 today. |
| Speech recognition | `AsrService`'s model calls | Core request queue; streaming chunking defined by the interface. |
| Speech synthesis | `TtsService`'s model calls | Core request queue; streaming chunks defined by the interface. |
| Image generation | `ImageService`'s model calls | Core job queue. |
| Video generation | `VideoJobService`'s model calls | Core job queue with persistent job storage. |

Each capability interface also lets a provider declare the model IDs it
lists in `/v1/models` (with `owned_by`, `root` and capability) and map
request model IDs to presets, covering H3. Providers also get request
validation hooks for model-specific features, such as which TTS voice modes
a variant supports (today a branch on `qwen3_tts::ModelVariant` in
`audio_tts_api.cpp`). The embeddings pilot keeps today's scheduling behind
the interface; a core admission queue, which does not exist today, would be
a separate later PR with its own scheduling and latency evidence.

`prompt`, `chat`, `bench`, `transcribe` and `video` measure or drive the
model without the server's scheduler today, and most of that code is
model-specific: about 1,300 of `bench.cpp`'s 1,764 lines
(`RunDeepSeekBenchmark`, `RunQwen38FlashNextBenchmark`, the Qwen 27B path)
and about 600 lines of per-model generation in `prompt.cpp`, plus `RunChat`.
The proposed default is that these drivers stay **model-owned**: each
extension registers command drivers through a `cli_drivers` entry point,
like kernel-bench cases, and the core keeps argument parsing, output
formats and the shared benchmark report. A core direct-session interface
that replaces the drivers is the alternative; the step 2 spike decides
between them, and step 10's scope follows that decision.

#### Text generation

The text interface is the successor of the `TextModelRunner` contract:

- `text_model_runner.hpp` is split into the runner contract (which moves to
  `include/gufo/ext/`, with its defaults in `gufo_ext_runtime`) and the pool,
  cache options and memory-budget code (which stay in `gufo_serving`).
  Dependencies on `ChatRequest`, `InitialOutputState`, `ContinuationState`,
  `ContinuationSnapshot`, the publication wait type and `ToolFormat` are
  replaced by neutral types in the API.
- `HipTextModelRunner`'s device probing becomes a core default
  implementation in `gufo_ext_runtime`. Only the model subclasses move into
  their model directories.
- The host memory-budget functions (`HostSnapshotBudgetBytes`,
  `HostSnapshotCeilingBytes`), which both the backing and the runners'
  `ResourceClaim` call, move to `gufo_ext_runtime` and are exposed as a
  budget query on the `Host` handle.
- Component cache backing (`CreateHipComponentCacheResources`) is core-owned
  cache-budget policy and moves next to `gufo_cache_hip`; it receives the
  budget ceiling and logging as injected callbacks, so the cache package
  does not depend on serving code. `BackingPlan` and
  a single neutral `ContinuationBackingClass` become part of the adapter
  contract, coordinated with the cache RFC, because the component cache is
  what its cutover moves every model onto.
- `CreateComponentCacheResources` and `ComponentCacheResources` (today in
  `component_text_cache.hpp`) move to the cache package as neutral types, so extensions never include `gufo_serving`
  headers.
- Model-specific diagnostics in the contract, such as `TextRunnerNgramRows`
  and `NgramRows()`, move behind an optional capability.
- Work on `text_model_runner.hpp` and the backing code is sequenced with the
  open cache RFC cards; the cache work takes precedence where they conflict,
  and this RFC's split follows it.
- The boundary stays at step level because the core keeps cross-request
  scheduling: `DecodeBatch` advances several requests in one call.

| Part | Owned by the extension | Notes |
| --- | --- | --- |
| Model | Weights, device state, context limits, sampling and reasoning defaults, position and YaRN policy | Replaces `TextModelPreset`; `create()` receives the resolved artifact set, options and core settings. |
| Inputs | Supported modalities, preprocessing (image decode, vision encoding), image cache, identity contribution | Today's `TextPromptContext::cache_identity`, `cache_prefixes` and `QwenImageContext`. |
| Prompt format | Chat template rendering, tokenizer, reasoning delimiters | Shared Qwen pieces come from family libraries. |
| Tool syntax | A tool-syntax implementation: grammar construction for forced tool decoding, a streaming marker scanner, a parser and a reasoning-separator rule | A code interface over an experimental grammar-builder API, not a static fragment. Schema helpers (`SchemaAccepts`, `PythonLiteralsToJson`, `QuoteTracker`) become host services. Client envelope closers stay in the core. |
| Execution | Prefill, in-pass checkpoints, decode step, decode batch, cancellation, logprobs, resource claims | Takes core-owned `SamplerState`, which the extension must apply, including during speculative verification. |
| Speculation | MTP, DFlash, DSpark and their drafts; the values each provider contributes to `--speculative` | Inside the extension; constraint and sampler state copies on rollback follow the core contract. Core primitives such as prompt lookup (`core/speculative/prompt_lookup.hpp`) are host services. |
| Continuation state | Either a `cache::Adapter` (component cache) or native snapshot and persistence support | Both are current behavior; dropping native snapshots depends on the cache RFC's all-model cutover. |
| Identity | Both compatibility identities (`cache::Adapter::CompatibilityIdentity()` and the persistence descriptor's identity) and the process-local identity | Built with a core-provided identity builder: the extension calls core-owned emitters (`Schema()`, `ArtifactId()`, `DraftArtifactId()`) and emits settings in today's order, reproducing today's bytes. Settings come from wherever they are owned: context tokens from the core, provider-declared options (`--draft-tokens`) and model-computed values (Flash-Next's decode concurrency) from the extension, and per-model process-local schema strings stay with the extension. Fingerprints stay core-computed; a core schema change is a deliberate identity bump recorded as such. |

Optional behavior is expressed through `TextRunnerCapabilities` and optional
sub-interfaces that the core discovers, not new mandatory virtual methods.
New core features arrive as new optional interfaces with defaults; older
extensions keep working without them.

The JSON tool-format fallback recognizes markers of every dialect today. With
per-extension tool syntax, whether it recognizes only the active model's
dialect or every built dialect is a behavior decision, listed as an
[open question](#to-resolve-before-dependent-implementation) and qualified
with the forced tool-choice suites (#315).

### Options

Options are declared per provider in the metadata manifest, with type, the
commands they apply to (for example `serve llm`, `bench`, `prompt`), help and
validation in code. Following the repository's one-canonical-option rule:

- **Shared names.** Several providers of the same capability may declare the
  same option name for the same command if the types agree; configure fails
  only on conflicting types. A second model that needs `--mtp-model` or
  `--draft-tokens` declares it instead of moving it to the core.
- **Capability options with provider values.** Options whose meaning is
  common but whose values are model-specific, such as `--speculative`, are
  core options whose valid values come from providers. `TextSpeculativeBackend`
  becomes a provider-contributed value set.
- **Parsing.** Each command's parser accepts the union of options declared
  by built providers for that command. After selection, options the selected
  provider does not declare are rejected with a message naming the provider.
- **No aliases.** Migration is the moment to unify `--mtp-model`,
  `--dflash-model` and `--dspark-model` if the maintainers agree; old names
  are not kept as aliases.
- **Nix serve module.** `mk-serve.nix`'s per-model arguments (draft model
  paths, modalities, speculative modes, TTS voices) stay as they are during
  migration and are generated from metadata in step 11b, before 1.0;
  `mkServe` is a public flake output, so renamed arguments carry `!`.

### Host services

Extensions receive a `Host` handle and compile against `include/gufo/ext/`:

- Device allocation, committed backing, weight upload, pinned staging and
  the transfer pool, through the same budgets serve and the cache use, and
  host memory-budget queries.
- The continuation cache package's public contracts (`cache::Adapter`,
  components, slots, completions, mutation guards).
- Sampling and constraint state, schema helpers, reasoning helpers and
  prompt lookup.
- The identity builder used for compatibility identities.
- Logging (moved out of `src/cli/serve/logging.hpp`), `--trace` events,
  cache metrics and diagnostics reporting.
- Platform leaves (`src/core/platform/`), re-exported under
  `include/gufo/ext/platform/`. Extensions follow the same rule as the rest
  of the tree and contain no `windows.h` or `_WIN32` checks.
- GGUF reading, JSON, image decoding, hashing, and audio decoding,
  resampling and WAV utilities (today `qwen3_asr::ResampleMono16k`,
  `AudioBuffer`, `qwen3_tts::DecodeBase64Wav` and EmbeddingGemma's audio
  decoder, used by core HTTP code).
- Shared kernels, HIP utilities and plan databases (GGML dequantization and
  GEMM, dispatch telemetry, the hipBLASLt plan database). These are
  device-code libraries rather than services; they start in `experimental/`
  and move to the stable surface once their interfaces have held. Moving
  reusable kernels here is how future models avoid re-tuning common
  operations per quantization.

### Family libraries

Code shared by several extensions of one family moves to
`src/families/<family>/`, as small libraries with their own manifests that
extensions name in `depends`, split by purpose so none becomes a second core.

The split below is provisional. Today's headers do not form a clean graph:
`hip/ops/token.hpp` includes `vision/rope.hpp`, `vision/encoder.hip` includes
`hip/ops/gemm.hpp`, `prefill_quant_gemm.hpp` reaches the 27B-specific
`QwenExecutionPolicy` through `hip/detail/attention_policy.hpp`, and
`vision/device_input.hpp` and Flash-Next include
`qwen/continuation_hooks.hpp`. `gemm_route.hpp`, `quant_ops.hpp` and
`continuation_hooks.hpp` have no assigned library yet. Step 1's report
includes a header-dependency analysis, and step 5 fixes the boundaries from
it: likely a `qwen_hip_common` library for execution policy, RoPE, GEMM
routing and continuation hooks, below both `qwen_kernels` and
`qwen_vision`, or `qwen_vision` merged into `qwen_kernels`. Family
libraries must not form cycles.

| Library | Contents | Used by |
| --- | --- | --- |
| `qwen_control_tokens` | Header-only shared control-token constants; keeps the `tools/gufo/control_tokens.py` mirror required by [the Qwen rules](../../src/models/qwen/AGENTS.md) | Qwen 27B, Flash-Next, Qwen3-TTS, Qwen-Image-2.1, MiniMax H3 |
| `qwen_text` | Tokenizer and chat template | Qwen 27B, Flash-Next, Qwen-Image-2.1 |
| `qwen_vision` | Vision encoder, image prompt, image cache | Qwen 27B, Flash-Next |
| `qwen_config` | CPU-only Qwen GGUF config extraction (called by the chat template's shape check, weights, MTP reference and `prompt`) | Qwen 27B, Flash-Next, Qwen-Image-2.1 (through `qwen_text`) |
| `qwen_kernels` | Shared Qwen HIP operators (including `prefill_quant_gemm.hpp`, `hip/ops.hpp` and `hip/ops/token.hpp`) | Qwen 27B, Flash-Next, Qwen3-TTS |

Moving the control tokens updates the links in the root `AGENTS.md` and
`src/models/qwen/AGENTS.md`.

Chat message, tool and tokenizer interfaces used by the server move into the
core API; only Qwen's implementations stay in `qwen_text`. Whether
out-of-tree extensions may depend on family libraries is an
[open question](#to-resolve-before-dependent-implementation).

### Tests, benchmarks, tools and documentation

- **Unit and operator tests** are registered through the manifests; the
  hosted `check-pr` and Windows target lists are generated from metadata at
  configure time, and per-model test presets are replaced by labels. Model API tests that need HTTP layers use a core
  `gufo_test_support` instead of including `src/cli/` headers.
  `gufo_test_support` is for in-tree tests only and is not part of the
  extension API; out-of-tree extensions test their HTTP behavior through the
  functional conformance suites. Tests that
  span extensions (Qwen framing goldens, cross-model chat-template tests)
  and the impact rules for family libraries live with the family library's
  manifest.
- **Functional registries.** The merged `devlab.json` and `impact.json` stay
  committed and drift-checked, because the Windows ZIP, devlab workflows and
  local harness runs read them from the source tree. Replacing the
  `deepseek-gpu` and `qwen-gpu-kernel-oracle` presets with labels updates
  the commands in `docs/TESTING.md`. Each extension ships fragments for both
  `devlab.json` and `impact.json`, each with its own schema:
  - `models`, profiles and `assets` entries are owned by their extension and
    must be unique; fragments may reference another extension's assets (the
    TTS round trip uses the ASR asset).
  - Core-path impact rules that apply to every model, such as
    `shared-runtime`, and suite `profiles` arrays (`image-inputs`,
    `tool-images`, `cache-stalls`, ...) select profiles by capability or
    flags such as `vision_required` instead of listing them, so new profiles
    join them automatically. Per-model workflow settings (`GUFO_ASSETS`,
    launch arguments, the Flash-Next benchmark command) come from the devlab
    fragments.
  - `representatives` and `disabled` stay in the core registry; they are
    release decisions, exempt from the isolation criterion.
  - The merged registries are generated from source manifests, and hosted CI
    and devlab workflows consume them.
  - Model-specific functional suites stay in `tests/functional/` as
    AGENTS.md requires; selecting them is driven by the fragments.
- **Benchmarks.** `prompt`, `chat`, `bench`, `transcribe` and `video` use
  extension-registered drivers (or direct sessions, per the step 2 spike).
  Routing them through providers is gated on benchmark continuity (see
  [success criteria](#success-criteria-and-evaluation)).
- **Tools.** `kernel-bench` calls individual kernels, so it is not part of
  the serving interface: extensions register kernel-bench cases through an
  experimental header (`include/gufo/ext/experimental/kernel_bench.hpp`)
  and a `kernel_bench` manifest entry point, compiled into
  `gufo-kernel-bench`'s generated registry and built only with
  `GUFO_BUILD_TOOLS`. Model-specific tools in
  `tools/` move into their extension, except modules of the shared
  `tools/gufo/` Python package (such as `h3_*.py`), which stay there to keep
  `gufo.*` imports working. `eval` stays a core HTTP client and its default
  dataset stays core-owned: `antirez-ds4.json` moves from
  `tests/models/deepseek_v4_flash/fixtures/` to a core fixture directory,
  with its Nix filter exception, so `gufo eval` keeps working in builds
  without DeepSeek.
- **Documentation and notices.** The docs index and `THIRD_PARTY_NOTICES.md`
  stay committed files, generated from metadata by a tool and drift-checked
  in hosted CI; adding a model regenerates them, an edit the goal allows as
  generated output. `/v1/models` entries come from providers.
- **Diagnostics.** `gufo diagnose` and `--version` list built extensions,
  their versions and their source (in-tree path or external directory and
  revision), so bug reports identify exactly what was compiled in.

## Interface stability policy

Extensions survive Gufo releases only if the interface changes rarely and
predictably. Today the runner contract changed in 12 commits in 30 days. The
policy is designed to get there, and the 1.0 criteria measure whether it has.
The extension API version is independent of Gufo's own version.

1. **Narrow surface where possible.** Extensions provide primitives; the core
   owns policy. Cache retention and budgets, scheduling and API surfaces stay
   in the core. Where a core feature needs model cooperation (logprobs,
   persistence), it is added as an optional capability with a default.
2. **Additive evolution.** New features arrive as optional sub-interfaces,
   capability flags or fields with defaults.
3. **Versioning before 1.0.** The API version is `0.MINOR.REVISION`. During
   0.x, a minor bump may break extensions; a revision bump is additive only.
   Extensions declare `>= 0.N.R` within one minor series, so additive
   changes do not break them. In-tree extensions are fixed in the same PR,
   as today.
4. **Versioning from 1.0.** Minor versions are additive. A breaking change
   bumps the major version; the core keeps an adapter from the previous major
   for at least 60 days where feasible, and the build reports the
   deprecation. Configure fails on an incompatible major. The
   [release policy](../RELEASING.md) defines Gufo's compatibility surface as
   the CLI, the HTTP API and public flake outputs, so extension API breaks do
   not use `!`. A PR that changes declarations in a `SURFACE` file carries
   an `ext-api` label and updates a checked `include/gufo/ext/CHANGES.md`
   with the version change; its title keeps its normal scope (for example
   `cache`), so core PRs that touch forwarded headers are not forced into
   one scope. This needs an
   amendment to `RELEASING.md` and `AGENTS.md`, made in step 3: the extension
   API and the flake `extensions` argument are outside Gufo's compatibility
   surface, ordinary PRs may change the extension API version and its
   changelog, and a check that receives the PR's diff requires the
   `ext-api` label and a `CHANGES.md` entry when declarations in `SURFACE`
   files change. User-facing
   changes made during migration (option unification, stricter Qwen 27B
   selection) are CLI changes and do carry `!`.
5. **In-tree canaries.** Every in-tree model uses only the public extension
   API. A change that breaks an extension breaks an in-tree model first, in
   maintainers' CI.
6. **External compatibility builds.** Known out-of-tree extensions (name,
   repository, revision) are built for every release. Their conformance
   suites run on a schedule rather than per release, because devlab runners
   download weights cold in every job.
7. **Experimental namespace.** New hooks start in `gufo::ext::experimental`
   with no guarantee. Graduation is an explicit decision in a reviewed PR,
   not a timer; a hook is eligible after 60 days without changes.

Criteria for API 1.0, all required:

- All in-tree models use only the public API, with the isolation check
  failing on any violation.
- At least one text-generation extension builds with zero
  `experimental/` includes, or a bounded, published list of experimental
  symbols a text extension may use at 1.0 exists and is enforced by the
  isolation check.
- The stable text-generation header has gone 60 days **and** at least five
  merged serving features that affected the extension API with only
  revision (additive) bumps. Header commits that do not change the API do not count.
- The out-of-tree pilot builds from an external directory and from Nix
  without changes for the same period.
- Every capability interface has a second provider besides its migrated
  model: a genuinely new model, or a minimal test-only provider built from
  an external directory and run through the conformance suite.

The promise is source compatibility. Rebuilding an extension against a new
compatible Gufo version must succeed without changes; binary compatibility is
not promised. Gufo releases frequently (about 30 release commits in the 30
days before this RFC), so stability windows are measured in time, not in
releases.

## Distribution

- **Official artifacts.** Nix outputs, Toolboxes OCI images and the Windows
  ZIP include the extensions listed in `extensions.lock`, built from source
  as today.
- **Nix.** `gufo.override { extensions = [ ext-a ext-b ]; }`, with each
  extension packaged as a flake exposing its source directory; its metadata
  manifest supplies `buildInputs` and installed data at evaluation time. A
  template flake documents the layout.
- **Containers.** Builder and runtime images (pinned toolchain and Gufo
  source, and a minimal runtime) would be produced by the Toolboxes
  repository, so an extension author can publish an image containing Gufo
  plus their extension with a short multi-stage Dockerfile. This is
  Toolboxes work, outside this repository.
- **Windows.** The Windows build accepts `GUFO_EXTENSION_DIRS` like any other
  CMake build; there is no Nix override path. Windows dependencies come from
  the root `vcpkg.json` in manifest mode, installed during `project()`
  before any extension metadata is read, so `vcpkg.json` stays core-owned:
  a pre-configure script (run by CI and documented for local Windows builds)
  checks it against the extensions' `packages.windows` entries (vcpkg port
  names and features), and an out-of-tree extension with a new Windows
  dependency needs a `vcpkg.json` change or its own overlay. The reusable Windows workflow
  takes only a release tag today; it needs an extension repository and
  revision input before third parties can produce their own ZIP with it.
- **Limitation.** An official artifact cannot gain a third-party model
  without a rebuild. With one model per process, users pick the image or
  build for the model they want, which keeps this acceptable until there is
  concrete demand for binary plugins.
- **Trust.** An extension is arbitrary native code at build and run time.
  Documentation must say so; official artifacts include only reviewed
  extensions.

## Quality tiers and conformance

| Tier | Where | Requirements |
| --- | --- | --- |
| Core | In-tree, maintained | Current repository rules: numerical quality tests, benchmarks, functional suites, release gate. |
| Community | Out-of-tree (in-tree acceptance is an open question) | Conformance suite for its capability; documented quality evidence; no release gate on speed. |

The conformance suite is assembled from existing functional suites per
capability: text generation includes correctness, tool calls (including
forced tool choice), continuation and cache suites (when the extension
declares continuation state), cancellation and device loss; other
capabilities use their existing API suites. A community extension benefits
automatically from shared-kernel and core improvements.

## Success criteria and evaluation

| Area | Required evidence |
| --- | --- |
| Isolation | The isolation check reports zero includes of, and symbol references to, model or family code outside the extensions that declare it, and zero model names in core-owned build, CI, Nix and registry files except generated files (including committed, drift-checked ones) and the release decisions in `extensions.lock`, `representatives` and `disabled`. Comments are not counted. |
| New in-tree model | A model in an existing capability can be built and tested with changes confined to its own directories. A migration cannot show this (it edits core files by design); it is shown by a genuinely new model or by the test-only second providers, and stays unproven until then. |
| Out-of-tree model | The pilot extension builds and passes its tests from an external directory and from a Nix flake input, from a source tree without its in-tree copy and with no diff to Gufo. |
| Correctness | Every migrated model passes its existing numerical, quality and functional checks; layout-only refactors are byte-identical under the lossless-precision gate. |
| Cache identity | A golden test records each model's compatibility identity bytes before migration and asserts they are unchanged after it. |
| Performance | Matched timings per request and phase within repository gates (5% and 10 ms) for each migrated text model: prefill, decode, TTFT, continuation restore; standard speed benchmark for non-text models. |
| Benchmark continuity | A matched main/PR A/B of `bench` and `prompt` output (same model, settings and toolchain) within the same gates, plus the HTTP standard speed benchmark, before and after moving each model's drivers. Retained `BENCHMARKS.md` numbers are HTTP comparisons at older revisions and are not reused as the baseline. |
| Build | Every core test and tool target builds with any single extension excluded. Configure fails clearly on schema errors, conflicting option types, missing dependencies, unsatisfiable conditions and incompatible API versions. Clean `release` build time grows by no more than 5%, and an incremental rebuild after touching one model source rebuilds only that model's targets and the link. |
| Diagnostics | `gufo diagnose` lists built extensions with versions and sources. |

## Risks

- **The runner contract keeps moving.** Core serving features keep reaching
  the per-model interface. Mitigation: optional capabilities with defaults,
  most runner hooks in `experimental/` at first, and measured 1.0 criteria
  instead of a date.
- **The first interface is wrong somewhere.** Migrating eight different
  models will find gaps. Mitigation: API 0.x; in-tree canaries.
- **Text generation does not fit one interface.** DeepSeek, Qwen 27B and
  Flash-Next differ in speculation, sampling fusion, tool syntax and
  continuation state. Mitigation: start from the existing runner contract,
  which already serves all three; keep speculation inside the extension
  under an explicit sampler-state contract.
- **Tool-call grammars are entangled with the schema compiler.** Extracting
  the Qwen and DSML dialects behind an interface means separating them from
  schema compilation inside a 2,500-line file. Mitigation: start with an experimental
  grammar-builder API that wraps the existing code, then extract per
  dialect, qualified with the forced tool-choice suites.
- **DeepSeek continuation and evidence.** DeepSeek relies on native
  snapshots, not the component cache, and its 81 GiB weights are disabled in
  devlab, so migration evidence must be gathered locally with dedicated host
  time. Mitigation: keep native snapshots supported until the cache RFC's
  cutover reaches DeepSeek; schedule DeepSeek's migration with that host
  time reserved.
- **Performance regressions from refactoring**, not from the boundary
  itself: extra copies, lost overlap or synchronization when code moves.
  Mitigation: matched A/B per migrated model before merge.
- **Family libraries become a second core.** Four extensions besides Qwen
  27B use Qwen code. Mitigation: small libraries by purpose, declared in
  `depends` and checked by the isolation check.
- **Untangling `gufo_core` and the executable.** Qwen 27B, H3, the
  speculative verifier, eval and test comparison code are compiled into
  `gufo_core`, and the serving layer is compiled into the executable and
  into test binaries. Moving HIP sources can change code generation through
  per-source flags. Mitigation: break the speculative dependency first,
  extract models and then split serving in their own PRs, with code-object
  and kernel-resource checks for moved HIP sources.
- **Mass include rewrite.** Publishing core headers under
  `include/gufo/ext/` changes many includes. Mitigation: mechanical PRs per
  model with no behavior change, after the headers are published.
- **Selection becomes stricter.** Artifacts accepted by today's Qwen 27B
  fallback may be rejected by an explicit key and probe. Mitigation: probe
  the shapes supported today and report clear errors.
- **Core layers become harder to change.** A stable API constrains future
  core refactors. Mitigation: narrow stable surface and the experimental
  namespace.
- **Option churn.** Unifying speculative-draft options changes user-facing
  flags. Mitigation: decide during migration and document once.
- **Out-of-tree quality.** Community extensions may be slow or wrong under
  Gufo's name. Mitigation: tiers, conformance suite, diagnose output that
  shows the extension source.

## Implementation plan

Each step is one or more small PRs, each naming the behavior it changes and
the models it affects. Steps land in order where the "Requires" column says
so; model migrations within steps 8 and 10 can proceed in parallel, and the
pilot path (steps 3, 7, 7b, 8, 9, 11) runs in parallel with the text path
(steps 4, 5, 6, 10), so the extension mechanism is validated before the
riskiest refactors finish. The
isolation check ratchets: it starts in report mode and becomes failing for
each model, and for `gufo_core`, in the PR that migrates it.

| Step | Work | Requires | Evidence per PR |
| --- | --- | --- | --- |
| 1 | Isolation check in report mode (`tools/ci/check-extension-isolation.py`): includes and symbol references across models, families, `src/cache`, `src/cli/`, all of `tests/` and all of `tools/` (core tests and tools such as `openai_chat_test`, `text_model_runner_test`, `exact_token_test`, `hipblaslt_runtime_cache_test` and `tune_hipblaslt` include specific models), model names in core-owned build, CI, Nix, `tools/ci` and registry files (including `GUFO_ASSETS` in workflows), and a header-dependency analysis of `src/models/qwen/`. | — | Hosted checks. |
| 1b | Identity golden test: expose the identity functions (today in an anonymous namespace in `inference_backend.cpp`) and record the compatibility identity bytes of Qwen 27B, Flash-Next and DeepSeek (persistence descriptor), with draft variants, before any code that writes them moves. | — | The test itself, run locally for DeepSeek. |
| 2 | Text interface spike: a throwaway branch moving Flash-Next behind a draft text interface, to validate the runner contract split, options and provider metadata before they are fixed. Not merged; findings update this RFC. | 1 | Flash-Next correctness and matched timings on the branch. |
| 3 | Extension infrastructure with no model changes: a minimal `gufo_ext_runtime` (`Registry`, `ProviderInfo`, API version, a minimal `Host` handle with logging moved out of `src/cli/serve/` into `gufo_logging`), the CLI parser that consumes provider-declared options, generated registration stubs per target so HIP-less builds never reference unbuilt `Create*` symbols, metadata schema and pure-CMake validation, `GufoExtension.cmake`, `extensions.lock`, generated registries and constants, `GUFO_EXTENSIONS`, `diagnose` output; Nix source filter admits `include/`, `schemas/` and `extensions.lock`; the `RELEASING.md` and `AGENTS.md` amendment. | 2 | Hosted checks; configure-failure tests for each validation rule. |
| 4 | Break the speculative dependency: move Qwen-specific speculation out of `src/core/speculative/` and use neutral tokenizer types in `draft_backend.hpp`. | 1b | All four text profiles plus `long-context` and `cache` (DeepSeek locally); matched timings against main. |
| 5 | Extract models from `gufo_core`: Qwen 27B into its extension library plus `src/families/qwen/` libraries with manifests (control tokens, text, vision, kernels); `GgufReader::ExtractModelConfig` becomes a free function in a CPU-only `qwen_config` library below `qwen_text` and `qwen_kernels`, with its test, while `ModelConfig` and `TextModelPreset` stay in core until step 10; H3 into its own library; `src/testing/compare` into its own target. Relink `gufo-kernel-bench` and `gufo_llm_cli` against the extracted libraries. Assign every core test and tool target that includes a model to that model's or family's manifest, or make it conditional on that extension being built; dialect tests (`json_constraint_test`, `openai_chat_test`) move with tool syntax in step 10. Move `src/eval/{dataset,extract,regrade}.cpp` into an eval target. Give DeepSeek a CPU-only text target (`chat_template.cpp`, `sampling.cpp`) for tests, without duplicating `tool_calls.cpp`. `tool_calls.cpp` stays in `gufo_http` until step 10. | 3, 4 | Build on Linux and Windows; affected unit tests; unchanged `kernel_resources_test` and identical ISA in stripped release (`NDEBUG`) code objects for moved HIP sources, with per-source flags and target-level options (includes, definitions) replicated, and unchanged (or deliberately changed and recorded) assertion and sanitizer state; functional correctness and matched timings for Qwen 27B and Flash-Next. |
| 6 | Serving split: extend `gufo_ext_runtime` (API types, runner defaults, neutral continuation state and snapshot bases, host services) and `gufo_serving` (scheduler, pool, component text cache, `InferenceBackend`), alongside the existing `gufo_http`, from sources compiled into `gufo` today; add the text-specific host services; neutral chat, tool, tokenizer and reasoning types. No behavior change. | 3, 5 | All four text profiles plus `long-context` and `cache` (DeepSeek locally, with reserved host time); matched timings against main. |
| 7 | Publish core headers used by models under `include/gufo/ext/` (stable, `platform/`, `experimental/`) per the inventory classification; rewrite model includes, one mechanical PR per model. Non-Qwen models need only step 3; Qwen-family models follow step 5. | 3 (5 for Qwen-family models) | Build on Linux and Windows; affected unit tests. |
| 7b | `gufo_test_support`: HTTP and serving fixtures for model API tests, so model tests stop including `src/cli/` headers. It wraps today's serving and HTTP code and does not need the serving split. | 3 | Affected model API tests. |
| 8 | Non-text capabilities, one PR each, starting with the pilot: embeddings (EmbeddingGemma 2, which has no Qwen or production CLI includes, owns its batch scheduler and is the embedding representative in devlab), then speech recognition (Qwen3-ASR, including `transcribe`), speech synthesis (Qwen3-TTS), image (Qwen-Image-2.1), video (MiniMax H3, including its `/v1/models` presets, preset routing and the `video` command). EmbeddingGemma's audio decoder stays in its extension; `ResampleMono16k`, `AudioBuffer` and `DecodeBase64Wav` move to host services in the ASR and TTS PRs. The pilot switches to extension-relative includes. TTS, Image and H3 also require step 5; H3's PR adds a platform leaf for its Windows media and process code. | 3, 7, 7b (and 5 for TTS, Image, H3) | The pilot keeps today's behavior. The model's API suites; standard speed benchmark. Qwen-Image and H3 have no devlab profile today, so their PRs add one or record the manual evidence. |
| 9 | Give every model a metadata-only `extension.json` before its code migrates, then generate `check-pr` and Windows target lists (and point the Windows ZIP at the generated `devlab.json`), test labels, the Nix source filter and inputs, `tools/ci` data, flake checks, devshell Python dependencies and the functional registries from metadata. | 8 (pilot) | Hosted checks and a devlab run selecting the pilot. |
| 10 | Text generation: split `text_model_runner.hpp` and `HipTextModelRunner`, add the experimental grammar-builder API and options with provider values; then migrate Qwen 27B, Flash-Next and DeepSeek V4 Flash, each moving its runner, tool syntax (DeepSeek's `tool_calls.cpp` leaves `gufo_http` here), input handling, artifact resolution and identity lines, and moving its `prompt`, `chat` and `bench` code into registered drivers. Sequenced with the cache RFC cards. | 6, 7 | All four text profiles plus `long-context` and `cache` for the shared split; per model: its modes with `long-context` and `cache`, forced tool-choice suites, identity golden test, matched timings per request and phase alternating main and PR for affected histories, benchmark continuity. DeepSeek evidence is local. |
| 11 | Out-of-tree builds: `GUFO_EXTENSION_DIRS`, the Nix `extensions` argument, a template repository and an extension input for the Windows workflow. Validate by building the pilot from a source tree where the in-tree copy is removed (and filtered out of Nix sources); `gufo_extension()` runs an include-allowlist check on external directories at configure time; the functional harness merges external registry fragments at run time from `GUFO_EXTENSION_DIRS`, with schema validation. Add test-only second providers per capability, or a genuinely new port, to validate the new-model goal. | 9 | Pilot tests from the external build. |
| 11b | Generate `mkServe`'s per-model arguments and validation (draft model paths, modalities, speculative modes, TTS voices) from metadata, together with the option-unification decision; renamed arguments are a public flake output change and carry `!`. | 9, 10 | Nix module evaluation checks; NixOS serve module tests. |
| 12 | API 1.0 when its criteria hold: publish the stability policy and start the external compatibility build list. | 8, 10, 11, 11b | Criteria recorded in this RFC. |

## Compatibility and rollout

- User-facing behavior does not change during migration, except for option
  unification if agreed and stricter selection of Qwen 27B artifacts. Those
  PRs carry `!` under the release policy and are announced once in the
  release notes; renamed options get no aliases.
- Cache compatibility identities stay byte-identical through the migration,
  checked by the identity golden test, so retained caches stay valid. A
  change to an identity is a deliberate, recorded decision.
- `extensions.lock` pins the extension list of official builds, so they do
  not change when a directory is added.
- Binary plugins are reconsidered when both hold: there is concrete demand
  for third-party models in official artifacts without a rebuild, and API 1.0
  has held for at least 90 days without a major change. The plugin layer
  would then be a C function table (ONNX Runtime-style), `libgufo_core` as a
  shared library, and a load-time toolchain check, built on the same C++
  interface without changing extension sources.

## Open questions

### Proposed decisions

| Decision | Proposed direction |
| --- | --- |
| Mechanism | Build-time source extensions; binary plugins deferred. |
| Compatibility promise | Source (API) only; 0.x until the measured 1.0 criteria hold. |
| Registration | Generated registry and constants from declarative metadata; no static-initializer registration. |
| Boundary | Step level per capability; the text interface succeeds the `TextModelRunner` contract; speculation and fused sampling inside the extension under the core's sampler-state contract. |
| Identity | Extension-ordered identity built with core-provided emitters, reproducing today's bytes. |
| Core ownership | HTTP/API surfaces, cross-request scheduling for text, cache policy, budgets and component cache backing, capabilities and their queue profiles, diagnostics, functional harness. |
| Target graph | Extensions link `gufo_ext_runtime` and `gufo_core`; `gufo_serving` sees them only through the registry, which each executable compiles. |
| Selection | Artifact keys plus a mandatory HIP-free probe; ambiguity is an error resolved with `--provider` or `GUFO_EXTENSIONS`. |
| One model per process | Unchanged; drafts, MTP heads and sidecars belong to their model's extension and artifact set. |
| Pilot | EmbeddingGemma 2. |

### To resolve before dependent implementation

- Whether the JSON tool-format fallback recognizes only the active model's
  dialect or every built dialect.
- Whether to unify speculative-draft options during migration, and their
  final names.
- Whether extension tests and docs move physically into the extension
  directory or stay under `tests/models/` and `docs/models/` with
  manifest-based registration.
- Whether out-of-tree extensions may depend on in-tree family libraries, and
  whether family libraries then fall under the stability policy.
- The name: "extension", "model module" or "plugin".
- Whether a community-tier extension can be accepted in-tree, and who
  maintains it when the original author leaves.
- Whether draft models from a different architecture (for example a DFlash
  draft) are ever loaded by a different extension than the target; the
  current answer is no.

## Evidence appendix

### Inventory commands

Run from the repository root at `5d66de8c`.

Model references outside model directories (the coupling inventory table):

```sh
for m in deepseek_v4_flash qwen38_flash_next 'models/qwen/|models::qwen[^0-9_]|qwen::' \
         qwen3_asr qwen3_tts embedding_gemma2 minimax_h3 qwen_image_21; do
  echo "== $m"
  grep -rlE "$m" src/cli src/core src/cache src/eval src/testing |
    while read -r f; do echo "  $f $(grep -cE "$m" "$f")"; done
done
```

Model mentions in the root build file (225 at `5d66de8c`; the pattern omits `models/qwen` and `ds4`, so it undercounts):

```sh
grep -cE 'minimax_h3|qwen3_tts|qwen3_asr|deepseek|qwen38|embedding_gemma|qwen_image' CMakeLists.txt
```

Cross-model, model-to-CLI and core includes per model directory:

```sh
for d in src/models/*/ tests/models/*/; do
  echo "== $d"
  grep -rhoE '#include "src/(models/[a-z0-9_]+|cli|cache|core/[a-z_0-9]+(/[a-z_0-9]+)?)' "$d" |
    sort | uniq -c | sort -rn
done
```

Runner contract churn (class body, then whole header):

```sh
git log --oneline --since=2026-09-30 \
  -L '/^class TextModelRunner/,/^};/:src/cli/serve/text_model_runner.hpp' |
  grep -cE '^[0-9a-f]{8} '
git log --oneline --since=2026-09-30 -- src/cli/serve/text_model_runner.hpp | wc -l
```

The coupling counts depend on the grep pattern and include comments; they
show where coupling lives and its scale, and step 1's isolation check
replaces them with a maintained report.

## References

- [Continuation cache RFC](../cache-redesign/RFC.md), especially the two-layer
  code structure and the adapter contract (`src/cache/adapter.hpp`).
- [Release process](../RELEASING.md).
- [vLLM plugin system](https://docs.vllm.ai/en/latest/design/plugin_system/).
- [SGLang: support new models](https://docs.sglang.io/docs/supported-models/support_new_models).
- [Modular MAX: serve custom model architectures](https://docs.modular.com/stable/develop/serve-custom-model-architectures/).
- [Triton Inference Server C API, `TRITONSERVER_ApiVersion`](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton_inference_server_230/user-guide/docs/cpp_api/function_tritonserver_8h_1ad8d9aeefc0090ebff033ab85a64898a9.html).
- [ONNX Runtime `OrtApiBase`](https://onnxruntime.ai/docs/api/c/struct_ort_api_base.html).
- [ggml: dynamic loading of backends (llama.cpp #10469)](https://github.com/ggml-org/llama.cpp/pull/10469)
  and [llama-cpp-python issue #2069](https://github.com/abetlen/llama-cpp-python/issues/2069).
- [Godot: what is GDExtension](https://docs.godotengine.org/en/stable/tutorials/scripting/gdextension/what_is_gdextension.html).
- [Zephyr modules](https://docs.zephyrproject.org/latest/develop/modules.html).
- [Linux: stable-api-nonsense](https://docs.kernel.org/process/stable-api-nonsense.html).
