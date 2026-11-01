# Model extensions implementation cards

This ledger implements the full [RFC](RFC.md), including its success criteria,
API stability policy and distribution work. Cards execute sequentially. Each
card has three distinct agents: implementation, independent review, and review
integration. Start the next card only after integration and required checks.
A report of a missing model, unavailable host or elapsed stability window is
pending evidence, never a quality pass. API 1.0 stays pending until its actual
time and feature criteria hold.

Current execution milestone: complete cards 01–14 with the three-agent workflow,
then stop. Cards 15–29 remain the documented later RFC scope and are not
authorized for this run. User decision (2026-10-31): the functional harness
(`tests/functional/run.py` rows, `prompt_lookup.py`, standard speed benchmarks
and main-control matched-timing comparisons) does NOT run during cards 06–14.
All functional evidence is deferred to one consolidated qualification window
after Card 14 verifies, scheduled solely by the orchestrator. Cards gate on
unit tests (including the identity goldens), CPU/hosted checks, static audits
and formatting; their functional requirements stay recorded as deferred,
pending evidence with exact commands — never as passes.

Statuses: pending, implementing, reviewing, integrating, complete. The agent
orchestrating this work records agent names, review findings, their disposition,
commands and retained evidence in the execution record. Splitting a card further
is allowed; dropping a requirement or replacing a gate with weaker evidence is
not. Dependencies below refer to card numbers. The ordering deliberately
serializes the RFC's parallel paths at the user's request.

## Shared acceptance gates

- Preserve behavior and cache identity bytes; the identity golden unit tests
  run in every card. Numerical quality and existing functional suites run on
  the candidate in the deferred post-Card-14 qualification window; matched
  main controls cover affected timing histories, with the same revision
  metadata, toolchain, settings, model,
  harness and cache history recorded. Text timings are per request and phase
  (5% and 10 ms), never averages. Include prefill, decode, TTFT and continuation
  restore; include `long-context` and `cache` for continuation changes and
  inspect loaded modes and actual drafts. Retain standard HTTP speed benchmarks
  and numerical quality tests. Investigate flags and fix confirmed regressions;
  noisy timing evidence stays explicitly inconclusive.
- In that qualification window, shared text changes need all four text
  profiles; model migrations need the affected model and modes. DeepSeek needs
  dedicated local time and actual weights (its Card 06 rows were dropped by
  user decision on host-capacity grounds and remain pending unless scheduled). Replay `tests/functional/pi_agent.py` for affected
  reported agent loops, retaining sessions and inspecting actual tool results. Avoid unrelated
  model sweeps and duplicated correctness runs on main.
- No functional-harness runs during a card (user decision 2026-10-31); they
  happen once, after Card 14, in the final qualification window. In-card
  evidence stays small: curated commands, verdicts, compact tables and short
  excerpts only; no raw server logs, per-row report trees or multi-megabyte
  artifacts under `docs/`; scratch above the size notes below lives in `/tmp`
  with mandatory cleanup.
- Functional test server disk caches always live in a dedicated
  `/tmp/<test-specific-dir>` and are always deleted when that test finishes,
  failures included; only durable report artifacts enter the card's evidence
  directory. Repository evidence trees never contain regenerable cache extents.
- `bench` and `prompt` routing changes require matched CLI output and timing
  controls as well as HTTP timings. Existing historical `BENCHMARKS.md` numbers
  are not migration baselines.
- Moved HIP sources retain per-source and target flags, includes, definitions,
  assertions and sanitizer state, or explicitly document intentional differences.
  Check unchanged kernel resources and identical ISA in stripped `NDEBUG`
  production code objects. A build directory named `release` proves nothing
  about its configured build type: audit the cache and compile commands.
- Run bounded hosted checks and affected unit tests. Linux and Windows builds
  are required where stated. Check any-single-extension-excluded builds for core
  tests and tools. Clean production release build time grows no more than 5%;
  touching one model recompiles only its targets and the final link.
- Default/discovery builds and shipped lock-mode builds have separate evidence.
  CPU lock builds and collision checks do not establish HIP release behavior.
  Run affected functional, numerical and performance gates on the actual
  lock-mode production artifact too, especially dialect selection and fallback.
  Report unshipped discovery independently.
- Ratchet isolation for each migrated model and core layer. No hidden baseline
  can excuse a newly migrated violation. Final isolation covers includes and
  symbol references, family dependency declarations and model knowledge in
  core build/CI/Nix/registries, with only generated output and explicit release
  decisions exempted. The initial scanner's lexical candidates are inventory
  evidence, not a proof of compiler-resolved symbol isolation.

## Decisions before dependent work

Card 03 records the spike's decisions on CLI drivers versus direct sessions,
runner split, provider metadata, option handling and draft ownership. Before
cards 06, 10, 18 or 23 fix their affected APIs, record decisions on:

- Active-model-only versus every-built-dialect JSON fallback (qualify forced
  tool choice in default and lock builds).
- Canonical speculative draft option names and `mkServe` argument changes;
  no aliases; public CLI/flake breaks carry `!`.
- Physical placement of model tests/docs versus manifest registration;
  model-specific functional suites stay in `tests/functional/`.
- Whether external extensions may depend on family libraries and the resulting
  stability scope. Resolve the Qwen header graph into acyclic libraries.
- Final terminology, community-tier in-tree acceptance and maintenance policy,
  and whether cross-architecture drafts stay inside their target extension.
- Existing cache RFC ownership and sequencing: its cutover takes precedence;
  native snapshots remain until that work's all-model cutover is proven.

## Sequential cards

Open a card for its detailed context, implementation plan, definition of done,
validation criteria and execution record. The scope remains the full original RFC;
the current implementation milestone stops after Card 14.

| Card | RFC step; dependencies | Detailed card | Status |
| --- | --- | --- | --- |
| 01 | 1; none | [Isolation inventory and implementation ledger](cards/01-isolation-inventory.md) | complete |
| 02 | 1b; 01 | [Exact continuation identity goldens](cards/02-identity-goldens.md) | complete |
| 03 | 2; 01–02 | [Throwaway Flash-Next interface spike](cards/03-flash-next-spike.md) | complete |
| 04 | 3; 03 | [Extension runtime, schema and explicit registration](cards/04-runtime-and-registration.md) | complete |
| 05 | 3; 04 | [Discovery, lock selection and provenance](cards/05-discovery-and-provenance.md) | complete |
| 06 | 4; 02, 05 | [Move model-specific speculation out of core](cards/06-model-owned-speculation.md) | pending |
| 07 | 5; 06 | [Extract acyclic Qwen family libraries](cards/07-qwen-family-libraries.md) | complete |
| 08 | 5; 07 | [Extract H3, evaluation and model-dependent targets](cards/08-h3-eval-and-test-targets.md) | complete |
| 09 | 6; 08 | [Separate serving policy from extension execution](cards/09-serving-runtime-boundary.md) | complete |
| 10 | 7; 09 | [Publish and measure the public extension API](cards/10-public-api-surface.md) | complete |
| 11 | 7; 10 | [Migrate models to public extension headers](cards/11-public-header-migrations.md) | complete |
| 12 | 7b; 11 | [Add in-tree model API test support](cards/12-model-api-test-support.md) | complete |
| 13 | 8 embeddings; 12 | [Migrate the EmbeddingGemma pilot](cards/13-embeddinggemma-pilot.md) | pending |
| 14 | 9; 13 | [Generate infrastructure from model metadata](cards/14-generated-infrastructure.md) | pending |
| 15 | 8 ASR; 14 | [Migrate speech recognition](cards/15-speech-recognition.md) | pending |
| 16 | 8 TTS; 15 | [Migrate speech synthesis](cards/16-speech-synthesis.md) | pending |
| 17 | 8 image; 16 | [Migrate image generation](cards/17-image-generation.md) | pending |
| 18 | 8 video; 17 | [Migrate video generation](cards/18-video-generation.md) | pending |
| 19 | 10 shared text; 18 | [Finalize the shared text-generation contract](cards/19-shared-text-contract.md) | pending |
| 20 | 10 Qwen 27B; 19 | [Migrate Qwen 27B text generation](cards/20-qwen27b-extension.md) | pending |
| 21 | 10 Flash-Next; 20 | [Migrate Flash-Next text generation](cards/21-flash-next-extension.md) | pending |
| 22 | 10 DeepSeek; 21 | [Migrate DeepSeek text generation](cards/22-deepseek-extension.md) | pending |
| 23 | 11; 22 | [Support external extension source trees](cards/23-external-extensions.md) | pending |
| 24 | 11; 23 | [Prove capability extensibility with second providers](cards/24-second-provider-conformance.md) | pending |
| 25 | 11b; 24 | [Generate Nix serving arguments and validation](cards/25-generated-serve-module.md) | pending |
| 26 | distribution/conformance; 25 | [Complete distribution recipes and conformance policy](cards/26-distribution-and-policy.md) | pending |
| 27 | success criteria; 26 | [Audit every RFC success criterion](cards/27-final-requirements-audit.md) | pending |
| 28 | 12 preparation; 27 | [Pin external compatibility fixtures and stability evidence](cards/28-external-compatibility-ledger.md) | pending |
| 29 | 12 graduation; 28 | [Graduate the extension API to 1.0](cards/29-api-one-graduation.md) | pending |

## Execution records

Card-specific records now live with their plans. Completed records describe
historical evidence and limitations; they are not instructions to redo completed
work. Cards 01–11 have now been through implementation, independent review and
review integration. See the
[three-agent workflow](cards/README.md#card-by-card-workflow) before continuing.

### Card 01

See [Card 01: execution record](cards/01-isolation-inventory.md#execution-record).

### Card 02

See [Card 02: execution record](cards/02-identity-goldens.md#execution-record).

### Card 03

See [Card 03: execution record](cards/03-flash-next-spike.md#execution-record).

### Card 04

See [Card 04: execution record](cards/04-runtime-and-registration.md#execution-record).

### Card 05

See [Card 05: execution record](cards/05-discovery-and-provenance.md#execution-record).

### Card 06

See [Card 06: execution record](cards/06-model-owned-speculation.md#execution-record).

### Card 07

See [Card 07: execution record](cards/07-qwen-family-libraries.md#execution-record).

### Card 08

See [Card 08: execution record](cards/08-h3-eval-and-test-targets.md#execution-record).
The extracted tiers are `gufo_minimax_h3`, `gufo_compare`, `gufo_eval` and the CPU-only
`gufo_deepseek_v4_flash_text`; the eval dataset is the core fixture
`tests/fixtures/eval/antirez-ds4.json`; the new
[`target-ownership.md`](target-ownership.md) contract and its
`extension_target_ownership_gate` own every model-dependent core test/tool. Its
unresolved gates are the deferred functional/performance half, Windows execution, the
asset-blocked H3/DeepSeek oracle rows, per-model extension exclusion (no
`extension.json` yet) and the shipped lock-mode artifact — none of which it claims.


### Card 09

See [Card 09: execution record](cards/09-serving-runtime-boundary.md#execution-record).
It extracted `gufo_serving` (scheduler, runner pool, component text cache,
`InferenceBackend`, continuation cache/store/identity), so the executable and the
model test trees no longer re-compile those seven sources; moved neutral
continuation state/snapshot bases, host snapshot budgets, the default HIP probe and
reasoning/tokenizer token types into `gufo_ext_runtime` (`src/ext/`); put the
backing/resource contracts in `src/cache/component_resources.hpp` with a
virtual `cache::Adapter::BackingPlan`; and placed the HIP backing provider in
`gufo_cache_hip` behind injected budget/log callbacks. Two permanent guards are new:
`--fail-on-serving-boundary` (`extension_serving_boundary_gate`) and the link-graph
guard (`cmake/TargetGraph.cmake`, `tools/ci/check-target-graph.py`,
`extension_link_graph_gate`, 24 protected targets), each with a run negative proof.
The blocking review (hosted `check-pr` include lost, an unselected layout suite, a
coverage-vacuous graph guard) plus the orchestrator-found `::@` item-group break and
two `codex exec review` findings (Windows system libraries, a ratio assertion that
fails at or below the 4 GiB host headroom) are fixed. Its unresolved gates — none of
which it claims — are the deferred functional/performance half, Windows execution,
the probe's HIP branch and asset-heavy serving rows, the low-memory budget branch,
per-model extension exclusion until `extension.json` exists, lock-mode runtime
qualification beyond `diagnose`, and the exempt ASR in-tree test edge until Card 12.

### Card 10

See [Card 10: execution record](cards/10-public-api-surface.md#execution-record).
It classified every model-used core/cache header into stable (`include/gufo/ext/`),
platform (`include/gufo/ext/platform/`), experimental (`include/gufo/ext/experimental/`),
host wrapper or private (the 2 test-only ASR edges to be removed in Card 12); published
44 forwarding headers; generated and verified the reproducible transitive `SURFACE`
closure (97 paths, 4,235 LoC, 31,365 tokens) with zero serving leaks and zero model
include leaks; bumped the extension API version to `0.1.2` (`kApiRevision = 2`, `kApiVersion = "0.1.2"`)
in `version.hpp` and `GufoExtension.cmake`; updated `CHANGES.md`; extended diff/policy tests
in `check_extension_api_test.py` and enforced rejection on missing `ext-api` label;
compiled representative HIP-less and HIP consumers; and passed all bounded CPU/GPU CTests
including continuation identity goldens. Its unresolved gates — none of which it claims —
are the deferred functional/performance half, Windows execution, model include migration
(deferred to Card 11), and the exempt ASR in-tree test edge until Card 12.

### Card 11

See [Card 11: execution record](cards/11-public-header-migrations.md#execution-record).
It migrated all eight model implementations (EmbeddingGemma, ASR, TTS, Image, H3,
Qwen 27B, Flash-Next, DeepSeek) from private `src/core/` and `src/cache/` include paths
to published `<gufo/ext/...>` headers (152 include directives across 95 files) with zero
non-include line changes; updated root `CMakeLists.txt` to include `${CMAKE_SOURCE_DIR}/include`;
preserved all 386 own root-relative includes and 65 family includes for their respective
cards; preserved the 2 test-only ASR serving edges for Card 12; added the permanent
`--fail-on-model-core-includes` isolation ratchet and CTest gate `extension_model_core_includes_gate`
with negative unit test proof; and verified continuation identity goldens and bounded
tests on both CPU and GPU presets. Its unresolved gates — none of which it claims — are
the deferred functional/performance half, native Windows execution, own and family coupling
(to be migrated with each model's card), and the exempt ASR test edge until Card 12.

### Card 12

See [Card 12: execution record](cards/12-model-api-test-support.md#execution-record).
It added the in-tree `gufo_test_support` STATIC library under `BUILD_TESTING` wrapping
HTTP and serving fixtures (`HttpRequest`, `HttpResponse`, `AsrService`, `AudioAsrApi`,
`AudioStream`, `TtsService`, `AudioTtsApi`, `EmbeddingService`, `EmbeddingsApi`, `ImageService`, `ImageApi`),
and migrated all four model API test suites (Qwen3-ASR, EmbeddingGemma 2, Qwen3-TTS, Qwen-Image-21)
from production `src/cli/serve/*.hpp` includes to `tests/support/model_api_test_support.hpp`;
removed production serving sources from test compile targets; tightened the serving boundary
ratchet in `tools/ci/check-extension-isolation.py` to forbid `src/cli/` includes from all
`BOUNDARY_LAYERS` (including model test trees); preserved `include/gufo/ext/SURFACE` (97 paths)
without leaking test machinery; proved exclusion under `BUILD_TESTING=OFF`; and passed bounded
CPU/GPU tests, continuation identity goldens, and format checks. Its unresolved gates — none of
which it claims — are the deferred functional/performance half (deferred to post-Card-14 qualification
window) and remaining CLI text runner coupling in text model integration harnesses.

