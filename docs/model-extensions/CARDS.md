# Model extensions implementation cards

This ledger implements the full [RFC](RFC.md), including its success criteria,
API stability policy and distribution work. Cards execute sequentially. Each
card has three distinct agents: implementation, independent review, and review
integration. Start the next card only after integration and required checks.
A report of a missing model, unavailable host or elapsed stability window is
pending evidence, never a quality pass. API 1.0 stays pending until its actual
time and feature criteria hold.

Statuses: pending, implementing, reviewing, integrating, complete. The agent
orchestrating this work records agent names, review findings, their disposition,
commands and retained evidence in the execution record. Splitting a card further
is allowed; dropping a requirement or replacing a gate with weaker evidence is
not. Dependencies below refer to card numbers. The ordering deliberately
serializes the RFC's parallel paths at the user's request.

## Shared acceptance gates

- Preserve behavior and cache identity bytes. Numerical quality and existing
  functional suites run on the candidate; matched main controls cover affected
  timing histories, with the same revision metadata, toolchain, settings, model,
  harness and cache history recorded. Text timings are per request and phase
  (5% and 10 ms), never averages. Include prefill, decode, TTFT and continuation
  restore; include `long-context` and `cache` for continuation changes and
  inspect loaded modes and actual drafts. Retain standard HTTP speed benchmarks
  and numerical quality tests. Investigate flags and fix confirmed regressions;
  noisy timing evidence stays explicitly inconclusive.
- Shared text changes need all four text profiles; model migrations need the
  affected model and modes. DeepSeek needs dedicated local time and actual
  weights. Replay `tests/functional/pi_agent.py` for affected reported agent
  loops, retaining sessions and inspecting actual tool results. Avoid unrelated
  model sweeps and duplicated correctness runs on main.
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

Each row is an executable card: its work and required evidence are both part of
acceptance. All rows use the three-agent workflow above.

| Card | RFC step; dependencies | Work and concrete acceptance evidence | Status |
| --- | --- | --- | --- |
| 01 | 1; none | Maintain this complete ledger. Add report-only isolation inventory for cross-model/family/core/cache/CLI/test/tool includes and lexical symbols, model mentions in build/CI/Nix/registries including workflow assets, and Qwen direct/transitive header dependencies. Add meaningful fixtures and hosted registration. Run report and focused CPU checks. | complete |
| 02 | 1b; 01 | Expose pre-migration identity functions without changing serialization; record independent exact compatibility-byte goldens for Qwen 27B, Flash-Next and DeepSeek persistence descriptors, including draft variants and process-local identity decisions. Run goldens, with DeepSeek locally; retain baseline bytes before moving writers. | complete |
| 03 | 2; 01–02 | In a separate throwaway jj change/workspace, move Flash-Next behind a draft text interface. Measure correctness and matched timings; decide runner split, options, metadata and CLI-driver/session alternative. Preserve findings in a spike report and update agreed design decisions; do not merge throwaway implementation. | complete |
| 04 | 3; 03 | Implement API 0.x metadata schema and pure-CMake validator, `GufoExtension.cmake`, deterministic per-executable registries/constants, provider information, runtime API version, minimal Host/logging extraction, HIP-less exclusion stubs and provider option parser. Validate required fields, names, missing/cyclic dependencies, versions, conditions, shared option types and forbidden core options. No production Python dependency. CPU tests cover every configure rejection and explicit registration. | pending |
| 05 | 3; 04 | Add discovery, `GUFO_EXTENSIONS`, lock mode and `extensions.lock`, excluded-key diagnostics, `diagnose` and `--version` provenance; respect `GUFO_PLATFORM_ONLY`. Add Nix source-filter admissions and release/default package distinction, unshipped/collision checks and CPU lock build. Amend `AGENTS.md`/`RELEASING.md` for extension API policy, ext-api label/change-log gate and compatibility scope. Schedule actual HIP lock-artifact gates, never claim default evidence covers them. | pending |
| 06 | 4; 02, 05 | Move Qwen-specific speculative code out of core; neutralize draft tokenizer types and preserve rollback sampler/constraint state. Coordinate cache RFC and keep native snapshots. All four profiles plus long-context/cache (DeepSeek locally), identity goldens and matched timings. | pending |
| 07 | 5; 06 | Extract acyclic Qwen family libraries (tokens with independent Python mirror, config/text/vision/kernels and any justified common layer), make GGUF config extraction CPU-only, extract Qwen 27B and relink CLI/kernel tools. Resolve header-graph unassigned pieces explicitly. Keep ModelConfig/TextModelPreset until text migration. Linux/Windows, unit, Qwen/Flash-Next functional/performance and moved-HIP ISA/resource gates. | pending |
| 08 | 5; 07 | Extract H3, comparison support and eval targets; move eval dataset to core fixture with Nix filter exception. Give DeepSeek CPU text target without duplicating tool calls (still in HTTP until text migration). Assign or condition every model-dependent core test/tool via manifests. Record sanitizer/assertion state. Linux/Windows, affected CPU/model checks, H3 moved-HIP gate and exclusion-build coverage. | pending |
| 09 | 6; 08 | Extract gufo_serving and extend gufo_ext_runtime with neutral runner/chat/tool/tokenizer/reasoning/continuation types, default HIP probing, host budgets, cache backing/resource contracts and injected logging/budget callbacks. Keep HTTP semantics/scheduler/pool/cache policy in core. All four profiles, long-context/cache, native DeepSeek snapshots, exact goldens and matched timings. | pending |
| 10 | 7; 09 | Classify every model-used core header: stable/platform/experimental forwarding API, host wrapper or private. Publish headers and generated transitive `include/gufo/ext/SURFACE`, measure paths/declarations and coupling before committing stability. Enforce changes via version/CHANGES/ext-api gate, including forwarded src/cache headers. Test transitive closure and declaration changes; record the measured surface and any narrowing required. | pending |
| 11 | 7; 10 | Mechanical public-header include migrations one model at a time, preserving root-relative own includes temporarily: EmbeddingGemma, ASR, TTS, Image, H3, Qwen 27B, Flash-Next, DeepSeek. Record per-model Linux/Windows and affected unit checks; keep each rewrite reviewable and ratchet only migrated boundaries. | pending |
| 12 | 7b; 11 | Add in-tree-only gufo_test_support for HTTP/serving fixtures, remove model-test production CLI includes. Public extension API excludes this support. Run all affected model API tests and report remaining coupling explicitly. | pending |
| 13 | 8 embeddings; 12 | Migrate EmbeddingGemma pilot, extension-relative includes, metadata/options/pure probe/artifact resolution/provider/model IDs. Preserve extension batch scheduling and audio decoder ownership. Move model production dependency/data/runtime-library declarations into manifests. Existing API/numerical quality/speed gates, default and lock artifact, isolated build and fail-on-violation ratchet. | pending |
| 14 | 9; 13 | Add metadata-only manifests for every remaining model before its code moves. Generate hosted/Windows target lists, labels, ZIP devlab registry input, Nix filters/packages/install checks/devshell/reference/check fragments, CI resource/vendoring data, docs/notices and committed functional registries with drift checks. Preserve public flake check names and test-source production-derivation independence. Hosted checks and pilot-selected devlab run. | pending |
| 15 | 8 ASR; 14 | Migrate ASR and transcribe driver; move shared AudioBuffer/ResampleMono16k into host service; preserve streaming and core queue semantics. Metadata/registry/isolation and Linux/Windows checks; existing API/numerical quality and standard speed on actual default/lock artifacts. | pending |
| 16 | 8 TTS; 15 | Migrate TTS, provider request validation and voice variants; move DecodeBase64Wav to host services and declare cross-extension ASR assets for round trip. API/streaming/cancellation, numerical quality, standard speed and Linux/Windows checks; ratchet isolation. | pending |
| 17 | 8 image; 16 | Migrate Qwen-Image provider and diffusers keys, artifacts and job queue integration. Add devlab profile or retain explicit manual API/quality/speed evidence. Linux/Windows and default/lock gates; ratchet isolation. | pending |
| 18 | 8 video; 17 | Migrate H3 manifest keys, presets/model-ID routing, video driver and persistent jobs. Put Windows media/process facilities in platform leaves, remove model platform branching. Add devlab profile or retain manual API/numerical quality/speed evidence. Validate data/runtime-library packaging on Linux/Windows and ratchet isolation. | pending |
| 19 | 10 shared text; 18 | Finalize public runner contract versus pool/cache policy split, optional diagnostics/capabilities and experimental grammar-builder, provider-valued options, neutral inputs and tool syntax. Preserve schema helpers/client envelope ownership and apply spike decisions. Shared four profiles, long-context/cache, forced tool choice, goldens, matched per-phase timings, default/lock dialect behavior. | pending |
| 20 | 10 Qwen 27B; 19 | Move runner, prompt formatting/tokenizer/reasoning, vision inputs/cache, artifact resolution, identity ordering, speculation and prompt/chat/bench drivers into extension. Replace permissive fallback with supported qwen35 key/shape probe and clear errors; document CLI break with `!`. Run Qwen modes, forced tool choice, long-context/cache, goldens, CLI/HTTP continuity and ISA/resource checks; isolation ratchet. | pending |
| 21 | 10 Flash-Next; 20 | Move runner, MTP ownership/options, input handling, tool syntax, resolution, identity/decode-concurrency contribution and CLI drivers into extension. Affected actual modes/drafts, forced tool choice, cache/long-context, goldens and matched CLI/HTTP performance, default/lock artifacts, isolation ratchet. | pending |
| 22 | 10 DeepSeek; 21 | Move runner, DSML grammar/scanner/parser/reasoning, tool_calls out of HTTP, artifacts/drafts/native snapshots/persistence identity and CLI drivers into extension. Reserve local model/host time. Actual DeepSeek modes, forced tool choice, long-context/cache, numerical quality, goldens and matched CLI/HTTP timings; isolation ratchet. | pending |
| 23 | 11; 22 | Implement external directories, configure-time allowlist (including inherited directory flags/root includes), Nix extensions override and template repository. Add external registry fragment schema/merge support. Externalize pilot and remove its in-tree copy from both raw and Nix source trees for proof; build/test with zero Gufo diff. Add Windows reusable-workflow repository/revision inputs and template third-party ZIP recipe. | pending |
| 24 | 11; 23 | Add genuinely external test-only second providers for every capability (or genuine new models), run conformance and prove adding an existing-capability model changes only its own directories plus allowed generated output/lock release decision. Include zero core/build/preset/workflow/Nix/registry source changes. Cover ambiguity, excluded keys, provider override and HIP-free probes; builds exclude any single extension. | pending |
| 25 | 11b; 24 | Generate mkServe model arguments and validation for drafts/modalities/speculation/voices, apply recorded canonical option decision without aliases; use `!` for renamed public flake arguments. Run module evaluation/NixOS serve checks, correct release/default distinctions and external extension configuration. | pending |
| 26 | distribution/conformance; 25 | Publish source/CMake/Nix/Windows recipes, arbitrary-native-code trust and tier policy, capability conformance suites and diagnostics provenance. Track Toolboxes builder/runtime images and lock-mode edge integration as an external deliverable; retain actual evidence or leave pending (repository-only changes do not prove images). Verify packaging/install/licensing and all generated-output drift gates. | pending |
| 27 | success criteria; 26 | Final requirement-by-requirement audit: zero prohibited coupling and model knowledge, any-single-extension-excluded core/tool builds, clean/incremental build bounds, every model's actual numerical/API/performance evidence, CLI benchmark continuity, unchanged cache goldens, diagnostic provenance, unshipped discovery and shipped HIP lock artifact evidence. Fix confirmed regressions before closing. | pending |
| 28 | 12 preparation; 27 | Pin independent external EmbeddingGemma and text compatibility fixtures by repository/revision, never rewrite fixtures alongside API changes. Run every-release compatibility builds and scheduled conformance. Track measured SURFACE churn, bounded experimental text symbols (or zero experimental includes), every second provider and source-only stability policy/adapters. Publish dated evidence ledger. | pending |
| 29 | 12 graduation; 28 | Require actual 60-day stable text header and pilot external+CMake/Nix period, at least five merged serving features affecting API with only additive revision bumps, all public-only in-tree models and external text fixture compatibility. Audit all 1.0 criteria, then graduate/publish policy. Experimental hooks require explicit reviewed graduation after their own 60 unchanged days. Do not simulate elapsed time or count non-API commits. Binary loading remains deferred until separate demand and 90-day stable-API criteria. | pending |

## Execution record

### Card 01

- Implementation agent: `/root/card01_implement`; independent review agent:
  `/root/card01_review`; integration agent: `/root/card01_integrate`.
- Changes: this ledger, report checker, focused fixture tests and hosted CTest
  registrations. No numerical/runtime/performance code changes.
- Initial local report: 1,186 inspected files, 8,229 literal includes, 1,086
  cross-owner includes, 1,039 lexical namespace-reference candidates, 12,491
  name-reference candidates (including Python/tools paths), 1,422
  configuration mentions and 59 Qwen headers. Counts change as files change;
  regenerate the report instead of treating these as an enforced baseline.
- Commands: `python3 tests/tools/check_extension_isolation_test.py` (2 passed);
  `python3 tools/ci/check-extension-isolation.py --json-report
  /tmp/gufo-extension-isolation-card01.json` (report completed, zero exit).
  Complete per-location evidence is reproducible with `--details` or
  `--json-report`; `/tmp` output is ephemeral, not a checked-in baseline.
- Environment reported by orchestration: local Strix Halo gfx1151, ROCm 7.2.3
  and 125 GiB RAM. `release` intentionally uses RelWithDebInfo; qualify actual
  production flags (`NDEBUG`, `BUILD_TESTING=OFF`), revision and toolchain. Model directory presence
  does not establish each model's availability or any quality/timing pass.
- Additional checks: `python3 tests/tools/check_docs_test.py` (11 passed),
  Python byte compilation passed; both extension documents passed local-link
  validation (38 links). `nix develop -c cmake --preset cpu-test` configured
  successfully (sndfile emitted missing optional `flac.pc` diagnostics).
- Whole-workspace `python3 tools/ci/check-docs.py` fails on 164 pre-existing
  broken links in the unrelated `hrx-system/` checkout; no task-owned links
  failed. No unrelated files were changed to suppress those diagnostics.
- Hosted CTest selection: `nix develop -c ctest --test-dir build/cpu-test
  -R '^(check_extension_isolation_test|extension_isolation_report|check_docs_test)$'
  --output-on-failure` passed all three tests.
- Review finding P2 (resolved): parent-relative include candidates were not
  normalized before inventory lookup, losing edges and header closure. Lexical
  POSIX normalization now resolves all candidates before lookup. A focused
  fixture proves cross-owner reporting and transitive Qwen closure through
  parent-relative includes. The three existing DeepSeek parent-relative
  includes now resolve to their actual inventoried headers.
- Review finding P2 (resolved): embedded C++ raw-string and Python fixture
  include text produced false dependency edges. Include extraction now accepts
  only C/C++ directives whose `#` lies outside string/comment tokens, preserving
  original offsets and line numbers. Fixtures cover multiline raw strings,
  continued ordinary strings, Python fixture strings, comment directives and
  actual directives in inactive conditional branches.
- Integration verification: `python3 tests/tools/check_extension_isolation_test.py`
  passed all 4 tests. The hosted CTest command above was rerun after integration
  and passed 3/3 tests (scanner fixtures, inventory report and docs fixtures).
  The report command above completed successfully with revised counts: 1,186
  files, 8,228 literal includes, 1,086 cross-owner includes, 1,039 namespace
  candidates, 12,502 name candidates, 1,422 configuration mentions and 59 Qwen
  headers. Name candidates include the new fixture source; they are inventory,
  not violations. Original lexical limitations and the unrelated whole-workspace
  documentation gap above remain explicit. Card 01 is complete; no runtime or
  GPU qualification is claimed by these repository checks.


### Card 02

- Implementation agent: `/root/card02_implement`; independent review agent:
  `/root/card02_review`; integration agent: `/root/card02_integrate`. All three
  stages complete. No commit/revision change by these agents; user-owned
  `RFC.md` unchanged.
- Extracted the original three durable identity writers and two process-local
  writers into the temporary serving-only `continuation_identity.*` seam.
  State ABI selection, fingerprint validation and every emitted field/value/order
  remain unchanged. Every existing executable/test-support compilation of
  `inference_backend.cpp` also compiles the new source. Model migrations will
  move each writer to its owner; this is not public extension API design.
- Captured baseline bytes from literal parent `d1304fd6` writer bodies before
  extracting production writers. Frozen capture source/cases/driver, reproduction
  commands, detailed coverage and SHA-256 inventory are retained in
  [evidence/card02/README.md](evidence/card02/README.md). The baseline compile
  used `nix develop -c c++ -std=c++20 -I. /tmp/card02_record.cpp -o
  /tmp/card02_record`, then `/tmp/card02_record`, GCC 16.2.0, local Linux x86-64.
  No HIP define, model weights or GPU inference are needed to execute these
  actual writers. DeepSeek executed locally and has no model-skip path.
- Model-owned CPU tests compare 39 independent exact byte fixtures (Qwen 20,
  Flash-Next 11, DeepSeek 8), including numerical-storage/draft/configuration
  distinctions and existing identity-neutral options. Validation tests check
  malformed main/draft hashes and exact errors. DeepSeek's existing writer-only
  support-validation omission stays documented; constructor DSpark validation
  is unchanged. Separate process-local tests account for address/null spelling
  instability rather than persisting pointer bytes.
- `nix develop -c cmake --preset cpu-test` and the matching `gpu-test` configure
  succeeded. CPU target build command:
  `nix develop -c cmake --build --preset cpu-test --target
  qwen27b_continuation_identity_test qwen38_flash_next_continuation_identity_test
  deepseek_v4_flash_continuation_identity_test --parallel 3` passed. Focused
  CTest command: `nix develop -c ctest --test-dir build/cpu-test -R
  '^(qwen27b|qwen38_flash_next|deepseek_v4_flash)_continuation_identity_test$'
  --output-on-failure` passed 3/3, with zero skips.
- Both changed production HIP translation units compiled successfully:
  `nix develop -c cmake --build --preset gpu-test --target
  CMakeFiles/gufo.dir/src/cli/serve/inference_backend.cpp.o
  CMakeFiles/gufo.dir/src/cli/serve/continuation_identity.cpp.o --parallel 2`.
  Existing missing-field-initializer warnings remain in unmodified DeepSeek
  sampler/runner and Flash-Next decode code. This is compile evidence, not a
  full production executable link or GPU inference result; a full GPU target
  dry run showed 136 unrelated pending compilation/link steps.
- Required `nix shell --inputs-from . nixpkgs#clang-tools -c python3
  tools/ci/check-format.py --fix`, followed by the same command without `--fix`,
  passed (708 files). `sha256sum -c
  docs/model-extensions/evidence/card02/SHA256SUMS` passed all 42 entries.
- Numerical execution, scheduling, payload serialization and identity values
  are unchanged; no full functional, numerical or performance matrix was run
  for this serialization-only seam. This card does not establish the later
  runner-migration or extension lock-artifact gates.

- Additional implementation verification: the same three identity targets built
  under `gpu-test`, and `nix develop -c ctest --preset gpu-full -R
  '^(qwen27b|qwen38_flash_next|deepseek_v4_flash)_continuation_identity_test$'
  --output-on-failure` passed 3/3 with zero skips. These standalone tests remain
  HIP/model-load-free even in that configuration. Existing CPU cache uses GCC
  15.3.0; current pinned capture/GPU configuration uses GCC 16.2.0, and both
  reproduce the exact baseline fixture bytes. No performance comparison uses
  mismatched toolchains. Both changed production translation units also compiled
  in the CPU configuration with the analogous object-target command.
- Independent review found one P2 validation gap and no production behavior
  defect: Qwen's null-draft process identity and Flash-Next's disabled-MTP process
  identity had inequality checks without exact serialization checks. Integration
  added literal schema/field/order expectations for both; the Qwen null spelling
  is obtained independently by streaming `static_cast<const void*>(nullptr)`,
  and disabled MTP still pins both draft token limit and concurrency fields.
  Review also verified the durable writers against parent `d1304fd6` and frozen
  capture, and checked all four CMake consumers of `inference_backend.cpp`.
- After integration, all three focused targets rebuilt under both `cpu-test`
  and `gpu-test`; the focused CTest commands above passed 3/3 in each, with zero
  skips. The required formatting fix/check passed all 708 C++ files, and the
  retained SHA-256 inventory passed all 42 entries again. These results cover
  exact identity serialization, not model inference or performance.


### Card 03

- Implementation agent: `/root/card03_implement`; independent review agent:
  `/root/card03_review`; integration agent: `/root/card03_integrate`.
  All three stages complete for the throwaway investigation. Throwaway jj change `ovmkszut` in
  `/tmp/gufo-card03-spike`, parent `79930b75`; production spike code is not
  merged into this workspace. See [SPIKE.md](SPIKE.md) for the implemented
  experiment, provisional decisions and remaining validation.
- Supplemental card 02 production evidence: orchestration built and linked
  the full production executable successfully (297/297 build steps) at
  `79930b75`, SHA-256
  `04a885b0d7847c5567c233cb1aeecafd1de3ca4862cec3034c505e0ce6dad8b6`.
  A stale release cache initially mixed old ROCm device libraries with the
  pinned toolchain; the successful build used `nix develop -c cmake --fresh
  --preset release -DCMAKE_HIP_COMPILER=/nix/store/92bpbb2rmrkszhw83x63x7qzj9jsz324-rocm-toolchain/bin/clang`,
  then `nix develop -c cmake --build --preset release --parallel 4`.
  Audited production cache: `BUILD_TESTING=OFF`, `RelWithDebInfo` with
  `NDEBUG`, GCC 16.2, HIP Clang 24, ROCm 10.1; `diagnose` passed gfx1151.
  Default embedded revision is `unknown`, so source revision and executable
  hash are retained explicitly. The original stale-cache failure establishes
  no runtime result; no comparison uses that artifact.

- Implementation handoff: finalized throwaway source
  `50ba836fc4ab2fe7003371f3f290bdd48ea68b0a`, final candidate SHA-256
  `0bf5c8653f071884f4f575dff3d51ccc1a3b640900ea293b72fc65b91c2b0a9d`.
  Runner/provider factory extraction, neutral value-type/header split and
  decisions are concrete; no spike code was merged. Six focused CPU tests,
  HIP-free provider-header compile and 716-file formatter check pass.
- Both modes pass all 17 affected functional sub-suites and preserve outputs,
  prefill/cache work and actual MTP counts (3,322 proposals / 2,491 accepted
  in each initial baseline/candidate run). Exact writer goldens pass locally.
  All captures and commands are retained in
  [card03 evidence](evidence/card03/README.md), including the contaminated first
  baseline and every later control.
- **Production performance parity unproven:** 1,663 strict request/phase observations
  per mode. Off repeated qualification: 1,640 passed / 23 inconclusive / zero
  reproduced failures. Initial repeated MTP qualification reproduced three
  phases of `long_schema_cold`; the original failed result is retained. Two
  focused exact-history pairs do not reproduce it, yielding combined 1,604
  passed / 59 inconclusive / zero reproduced failures. Initial stall cause and
  parity remain unproven; third-agent integration retains the combined evidence
  as inconclusive, with no failure confirmed across expanded controls.
  Thresholds remain 5% and 10 ms.
- Source/compile audit: all 281 common normalized compilation commands match;
  all 74 gfx1151 resource notes and 71 `.text` sections match. Three MMQ text
  sections and seven constant sections differ; complete source-path/relocation
  investigation remains explicit. No universal ISA identity is claimed.
- P1 required-check finding resolved: integration ran the prepared standard
  HTTP speed script, all matched CLI controls and candidate GPU prefill check.
  All eight HTTP request/history/output/work/draft comparisons match; all 70
  available individual phases pass 5% and 10 ms (two fields unavailable). Both
  CLI modes preserve exact bench/prompt output hashes and all six CLI phases
  pass those margins, using durations derived from printed rounded rates.
  All 39 new raw capture files and their comparisons/invocations are retained in
  the speed archive and its exact hash inventory. No compile/test job overlapped
  timings. Actual per-arm compiler/source/binary/model provenance overrides raw
  model-bench's editing-host autodetection; see the evidence README.
- Numerical scopes: independent CPU `qwen38_flash_next.mtp_sampling` checks
  proposal/rejection/residual/RNG/stop contracts. Candidate GPU 16-token
  batched-versus-sequential prefill passes existing bounds (finite, winner rank
  1, RMSE 0.389394, cosine 0.990998, max error 2.20532). No independent full-model
  GPU forward oracle ran or is claimed. These checks plus unchanged arithmetic
  bodies and exact serving outputs/work provide scoped spike evidence, without
  waiving later migrations' model-specific quality gates.
- P2 metadata finding resolved with a retained HIP-free executable probe:
  artifact key, both canonical option names, speculative value, every typed
  option field and factory signature checked. It validates shape and
  representability only. Production metadata selection/parser/CLI-driver
  registration remain dependent-card gates; actual factory/options already
  execute on serving. Probe compiled/executed after all timing jobs finished.
- P2 timing disposition: the original failed MTP qualifier remains unchanged.
  Initial delays remain unexplained and **performance parity is unproven**;
  expanded exact-history controls support an inconclusive disposition, never a
  parity pass. Card 03 completes the throwaway experiment, measurements,
  decisions and review integration; it does not qualify a production migration.
  No source was merged from the throwaway workspace, no threshold widened and
  no arithmetic/performance fix applied. All later production gates remain.
