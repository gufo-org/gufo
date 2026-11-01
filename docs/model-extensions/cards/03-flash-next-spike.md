# Card 03: Throwaway Flash-Next interface spike

Status: **complete**. RFC step and dependencies: **2; 01–02**.

Current milestone (stop after Card 14).

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

A concrete throwaway experiment determines whether the text seam is practical before production APIs harden. The spike is investigation, and its inconclusive timing/ISA observations cannot qualify a later production migration.

This card is complete. The plan below describes its original intended work; use the execution record as historical evidence rather than rerunning it automatically.

### Original scope

In a separate throwaway jj change/workspace, move Flash-Next behind a draft text interface. Measure correctness and matched timings; decide runner split, options, metadata and CLI-driver/session alternative. Preserve findings in a spike report and update agreed design decisions; do not merge throwaway implementation.

## Implementation plan

1. Create a separate jj workspace/change for Flash-Next behind a draft neutral runner/provider interface; keep its implementation out of the production workspace.
2. Exercise provider metadata, typed options, factory shape and HIP-free header/probe compilation. Compare model-owned CLI drivers with the direct-session alternative.
3. Measure both actual Flash-Next modes, independent numerical checks, identity goldens, HTTP functional suites, standard HTTP speed and matched bench/prompt controls with full source/binary provenance.
4. Audit compile commands, moved code resources and production code objects; retain every failed/contaminated/control run and record provisional API decisions in SPIKE.md.

## Definition of done

- [x] SPIKE.md records runner/driver split, option handling, metadata/model-ID and draft ownership decisions with limitations.
- [x] Actual functional output/work/draft observations and quality scopes are retained independently of timings.
- [x] No throwaway implementation is merged and no inconclusive performance/ISA comparison is promoted to parity.

## Validation criteria

- Reproduce the focused tests, functional invocations, numerical checks and matched speed controls from ../evidence/card03/README.md when new spike evidence is needed.
- For every timing arm verify pinned toolchain, actual mode/draft counts, cache history and no competing GPU jobs; apply the shared strict per-phase margins.
- Inspect full compile/resource/ISA comparison and unresolved relocation/path differences. Completion means a reviewed experiment and recorded decisions, not universal ISA identity or production performance parity.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

- Implementation agent: `/root/card03_implement`; independent review agent:
  `/root/card03_review`; integration agent: `/root/card03_integrate`.
  All three stages complete for the throwaway investigation. Throwaway jj change `ovmkszut` in
  `/tmp/gufo-card03-spike`, parent `79930b75`; production spike code is not
  merged into this workspace. See [SPIKE.md](../SPIKE.md) for the implemented
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
  [card03 evidence](../evidence/card03/README.md), including the contaminated first
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
