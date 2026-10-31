# Card 05: Discovery, lock selection and provenance

Status: **implementing**. RFC step and dependencies: **3; 04**.

Current milestone (stop after Card 14).

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

Card 04 supplies explicit registries; this card selects discovered source extensions and reports truthful provenance. Legacy models still have no production manifests, so the empty lock is a bootstrap and cannot claim model exclusion. Implementation checks are complete; independent review and third-agent integration are pending.

### Original scope

Add discovery, `GUFO_EXTENSIONS`, lock mode and `extensions.lock`, excluded-key diagnostics, `diagnose` and `--version` provenance; respect `GUFO_PLATFORM_ONLY`. Add Nix source-filter admissions and release/default package distinction, unshipped/collision checks and CPU lock build. Amend `AGENTS.md`/`RELEASING.md` for extension API policy, ext-api label/change-log gate and compatibility scope. Schedule actual HIP lock-artifact gates, never claim default evidence covers them.

## Implementation plan

1. Review existing cmake/GufoExtensionDiscovery.cmake and root wiring before editing: discover in-tree model/family manifests, exact dependency-complete selectors, lock mode, excluded metadata and platform-only early return.
2. Verify default/lock CLI provenance in --version and diagnose, excluded-key errors, structural checks on unselected manifests and shipped/unshipped collisions without executing excluded build files.
3. Verify Nix source admissions, development/default versus release packages, release Windows selection and API policy updates to AGENTS.md, RELEASING.md and PR diff gates.
4. Start the pending independent read-only review, then have a third agent integrate findings and rerun affected fixtures. Record the future real HIP lock-artifact gates separately from the CPU bootstrap evidence.

## Definition of done

- [ ] Discovery/selection, unshipped reporting, collisions, deterministic ordering and glob reconfiguration have executable coverage.
- [ ] API label/version/CHANGES/SURFACE policy checks include forwarded declarations and CMake/schema contracts.
- [ ] Independent review and integration are completed and recorded before changing implementing to complete; no actual HIP/Windows qualification is inferred from wiring.

## Validation criteria

- Reproduce the retained commands and hashes in ../evidence/card05/README.md: focused CTests, default/lock CPU binary CLI checks, platform-only build and Nix evaluations; require no skipped selected checks.
- Run python3 tests/tools/check_extension_api_test.py and python3 tests/tools/linux_artifact_test.py; preserve all initial failures and final dispositions.
- Inspect actual binaries, selection provenance and Nix source-filter outputs. Schedule functional/numerical/performance evidence on the actual future lock-mode HIP artifact; collision checks are not proof of runtime equivalence.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

- Implementation agent: `/root/card05_implement`; implementation/checks complete.
  Independent review and third-agent integration remain pending. User-owned
  `RFC.md` is unchanged. See [INFRASTRUCTURE.md](../INFRASTRUCTURE.md) for exact
  selection/lock semantics and the truthful empty-lock legacy bootstrap.
- API 0.1.1 adds excluded metadata and selection-mode provenance; a first
  transitive SURFACE inventory and real PR diff/policy gate cover the current
  six header/definition paths plus CMake/schema contracts. Card10 still owns
  broader header classification and declaration/coupling measurements.
- Discovery/selection is wired to actual `gufo` registry generation and link;
  no production model manifests were fabricated. Exact dependency-complete
  selectors, future-API unselected metadata, platform exclusions, collisions,
  conditional HIP stubs, glob reconfiguration and build ordering have executable
  configure/build/runtime coverage. Card04 fixture metadata/targets now live in
  an isolated CMake subproject, preserving real-extension name freedom.
- Required formatter passes all 725 C++ files. Focused hosted CTest passes 12/12
  with zero skips (including nine discovery groups and current schema/policy
  fixtures); actual CPU default/lock executables and the platform-only leaf/cache
  build pass. Nix eval confirms source admissions and release/default flags.
  Real lock CLI provenance and multiline artifact regression checks pass.
  Commands, binary hashes, retained initial fixture-expectation failure and
  final results are in [evidence/card05](../evidence/card05/README.md).
- No production model arithmetic/scheduling/routing changes, no GPU/model
  qualification claimed. Default and actual shipped HIP artifact gates remain
  separate future requirements. Windows workflow wiring is not Windows
  execution evidence. Independent review and third-agent integration must
  complete before the card's status changes to complete.
