# Card 05: Discovery, lock selection and provenance

Status: **complete**. RFC step and dependencies: **3; 04**.

Current milestone (stop after Card 14).

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

Card 04 supplies explicit registries; this card selects discovered source extensions and reports truthful provenance. Legacy models still have no production manifests, so the empty lock is a bootstrap and cannot claim model exclusion. Implementation, independent review and third-agent integration are complete; the remaining gates are the real lock-mode HIP artifacts and Windows execution, which this card never claims.

### Original scope

Add discovery, `GUFO_EXTENSIONS`, lock mode and `extensions.lock`, excluded-key diagnostics, `diagnose` and `--version` provenance; respect `GUFO_PLATFORM_ONLY`. Add Nix source-filter admissions and release/default package distinction, unshipped/collision checks and CPU lock build. Amend `AGENTS.md`/`RELEASING.md` for extension API policy, ext-api label/change-log gate and compatibility scope. Schedule actual HIP lock-artifact gates, never claim default evidence covers them.

## Implementation plan

1. Review existing cmake/GufoExtensionDiscovery.cmake and root wiring before editing: discover in-tree model/family manifests, exact dependency-complete selectors, lock mode, excluded metadata and platform-only early return.
2. Verify default/lock CLI provenance in --version and diagnose, excluded-key errors, structural checks on unselected manifests and shipped/unshipped collisions without executing excluded build files.
3. Verify Nix source admissions, development/default versus release packages, release Windows selection and API policy updates to AGENTS.md, RELEASING.md and PR diff gates.
4. Start the pending independent read-only review, then have a third agent integrate findings and rerun affected fixtures. Record the future real HIP lock-artifact gates separately from the CPU bootstrap evidence.

## Definition of done

- [x] Discovery/selection, unshipped reporting, collisions, deterministic ordering and glob reconfiguration have executable coverage. `extension_discovery_test` is 13 cases: selection (default, lock, explicit, explicitly empty), glob reconfiguration on a new manifest, dependency-first build order, canonical lexical exclusion, configure-time `unshipped` STATUS lines, shipped/unshipped collisions for keys/model IDs/options with deterministic owner attribution, missing lock, unknown/duplicate/empty selector entries, unavailable platforms, unselected future API, disabled target and missing build manifest.
- [x] API label/version/CHANGES/SURFACE policy checks include forwarded declarations and CMake/schema contracts. `extension_api_surface` (6-path transitive closure including forwarded `src/core/logging/logging.hpp`), `extension_api_policy_test` (4 groups), and the real PR diff gate passing with the `ext-api` label while `--labels '[]'` exits 1. The schema is a `CONTRACTS` path and now agrees with configure on the artifact-key prefix rule.
- [x] Independent review and integration are completed and recorded before changing implementing to complete; no actual HIP/Windows qualification is inferred from wiring. See [review and integration](../evidence/card05/review-and-integration.md). HIP and Windows remain explicitly unqualified below.

## Validation criteria

- Reproduce the retained commands and hashes in ../evidence/card05/README.md: focused CTests, default/lock CPU binary CLI checks, platform-only build and Nix evaluations; require no skipped selected checks.
- Run python3 tests/tools/check_extension_api_test.py and python3 tests/tools/linux_artifact_test.py; preserve all initial failures and final dispositions.
- Inspect actual binaries, selection provenance and Nix source-filter outputs. Schedule functional/numerical/performance evidence on the actual future lock-mode HIP artifact; collision checks are not proof of runtime equivalence.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

- Three distinct agents. Implementation: `/root/card05_implement`.
  Independent review: pi `reviewer` (static audit of the actual worktree and the
  retained evidence) plus pi `delegate` (read-only execution pass that ran every
  retained command). Integration: pi `worker` (sole writer). User-owned `RFC.md`
  is unchanged; see [INFRASTRUCTURE.md](../INFRASTRUCTURE.md) for exact
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
  with zero skips (including the then-nine discovery groups, now 13 after
  integration, and current schema/policy fixtures); actual CPU default/lock executables and the platform-only leaf/cache
  build pass. Nix eval confirms source admissions and release/default flags.
  Real lock CLI provenance and multiline artifact regression checks pass.
  Commands, binary hashes, retained initial fixture-expectation failure and
  final results are in [evidence/card05](../evidence/card05/README.md).
- Review verdict and dispositions. The static audit found no correctness defect
  and blocked narrowly: no hosted-CI execution (P1) and the missing third-agent
  pass. Its P2 items were each resolved, rejected with evidence, or deferred; the
  execution pass reproduced every implementation claim, including the negative
  API-gate probe failing as required, and proved the stale schema title and the
  missing artifact-key `pattern` predate this card. Integration then added four
  discovery cases and one selector-failure case, made the shipped/unshipped
  collision report name the first recorded owner behind a mode-neutral `gufo:`
  prefix, gave a stray `;` its own diagnostic, removed the duplicated lock read
  and conflict rule, merged the appended `namespace gufo::ext` block, aligned the
  schema with the 0.1.1 title and the configure-time key-prefix rule, renamed the
  PR workflow job that also runs the API gate, and documented the ordering,
  lock-in-every-mode and excluded-metadata contracts. Full table: integration
  evidence `docs/model-extensions/evidence/card05/review-and-integration.md`.
- Integration validation: focused CTest still 12/12 with zero skips; discovery
  is now 13 cases; metadata 12 groups plus the schema group; policy 4 groups;
  Linux artifact 6 tests; formatter 725 C++ files; a fresh lock-mode CPU
  executable reproduces the CLI provenance byte-for-byte (`7da13ad0…`, fresh
  directory hashes are not expected to match older ones because the build path
  is embedded); the platform-only build still ignores the deliberately bogus
  selector; Nix default/release flags and source admissions reproduce against
  store path `/nix/store/22i36v145df5wkscvvcyjdk7kq992kc2-source`.
- The hosted gate itself now has executed evidence: `nix build
  .#checks.x86_64-linux.pr` exits 0 with `100% tests passed out of 89`, the
  mandatory lock-mode build reporting `Extension selection: lock`, `check-docs`
  passing 150 markdown files (903 links pre-commit, 904 after) and
  `check-format` 725 C++ files
  (`docs/model-extensions/evidence/card05/hosted-pr-check.txt`).
- Evidence is now durable in-repository instead of `/tmp`-bound: the two raw
  reviews plus the integration record and both retained initial failures live
  under `docs/model-extensions/evidence/card05/`. One handoff gotcha is recorded
  there too: the Nix source filter takes tracked files only, so a hosted check
  run before new evidence files are committed cannot see them, and
  `cmake --build --preset X -B <dir>` silently ignores `-B` and builds the
  preset's own directory.
- No production model arithmetic/scheduling/routing changes, no GPU/model
  qualification claimed. Default and actual shipped HIP artifact gates remain
  separate future requirements. Windows workflow wiring is not Windows
  execution evidence. The two-level `src/families/*/*/extension.json` glob is
  unproven until Card 07 creates `src/families/`.
- Closure. The orchestrator re-ran the focused CTest (12/12, zero skipped), the
  13 discovery cases, metadata (12 plus schema), policy (4), Linux artifact (6),
  the extension-API gate in both directions (exit 0 with the `ext-api` label,
  exit 1 without it), the drift inventory (6 paths, unchanged by `--write`) and
  the formatter (725 files), then re-ran `nix build .#checks.x86_64-linux.pr`
  on the committed tree (exit 0, documentation check clean). Details:
  [review and integration](../evidence/card05/review-and-integration.md).
  The 19 task-owned paths are committed as
  `fix(extensions): tighten selection diagnostics and metadata key contract`.
  `docs/model-extensions/RFC.md` is unchanged and was not part of that commit.
