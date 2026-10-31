# Card05 evidence

Implementation evidence first, then the independent review and third-agent
integration in `review-and-integration.md`, with the verbatim transcripts
`review/independent-static-review.md` and
`review/independent-execution-review.md`. Those three files are referenced as
plain paths because the Nix source filter takes tracked files only, so a link
here would dangle in any hosted check that runs before they are committed; the
navigable links live inside `review-and-integration.md` itself. The worktree
parent during implementation was `3d2ce45a`.
The user-owned RFC SHA-256 remains
`2ce2becbc0bf0f844702b624de6827c2bae6eba921ab8a1ea883659fa2f96448`.

The lock is deliberately empty: there are no migrated production manifests yet.
Legacy models remain linked/routed exactly as before. The real CPU executable
therefore truthfully reports zero built extension providers in both modes;
synthetic fixtures exercise nonempty production discovery/build/registration.
No dummy production provider was added to make the bootstrap look migrated.

## Checks

- `nix develop -c cmake --preset cpu-test`, then
  `nix develop -c cmake --build --preset cpu-test --target gufo extension_runtime_test extension_role_test --parallel 4`
  built the real executable and isolated Card04 fixtures.
- `nix develop -c ctest --test-dir build/cpu-test -R '^(extension_.*|gufo_version|strix_diagnose_help|strix_diagnose_json)$' --output-on-failure`
  passed 12/12, zero skips; retained in `focused-ctest.txt`. Discovery contains
  nine configure/build/runtime cases, metadata contains twelve test groups
  (many rejection subcases), and policy contains four groups. The schema uses
  jsonschema only in hosted tests. Card04 fixtures are an isolated CMake
  subproject, avoiding production alpha/beta/base name/target contamination.
  After final warning/sanitizer forwarding and policy-token fixture additions,
  the isolated fixture rebuild plus four affected runtime/policy tests passed
  again (`final-fixture-policy.txt`).
- `nix develop -c cmake -S . -B build/card05-lock -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=OFF -DENGINE_ENABLE_HIP=OFF -DGUFO_EXTENSIONS_FROM_LOCK=ON`,
  then `nix develop -c cmake --build build/card05-lock --target gufo --parallel 4`
  configured and linked the actual CPU lock executable. A final incremental
  build and `nix develop -c python3 tests/extensions/cli_provenance_test.py build/card05-lock/gufo`
  passed after source changes. `lock-cli.txt`, version/JSON captures and binary
  hashes retain runtime evidence. These are CPU artifacts, never HIP release
  qualification. Production `BUILD_TESTING=OFF` config does not discover Python.
- `nix develop -c cmake -S . -B build/card05-platform -GNinja -DGUFO_PLATFORM_ONLY=ON -DBUILD_TESTING=OFF -DGUFO_EXTENSIONS=missing -DGUFO_EXTENSIONS_FROM_LOCK=ON`
  and `nix develop -c cmake --build build/card05-platform --target gufo_platform gufo_cache --parallel 3`
  passed. Deliberately conflicting/unknown selectors were never read because
  the platform-only early return precedes discovery.
- `nix eval --json .#packages.x86_64-linux.default.cmakeFlags` and its `release`
  counterpart prove default discovery/release lock flags. `nix eval --raw
  .#packages.x86_64-linux.default.src` plus direct filesystem inspection proves
  production source admits `include/`, `schemas/`, `extensions.lock` and the
  existing DeepSeek eval fixture, while excluding new test sources. Exact
  flags/admissions are retained as JSON. No GPU Nix package build is claimed.
- `python3 tools/ci/check-extension-api.py --base 3d2ce45a --labels '["ext-api"]'`
  passes the actual diff gate (`api-diff-gate.txt`); four policy groups cover
  forwarded core/cache declarations, deletion, CMake changes, required label,
  version/change-log obligations, comment-only changes, strings and closure.
  The source inventory has six current paths. Runtime numeric constants and
  CMake API version must agree. The PR workflow runs this gate with actual
  base SHA/label data; hosted CTest checks inventory drift and fixtures.
- `python3 tests/tools/linux_artifact_test.py` passes 6/6, including multiline
  extension provenance and revision validation. `linux-artifact-tests.txt`
  retains the output. Linux artifact manifests retain full version provenance;
  first-line revision validation remains strict.
- Required `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py --fix`,
  followed by the command without `--fix`, passed all 725 C++ files. Scoped
  links pass in the six task-owned documents (18 local links); see retained logs.
  The known unrelated whole-workspace hrx-system broken links were not changed.

The initial expanded fixture test expected dependency-first ordering even for
excluded manifests, whose missing dependencies are intentionally not visited.
`initial-fixture-order-failure.txt` retains that failure. The expectation now
matches canonical lexical exclusion order; final discovery and full focused
CTest runs pass. No runtime defect, model failure or timing threshold was
hidden by that fixture correction. Integration made that ordering contract
explicit in [INFRASTRUCTURE.md](../../INFRASTRUCTURE.md): built extensions are
dependency-first, excluded extensions are canonical lexical.

Integration retained one more failure of its own:
`initial-unshipped-reporting-failure.txt`, where a newly added test wrongly
expected configure-time "unshipped" reporting to follow the selection mode
instead of `extensions.lock` membership. The test expectation was corrected and
the implementation was left unchanged, because the lock-independent meaning is
the intended contract.

## Review and integration

Two distinct read-only reviewers and one integration agent closed the card.
The static audit found no correctness defect and blocked narrowly on the
unexecuted hosted-CI gate; the execution pass reproduced every claim above, in
cluding the negative API-gate probe failing as required. Integration added four
discovery cases (explicit empty selector, missing lock in every mode,
configure-time unshipped reporting, deterministic collision attribution) and one
selector-failure case, fixed the collision-owner attribution and its mode-neutral
prefix, removed the duplicated lock read, agreed the metadata schema with the
configure-time artifact-key rule and the 0.1.1 version, and documented the
ordering, lock-in-every-mode and excluded-metadata contracts. Focused CTest is
still 12/12 with zero skips, discovery is now 13 cases, and the formatter still
passes 725 C++ files. See `review-and-integration.md` for the per-finding
dispositions, exact commands, fresh artifact hashes and pending gates.

## Scope and remaining gates

No model arithmetic, scheduling, request routing or cache identity changed.
No full functional/numerical/performance model sweep was run for metadata and
diagnostics infrastructure. Actual shipped HIP artifacts still need affected
functional, numerical, dialect/fallback/forced-tool-choice and per-request
performance evidence as providers migrate. CPU lock builds/collision checks do
not prove equivalence with default builds. Windows release/default paths and
version checks are wired but Windows execution is not claimed. Actual Toolboxes
image publication remains outside this repository implementation evidence.
Card10 still owns broader public-header classification/declaration measurements;
API1.0 graduation and independent pinned external compatibility builds remain
later cards. Card06 was not started.
