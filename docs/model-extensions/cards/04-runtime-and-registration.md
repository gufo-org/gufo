# Card 04: Extension runtime, schema and explicit registration

Status: **complete**. RFC step and dependencies: **3; 03**.

Current milestone (stop after Card 14).

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

Build-time extensions need deterministic explicit registration and CPU-visible unavailable metadata before any production provider migrates. Production configuration must work with CMake 3.21 and no Python runtime dependency.

This card is complete. The plan below describes its original intended work; use the execution record as historical evidence rather than rerunning it automatically.

### Original scope

Implement API 0.x metadata schema and pure-CMake validator, `GufoExtension.cmake`, deterministic per-executable registries/constants, provider information, runtime API version, minimal Host/logging extraction, HIP-less exclusion stubs and provider option parser. Validate required fields, names, missing/cyclic dependencies, versions, conditions, shared option types and forbidden core options. No production Python dependency. CPU tests cover every configure rejection and explicit registration.

## Implementation plan

1. Define API 0.x schema, version and registry headers in include/gufo/ext, schemas/extension.schema.json and src/ext/runtime.cpp; keep capability kinds core-owned.
2. Implement pure-CMake validation and target/HIP helpers in cmake/GufoExtension.cmake. Generate per-executable registry calls and per-extension constants; reference archive entries directly.
3. Extract minimal borrowed Host/logging services into core/runtime without a serving dependency, preserving shared logger state and forwarding compatibility headers.
4. Implement typed union/provider-specific option parsing and core option reservations. Add isolated synthetic fixtures for dependencies, conditions, versions, identifiers, inert strings and unavailable HIP entries.

## Definition of done

- [x] All schema/configure rejection categories have fixtures, including false-like metadata, keywords and recursive JSON NUL rejection.
- [x] CPU registry links available providers and metadata stubs without unavailable HIP factory/entry references.
- [x] Warnings, assertions, sanitizers and HIP options are applied by target role; no static initializers/whole-archive or production Python dependency.

## Validation criteria

- Run tests/extensions focused CTest selection recorded in ../evidence/card04/README.md, including available-provider and HIP-less variants; all must pass without model loads.
- Inspect linked CPU symbols to prove unavailable factories are absent, and build platform-only leaves/cache and logging/runtime production targets.
- Run formatter and full hosted schema validation. These checks establish infrastructure behavior only, not migrated-model, Windows or HIP release-artifact correctness.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

- Implementation agent: `/root/card04_implement`; independent review agent:
  `/root/card04_review`; third integration agent: `/root/card04_integrate`.
  Review findings are integrated and scoped checks pass. No commit/revision
  switch was made; user-owned `RFC.md` remains unchanged.
- Added API 0.1.0 metadata/registry, core-owned capability kinds, runtime version,
  typed provider option parsing and a borrowed Host/logging handle. Moved the
  existing logger/trace implementation to `src/core/logging/`, kept old headers
  forwarding and the single shared state, and gave `gufo_logging` its explicit
  platform dependency. No production model behavior or command parser changed.
- Added the full metadata schema and pure-CMake helpers, validation and
  deterministic per-executable registration/constants. Explicit unconditional
  extension `registration_target`/`entry` plus per-provider `target`/`entry`
  resolve the RFC example's implicit availability mapping: generated glue
  references only available provider entries and otherwise registers metadata
  stubs. Actual CPU binary symbol inspection proves unavailable HIP factory and
  registration symbols are absent. No static registration or whole-archive.
- Representation decisions and rationale (revision-floor version spelling,
  target conjunctions/roles, option provider membership, explicit-only values,
  defaults in code and the central core-owned option catalog) are in
  [INFRASTRUCTURE.md](../INFRASTRUCTURE.md). Schema accepts the broader optional
  packaging/docs/CI fields needed by later cards. Full schema validation runs
  in bounded hosted CPU checks with `jsonschema`; production configuration
  uses only CMake 3.21 features and does not discover Python.
- Independent review reproduced three P2 issues: CMake false-like metadata
  bypassed owner/selection checks, C++ keywords produced invalid generated
  identifiers, and JSON NUL silently truncated decoded metadata. Integration
  uses property presence/explicit emptiness, a deterministic `extension_`
  namespace prefix plus keyword rejection for entry symbols, and predecode
  escape-aware NUL rejection aligned with the recursive schema. Compiled edge
  fixtures cover valid false-like extension/provider/target/entry values,
  keyword namespace names, selective provider options and literal escape help;
  nested NUL and first-owner duplicate cases have configure regressions.
- [Evidence/card04](../evidence/card04/README.md) retains exact commands, 113
  configure diagnostics plus three compile/run commands, final 8/8 focused
  CTest output and actual linked
  symbol probes/binary hashes. Tests exercise ordinary static archive
  registration, dependency/deterministic ordering, every configure-validation
  category, typed values/union/selected-provider errors, inert metadata strings,
  disabled HIP metadata and optimized test assertions. Synthetic available
  provider variant passes 2/2; platform-only platform/cache build and production
  logging/runtime library targets pass. Required formatter passes 723 files.
- Existing CPU cache initially mixed GCC 15.3 with new GLIBC 2.43 codec
  dependencies; refreshed with `nix develop -c cmake --fresh --preset cpu-test`
  under pinned GCC 16.2, then all focused targets/checks passed. No source
  workaround was introduced. Production `include/` Nix source admission moved
  forward from card 05 because this card's always-built runtime requires it;
  hosted check/format/static-analysis inputs also admit the new public headers,
  and hosted tests admit the schema.
- Card 05 still owns discovery/selection, discovered-but-unselected diagnostics,
  lock mode, CLI provenance and release policy. No production model migration,
  timing/numerical gate, full GPU execution, Windows execution or lock-artifact
  qualification is claimed or waived. The execution milestone remains card 14.
