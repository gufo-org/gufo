# Card 01: Isolation inventory and implementation ledger

Status: **complete**. RFC step and dependencies: **1; none**.

Current milestone (stop after Card 14).

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

The RFC starts with implicit coupling rather than an enforceable boundary. Inventory must precede target moves, and must distinguish lexical candidates from compiler-resolved dependencies. This ledger must retain the complete RFC scope even where work is serialized.

This card is complete. The plan below describes its original intended work; use the execution record as historical evidence rather than rerunning it automatically.

### Original scope

Maintain this complete ledger. Add report-only isolation inventory for cross-model/family/core/cache/CLI/test/tool includes and lexical symbols, model mentions in build/CI/Nix/registries including workflow assets, and Qwen direct/transitive header dependencies. Add meaningful fixtures and hosted registration. Run report and focused CPU checks.

## Implementation plan

1. Inventory includes and lexical references across models, families, core, cache, CLI, tests and tools using tools/ci/check-extension-isolation.py; include model mentions in build, CI, Nix, registries and workflow assets.
2. Trace the direct and transitive Qwen header graph, preserving source locations and owner assignments; normalize parent-relative paths without turning fixture text into edges.
3. Add scanner fixtures in tests/tools/check_extension_isolation_test.py and bounded registration in cmake/Checks.cmake; retain reproducible report commands and document lexical limits.
4. Maintain dependencies, review dispositions and all later acceptance gates in the card ledger.

## Definition of done

- [x] All coupling categories and Qwen closure are reportable with file/line evidence.
- [x] Fixtures prove real directives, parent traversal, comments/string exclusions and transitive edges.
- [x] Hosted report remains report-only; numerical/runtime claims are absent.

## Validation criteria

- Run python3 tests/tools/check_extension_isolation_test.py; all fixtures must pass.
- Run python3 tools/ci/check-extension-isolation.py --json-report /tmp/gufo-extension-isolation.json; inspect category counts and representative paths, expecting a successful report rather than zero coupling.
- Run the focused hosted CTests recorded below plus python3 tests/tools/check_docs_test.py; retain scanner limitations and unrelated documentation failures.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

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
