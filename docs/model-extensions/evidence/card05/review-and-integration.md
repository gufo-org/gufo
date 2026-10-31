# Card 05 — independent review and third-agent integration

Review phase: two distinct read-only agents.

- `reviewer` (static audit, no shell in that session):
  [review/independent-static-review.md](review/independent-static-review.md).
  Verdict "BLOCK (narrow)": no correctness defect found; blocking only on the
  unexecuted hosted-CI path and on the missing third-agent integration.
- `delegate` (independent execution pass, read-only, all commands run):
  [review/independent-execution-review.md](review/independent-execution-review.md).
  Reproduced every retained implementation claim, including the negative
  API-gate probe failing as required.

Integration phase: `worker` (sole writer). This file records its dispositions,
commands and results. The two reports above are verbatim transcripts, kept so
the findings stay durable instead of living under `/tmp` or `~/.pi`.

## Finding dispositions

| # | Finding | Disposition | Change | Proving check |
| --- | --- | --- | --- | --- |
| F-A | P1: hosted `mkTestCheck` lock-build block and `extension_discovery_test` in the pr list were never executed | RESOLVED | none needed; executed the gate | `nix build .#checks.x86_64-linux.pr` exit 0: `100% tests passed out of 89` including `extension_discovery_test` (40.82 s), plus the mandatory lock-mode build (`Extension selection: lock`, `-- gufo: extension mode=lock; included=[]; excluded=[]`). Retained in [hosted-pr-check.txt](hosted-pr-check.txt) |
| F-B | No positive coverage for an explicit empty `GUFO_EXTENSIONS=` | RESOLVED | `test_explicit_empty_selector_selects_none`: mode `explicit`, zero built, all three discovered names excluded with reason `not selected in explicit mode` | `extension_discovery_test` (13 cases) |
| F-C | No coverage for a missing `extensions.lock` | RESOLVED | `test_missing_lock_is_an_error_in_every_mode`: `extension lock: missing` in both discovery and lock mode | same |
| F-D | Configure-time `unshipped` STATUS line unasserted | RESOLVED | `test_unshipped_reporting_at_configure_time`: all names unshipped with an empty lock; only `beta` once locked | same; initial wrong expectation retained in `initial-unshipped-reporting-failure.txt` |
| F-E | Collision gate overwrote the same-tier owner before reporting; `extension lock:` prefix in discovery builds | RESOLVED | `cmake/GufoExtensionDiscovery.cmake`: report before recording, keep the first recorded owner per tier, prefix is now `gufo:` | `test_collision_names_the_first_recorded_owner` asserts `(alpha, gamma)`, i.e. the first recorded holder, not the last seen |
| F-F | `GUFO_EXTENSIONS=alpha;;beta` reported a misleading fault | RESOLVED, symptom corrected | dedicated `empty entry (stray ';')` diagnostic before the unknown-name check | new case in `test_selection_failures` |
| F-G | `_gufo_extension_lock` read twice per configure, duplicate `CMAKE_CONFIGURE_DEPENDS`, duplicated conflict rule | RESOLVED | one lock read reused after validation; second conflict check removed; validation strength unchanged | `extension lock: unknown/missing dependency` cases in `test_selection_failures`, `test_missing_lock_is_an_error_in_every_mode` |
| F-H | Second `namespace gufo::ext` block appended in `src/ext/runtime.cpp` | RESOLVED | merged into the existing block | formatter + `extension_runtime_test` |
| F-I | `pr-title.yml` job named "Validate PR title" while running the API gate too | RESOLVED | job name `Validate PR title and extension API`, `timeout-minutes` 2 → 5 | YAML inspection; gate itself proven by F-A/`extension_api_policy_test` |
| F-J | Implemented behavior undocumented: excluded ordering, lock validated in every mode, empty-selector wording | RESOLVED | ordering contract + lock-in-every-mode clauses in `INFRASTRUCTURE.md`; "in every configure mode" and stray-`;` rejection in `docs/RELEASING.md`. `*`/empty-list wording was already correct there and is now "explicitly empty" | doc review; `check_docs_test` |
| F-K | Docs claimed `diagnose` exposes excluded model IDs/options; code exposes keys only | RESOLVED (docs-only) | `INFRASTRUCTURE.md` now states `diagnose` reports excluded provenance and artifact keys only, with the full sets internal to the collision gate and `ExcludedKeyDiagnostic()` | `tests/extensions/cli_provenance_test.py` unchanged and passing |
| F-L | SURFACE/policy hygiene | RESOLVED | `python3 tools/ci/check-extension-api.py --write` run after edits: SURFACE unchanged at 6 paths; `include/gufo/ext/CHANGES.md` 0.1.1 entry extended with the `api_version` major broadening, the artifact-key prefix rule and the diagnostic attribution. No new API version (this PR is already the 0.1.1 revision bump with the `ext-api` label) | `extension_api_surface`, `extension_api_policy_test`, positive/negative diff gate |
| F-M | Schema `title` still `API 0.1.0` | RESOLVED | title → `Gufo extension metadata API 0.1.1`. The execution pass proved the staleness predates Card 05; Card 05 edited this same CONTRACTS file, so the fix is in scope | `extension_schema_test` |
| F-N | Schema accepted artifact keys that configure rejects | RESOLVED | added `"pattern": "^(gguf|safetensors|manifest|diffusers):.+$"` to provider `keys.items`, matching `cmake/GufoExtension.cmake` | three new `artifact_key` subTests in `SchemaTests`; existing valid fixtures still validate |
| F-O | `manifest.json["version"]` became multi-line | REJECTED as a code change, recorded | Consumer audit: only `tools/ci/linux_artifact.py` `verify()` (and `tests/tools/linux_artifact_test.py`) reads `manifest['version']`, comparing it verbatim and validating the first line's revision. `tools/ci/devlab_artifact.py` reads a different descriptor (`digest`, `artifact`). No tracked consumer tokenizes it (`git ls-files homelab` → 0 files, so `homelab/` is not repository content). Adding `version_first_line` would be an unused field; the full multi-line provenance is what the RFC's diagnostics requirement wants | `python3 tests/tools/linux_artifact_test.py` (6 tests) |
| F-P | Two-level `src/families/*/*/extension.json` glob unproven (`src/families/` does not exist) | DEFERRED to Card 07 | nothing fabricated; recorded as a Card 05 limitation and a Card 07 dependency | — |
| E-1 | `cmake --build --preset X -B <dir>` silently ignores `-B` and builds the preset's dir | REJECTED as a product defect, recorded as a verification-recipe caveat | The retained implementation README commands do not use `-B`; this record and future re-verification use the positional `cmake --build <dir>` form. The execution pass verified no retained binary hash changed (`build/cpu-test/gufo` still `22ce4bd9…`) | this record |
| E-2 | An external `jj describe` set the empty `@` description to "feat(extensions): card 5 follow-ups" | NOT INTEGRATION'S ACTION | left for the orchestrator, which owns the commit; no content change (`@` was empty) | `jj status` |
| E-3 | `nix-source-admissions.json` recorded booleans with no store path or command | RESOLVED | fresh capture with the command and store path below | `nix eval` block above |
| F-Q | The retained `docs-check.txt` line ("PASS: 6 task-owned documents, 18 local links") does not match any checker under `tools/ci/` | REJECTED as a defect, recorded | `tools/ci/check-docs.py` is whole-repository (`--root` only) and prints "PASSED: N markdown files and M local links/anchors"; the retained line came from an ad-hoc scoped grep, not a committed tool. The hosted `check_docs_test` is the real gate | hosted run: `PASSED: 150 markdown files and 903 local links/anchors`; locally `python3 tools/ci/check-docs.py --root .` reports 164 errors, every one of them under the untracked `hrx-system/` tree (0 outside it), matching HANDOFF |

## Integration validation (all commands from the repository root)

Fresh directories only: `build/card05-int-cpu`, `build/card05-int-lock`,
`build/card05-int-platform`. No pre-existing build directory was reused,
reconfigured or deleted. Raw logs: `/tmp/card05-int/`.

| Command | Result |
| --- | --- |
| `nix develop -c cmake --preset cpu-test -B build/card05-int-cpu` | exit 0 |
| `nix develop -c cmake --build build/card05-int-cpu --target gufo extension_runtime_test extension_role_test --parallel 16` | exit 0, 181 edges |
| `nix develop -c ctest --test-dir build/card05-int-cpu -R '^(extension_.*\|gufo_version\|strix_diagnose_help\|strix_diagnose_json)$' --output-on-failure` | exit 0, `100% tests passed out of 12`, zero "Not Run"/skipped/disabled lines, 54.83 s. Names: `gufo_version`, `strix_diagnose_help`, `strix_diagnose_json`, `extension_isolation_report`, `extension_runtime_test`, `extension_role_test`, `extension_cli_provenance_test`, `extension_discovery_test`, `extension_api_policy_test`, `extension_api_surface`, `extension_metadata_test`, `extension_schema_test` |
| `nix develop -c python3 tests/extensions/discovery_test.py -v` | exit 0, `Ran 13 tests` OK (was 9; four new cases) |
| `nix develop -c python3 tests/extensions/metadata_test.py` | exit 0, `Ran 12 tests` OK |
| `nix develop -c python3 tests/extensions/metadata_test.py --schema` | exit 0, `Ran 1 test` OK (includes the new artifact-key subTests) |
| `nix develop -c python3 tests/tools/check_extension_api_test.py` | exit 0, `Ran 4 tests` OK |
| `nix develop -c python3 tests/tools/linux_artifact_test.py` | exit 0, `Ran 6 tests` OK |
| `nix develop -c python3 tools/ci/check-extension-api.py` | exit 0, `PASS: extension API surface (6 paths)` |
| `nix develop -c python3 tools/ci/check-extension-api.py --write` | exit 0, SURFACE unchanged (`jj diff -- include/gufo/ext/SURFACE` → 0 files) |
| `nix develop -c python3 tools/ci/check-extension-api.py --base 3d2ce45a --labels '["ext-api"]'` | exit 0, `PASS: extension API surface (6 paths)` |
| `nix develop -c python3 tools/ci/check-extension-api.py --base 3d2ce45a --labels '[]'` | exit 1, `extension API changes require the ext-api PR label` (required failure) |
| `nix develop -c cmake -S . -B build/card05-int-lock -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=OFF -DENGINE_ENABLE_HIP=OFF -DGUFO_EXTENSIONS_FROM_LOCK=ON` | exit 0; `CMakeCache.txt` shows `BUILD_TESTING:BOOL=OFF`, `CMAKE_BUILD_TYPE:STRING=RelWithDebInfo`, `ENGINE_ENABLE_HIP:BOOL=OFF`, `GUFO_EXTENSIONS_FROM_LOCK:BOOL=ON` |
| `nix develop -c cmake --build build/card05-int-lock --target gufo --parallel 16` | exit 0 |
| `nix develop -c python3 tests/extensions/cli_provenance_test.py build/card05-int-lock/gufo` | exit 0, `PASS: actual gufo CLI provenance (lock)` |
| `build/card05-int-lock/gufo --version` | line 1 `gufo version development (unknown)`, then `Extension API: 0.1.1`, `Extension selection: lock`, `Built extensions: none` |
| `build/card05-int-lock/gufo diagnose --section extensions --json` | `{"extensionApiVersion":"0.1.1","selectionMode":"lock","builtExtensions":[],"excludedExtensions":[],"providers":[]}` |
| `nix develop -c cmake -S . -B build/card05-int-platform -GNinja -DGUFO_PLATFORM_ONLY=ON -DBUILD_TESTING=OFF -DGUFO_EXTENSIONS=missing -DGUFO_EXTENSIONS_FROM_LOCK=ON` then `--build ... --target gufo_platform gufo_cache --parallel 8` | both exit 0; the only mention of the two bogus selectors in the configure log is CMake's `unused-cli` warning, i.e. the platform-only return precedes discovery |
| `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py` (no `--fix`) | exit 0, `PASS: Checked 725 C++ files` |
| `nix eval --json .#packages.x86_64-linux.{default,release}.cmakeFlags` | exit 0; `-DGUFO_EXTENSIONS_FROM_LOCK=OFF` vs `ON`, `-DGUFO_RELEASE_VERSION=development` vs `0.28.0` |
| `p=$(nix eval --raw .#packages.x86_64-linux.default.src)` | `p=/nix/store/22i36v145df5wkscvvcyjdk7kq992kc2-source`; present: `include/gufo/ext/registry.hpp`, `include/gufo/ext/SURFACE`, `schemas/extension.schema.json`, `extensions.lock`, `tests/models/deepseek_v4_flash/fixtures/antirez-ds4.json`; absent: `tests/extensions/discovery_test.py`, `docs/model-extensions/RFC.md` |

Artifact hashes:

```
7da13ad01cc254e73baaa8de444a242ab10713fb7ce068bade5b1e0084ddb7da  build/card05-int-lock/gufo (this pass, fresh)
22ce4bd9b95f56c797241ff83613fff33784bb9450b531e46fcb0d8c523b326d  build/cpu-test/gufo        (unchanged, matches binary-sha256.txt)
6d5ae9cc3d79b1e14f5375f5574e0242165757d426e8d89b282a00fee5d15423  build/card05-lock/gufo     (unchanged, matches binary-sha256.txt)
```

Cross-build-directory bit-identity is neither expected nor claimed: each
executable embeds its own absolute build directory. The claim re-proved here is
the CLI contract and the unchanged retained artifacts, not hash equality across
directories.

## Hosted check (run last, after every code and test edit)

`nix build .#checks.x86_64-linux.pr` exercises the Card 05 additions to the
branch-protection path: the mandatory lock-mode build block in `flake.nix`'s
`mkTestCheck` and `extension_discovery_test` inside `gufo_pr_tests`. It exited 0
with `100% tests passed out of 89`, `check-docs` "PASSED: 150 markdown files and
903 local links/anchors", `check-format` "PASS: Checked 725 C++ files", and the
lock-mode block reporting `"selectionMode":"lock"` and `Extension selection:
lock`. Two earlier attempts are retained in
[hosted-pr-check.txt](hosted-pr-check.txt): they failed only because new
uncommitted evidence files were not in the Nix source snapshot yet, which is a
handoff artifact of this three-agent sequence rather than a product defect. The
only post-run documentation edits were filling the hosted result into this file
and into `hosted-pr-check.txt`; they add no links, and the documentation
reference check above was re-run afterwards over the whole worktree.

## Still pending (not passes)

- Real lock-mode HIP production artifacts still need their functional,
  numerical, dialect/fallback/forced-tool-choice and per-request performance
  gates as providers migrate (cards 13, 20–22). A CPU lock build and the
  collision gate prove neither runtime equivalence nor performance.
- Windows: `windows-build.yml` selection wiring only. No Windows execution
  evidence exists or is claimed.
- The two-level family glob is unproven until Card 07 creates `src/families/`.
- The hosted run's source snapshot already contained every code, test and
documentation edit except the final numbers written into this file and into
`hosted-pr-check.txt`; those additions introduce no new links. The gate was
re-run after committing, on the committed tree, and passed (`check-docs` 150
files / 904 links, `HOSTED_EXIT=0`); see the post-commit section of
`hosted-pr-check.txt`. A routine post-commit CI run remains the external
authority.

## Orchestrator closure verification

The orchestrator re-ran the decisive checks itself against the integrated tree
(fresh logs under `/tmp/card05-own/`, not agent reports), after the integration
agent handed off editing ownership:

- `ctest --test-dir build/card05-int-cpu -R '^(extension_.*|gufo_version|strix_diagnose_help|strix_diagnose_json)$'`
  → exit 0, 12/12, zero `Skipped`/`Not Run` lines (54.98 s).
- `python3 tests/extensions/discovery_test.py -v` → `Ran 13 tests ... OK`.
- `python3 tests/extensions/metadata_test.py` → `Ran 12 tests ... OK`;
  `--schema` → `Ran 1 test ... OK`.
- `python3 tests/tools/check_extension_api_test.py` → `Ran 4 tests ... OK`.
- `python3 tests/tools/linux_artifact_test.py` → `Ran 6 tests ... OK`.
- `python3 tools/ci/check-extension-api.py --base 3d2ce45a --labels '["ext-api"]'`
  → exit 0, `PASS: extension API surface (6 paths)`; the same call with
  `--labels '[]'` → exit 1 with `extension API changes require the ext-api PR
  label`; the drift-only invocation → exit 0; `--write` produced no `SURFACE`
  diff.
- `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py`
  → `PASS: Checked 725 C++ files`.
- Post-commit `nix build .#checks.x86_64-linux.pr` → exit 0 (documented above).
- `sha256sum docs/model-extensions/RFC.md` →
  `2ce2becbc0bf0f844702b624de6827c2bae6eba921ab8a1ea883659fa2f96448`, unchanged.

Verification-harness gotcha worth keeping: detaching these commands from a
sandboxed shell that exports a private `TMPDIR` makes every `nix develop` child
fail with `FileNotFoundError` under that directory once the parent call returns.
Re-running with an explicit `TMPDIR` outside the sandbox reproduced all results.

Closure result: the review-integration paths were committed as a single Conventional
Commit (`fix(extensions): tighten selection diagnostics and metadata key contract`)
containing exactly the 19 task-owned files. `docs/model-extensions/RFC.md` is
byte-identical to the recorded SHA-256 and was not touched by that commit; it
already entered history in the implementation commit `381c1648`, made by the user
before this phase.
- Independent pinned external compatibility builds and API 1.0 criteria are
  later cards (28, 29).
