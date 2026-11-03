## Review

Type: independent read-only review (phase 2 of 3), Card 05 — Discovery, lock selection and provenance.

**Tooling caveat up front:** this reviewer session has file-inspection tools only (`read`, `grep`, `find`, `ls`, `watchdog_diff`) and no shell. I therefore executed **no** CMake/CTest/Nix/git command. Everything below is a static audit of the actual worktree content plus cross-checks of the retained evidence files. Section C lists the exact commands the supervisor/integration agent must run; section B marks every verdict that depends on them as PENDING rather than passing it.

State verified: worktree `@` empty; Card 05 content is `@-` (kxnltrzq / 381c1648); parent `@--` = 3d2ce45a (Card 04, complete); `docs/model-extensions/RFC.md` SHA-256 = `2ce2becbc0bf0f844702b624de6827c2bae6eba921ab8a1ea883659fa2f96448` (matches HANDOFF; untouched).

### - Correct (verified in this review)

- **Closure claim is true.** `include/gufo/ext/{core_options.def,host.hpp,options.hpp,registry.hpp,version.hpp}` transitively includes only `<std…>` plus `"src/core/logging/logging.hpp"`, which includes only std headers. Closure = exactly the 6 paths in `include/gufo/ext/SURFACE`. The card record's "six header/definition paths" is accurate.
- **API 0.1.0 → 0.1.1 is genuinely additive.** `registry.hpp` only adds `ExcludedExtension`, `AddExcluded`, `Excluded()`, `ExcludedKeyDiagnostic()`, `ExtensionSelectionMode()`; `src/ext/runtime.cpp` only adds the two latter definitions; `version.hpp` adds numeric constants matching `kApiVersion = "0.1.1"` and `GUFO_EXTENSION_API_VERSION "0.1.1"` in `cmake/GufoExtension.cmake`. No removed/renamed/re-typed public declaration found, so a revision bump (not a minor bump) is the correct classification, and `include/gufo/ext/CHANGES.md` has a matching `## 0.1.1` entry.
- **First line of `--version` is preserved.** `src/cli/main.cpp:28-32` keeps `gufo version … (…)` as line 1 and appends `ExtensionDiagnostics(false)`; `tools/ci/linux_artifact.py` still validates `splitlines()[0]`.
- **Hardware JSON schema preserved.** `src/cli/diagnose/diagnose.cpp:412-422`: `--section extensions` is a new branch; `all`/human appends provenance; `json_mode` for hardware sections is untouched. Covered by the existing `strix_diagnose_help`/`strix_diagnose_json` tests and by `cli_provenance_test.py`'s `--section platform --json` assertion.
- **Policy statements match RFC stability item 4.** `AGENTS.md` and `docs/RELEASING.md:122-172` state the extension API + experimental flake `extensions` argument are outside Gufo's compatibility surface, that an extension break does not carry `!`, that ordinary PRs may bump the API version/changelog, and that the `ext-api` label + version + `CHANGES.md` entry are required for SURFACE/CMake/schema declaration changes. Nothing overclaims HIP/Windows execution; `#release`/lock-mode and "Windows workflow wiring is not execution evidence" are stated plainly.
- **`tools/ci/check-extension-api.py` is a real gate, not a rubber stamp.** It fails closed on: SURFACE drift, missing `kApi*` numeric agreement, CMake/runtime version mismatch, missing `ext-api` label, unchanged version, non-increasing version (numeric tuple compare), missing `## <version>` heading, deleted files (`read_new` → `""`), forwarded-declaration changes (via SURFACE), and CMake/schema token changes (`CONTRACTS`). Comments/whitespace are excluded while strings, raw strings and CMake bracket comments/literals are handled. An invalid `--base` hard-fails via `git rev-parse --verify … check=True` rather than silently treating every file as added. A broken base therefore produces spurious *failures*, not silent passes.
- **`tests/tools/check_extension_api_test.py` contains real assertions** (4 tests) including the 3-error "no label / no bump / no CHANGES" case and a genuine 3-level transitive-closure fixture. The evidence's "four policy groups" matches.
- **Truthful bootstrap is truthful.** `find src` shows **zero** `extension.json`, and `src/families/` does not exist at all. `extensions.lock` is `{"schema":1,"extensions":[]}` and the card/evidence/INFRASTRUCTURE/RELEASING all say the lock excludes nothing yet and that legacy models still link and route normally. No dummy production provider was fabricated.
- **Selection semantics hold up under reading.** Unset vs `*` vs explicit semicolon list vs empty string are distinguished by explicit `STREQUAL "*"` comparisons (not CMake truthiness), so false-like names (`off`, `no`, `false`, `notfound`) survive; the duplicate check uses `list(REMOVE_DUPLICATES)` comparison; ordering is re-derived from `GUFO_EXTENSION_ORDER` (dependency-first, lexically sorted traversal of sorted names), not from `file(GLOB)` order; both globs are exactly `src/models/*/extension.json` and `src/families/*/*/extension.json` with `CONFIGURE_DEPENDS`; `GUFO_PLATFORM_ONLY` genuinely `return()`s at `CMakeLists.txt:54-69` before `include(cmake/GufoExtensionDiscovery.cmake)` at line 101.
- **The shipped/unshipped collision gate does run in lock mode over unselected manifests** (loop over `GUFO_EXTENSION_NAMES`, tier = lock membership, not selection). That answers the "does a lock build really check unshipped metadata" question affirmatively.
- **Fixture recursion is not possible.** `tests/extensions/fixture/CMakeLists.txt` sets `GUFO_EXTENSION_FIXTURE ON` before `add_subdirectory(…/tests/extensions)`, and the parent forwards compiler/warnings/sanitizer settings. The `PROJECT_SOURCE_DIR` override dance in both the fixture and `discovery_test.py` is deliberate and consistent with `src/ext/CMakeLists.txt` using `${PROJECT_SOURCE_DIR}`.
- **Evidence artifacts are internally consistent** with what they claim: `focused-ctest.txt` really shows 12/12 with the expected 12 names (8 `extension_*` + `extension_isolation_report` + `gufo_version` + 2 `strix_diagnose_*`), `discovery-tests.txt` shows `Ran 9 tests … OK` against 9 real `test_*` methods, `format-check.txt` shows 725 files, `nix-release-flags.json` shows `-DGUFO_EXTENSIONS_FROM_LOCK=ON` with `releaseVersion 0.28.0` while `nix-default-flags.json` shows `OFF` with `development`.
- **Menu ordering is right:** `gufo_build_discovered_extensions()`/`gufo_target_extensions(gufo …)` run at `CMakeLists.txt:446-448`, after `src/ext` (398) and after `option(GUFO_ENABLE_WARNINGS/SANITIZERS)` (190), so role/sanitizer forwarding resolves correctly.

### - Finding: P1, hosted CI path never executed (evidence gap)

`flake.nix:524-529` adds a new mandatory block to **every** hosted CPU check (`mkTestCheck`, used by both `checks.x86_64-linux.test` and `.pr`):

```
cmake "$src" -B ../lock-build … -DBUILD_TESTING=OFF -DGUFO_EXTENSIONS_FROM_LOCK=ON
cmake --build ../lock-build --target gufo …
../lock-build/gufo diagnose --section extensions --json
../lock-build/gufo --version | grep -Fx 'Extension selection: lock'
```

and `cmake/Checks.cmake:62` newly registers `extension_discovery_test` in the hosted `gufo_pr_tests` list. `docs/model-extensions/evidence/card05/README.md` records only `nix develop`-local runs (focused CTest regex, hand-configured `build/card05-lock`). No `nix build .#checks.x86_64-linux.pr` output exists anywhere in `evidence/card05/`.

Impact: the branch-protection gate that runs for every future PR is itself unverified. `extension_discovery_test` shells out to `cmake`/`ninja` and performs ~10 nested configure+build cycles inside a Nix build sandbox; the flake provides both (`pkgsSys.cmake`, `pkgsSys.ninja`) and `testSource` admits `cmake/`, `schemas/`, `src/`, `tests/`, `tools/`, so it *should* work — but that is an argument, not evidence. This is exactly the "a planned command is not a result" rule in cards/README.md.

Smallest fix: run `nix build .#checks.x86_64-linux.pr`, retain the log as `evidence/card05/hosted-pr-check.txt`, and record that the pre-existing `check_docs_test` failures (164 broken links under the unrelated `hrx-system/` tree, per HANDOFF) are separate if they still block. If the full check cannot be run, at minimum execute the four lock-build lines above verbatim on the filtered source and retain that output.

### - Finding: P2, documented empty-selector behavior has no executable coverage

`docs/RELEASING.md:166-167` states "`*` selects discovery and an empty list selects none", and `cmake/GufoExtensionDiscovery.cmake` implements it (`set(selected "${GUFO_EXTENSIONS}")` with an empty `GUFO_EXTENSIONS`; the duplicate check compares `""` to `""` and does not fire). I traced the path and found no defect, but `tests/extensions/discovery_test.py` only exercises the empty value **together with** `GUFO_EXTENSIONS_FROM_LOCK=ON` (expecting `conflicts`). Nothing asserts that `-DGUFO_EXTENSIONS=` alone configures successfully with zero `builtExtensions` and every discovered name in `excludedExtensions`. Likewise there is no case for `GUFO_EXTENSIONS_FROM_LOCK=ON` with `extensions.lock` **missing** (`FATAL_ERROR "extension lock: missing …"`).

Fix: add both cases to `test_selection_failures`/a new method. Note this sits just outside DoD box 1's enumerated list (discovery/selection, unshipped reporting, collisions, ordering, glob reconfiguration), all five of which *are* covered — hence P2, not P1.

### - Finding: P2, schema/API drift in a `CONTRACTS` file

`schemas/extension.schema.json` is a public contract in `check-extension-api.py`'s `CONTRACTS` set and appears in `evidence/card05/api-diff-gate.txt`'s changed-path list (so exactly one non-comment line changed in this PR). Two inconsistencies visible in the current file:

1. `schemas/extension.schema.json:620` — `"title": "Gufo extension metadata API 0.1.0"`, while `version.hpp`, `cmake/GufoExtension.cmake` and `CHANGES.md` are at 0.1.1. The schema is the reference document out-of-tree authors read.
2. `schemas/extension.schema.json:160-166` — `keys.items` has `type: string, minLength: 1` but **no `pattern`**, while `cmake/GufoExtension.cmake` hard-fails on `NOT key MATCHES "^(gguf|safetensors|manifest|diffusers):.+$"`. The schema therefore accepts artifact keys that configure rejects, so `python3 tests/extensions/metadata_test.py --schema` passes metadata that cannot build.

Fix: identify the one changed schema line first (`git show 3d2ce45a:schemas/extension.schema.json | diff - schemas/extension.schema.json` — I could not run this), update the title to 0.1.1, and add the key `pattern` (additive; keep CMake as the production authority). If (2) predates Card 05, record it as an adjacent consistency note rather than a Card 05 regression.

### - Finding: P2, excluded-list ordering is a de-facto contract that is undocumented

The retained initial failure (`evidence/card05/initial-fixture-order-failure.txt`) shows the test expected dependency order `['base','alpha','beta']` but got lexical `['alpha','base','beta']`. The disposition was to change the expectation to lexical order. That is consistent with the implementation (`gufo_discover_extensions` ends with `list(SORT excluded)`) and defensible — excluded extensions are not built, so dependency order is meaningless for them, and the *built* order is still dependency-first (`test_explicit_reverse_order_builds_dependencies_first`).

I found no defect hidden by the correction. But the resulting contract ("built = dependency-first; excluded = lexical") is stated nowhere in `docs/model-extensions/INFRASTRUCTURE.md` or RELEASING.md, which is why it was re-litigated once already.

Fix: add one clause to the "Discovery and selection (card05)" section. This also closes the reviewer question "was the fixture expectation corrected or the behavior papered over?" with a durable answer.

### - Finding: P2, configure-time "unshipped" reporting is untested

`gufo_discover_extensions` emits `message(STATUS "gufo: unshipped discovered extension ${name}")` for every non-locked extension. The card's DoD asks for "unshipped reporting"; the tests assert the *runtime* `excludedExtensions` list, never the configure-time STATUS line. Low risk, but a requirement whose only assertion is indirect. Fix: assert the string in a `configure()` return value for one lock-mode case (the helper already returns `result.stdout`).

### - Finding: P2, collision gate scope and a misleading owner field

In the shipped/unshipped collision loop, the same-tier owner property is written **before** the cross-tier lookup:

```
set_property(GLOBAL PROPERTY "GUFO_DISCOVERY_${key}_${tier}" "${name}")
… get_property(collision GLOBAL PROPERTY "GUFO_DISCOVERY_${key}_${other}" SET)
```

Consequences: (a) two extensions in the *same* tier sharing an artifact key, model ID or option name are never flagged at configure time (last writer silently wins); (b) a reported collision always names the current `${name}` as one side, so the message can attribute the other side incorrectly after an overwrite. Artifact-key overlap between providers is explicitly allowed later ("Keys may overlap between providers for later probe-based selection"), and same-tier option-type conflicts are caught at runtime by `Registry::AddProvider`, so this is not a blocker — but the RFC's `disabled` tier is also not modelled yet, and the collision message prefix is `"extension lock: …"` even in discovery mode.

Fix: keep the cross-tier gate as-is, but record same-tier occurrences (or at least do not overwrite the owner before reporting), and use a mode-neutral message prefix.

### - Finding: P2, `extensions.lock` is validated in every mode (correct, undocumented)

`gufo_discover_extensions` validates locked names and locked dependency completeness unconditionally, including pure discovery builds. It is tested (`self.lock(["alpha"])` then expecting `missing dependency base` without `FROM_LOCK`) and it is a good property — a stale lock fails development builds too. RELEASING.md:164-165 only says "An unavailable platform or missing dependency in the lock is an error" without saying "in every mode". Fix: one clause in RELEASING.md/INFRASTRUCTURE.md.

### - Finding: P2, artifact manifest `version` field changed shape

`tools/ci/linux_artifact.py:137` now stores the **full multi-line** `--version` output in `manifest.json["version"]`, and `verify()` (line 151) compares that whole string. The new test `test_multiline_extension_provenance_retains_revision_validation` covers the in-repo consumer, and first-line revision validation is correctly preserved. However `manifest.json` is a persisted artifact read by harnesses outside this repository (devlab/homelab). Nothing in the tree still parses `version` as a single token, but the field's *shape* changed without a companion `version_first_line`. Fix: add `version_first_line` (additive) or confirm no external consumer splits on whitespace, and note the shape change in the evidence README.

### - Finding: P2, diagnostics surface asymmetry for excluded metadata

`infrastructure.md` says excluded extensions record "version/source/reason plus keys, model IDs and option names"; `Registry` does record all three, but `src/cli/extension_diagnostics.cpp:38-48` exposes only `keys` in JSON and the human output prints only name/version/source/reason. Either intentional (provenance only) or incomplete. Fix: state the intent, or include `model_ids`/`options`.

### - Finding: P2, nits

- `src/ext/runtime.cpp` appends a **second** `namespace gufo::ext { … }` block for the new code instead of extending the existing one. Harmless; inconsistent with the file's structure.
- `_gufo_extension_lock` is called twice per configure (`gufo_discover_extensions`, before and after `gufo_extensions_validate`), re-reading and re-validating the same file and appending duplicate `CMAKE_CONFIGURE_DEPENDS` entries.
- `set(unique "${selected}")` + `list(REMOVE_DUPLICATES)` reports `"GUFO_EXTENSIONS contains duplicate names"` for a stray separator such as `alpha;;beta`, where the real fault is an empty list element. Misleading diagnostic.
- `src/families/` does not exist yet, so `src/families/*/*/extension.json` under `CONFIGURE_DEPENDS` is only proven for the one-level `src/models/<name>` case (`test_default_lock_excluded_runtime` adds `gamma`). Card 07 creates the two-level tree; if a fresh configure is needed there, document it rather than rediscovering it.
- `.github/workflows/pr-title.yml`: the job is still named `Validate PR title` with `timeout-minutes: 2` while now also running the API gate. The label trigger set (`labeled`, `unlabeled`, `synchronize`) is correct for an `ext-api`-label gate — good — but the job name/scope no longer matches.

---

## B. DoD / validation verdict

| Item | Verdict | Basis |
| --- | --- | --- |
| DoD 1 — discovery/selection, unshipped reporting, collisions, deterministic ordering, glob reconfiguration have executable coverage | **PASS** (static), execution PENDING | All five named behaviors have real tests: `test_default_lock_excluded_runtime` (selection + lock + reconfigure-on-new-glob-match + idempotent re-read), `test_explicit_reverse_order_builds_dependencies_first`, `test_selection_failures` (7 negative rules), `test_unselected_api_and_dependency`, `test_shipped_unshipped_collisions` (keys/model_ids/options), `test_disabled_target_dependency`, `test_missing_build_manifest`, `test_platform_selection`, `test_false_like_name_and_production_test_isolation`. `discovery-tests.txt` shows `Ran 9 tests … OK`. Configure-time unshipped STATUS reporting and the explicit-empty-list mode are the gaps (P2 above). |
| DoD 2 — API label/version/CHANGES/SURFACE policy includes forwarded declarations and CMake/schema contracts | **PASS** (static), execution PENDING | `SURFACE` = verified transitive closure (6 paths, incl. forwarded `src/core/logging/logging.hpp`); `CONTRACTS` = `cmake/GufoExtension.cmake`, `cmake/GufoExtensionDiscovery.cmake`, `schemas/extension.schema.json`; gate enforces label/version-increase/CHANGES/drift and is wired to the real PR event with real base SHA and `toJSON(labels)`; 4 policy fixtures assert all three error classes plus deletion, CMake contract, comment-only and transitive-closure behavior. Schema/API title drift and the missing key `pattern` are P2 nits inside this box. |
| DoD 3 — independent review and integration completed and recorded before status → complete; no HIP/Windows qualification inferred from wiring | **PENDING → cannot pass yet** | The independent review is this document. **Third-agent integration has not run and the execution record has not been updated**, so the box is unfulfilled by definition. No overclaiming was found: evidence/README, card record, INFRASTRUCTURE and RELEASING all keep HIP/Windows/numerical/performance qualification deferred. |
| VC 1 — reproduce retained commands and hashes; no skipped selected checks | **PENDING** | Not executable from this session. Statically the retained artifacts are self-consistent (12/12 names and counts, 9 discovery cases, 6 SURFACE paths, 725-file claim). Also note `binary-sha256.txt` records local binaries whose hashes depend on embedded absolute paths; treat "reproduce the hashes" as "record fresh hashes + re-prove the CLI contract", and say so explicitly rather than implying bit-identity. |
| VC 2 — run `tests/tools/check_extension_api_test.py` and `tests/tools/linux_artifact_test.py`; preserve initial failures and dispositions | **PENDING** (execution) / **PASS** (artefact quality) | Both files contain real assertion-bearing tests with the claimed counts (4 policy groups; 6 artifact tests incl. multiline provenance and first-line-revision strictness). The retained initial fixture failure is preserved verbatim in `evidence/card05/initial-fixture-order-failure.txt` with a stated, defensible disposition (P2 ordering-doc note). |
| VC 3 — inspect actual binaries, selection provenance, Nix source-filter outputs; schedule future HIP gates | **PENDING** (execution) | Files exist and agree with each other (`configuration.json` shows `BUILD_TESTING=OFF`/`FROM_LOCK=ON` for the lock dir, `ENGINE_ENABLE_HIP=OFF`; `lock-version.txt`/`lock-diagnose.json` show `selectionMode: lock` with zero built extensions). Weakness: `nix-source-admissions.json` records only booleans with **no store path and no command**, so it is not reproducible evidence as-is — re-capture with the `nix eval --raw …src` path and the exact existence checks. Future HIP lock-artifact functional/numerical/performance gates are scheduled and explicitly not claimed. |

## C. Reproduction log

No commands executed (no shell in this session). Static inspection performed: `AGENTS.md`, `docs/model-extensions/{CARDS.md,l README.md, 05-…, RFC.md, INFRASTRUCTURE.md, HANDOFF.md}`, all 20 files in `evidence/card05/`, `CMakeLists.txt`, `cmake/{GufoExtension,GufoExtensionDiscovery,Checks}.cmake`, `include/gufo/ext/*`, `src/ext/{CMakeLists.txt,runtime.cpp}`, `src/cli/{main.cpp,extension_diagnostics.cpp,diagnose/diagnose.cpp}`, `schemas/extension.schema.json`, `tools/ci/{check-extension-api.py,linux_artifact.py}`, `tests/tools/{check_extension_api_test.py,linux_artifact_test.py}`, `tests/extensions/*`, `flake.nix`, `.devops/nix/package.nix`, `.github/workflows/{pr-title,windows-build}.yml`, `docs/RELEASING.md`.

Commands the supervisor/integration agent must run and retain (from repo root; new build dirs only — do **not** touch `build/cpu-test`, `build/card05-lock`, `build/card05-platform`):

```
nix develop -c cmake --preset cpu-test -B build/card05-rev-cpu
nix develop -c cmake --build --preset cpu-test -B build/card05-rev-cpu --target gufo extension_runtime_test extension_role_test --parallel 16
nix develop -c ctest --test-dir build/card05-rev-cpu -R '^(extension_.*|gufo_version|strix_diagnose_help|strix_diagnose_json)$' --output-on-failure
# expect 12/12, zero skipped, same 12 names as evidence/card05/focused-ctest.txt

nix develop -c cmake -S . -B build/card05-rev-lock -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_TESTING=OFF -DENGINE_ENABLE_HIP=OFF -DGUFO_EXTENSIONS_FROM_LOCK=ON
nix develop -c cmake --build build/card05-rev-lock --target gufo --parallel 16
nix develop -c python3 tests/extensions/cli_provenance_test.py build/card05-rev-lock/gufo
build/card05-rev-lock/gufo --version ; build/card05-rev-lock/gufo diagnose --section extensions --json
sha256sum build/card05-rev-lock/gufo   # compare to binary-sha256.txt and state match/mismatch reason

nix develop -c cmake -S . -B build/card05-rev-platform -GNinja -DGUFO_PLATFORM_ONLY=ON \
  -DBUILD_TESTING=OFF -DGUFO_EXTENSIONS=missing -DGUFO_EXTENSIONS_FROM_LOCK=ON
nix develop -c cmake --build build/card05-rev-platform --target gufo_platform gufo_cache --parallel 8

python3 tests/tools/check_extension_api_test.py        # expect Ran 4 tests OK
python3 tests/tools/linux_artifact_test.py             # expect Ran 6 tests OK
python3 tools/ci/check-extension-api.py --base 3d2ce45a --labels '["ext-api"]'
python3 tools/ci/check-extension-api.py --base 3d2ce45a --labels '[]'   # must FAIL: label required
python3 tools/ci/check-extension-api.py                                # drift-only, must PASS (6 paths)

nix eval --json .#packages.x86_64-linux.default.cmakeFlags
nix eval --json .#packages.x86_64-linux.release.cmakeFlags
p=$(nix eval --raw .#packages.x86_64-linux.default.src)
for f in include/gufo/ext/registry.hpp schemas/extension.schema.json extensions.lock \
         tests/models/deepseek_v4_flash/fixtures/antirez-ds4.json tests/extensions/discovery_test.py; do
  printf '%s: ' "$f"; test -e "$p/$f" && echo present || echo absent; done   # record "$p" itself

nix build .#checks.x86_64-linux.pr     # P1: the new lock-build block + discovery_test in the pr suite
git show 3d2ce45a:schemas/extension.schema.json | diff - schemas/extension.schema.json  # the 1 changed line
nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py          # no --fix
```

## D. Unverified / limitations

- **No execution at all.** Every numeric claim in the evidence README is unverified by me beyond internal consistency. Section C is mandatory, not optional.
- **The `@-` diff was not read as a diff.** I inspected the resulting worktree files, and I took the changed-path/changed-line inventory from `jj diff --stat` output supplied in the task framing plus `evidence/card05/api-diff-gate.txt`. I therefore could not verify that, e.g., `tests/extensions/metadata_test.py` (+4/−?) and `tests/extensions/runtime_test.cpp` (+10/−?) contain no weakened assertions, or identify the single changed schema line. A reviewer with a shell should run `jj diff -r @- -- tests/extensions schemas` and confirm no assertion was relaxed except the documented excluded-order expectation.
- **Delta claims about deleted/modified public declarations** rest on reading the current headers plus the "additive" test result; the diff gate itself is the authoritative check and must be run (section C).
- **Hosted CI remains the largest risk** (P1). `check-pr` also contains the pre-existing `check_docs_test`, which HANDOFF says reports 164 broken links under the unrelated `hrx-system/` tree; if `nix build .#checks.x86_64-linux.pr` fails, that failure must be reported as pre-existing and *not* folded into a Card 05 pass.
- **Windows** is untouched by design: `windows-build.yml` now selects `FROM_LOCK=OFF` for development and `ON` for releases and tolerates historical tags, but nothing here is Windows execution evidence, and the card does not claim otherwise.
- **Not examined in depth:** `tools/ci/check-format.py` behaviour on the new CMake/Python files (unaffected by the formatter's C++ file list), `src/cli/extension_diagnostics.hpp`, `include/gufo/ext/{host,options}.hpp` beyond the include closure, and any external (devlab/homelab) consumer of `manifest.json["version"]` — the last is the one I would most want checked before closing the P2 about the moved artifact field.

**Merge verdict: BLOCK (narrow).** I found **no correctness defect** in the implementation and would not ask for redesign. Blocking is limited to (1) the unexecuted hosted-CI path (P1) — a gate that will run on every future PR — and (2) DoD box 3, which requires the third integration agent plus execution-record update before the card leaves `implementing`. All P2 items are report-only and can be resolved or explicitly deferred with evidence by that agent.