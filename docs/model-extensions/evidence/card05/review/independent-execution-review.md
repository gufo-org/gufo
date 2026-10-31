# Card 05 — independent review EXECUTION PASS (raw results)

Run at 2026-10-31 06:17–06:35 on /home/mixer/gufo (jj `@-` = kxnltrzq / git 381c1648, parent `@--` = 3d2ce45a).
Read-only pass: no repository file was created, modified, deleted or formatted; no jj/git write operation was run by this pass.
Raw logs: `/tmp/gufo-card05-repro/` (numbered 01–17 + `schema-before.json`).
Fresh build dirs created by this pass: `build/card05-repro-cpu`, `build/card05-repro-lock`, `build/card05-repro-platform` (all CMake build dirs, ignored by jj).

## Headline facts first

1. **All 12 focused CTests reproduce, zero skips.** (`/tmp/gufo-card05-repro/06-ctest-focused.log`, `CTEST_EXIT=0`)
2. **The retained command form `cmake --build --preset cpu-test -B <dir>` is broken**: `cmake --build` does not accept `-B`, prints `Unknown argument -B` and silently builds the *preset's* binaryDir (`build/cpu-test`), not the fresh dir.
   - Consequence: step A as written rebuilt **pre-existing `build/cpu-test`** (276 ninja edges, `/tmp/gufo-card05-repro/02-build-cpu.log` line 5 `Unknown argument -B`, line `[1/5] ... cmake -E create_symlink gufo /home/mixer/gufo/build/cpu-test/gufo-server`).
   - Damage check: `sha256sum build/cpu-test/gufo` = `22ce4bd9b95f56c797241ff83613fff33784bb9450b531e46fcb0d8c523b326d`, i.e. **identical to the retained hash**, file mtime still `10-31 05:39`. `jj status` = "The working copy has no changes." No tracked file and no retained evidence hash changed.
   - This pass re-ran the build correctly with the positional form `cmake --build build/card05-repro-cpu ...` (step A' below). The `-B` form appears in the reviewer's command list, not in `docs/model-extensions/evidence/card05/README.md` (whose recorded command has no `-B`).
3. **An external `jj` write op touched the workspace during this review window** (not by this pass): `jj op log` shows `e6521626a984 ... describe commit 73aa17d3...` / `args: jj describe -m 'feat(extensions): card 5 follow-ups'` at 06:06:45. Effect: the empty working-copy commit `@` (twvsqlym) went from `(empty) (no description set)` / commit 73aa17d3 to `(empty) feat(extensions): card 5 follow-ups` / commit 3dd431c7. `jj diff -r @ --stat` = `0 files changed` — content is untouched, only a description string on an empty changeset. Flagging for the orchestrator/integration agent to accept or clear (`jj describe -m ''`).
4. Every other retained claim below reproduced, including the negative API-gate probe failing as required.

## Per-step results

### A. Build in the fresh cpu-test dir
- As instructed: `nix develop -c cmake --build --preset cpu-test -B build/card05-repro-cpu --target gufo extension_runtime_test extension_role_test --parallel 16` → built `build/cpu-test` instead (see headline 2). Log `/tmp/gufo-card05-repro/02-build-cpu.log`; no `FAILED:`/`ninja: build stopped`.
- Corrected (`A'`): `nix develop -c cmake --build build/card05-repro-cpu --target gufo extension_runtime_test extension_role_test --parallel 12` → exit 0, log `/tmp/gufo-card05-repro/03-build-repro-cpu.log`, last edges:
  ```
  [179/181] Linking CXX static library libgufo_http.a
  [180/181] Linking CXX executable gufo
  ```
  Artifacts produced: `build/card05-repro-cpu/gufo` (113,703,520 B, 06:27) and the isolated Card04 fixture project `build/card05-repro-cpu/tests/extensions/isolated-fixture/extensions/{extension_runtime_test,extension_role_test,libalpha_cpu.a}` (06:26).

### B. Focused CTest
`nix develop -c ctest --test-dir build/card05-repro-cpu -R "^(extension_.*|gufo_version|strix_diagnose_help|strix_diagnose_json)$" --output-on-failure` → `CTEST_EXIT=0`, **`100% tests passed out of 12`**, zero "Not Run"/skipped/disabled lines:

```
 1/12 Test   #8: gufo_version .....................   Passed    0.01 sec
 2/12 Test  #11: strix_diagnose_help ..............   Passed    0.00 sec
 3/12 Test  #12: strix_diagnose_json ..............   Passed    0.00 sec
 4/12 Test  #36: extension_isolation_report .......   Passed    3.33 sec
 5/12 Test #109: extension_runtime_test ...........   Passed    0.00 sec
 6/12 Test #110: extension_role_test ..............   Passed    0.00 sec
 7/12 Test #111: extension_cli_provenance_test ....   Passed    0.05 sec
 8/12 Test #112: extension_discovery_test .........   Passed   33.81 sec
 9/12 Test #113: extension_api_policy_test ........   Passed    0.03 sec
10/12 Test #114: extension_api_surface ............   Passed    0.02 sec
11/12 Test #115: extension_metadata_test ..........   Passed   10.01 sec
12/12 Test #116: extension_schema_test ............   Passed    0.10 sec
```
Same 12 names as `evidence/card05/focused-ctest.txt`. `discovery_test.py` contains exactly 9 `def test_*` methods (`test_default_lock_excluded_runtime`, `test_false_like_name_and_production_test_isolation`, `test_explicit_reverse_order_builds_dependencies_first`, `test_selection_failures`, `test_unselected_api_and_dependency`, `test_shipped_unshipped_collisions`, `test_disabled_target_dependency`, `test_missing_build_manifest`, `test_platform_selection`), matching `Ran 9 tests` in `discovery-tests.txt`. Log `/tmp/gufo-card05-repro/06-ctest-focused.log`.

### C. Lock-mode CPU artifact (fresh dir)
Chain log `/tmp/gufo-card05-repro/04-lock-chain.log`: `CONFIGURE_LOCK_EXIT=0`, `BUILD_LOCK_EXIT=0`, `PROVENANCE_EXIT=0` (`PASS: actual gufo CLI provenance (lock)`).

`grep -E '^(BUILD_TESTING|CMAKE_BUILD_TYPE|ENGINE_ENABLE_HIP|GUFO_EXTENSIONS|GUFO_EXTENSIONS_FROM_LOCK):' build/card05-repro-lock/CMakeCache.txt`:
```
BUILD_TESTING:BOOL=OFF
CMAKE_BUILD_TYPE:STRING=RelWithDebInfo
ENGINE_ENABLE_HIP:BOOL=OFF
GUFO_EXTENSIONS:STRING=*
GUFO_EXTENSIONS_FROM_LOCK:BOOL=ON
```

`./build/card05-repro-lock/gufo --version` (verbatim, byte-identical to retained `lock-version.txt`):
```
gufo version development (unknown)
Extension API: 0.1.1
Extension selection: lock
Built extensions: none
```

`./build/card05-repro-lock/gufo diagnose --section extensions --json` (verbatim, byte-identical to retained `lock-diagnose.json`):
```
{"extensionApiVersion":"0.1.1","selectionMode":"lock","builtExtensions":[],"excludedExtensions":[],"providers":[]}
```

`./build/card05-repro-lock/gufo diagnose` (human, tail; hardware section still present and unmodified, provenance appended after it):
```
  [PASS] toolchain: Toolchain and driver stack verified
      - amdgpuStatus: loaded
      - kernelRelease: 7.2.9
      - rocmVersion: 7.2.3
Extension API: 0.1.1
Extension selection: lock
Built extensions: none
```

Hashes:
```
435d3248e8593de5e3e86647bf7900a190b5b20ca5561d8dde13bb4ff8024abe  build/card05-repro-lock/gufo   (this pass, fresh)
22ce4bd9b95f56c797241ff83613fff33784bb9450b531e46fcb0d8c523b326d  build/cpu-test/gufo            (matches retained)
6d5ae9cc3d79b1e14f5375f5574e0242165757d426e8d89b282a00fee5d15423  build/card05-lock/gufo         (matches retained, untouched)
```
The fresh lock binary hash **does not equal** `6d5ae9cc…`. Reason observed, not assumed: each binary embeds its own absolute build directory (`strings` hit `/home/mixer/gufo/build/card05-repro-lock` vs `/home/mixer/gufo/build/card05-lock`), sizes 113,691,088 vs 113,689,472 (1,616 B apart). Same source revision, same compiler, same cache settings. **Bit-identity between build dirs was NOT observed and is not claimed.** The two retained artifacts themselves are unchanged.

### D. Platform-only build
`nix develop -c cmake -S . -B build/card05-repro-platform -GNinja -DGUFO_PLATFORM_ONLY=ON -DBUILD_TESTING=OFF -DGUFO_EXTENSIONS=missing -DGUFO_EXTENSIONS_FROM_LOCK=ON` → `CONFIGURE_PLATFORM_EXIT=0`; `nix develop -c cmake --build build/card05-repro-platform --target gufo_platform gufo_cache --parallel 8` → `BUILD_PLATFORM_EXIT=0` (`[22/22] Linking CXX static library src/cache/libgufo_cache.a`).
Proof the bogus selector `missing` was never read — the only mention of the two options in the whole configure log is CMake's unused-variable warning (`/tmp/gufo-card05-repro/05-platform-chain.log` lines 21–27):
```
CMake Warning (unused-cli):
  Manually-specified variables were not used by the project:

    GUFO_EXTENSIONS
    GUFO_EXTENSIONS_FROM_LOCK
```
No `discovery`/`unshipped`/`extension lock:` output exists in that log.

### E. Python tool tests
- `python3 tests/tools/check_extension_api_test.py` → exit 0, `Ran 4 tests in 0.001s / OK` (`/tmp/gufo-card05-repro/10-api-policy-tests.log`).
- `python3 tests/tools/linux_artifact_test.py` → exit 0, `Ran 6 tests in 0.003s / OK`; its captured `--version` sample shows `Extension API: 0.1.1`, `Extension selection: discovery`, `Built extensions: none` (`/tmp/gufo-card05-repro/11-linux-artifact-tests.log`).

### F. API diff gate — positive, negative, drift
```
F1 python3 tools/ci/check-extension-api.py --base 3d2ce45a --labels '["ext-api"]'   exit=0
   Extension API changed paths: cmake/GufoExtension.cmake, cmake/GufoExtensionDiscovery.cmake,
     include/gufo/ext/registry.hpp, include/gufo/ext/version.hpp, schemas/extension.schema.json
   PASS: extension API surface (6 paths)
F2 python3 tools/ci/check-extension-api.py --base 3d2ce45a --labels '[]'            exit=1
   extension API changes require the ext-api PR label
   Extension API changed paths: <same 5 paths>
F3 python3 tools/ci/check-extension-api.py                                          exit=0
   PASS: extension API surface (6 paths)
```
The negative probe **fails as required** — no headline finding here. Logs 12/13/14. F1's output is byte-identical to retained `api-diff-gate.txt`.

### G. Nix provenance
```
default.cmakeFlags  : ["-DCMAKE_BUILD_TYPE=RelWithDebInfo","-DBUILD_TESTING=OFF",
                       "-DGUFO_EXTENSIONS_FROM_LOCK=OFF","-DGUFO_RELEASE_VERSION=development",
                       "-DGUFO_REVISION=381c164", ...ffmpeg..., "-DCMAKE_HIP_COMPILER=.../rocm-toolchain/bin/clang"]
release.cmakeFlags  : same but "-DGUFO_EXTENSIONS_FROM_LOCK=ON","-DGUFO_RELEASE_VERSION=0.28.0","-DGUFO_REVISION=381c164"
default.releaseVersion : null
release.releaseVersion : "0.28.0"
default.src            : /nix/store/agw3z9x0a39jykwkk7j8n2fav0fhd260-source   (store path retained evidence omitted)
  include/gufo/ext/registry.hpp                          present
  schemas/extension.schema.json                          present
  extensions.lock                                        present
  tests/models/deepseek_v4_flash/fixtures/antirez-ds4.json present
  tests/extensions/discovery_test.py                     absent
  docs/model-extensions/RFC.md                           absent   (extra probe)
  cmake/GufoExtensionDiscovery.cmake                     present   (extra probe)
```
Exactly matches `evidence/card05/nix-source-admissions.json`. One difference vs `evidence/card05/nix-default-flags.json`: retained `GUFO_REVISION=3d2ce45-dirty`, now `381c164` — explained by the worktree state (the Card 05 work is now committed as 381c164 and the working copy is clean), not a flake change. Logs 15/16/17.

### H. Schema delta (exact)
`diff -u /tmp/gufo-card05-repro/schema-before.json schemas/extension.schema.json` — exactly one changed line:
```
@@ -16,7 +16,7 @@
     "api_version": {
       "type": "string",
-      "pattern": "^>= 0\\.[0-9]+\\.[0-9]+$"
+      "pattern": "^>= [0-9]+\\.[0-9]+\\.[0-9]+$"
     },
```
So Card 05 broadened the accepted `api_version` major from `0` to any digit.
- `"title": "Gufo extension metadata API 0.1.0"` (current line 620): **pre-existing at 3d2ce45a** (verified by loading `/tmp/gufo-card05-repro/schema-before.json` → `before title: Gufo extension metadata API 0.1.0`). It is now stale relative to the 0.1.1 bump, but was not introduced by this card.
- Absence of a `pattern` on provider `keys.items` (current lines ~159–166: `{"type":"string","minLength":1}` + `uniqueItems`/`minItems`) is also **pre-existing**; `grep -E 'gguf\|safetensors|diffusers' schemas/extension.schema.json` returns nothing, while `cmake/GufoExtension.cmake:263` enforces `if(NOT key MATCHES "^(gguf|safetensors|manifest|diffusers):.+$")`. The schema/CMake divergence therefore predates Card 05; Card 05 did edit this CONTRACTS file (the `api_version` line above).

### I. Diff hygiene — `jj diff -r @- -- tests/extensions tests/tools tools/ci`
Full diff kept at `/tmp/gufo-card05-repro/09-tests-tools.diff` (600 lines). **Every removed line** in that scope (5 groups, no others):
1. `tests/extensions/CMakeLists.txt`: the old `extension_metadata_test` / `extension_schema_test` `add_test` block was relocated — both are still registered in the current file (`grep` confirms `add_test(NAME extension_metadata_test ...)` and `add_test(NAME extension_schema_test ... --schema)` present). No test dropped from ctest; the 12-test CTest list above includes both.
2. `tests/extensions/metadata_test.py`:
   ```
   -        for api in (">= 0.2.0", ">= 0.1.1", ">= 1.0.0", "0.1", ">= 0.01.0"):
   +        for api in (">= 0.2.0", ">= 0.1.2", ">= 1.0.0", "0.1", ">= 0.01.0"):
   -                self.invalid(..., "API" if api.startswith(">= 0.") else "api_version")
   +                self.invalid(..., "API" if api.startswith(">= ") else "api_version")
   ```
   Same number of `self.invalid` cases (5); the substitution keeps the invalid set valid-relative-to-runtime after the 0.1.1 bump. Not a weakening; no case removed.
3. `tests/extensions/runtime_test.cpp`: `Require(RuntimeApiVersion() == "0.1.0")` → `"0.1.1"`, and `Extensions().size() == 2` → `== 3` with three named extensions (`base`, `alpha`, `beta`). Strengthened, not weakened.
4. `tools/ci/linux_artifact.py`: `if not version.endswith(f'({revision[:7]})'):` and `if version != manifest['version'] or not version.endswith(...)` replaced (multiline-manifest handling); first-line revision strictness is retained and covered by `test_multiline_extension_provenance_retains_revision_validation` (one of the 6 passing artifact tests).

No removed `assert`/`Require`/`self.assert*` beyond the substitutions above. The only documented expectation flip (`evidence/card05/initial-fixture-order-failure.txt`, excluded-order lexical vs dependency-first) is not present as a removal in this scope because the discovery test file is new in this card.

### J. Formatter (ran it)
`nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py` (no `--fix`) → `FORMAT_EXIT=0`, `PASS: Checked 725 C++ files` (`/tmp/gufo-card05-repro/07-format-check.log`). Matches retained `format-check.txt`.

## Claim → reproduction table

| Retained claim (evidence/card05/README.md + files) | Reproduced? | What was seen |
| --- | --- | --- |
| Focused CTest 12/12, zero skips | YES | `100% tests passed out of 12`, 12 named tests, no Not Run/Skipped (fresh dir `build/card05-repro-cpu`) |
| Nine discovery cases | YES | `extension_discovery_test` Passed 33.81 s; file has exactly 9 `def test_*` |
| Four API policy groups | YES | `Ran 4 tests ... OK` |
| Six linux-artifact tests | YES | `Ran 6 tests ... OK` |
| Formatter, 725 C++ files | YES | `PASS: Checked 725 C++ files`, exit 0 |
| Real lock-mode CPU executable + CLI provenance | YES | fresh configure+build exit 0; `cli_provenance_test.py` PASS; `--version`/diagnose JSON byte-identical to retained captures |
| Lock dir really is `BUILD_TESTING=OFF` / `FROM_LOCK=ON` / `HIP=OFF` / RelWithDebInfo | YES | quoted `CMakeCache.txt` lines |
| Platform-only early return ignores conflicting/unknown selectors | YES | CMake `unused-cli` warning names `GUFO_EXTENSIONS`/`GUFO_EXTENSIONS_FROM_LOCK`; both targets built, exit 0 |
| `--version` first line preserved | YES | line 1 `gufo version development (unknown)` |
| API diff gate passes with label, 6-surface inventory | YES | exit 0, `PASS: extension API surface (6 paths)` |
| Nix default vs release flags (discovery vs lock), releaseVersion dev/0.28.0 | YES | quoted eval output |
| Nix source admissions (5 keys) | YES | identical present/absent set; fresh store path `/nix/store/agw3z9x0a39jykwkk7j8n2fav0fhd260-source` recorded |
| `binary-sha256.txt` artifacts | PARTIAL | both retained artifacts unchanged and matching their recorded hashes; the *fresh* lock binary differs (`435d3248…`) because it embeds a different build-dir path — bit-identity across build dirs not observed |
| Retained `nix-default-flags.json` byte-identical today | NO (explained) | `GUFO_REVISION` now `381c164` vs retained `3d2ce45-dirty` (worktree is now committed/clean) |
| "Schema title/API drift" review finding | NOT INTRODUCED BY CARD 05 | title `0.1.0` already at 3d2ce45a; Card 05's only schema edit was the `api_version` pattern line |
| "Schema accepts keys CMake rejects" review finding | PRE-EXISTING | no key `pattern` in schema at either revision; CMake check at `cmake/GufoExtension.cmake:263` |
| Hosted `nix build .#checks.x86_64-linux.pr` (new lock-build block + `extension_discovery_test` in `gufo_pr_tests`) | NOT RUN — excluded by this pass's constraints | assigned to another agent; still PENDING |
| Any HIP/GPU, Windows, functional/numerical/performance gate | NOT RUN (out of scope by instruction) | still PENDING by card design |

## Discrepancies and observations (facts only, no disposition)

1. `cmake --build --preset <p> -B <dir>` ignores `-B` and builds the preset dir. Retained evidence README does not use `-B`, so the recorded commands are self-consistent; any future re-verification recipe must use `cmake --build <dir>` or a distinct preset. This pass's execution of the `-B` form wrote build outputs into the untracked `build/cpu-test` tree (no tracked file, no retained binary hash changed; verified by hash + mtime + `jj status`).
2. External `jj describe` at 06:06:45 set `@`'s description to "feat(extensions): card 5 follow-ups" (op `e6521626a984`). `@` is still empty (`0 files changed`). Not performed by this pass.
3. Empty-selector (`-DGUFO_EXTENSIONS=` alone, lock mode OFF) has no positive coverage: `tests/extensions/discovery_test.py:147` only exercises it together with `-DGUFO_EXTENSIONS_FROM_LOCK=ON` expecting `conflicts`. The conflict rule itself is at `cmake/GufoExtensionDiscovery.cmake:39-41` and duplicated at `:70-73`.
4. No discovery test covers a **missing `extensions.lock`** with `GUFO_EXTENSIONS_FROM_LOCK=ON` (grep for `missing` in the test file shows only `unknown extension missing`, `missing selected dependency base`, `missing dependency base`, `test_missing_build_manifest`).
5. Selection-failure coverage that does exist and passed: `unknown extension missing`, `missing selected dependency base`, `duplicate names`, lock/selector conflict, unavailable-on-platform (explicit and `*`), lock dependency completeness.
6. `nix-source-admissions.json` in the retained evidence lists booleans without the store path or command; this pass captured `/nix/store/agw3z9x0a39jykwkk7j8n2fav0fhd260-source` plus per-path presence, which the integration agent may want to persist.
7. Retained `default-version.txt` / `lock-version.txt` show `Extension selection: discovery|lock` and API `0.1.1` — reproduced byte-for-byte from a fresh binary.

## Commands not run (and why)
- `nix build .#checks.x86_64-linux.pr` and any `nix build .#packages...` — explicitly excluded for this pass ("another agent owns that"). The flake's new mandatory lock-build block and `extension_discovery_test` in `cmake/Checks.cmake` therefore remain **unexecuted evidence**.
- GPU/HIP builds, `tests/functional/**`, model downloads, HTTP servers, full ctest sweeps — excluded by constraint; the card defers them.
- `check-format.py --fix` — excluded (mutating).

## Workspace state at hand-off
`jj status`: "The working copy has no changes." `@` = twvsqlym (empty, description noted above), `@-` = kxnltrzq/381c1648 unchanged. `sha256sum docs/model-extensions/RFC.md` = `2ce2becbc0bf0f844702b624de6827c2bae6eba921ab8a1ea883659fa2f96448` (unchanged, matches HANDOFF). Nothing was fixed by this pass.
