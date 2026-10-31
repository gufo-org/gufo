# Card 08 — gates run by the implementation agent

Working copy `@ uytorytn 1cac04bf`, parent `zwllvvnv ec59bed0` (Card 07).
Host: Linux x86-64, gfx1151, 125 GiB RAM (≥112 GiB available before every GPU
step), `llama-swap` services untouched, scratch under `/tmp/card08-*` and
removed after each step. Every command below was actually run; a command that
was not run appears only in [pending-commands.md](pending-commands.md).

## Static and repository gates

| Gate | Command (verbatim) | Result |
| --- | --- | --- |
| Formatting | `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py` | `PASS: Checked 730 C++ files`, exit 0 |
| Extension API | `python3 tools/ci/check-extension-api.py --write` then `python3 tools/ci/check-extension-api.py` | `PASS: extension API surface (6 paths)` both, exit 0/0, no surface diff → no `ext-api` label needed |
| Docs | `python3 tools/ci/check-docs.py` | `PASSED: 267 markdown files and 1477 local links/anchors … zero broken links or invalid JSON blocks`, exit 0 (final numbers after these evidence files landed; an earlier run reported 262/1469) |
| Isolation (report) | `python3 tools/ci/check-extension-isolation.py --mode report` | exit 0, `files: 1249`, `family_libraries: 6`, `family_edges: 7` |
| Isolation (ratchets) | `python3 tools/ci/check-extension-isolation.py --fail-on-core-model-includes --fail-on-family-graph` | `PASS: core boundary` + `PASS: family graph (6 libraries, 7 declared edges, acyclic)`, exit 0 |
| **New** target ownership | `python3 tools/ci/check-extension-isolation.py --fail-on-target-ownership` | `PASS: target ownership (17 libraries, 183 dependency edges)`, exit 0 |
| Ownership graph | `python3 tools/ci/check-extension-isolation.py --target-ownership-report` | 183 edges, no findings; the per-library view is [target-ownership.md](../../target-ownership.md) |
| Ownership unit tests | `python3 tests/tools/check_extension_isolation_test.py` | `Ran 18 tests … OK`, exit 0 (3 new fixture cases + repository assertion) |

## Build gates (candidate configurations)

The Card 07 directories are parent-revision baselines and were never built
into. CMake refuses a copied `CMakeCache.txt`, so each candidate is a fresh
configure; unchanged translation units came from the installed compiler cache.

| Target | Command (verbatim) | Result |
| --- | --- | --- |
| CPU test config | `nix develop -c cmake -S . -B build/card08-cpu -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON -DENGINE_ENABLE_HIP=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DGUFO_ENABLE_WARNINGS=ON` then `nix develop -c cmake --build build/card08-cpu --parallel 10` | configure exit 0; build exit 0, 0 `error:` lines |
| HIP test config | `nix develop -c cmake -S . -B build/card08-gpu -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON -DENGINE_ENABLE_HIP=ON -DGUFO_BUILD_TOOLS=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DGUFO_ENABLE_WARNINGS=ON` then `nix develop -c cmake --build build/card08-gpu --parallel 12` | configure exit 0; build exit 0, 0 `error:` lines |
| Production config | `nix develop -c cmake -S . -B build/card08-release -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=OFF -DENGINE_ENABLE_HIP=ON -DGUFO_BUILD_TOOLS=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DGUFO_ENABLE_WARNINGS=ON -DGUFO_FFMPEG_EXECUTABLE=ffmpeg -DGUFO_FFPROBE_EXECUTABLE=ffprobe` then `nix develop -c cmake --build build/card08-release --parallel 12` | configure exit 0; build exit 0. Cache values mirror `build/card07-release` (including the FFmpeg executables) so flag comparison is meaningful |

First build attempts failed on three qwen module tests with
`undefined reference to gufo::testing::CompareLogits`; those targets (plus
`qwen_oracle_test`) now link `gufo_compare` explicitly instead of relying on
`gufo_core`, and the rebuilds above are clean.

## Exclusion and condition cases (configure evidence)

No `extension.json` exists yet (`cmake/GufoExtensionDiscovery.cmake` globs
`src/models/*/extension.json` and `src/families/*/*/extension.json`), so
extension-level exclusion cannot be exercised per model until Card 09+. The
real exclusions available today were configured:

| Case | Command (verbatim) | Result |
| --- | --- | --- |
| Model-tier exclusion | the CPU test config above (`-DENGINE_ENABLE_HIP=OFF`) | configures, builds, and runs the 25 selected tests; `gufo`, `gufo_http`, `gufo_llm_cli`, eval, comparison and the DeepSeek text tier all exist without any DeepSeek/HIP model target |
| Platform-only (Windows shape) | `-DGUFO_PLATFORM_ONLY=ON` inside `nix build .#packages.x86_64-linux.windows-platform-tests` | fails, pre-existing and outside this card: see *Windows gap* below |
| Explicit empty selection | `nix develop -c cmake -S . -B /tmp/card08-excl -DENGINE_ENABLE_HIP=ON -DGUFO_EXTENSIONS="" -DGUFO_EXTENSIONS_FROM_LOCK=OFF` | `gufo: extension mode=explicit; included=[]; excluded=[]`, configure exit 0 (scratch removed) |
| Unknown extension name | `nix develop -c cmake -S . -B /tmp/card08-excl -DENGINE_ENABLE_HIP=ON -DGUFO_EXTENSIONS=minimax_hip_typo` | configure exit 1 from `cmake/GufoExtensionDiscovery.cmake:93` — a typo cannot silently select nothing (scratch removed) |

## Hosted gates

| Gate | Command (verbatim) | Result |
| --- | --- | --- |
| Hosted PR check | `nix build .#checks.x86_64-linux.pr` | see *Hosted outcome* at the end of this file |
| H3 manifest | `nix build .#checks.x86_64-linux.h3-manifest --no-link --print-out-paths` | `PASS: MiniMax H3 manifest tests clean`, exit 0 |
| Nix source admission | `nix eval --raw .#packages.x86_64-linux.release.src` then `find $src/tests -type f` | the production source admits exactly one test-tree file: `tests/fixtures/eval/antirez-ds4.json`; the old `tests/models/deepseek_v4_flash/fixtures/antirez-ds4.json` is gone and no model test file enters the derivation |
| Snapshot honesty | Pre-commit Nix evaluations | coverage not established (dirty git tree → Nix snapshot from tracked parent paths); P7 is authoritative after commit |

Hosted outcome: **coverage not established (dirty git tree → Nix snapshot from tracked parent paths)**.
The earlier command returned exit 0 and hosted PASS lines, but these do not prove
coverage of the candidate. Nix printed:

```text
warning: Git tree '/home/mixer/gufo' is dirty
```

The earlier snapshot claim is withdrawn. Only the post-commit P7 source proof
and hosted rerun can establish candidate coverage; no hosted pass is claimed.

### Windows gap (pre-existing, outside Card 08)

`nix build .#packages.x86_64-linux.windows-platform-tests` fails during CMake
generate:

```
CMake Error at cmake/PlatformTests.cmake:17 (add_executable):
  Cannot find source file:
    tests/core/platform_pages_test.cpp
Call Stack (most recent call first):
  CMakeLists.txt:66 (include)
```

`cmake/PlatformTests.cmake` registers three platform tests
(`platform_fs_test`, `platform_mapped_file_test`, `platform_pages_test`) while
the `src` fileset of `.devops/nix/windows-platform-tests.nix` admits only two
of them. Neither file is touched by this card
(`jj diff --from zwllvvnv --to @ --name-only` lists no `PlatformTests`,
`windows-platform-tests` or `platform_*` path), so the breakage is on the
parent revision and needs its own change. Consequences for this card: the
Windows cross-built platform/cache gate could not be exercised, the full
Windows build and CTest run stays with
`.github/workflows/windows-build.yml`, and the Windows-facing part of Card 08 —
the `"$<$<PLATFORM_ID:Windows>:media_windows.cpp>"` /
`"$<$<NOT:$<PLATFORM_ID:Windows>>:media_posix.cpp>"` pair and the private
`GUFO_FFMPEG_EXECUTABLE` / `GUFO_FFPROBE_EXECUTABLE` definitions — moved
verbatim into `src/models/minimax_h3/CMakeLists.txt`, with the platform-only
early `return()` in the root file untouched above it.
