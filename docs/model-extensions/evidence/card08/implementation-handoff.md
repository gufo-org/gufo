Implemented Card 08 (RFC step 5) in the working copy; nothing committed, nothing staged.

**What changed**

1. **New targets with their own directories**: `gufo_minimax_h3` (`src/models/minimax_h3/CMakeLists.txt`, HIP sources join under `ENGINE_ENABLE_HIP`, ffmpeg defines now private), `gufo_compare` (`src/testing/compare`), `gufo_eval` (`src/eval/{dataset,extract,regrade}.cpp`), and `gufo_deepseek_v4_flash_text` (CPU-only `chat_template.cpp` + `runtime/sampling.cpp`, shipped `-O3 -ffast-math` preserved, `-UNDEBUG` only under `BUILD_TESTING`). `gufo_core` no longer compiles any of them; consumers relink instead of duplicating (`openai_chat_test` compiles zero model/HTTP copies now; `tool_calls.cpp` stays in `gufo_http` with recorded justification).
2. **Eval dataset → core fixture** `tests/fixtures/eval/antirez-ds4.json` (jj rename), references updated in root CMake, `src/cli/eval/eval.cpp`, ds4 test/CMake/python, install rule, and the `.devops/nix/package.nix` filter (verified: the production source now admits exactly that one test-tree file).
3. **Ownership made machine-checked**: 11 `library.json` manifests (8 models incl. the DeepSeek text tier, eval, compare) with `consumers`/`checks`/`tools`/`compiled_elsewhere`; new gate `--fail-on-target-ownership` (+`--target-ownership-report`) registered as CTest `extension_target_ownership_gate`; 3 new fixture tests + repository assertion; `docs/model-extensions/target-ownership.md` documents rules, claims, conditions and limits.
4. **New parity tool** `tools/prof/card08_parity.py` derives the 27 target-moved sources from `compile_commands.json` and audits flags, host objects and gfx1151 device sections with a stale-proof justification table.

**Validation (verbatim results in `docs/model-extensions/evidence/card08/`)**

- Static: format `730 files`; ext-api `6 paths` no surface diff; docs `267 files/1477 links`; isolation core-boundary + family-graph PASS; **target ownership PASS (17 libraries, 183 edges)**; `check_extension_isolation_test.py` 18 tests OK.
- Builds `build/card08-{cpu,gpu,release}` all exit 0 (parent `build/card07-*` untouched).
- CTest: CPU selection `100% out of 25`; GPU selection `100% out of 30` — three continuation identity goldens passed; **9 H3 rows Skipped (exit 77, missing weights) recorded as pending, not passes**.
- Parity vs parent: flags `compared=27 failures=0` (21 identical, 6 justified), object `compared=27 failures=0` (27 code identical/move-artifact), device `compared=8 failures=0` (24 `.text/.rodata/.note` sections byte-identical); kernel resources `0 worse, 2692 same`, registered baseline FAIL rows identical to parent (pre-existing).
- Hosted: `nix build .#checks.x86_64-linux.pr` exit 0 (format/dependency/docs/**hosted CPU contract suite**/mkGufoServe all PASS); `checks…h3-manifest` PASS.

**Open items for the orchestrator**

- `nix build .#packages.x86_64-linux.windows-platform-tests` fails **on parent inputs too** (`cmake/PlatformTests.cmake` needs `tests/core/platform_pages_test.cpp`; the Nix fileset admits only two of three) — outside Card 08 scope, needs its own change; full Windows gate stays CI-only.
- Functional/perf rows P1–P7 and asset-gated H3 rows P8 remain **PENDING** with exact commands; extension-level exclusion builds can't be exercised per model until `extension.json` exists (Card 09+), so I recorded the available exclusions (HIP-off, platform-only, empty/typo `GUFO_EXTENSIONS`).
- `docs/model-extensions/evidence/card08/brief-worker.md` is my orchestration brief — decide whether it belongs in the commit.
- Card 08 file and `CARDS.md`/`cards/README.md` execution-record edits were intentionally left undone.

**editing ownership released** — I will make no further edits; the working copy `@ uytorytn 61f17828` is ready for the reviewer agent.