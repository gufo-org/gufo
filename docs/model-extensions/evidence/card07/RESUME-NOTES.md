# Card 07 — RESUME NOTES (implementation, resumed 2026-10-31 11:57)

Baseline: jj working copy `@` = `zwllvvnv 6b9217ec` (no description), parent `@-` =
`kppqwqnv 11af44f9` "refactor(extensions): move qwen-specific speculation into model-owned
code" (Card 06). RFC.md sha256 verified unchanged:
`2ce2becbc0bf0f844702b624de6827c2bae6eba921ab8a1ea883659fa2f96448`.
Host: Strix Halo gfx1151, 125 GiB RAM. Toolchain: repo-pinned `nix develop`.

History: the first implementation agent (started 11:06) landed the whole refactor and both
dev builds, then died on infrastructure at ~11:50 without writing any evidence besides this
file (which was stale — this revision replaces it). A second agent resumed at 11:57.
**No commits were made by anyone.** Briefs: `/tmp/card07-orch/WORKER-BRIEF.md`,
`/tmp/card07-orch/RESUME-BRIEF-2.md` (ephemeral; if missing, the card file
`docs/model-extensions/cards/07-qwen-family-libraries.md` plus `../CARDS.md` and
`../README.md` are the contract).

## Done and verified (do not redo)

- [x] A. Analysis: 60 Qwen headers inventoried and assigned; library graph acyclic.
      Helper + tables: `/tmp/card07-work/{qwen_split.py,report.txt,move-plan.txt,rewrite-plan.txt}`.
      Pre-move isolation snapshot: `/tmp/card07-work/isolation-before.{txt,json}`.
- [x] B/C. `src/families/qwen/` exists with `control_tokens` (INTERFACE), `hip_common`
      (INTERFACE: continuation_hooks, execution_policy, attention_policy, recurrent_state,
      gemm_route, quant_ops, rope), `config`, `text`, `kernels`, `vision`; each has
      `library.json` + `CMakeLists.txt`; family `README.md` and family `AGENTS.md` written.
- [x] D. `gufo_qwen_config` holds the free `ExtractQwenConfig`; `GgufReader::ExtractModelConfig`
      removed; callers (`weights.cpp`, `chat_template.cpp`, `mtp_reference.cpp`,
      `src/cli/prompt/prompt.cpp`) updated; config test moved to
      `tests/families/qwen/qwen_config_test.cpp`.
- [x] E/F. `gufo_qwen_text`, `gufo_qwen_kernels`, `gufo_qwen_vision`, `gufo_qwen27b` created;
      `gufo_core` compiles no Qwen sources; consumers (CLI, kernel-bench, tools, tests) relinked;
      includes rewritten across `src/`, `tests/`, `tools/`, `docs/models/**`.
- [x] Functional registries (`tests/functional/impact.json`, `devlab.json`, `test_impact.py`,
      `test_devlab.py`) updated for the new paths. **The harnesses themselves were not run**
      (user decision 2026-10-31: no functional runs during cards 06–14).
- [x] `tools/ci/check-extension-isolation.py` gained `family_analysis()`,
      `cycle_violations()`, `QWEN_PREFIXES` (header closure now covers
      `src/families/qwen/**` too) and `--fail-on-family-graph`.
- [x] `extension_family_graph_gate` registered in `CMakeLists.txt` (11:31:58) and added to
      `gufo_pr_tests` in `cmake/Checks.cmake` (11:32:03).
- [x] Builds: `build/card07-cpu` (cpu-test) and `build/card07-gpu` (gpu-test) both built with
      EXIT=0 (`/tmp/card07-work/{configure-cpu,configure-gpu,build-cpu3,build-gpu}.log`).
      Earlier `build-cpu.log`/`build-cpu2.log` failures were fixed and are history only.
- [x] Test runs (selection not attributable): `/tmp/card07-work/ctest-fast.log` 100% of 64,
      `/tmp/card07-work/cpu-tests-targeted.log` 100% of 11.

## Open at resume time

1. `build/card07-gpu/CTestTestfile.cmake` is from 11:29, i.e. older than the
   `extension_family_graph_gate` registration (11:31:58): the gate is not in that CTest list
   until CMake reconfigures. Fix by rebuilding (ninja reconfigures) and re-listing.
2. `tests/tools/check_extension_isolation_test.py` has no family-rule fixtures (file mtime
   08:08). Add: clean graph passes; missing/unused declared dependency fails; cycle fails;
   family→model include fails; header outside any manifested library fails; plus a
   repository-level family-graph test.
3. No gate evidence exists in this directory yet (see README.md status table).

## Gates status (see README.md for the full table)

None recorded yet. Nothing had been written to `docs/` by the first agent.

## Next exact commands

    nix develop -c cmake --build build/card07-gpu -j 16      # reconfigures + registers the gate
    nix develop -c ctest --test-dir build/card07-gpu -N -R 'family'
    nix develop -c ctest --test-dir build/card07-gpu -R '^(extension_family_graph_gate|check_extension_isolation_test|extension_isolation_report|extension_core_model_includes_gate)$' --output-on-failure

Then the pinned identity/unit selections, `build/card07-release` (production NDEBUG) plus the
compile-flag and ISA audits, formatting/API/docs checks, `gufo diagnose`, and this directory's
`README.md`, `pending-commands.md`, `handoff.md`.

## Known-state warnings for a resuming session

- Tree state matches the "Done" list above; both dev build directories are green.
- ISA baseline for the production audit: `build/card06-release/**/gufo_core.dir/**.o` and
  `build/card06-release/gufo` (pre-Card-07 production build; every object is older than the
  first Card 07 edit at 11:06).
- Do not run functional harnesses (`tests/functional/run.py`, `prompt_lookup.py`, `compare.py`),
  `tools/bench/model-bench.py`, devlab, or hosted `nix build .#checks.x86_64-linux.pr`.
- Scratch lives in `/tmp/card07-work` and `/tmp/card07-orch`; both must be deleted at the end.
