# Card 09 integration — edits complete, editing ownership released

Integration agent /root has completed the available edits and releases editing
ownership to the orchestrator. No commits, pushes, servers, functional harness,
benchmarks or full suite. No live handles. No HANDOFF.md access or worktree edits.

Review remains BLOCK pending orchestrator validation: F1 target resolution, F2
complete corrected GPU/CPU verdicts, F3 real CMake rename/revert negative proof,
refreshed HIP reports/gates, hosted and formatting checks. Nix daemon access was
denied once and not retried. Host checks/negative expectations and the final
RFC/API/docs verdicts are in integration-reruns.txt. DoD 2 is partial; DoD 3 has
historical identities only. No deferred qualification is claimed as passed.

Requested `jj diff --name-only` failed before producing a list: jj attempted to
snapshot into read-only .git/objects (exact denial retained). No retry. Full
working-copy changed-path inventory and staged-state verification remain pending
for the orchestrator. The integration-owned paths below were directly edited;
the historical worker inventory follows and is not a fresh jj result.

- `CMakeLists.txt`
- `cmake/TargetGraph.cmake`
- `tools/ci/check-target-graph.py`
- `tests/tools/check_target_graph_test.py`
- `src/ext/continuation.hpp`
- `src/ext/host_budget.hpp`
- `src/ext/device_probe.hpp`
- `src/ext/text_contracts.hpp`
- `src/cache/component_resources.hpp`
- `src/core/hip/component_cache_resources.hpp`
- `docs/TESTING.md`
- `docs/model-extensions/target-ownership.md`
- `docs/model-extensions/cards/09-serving-runtime-boundary.md`
- `docs/model-extensions/cards/README.md`
- `docs/model-extensions/CARDS.md`
- `docs/model-extensions/evidence/card09/RESUME-NOTES.md`
- `docs/model-extensions/evidence/card09/build-and-tests.txt`
- `docs/model-extensions/evidence/card09/integration-reruns.txt`
- `docs/model-extensions/evidence/card09/moved-source-flags.txt`
- `docs/model-extensions/evidence/card09/negative-proofs.txt`
- `docs/model-extensions/evidence/card09/pending-commands.txt`
- `docs/model-extensions/evidence/card09/logs/selection.txt`
- `docs/model-extensions/evidence/card09/logs/gpu-test-flag-diff.txt`
- `docs/model-extensions/evidence/card09/logs/production-flag-diff.txt`
- `docs/model-extensions/evidence/card09/logs/ctest-gpu-final.txt`
- `docs/model-extensions/evidence/card09/logs/ctest-cpu-final.txt`
- `docs/model-extensions/evidence/card09/logs/ctest-gpu-worker-tail.txt`
- `docs/model-extensions/evidence/card09/logs/ctest-cpu-worker-tail.txt`

Scratch to delete at closure: /tmp/card09-integration/ (scripts and logs),
/tmp/card09-logs/, /tmp/card09-backup/, /tmp/card09-review/. The requested
/tmp/card09-integration/rename-proof was not created. Retain evidence before
cleanup; no scratch was removed here. No active subprocess/session handles.

## Historical worker handoff (superseded where integration corrects it)

# Card 09 worker — handoff (editing ownership released)

Working copy `@` (`prusoklk 41e532c8`), parent `uytorytn c03aaa23` (Card 08).
**Nothing was committed**: no `jj commit`, `jj new`, `jj squash`, no `git commit`.
`docs/model-extensions/RFC.md` is byte-identical to the baseline
(`sha256 2ce2becbc0bf0f844702b624de6827c2bae6eba921ab8a1ea883659fa2f96448`).
`HANDOFF.md` was never read or written; `.claude/worktrees/` untouched.
This agent now holds no editing intent — reviewer and integration agents may
edit freely.

## All seven plan phases are implemented

| Phase | Result |
| --- | --- |
| 1 baseline capture | card08 compile lines captured before any edit (`/tmp/card09-logs/baseline-flags-raw.txt`, used by the retained audit) |
| 2 neutral continuation bases | `src/ext/continuation.hpp`, aliases in serving headers |
| 3 host budgets + default device probe | `src/ext/host_budget.{hpp,cpp}`, `src/ext/device_probe.{hpp,cpp}` |
| 4 backing/resource contracts + moved HIP provider | `src/cache/component_resources.hpp`, `Adapter::BackingPlan`, `src/core/hip/component_cache_resources.{hpp,cpp}` |
| 5 `gufo_serving` extraction | new static library; the executable, the five serving test targets in the root `CMakeLists.txt` and three model test trees relinked |
| 6 guards | `--fail-on-serving-boundary`, `cmake/TargetGraph.cmake` + `tools/ci/check-target-graph.py`, `ext_runtime_neutral_contracts_test` |
| 7 gates + docs | results below |

## Changed paths (50, all task-owned)

Build/graph: `CMakeLists.txt`, `cmake/Checks.cmake`, `cmake/TargetGraph.cmake` (new),
`src/ext/CMakeLists.txt`.
Neutral runtime: `src/ext/continuation.hpp`, `host_budget.{hpp,cpp}`,
`device_probe.{hpp,cpp}`, `text_contracts.hpp` (all new).
Cache/HIP: `src/cache/adapter.hpp`, `src/cache/component_resources.hpp` (new),
`src/core/hip/component_cache_resources.{hpp,cpp}` (new).
Serving: `src/cli/serve/{continuation_cache,component_text_cache,text_model_runner}.hpp`,
`src/cli/serve/{text_model_runner,inference_backend}.cpp`.
Models: `src/models/qwen/continuation_{adapter,layout}.hpp`,
`src/models/qwen38_flash_next/continuation_{adapter,layout}.hpp`,
three `library.json` manifests, three DeepSeek sources that now include
`src/core/logging/logging.hpp`.
Tests: `tests/extensions/CMakeLists.txt`,
`tests/extensions/ext_runtime_neutral_contracts_test.cpp` (new),
`tests/cache/adapter_test.cpp`, `tests/models/{qwen27b,qwen38_flash_next,deepseek_v4_flash}/CMakeLists.txt`,
`tests/tools/check_extension_isolation_test.py`,
`tests/tools/check_target_graph_test.py` (new).
Tools: `tools/ci/check-extension-isolation.py`, `tools/ci/check-target-graph.py` (new).
Docs: `docs/DEVELOPMENT.md`, `docs/TESTING.md`,
`docs/model-extensions/target-ownership.md`,
`docs/model-extensions/cards/09-serving-runtime-boundary.md`,
this directory.

Deliberately unchanged: `include/gufo/ext/**` and `SURFACE` (Card 10 publishes
them; `python3 tools/ci/check-extension-api.py` → `PASS: extension API surface
(6 paths)`), the `CARDS.md` row 09 status and the `cards/README.md` "Cards 09–29
are pending" sentence — those flip at closure, not by the implementer.

## Checks run (command → exit → verdict)

```sh
nix develop -c cmake --preset gpu-test  -B build/card09-gpu              # 0
nix develop -c cmake --build build/card09-gpu --parallel 16              # 0  full tree
nix develop -c cmake --preset cpu-test  -B build/card09-cpu              # 0
nix develop -c cmake --build build/card09-cpu --parallel 16              # 0  full tree
nix develop -c cmake -S . -B build/card09-release --preset release       # 0
nix develop -c cmake --build build/card09-release --target gufo --parallel 16   # 0
cmake -S . -B build/card09-lock -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_TESTING=OFF -DENGINE_ENABLE_HIP=ON -DGUFO_EXTENSIONS_FROM_LOCK=ON     # 0
nix develop -c cmake --build build/card09-lock --target gufo --parallel 16       # 0
R='^(continuation_cache_test|…|check_docs_test)$'   # /tmp/card09-logs/selection.txt
nix develop -c ctest --test-dir build/card09-gpu -R "$R" --output-on-failure    # 0  42/42
nix develop -c ctest --test-dir build/card09-cpu -R "$R" --output-on-failure    # 0  42/42
python3 tools/ci/check-target-graph.py --report build/card09-gpu/gufo-link-graph.json --root .   # 0  24 protected targets
python3 tools/ci/check-target-graph.py --report build/card09-cpu/gufo-link-graph.json --root .   # 0  HIP tiers listed absent
python3 tools/ci/check-extension-isolation.py --fail-on-serving-boundary        # 0
python3 tools/ci/check-extension-isolation.py --fail-on-target-ownership        # 0
python3 tools/ci/check-extension-api.py                                         # 0  SURFACE unchanged
python3 tests/tools/check_target_graph_test.py                                  # 0  8 tests
python3 tests/tools/check_extension_isolation_test.py                           # 0  22 tests
nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py  # 0  740 files
./build/card09-release/gufo diagnose                                            # 0
./build/card09-lock/gufo diagnose                                               # 0
```

Negative proofs (injected, verified failing, reverted, re-gated):
`docs/model-extensions/evidence/card09/negative-proofs.txt`.

## Evidence

- `docs/model-extensions/evidence/card09/build-and-tests.txt` — configurations,
  the 42-test selection, production artifacts, `gufo diagnose` comparison.
- `moved-source-flags.txt` — per-source compile-line parity audit and the one
  intentional deviation.
- `negative-proofs.txt` — both new gates proven to fail on injected violations.
- `pending-commands.txt` — the deferred functional/timing/perf/Windows/lock
  commands, verbatim, marked PENDING.
- `logs/` (committed, < 100 KiB total) — the two flag-diff audits, the
  link-graph and serving-boundary gate output before and after the injected
  violation, the 42-test selection regex, both final ctest transcripts and the
  `gufo diagnose` diffs against Card 08.
- `flag-audit.py` — reproduces the parity audit from two
  `compile_commands.json` files.
- Bulk build logs stay scratch in `/tmp/card09-logs/` (`build-*.log`,
  `configure-*.log`, `format*.txt`) and are removed at closure.
- Crash-recovery snapshot: `/tmp/card09-backup/progress.patch` (full `jj diff`
  of this working copy) plus pre-injection originals used by the negative proofs.

## Limitations and deviations to review

1. **`-UNDEBUG` in test builds.** The seven serving sources now compile once in
   `gufo_serving`, which adds `-UNDEBUG` under `BUILD_TESTING`. A test-preset
   `gufo` therefore asserts where the card08 executable did not. Production
   presets (`BUILD_TESTING=OFF`) are provably unchanged — the release and lock
   audits add nothing and remove only unused executable defines.
2. **Test-tree exemption.** `--fail-on-serving-boundary` skips any path with a
   `tests` component, because `src/models/qwen3_asr/tests/audio_api_test.cpp`
   still includes `src/cli/serve/{asr_service,audio_asr_api}.hpp` and
   `src/models/qwen3_asr/CMakeLists.txt` compiles those two serving sources.
   Card 12 (`gufo_test_support`) removes it; tighten the exemption then.
3. **Interim `gufo_serving` edges.** It links `gufo_qwen27b`,
   `gufo_qwen38_flash_next`, `gufo_deepseek_v4_flash`, `gufo_cache_hip` and
   `hip::host` under `ENGINE_ENABLE_HIP`, and is excluded from the link-graph
   protected set. The direction it *is* guarding (nothing below
   `gufo_ext_runtime` reaches serving/HTTP) is enforced and negative-proved.
4. **`ext_runtime_neutral_contracts_test`** runs in the isolated extension
   fixture project, so its device-probe assertions must hold on the HIP-free
   code path too; the capture/backing contract is covered by
   `cache_adapter_test` because `gufo_ext_runtime` does not link the cache tier.
5. **Device probe diagnostics** now report `HIP_CHECK` file/line from
   `src/ext/device_probe.cpp` instead of `inference_backend.cpp`.
6. **Functional/performance/Windows/lock-runtime** validation is deferred,
   recorded as PENDING, never as a pass.

## Next steps for the orchestrator

Independent review (`input-brief-reviewer.md`), third-agent integration
(`input-brief-integration.md`), then closure: re-run the pinned gate set once,
run `codex exec review --dangerously-bypass-approvals-and-sandbox 'review the
diff of current jj revision, you can use jj diff to see changes'`, fix
critical/major findings, flip the `CARDS.md`/`cards/README.md` status rows,
commit only the task-owned paths above, and clean `/tmp/card09-*`.
