> Instructions given to the implementation agent, written before it ran. This is a
> brief, not a result. Results go in the other files of this directory.

# Worker brief — Card 14: Generate infrastructure from model metadata

You are the **implementation agent** (`worker`) for Card 14 of the model-extensions RFC in
`/home/mixer/gufo` (jj working copy `@`, parent commit `qupqlxsk 642af9d6`). You have
**sole editing ownership** of the shared worktree while you run.

## Absolute constraints

1. **Never run `jj commit`, `jj new`, `jj squash`, `git commit`, or anything that
   changes revision state.** The orchestrator commits. Read-only commands (`jj st`,
   `jj diff`, `jj log`) are permitted.
2. `docs/model-extensions/RFC.md` must stay **byte-identical and uncommitted**.
   Do not read or write `HANDOFF.md` (deleted; its decisions now live in `CARDS.md` and `cards/README.md`).
3. **Scope discipline**: during this card run only unit/identity/CPU/hosted/static
   gates. Functional harness runs (`tests/functional/run.py` rows), main-control timings,
   and standard speed benchmarks are **deferred to the single post-Card-14 qualification window**
   per standing user decision. Record them as PENDING with their exact commands in
   `docs/model-extensions/evidence/card14/pending-commands.txt`, never as passes.
4. Scratch goes in `/tmp/card14-*` only (logs, caches, temporary build trees) and MUST
   be deleted when the test finishes, failures included.
   No evidence file over 1 MB under `docs/`.
   No raw full build logs, no per-row report trees under `docs/`.
5. Keep tool output small: redirect build and test logs to files in `/tmp/card14-logs/`
   and inspect using `tail -n 40` or `grep`. Never output multi-megabyte tool outputs.
6. Host RAM: Before running any GPU-heavy operation, check `free -g` and keep at least
   15 GiB headroom. The `llama-swap.service` must stay running; do not stop services.
7. Preserve unrelated work and user changes.

## Objectives & Context

Card 14 implements RFC Step 9:
"Give every model a metadata-only `extension.json` before its code migrates, then generate
`check-pr` and Windows target lists (and point the Windows ZIP at the generated `devlab.json`),
test labels, the Nix source filter and inputs, `tools/ci` data, flake checks, devshell
Python dependencies and the functional registries from metadata."

Before remaining implementations move in later cards (cards 15–22), every model needs
complete metadata so generic tooling can replace hand-maintained lists across build, CI,
Nix, docs, and functional test registries. Metadata-only registration must remain truthful
while legacy code is still linked through old paths.

All packaging and CI metadata fields are already specified in `schemas/extension.schema.json`:
- `packages` (`pkg_config`, `nix`, `windows`)
- `compile_definitions`
- `runtime_libraries`
- `install_data`
- `licenses`
- `production_paths` (`include`, `exclude`)
- `vendored_paths`
- `kernel_resources`
- `docs` (`title`, `inputs`, `guide`)
- `tests` (`target`, `ctest`, `check_pr`, `windows_build`, `exclude_os`)
- `test_labels`
- `fixtures`
- `bundled_tests`
- `functional`
- `tools`
- `python_checks` (`name`, `tests`, `sources`)

## Step-by-Step Implementation Plan

### Step 1: Inventory & Model Metadata Manifests (`extension.json`)
Inventory every model and declare complete, valid `extension.json` files for:
1. `src/models/deepseek_v4_flash/extension.json`
2. `src/models/minimax_h3/extension.json`
3. `src/models/qwen/extension.json` (Qwen 27B)
4. `src/models/qwen38_flash_next/extension.json`
5. `src/models/qwen3_asr/extension.json`
6. `src/models/qwen3_tts/extension.json`
7. `src/models/qwen_image_21/extension.json`
Also update `src/models/embedding_gemma2/extension.json` to include its packaging, docs, tests, fixtures, and labels.

**Crucial Truthfulness Constraint**:
Do NOT pretend unmigrated models already have factories or `extension.cmake`. Unmigrated
models still compile through root `CMakeLists.txt` and legacy libraries.
In `cmake/GufoExtensionDiscovery.cmake`, ensure that extensions without `extension.cmake`
(or unmigrated metadata-only extensions) are marked `GUFO_EXTENSION_METADATA_ONLY = TRUE`
and are NOT added to `gufo_selected_extensions` in discovery mode, so `gufo_build_discovered_extensions()`
and `gufo_target_extensions()` only build migrated extensions (`embedding_gemma2`), while
all metadata is still discovered, validated, and available for tooling.

Verify that all manifests strictly conform to `schemas/extension.schema.json` via
`nix develop -c python3 tests/extensions/metadata_test.py --schema`.

### Step 2: Infrastructure Generator & Drift Checker Tool
Implement a generator and drift-checking tool (e.g. `tools/ci/generate-infrastructure.py` or
integrated into `tools/ci/`) with `--check` (default) and `--write` modes.

The tool must read all `extension.json` manifests and generate / check drift for:
1. **Bounded hosted and Windows target lists, CTest labels**:
   - `cmake/Checks.cmake`: Generate the model test targets and ctest regex patterns from `manifest["tests"]` with `check_pr: true` and `exclude_os: ["windows"]`.
   - Windows build target lists and bundled test assets.
2. **Nix production filters and declarations**:
   - Nix source filter paths in `.devops/nix/package.nix` / `flake.nix` (`production_paths`, `install_data`, `licenses`, `fixtures`).
   - Editing model test/functional/tool sources must leave production derivations unchanged!
   - Devshell Python dependencies, reference packages, and flake check fragments.
3. **Tools CI & Documentation**:
   - `tools/ci/check-format.py`: Vendored paths table derived from `manifest["vendored_paths"]`.
   - `tools/ci/check-docs.py`: Required model docs derived from `manifest["docs"]`.
   - `docs/models/README.md`: Models index table generated from `manifest["docs"]`.
   - `THIRD_PARTY_NOTICES.md`: Model component notices / licenses table.
4. **Committed Functional Registries**:
   - `tests/functional/devlab.json` and `tests/functional/impact.json`: Merged from model manifest data / fragments.
   - Point Windows ZIP devlab registry input to generated registry data.
   - Second run of generator produces no diff (`--check` exits 0).

### Step 3: Conformance Fixtures & Drift Failure Tests
Add test fixtures and unit tests in `tests/tools/` (e.g. `tests/tools/generate_infrastructure_test.py` or similar):
- Missing metadata fixture -> fails with clear error.
- Duplicate metadata fixture -> fails with clear error.
- Deliberate stale fixture / drift -> fails with non-zero exit code.
- Conditional metadata (HIP / Linux / Windows) properly handled.

### Step 4: Focused Verification Gates
Run the required gates and record evidence:
1. `python3 tools/ci/generate-infrastructure.py --check` (clean, 0 drift).
2. Generator unit tests: `python3 -m unittest discover -s tests/tools -p "*infrastructure*"`.
3. Metadata schema and configure tests:
   `nix develop -c python3 tests/extensions/metadata_test.py --schema`
   `nix develop -c python3 tests/extensions/metadata_test.py`
   `nix develop -c python3 tests/extensions/discovery_test.py`
4. Extension isolation & target ownership gates:
   `python3 tools/ci/check-extension-isolation.py --fail-on-core-model-includes --fail-on-serving-boundary --fail-on-model-core-includes --fail-on-extension-boundary --fail-on-target-ownership --fail-on-family-graph`
   `python3 tests/tools/check_extension_isolation_test.py`
5. Docs and formatting checks:
   `python3 tools/ci/check-docs.py`
   `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py`
6. Bounded hosted CPU contract and check-pr targets:
   `nix develop -c cmake --preset cpu-test`
   `nix develop -c cmake --build --preset cpu-test --target check-pr`
   `nix build .#checks.x86_64-linux.pr`
7. Derivation independence proof:
   Compare `nix eval --raw .#packages.x86_64-linux.default.drvPath` before and after an edit to a model test file; verify it is unchanged.
8. Continuation identity goldens (GPU preset, weight-free):
   `nix develop -c ctest --preset gpu-full -R 'continuation_identity' --output-on-failure`

### Step 5: Evidence & Handoff
Write durable evidence in `docs/model-extensions/evidence/card14/`:
- `build-and-tests.txt`: exact commands, exit codes, and outputs (truncated where appropriate).
- `pending-commands.txt`: exact deferred commands for functional harness and speed benchmarks.
- `RESUME-NOTES.md`: state summary and explicit release of editing ownership.
- Update `docs/model-extensions/cards/14-generated-infrastructure.md` with preliminary status and details.
