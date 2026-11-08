# Card 23: Support external extension source trees — Evidence Ledger

## Summary of Accomplishments

1. **`GUFO_EXTENSION_DIRS` Discovery**:
   - Added cache string `GUFO_EXTENSION_DIRS` in `cmake/GufoExtensionDiscovery.cmake`.
   - Supports passing one or multiple external directories (direct extension root or directory containing multiple extensions).
   - Resolves paths and marks external extensions with `GUFO_EXT_${name}_IS_EXTERNAL TRUE`.

2. **Configure-time Include Allowlist & Sandbox**:
   - In `cmake/GufoExtension.cmake` and `cmake/GufoExtensionDiscovery.cmake`:
     - Overwrites directory property `INCLUDE_DIRECTORIES` in external wrappers to only `${PROJECT_SOURCE_DIR}/include`.
     - Automatically configures `${GUFO_EXTENSION_BUILD_ROOT}` and system include `${PROJECT_SOURCE_DIR}/include`.
     - Prohibits linking private in-tree libraries (`gufo_core`, `gufo_cache`, `gufo_serving`, `gufo_http`, `gufo_test_support`, `gufo_qwen_config`, etc.).
     - Inspects directory and target `INCLUDE_DIRECTORIES`, failing with a fatal error if any include path reaches `${PROJECT_SOURCE_DIR}/src` or `${PROJECT_SOURCE_DIR}` directly.

3. **Nix Flake Support**:
   - Added `extensions ? [ ]` parameter in `.devops/nix/package.nix`.
   - Appends `-DGUFO_EXTENSION_DIRS=${lib.concatStringsSep ";" (map toString extensions)}` to `cmakeFlags` when non-empty.
   - Compatible with local paths, remote flake inputs, or `pkgs.fetchFromGitHub`.

4. **Reusable Windows CI Workflows**:
   - Added inputs `extension-repo` and `extension-ref` to `.github/workflows/windows-build.yml`.
   - Checks out the external repository and adds `-DGUFO_EXTENSION_DIRS` to CMake configuration flags.

5. **Blueprint Template Repository**:
   - Authored `examples/extension_template/` with `extension.json`, `extension.cmake`, `src/template_model.cpp`, and `README.md` (including third-party Windows ZIP packaging recipes).

6. **Automated Verification**:
   - Added `test_external_extension_discovery` and `test_external_extension_allowlist_violation` in `tests/extensions/discovery_test.py`.
   - Verified that external extensions build and register cleanly, and unauthorized include/library attempts are strictly rejected.

## Gates Status

- `tools/ci/check-format.py`: PASS (850 files)
- `python3 tools/ci/check-docs.py`: PASS (329 files, 1613 links)
- `python3 tools/ci/check-extension-api.py`: PASS (120 surface paths)
- `python3 tools/ci/check-extension-isolation.py`: PASS (0 boundary violations)
- `nix develop -c python3 tests/extensions/discovery_test.py`: PASS (15/15 tests)
- `check-pr`: PASS (97/97 suites passed)
- `nix build && ./result/bin/gufo diagnose`: PASS (8/8 providers active)

## Review corrections

The review found root includes exported by both `gufo_logging` and
`gufo_platform`, expression-wrapped links/includes bypassing validation, and
missing inherited flags. Both root include requirements are now private.
`gufo_ext_runtime` publishes the forwarded `SURFACE` headers through a separate
build include tree, removing stale copies when the surface changes. Published
forwarded headers (including `logging.hpp`, used by `gufo/ext/host.hpp`) remain
part of the source API; unpublished headers such as `logging/trace.hpp` are
inaccessible through this include tree.

Validation runs at the end of configuration over external targets and helper
subdirectories. It checks supported generator-expression payloads, rejects
unknown expressions, resolves aliases, checks dependency usage requirements,
and inspects source settings, language/configuration flags, standard include
directories and compiler include-path environment variables. Paths are
normalized and existing symlinks resolved. Link-only edges are checked for
prohibited dependencies without inheriting compilation settings.

The Windows external checkout now uses the workflow's pinned checkout action
with credentials disabled; its repository input reaches PowerShell through an
environment variable. The template documents these checks as a source API
boundary for trusted CMake manifests.

Review validation on Linux with the pinned Nix CMake/GCC toolchain:

- Discovery suite: 18 tests, including template compilation, unpublished-header
  rejection and configure-time regression cases for the reported bypasses.
- Metadata suite: 12 tests passed.
- Isolated runtime, role and neutral public-header contracts: 3 tests passed.
- Extension API policy tests: 7 tests passed; the PR surface gate passes with
  the existing 0.1.8 API bump and `ext-api` label.
- Documentation, infrastructure generation and isolation checks passed (zero
  strict boundary violations); `git diff --check` passed.

Windows and GPU builds were not run for these review corrections.
