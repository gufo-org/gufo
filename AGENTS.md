# AGENTS.md

## Platform and build

Production packages target Linux x86-64 AMD Strix Halo (`gfx1151`); Windows
is a supported native build target for development. CMake owns
compiler flags, dependencies and installation; Nix supplies the pinned toolchain.
Linux source-build prerequisites are in [docs/install.md](docs/install.md#build-from-source).

```sh
nix build                              # development package, all discovered extensions
nix build .#release                    # shipped lock-mode package, no tests/tools
./result/bin/gufo diagnose
nix flake check                       # hosted CPU/repository checks
nix develop                            # GPU development and reference tools

# Same production build without Nix, with the documented dependencies installed
cmake --preset release                # discovery build
# Add -DGUFO_EXTENSIONS_FROM_LOCK=ON for the shipped extension selection.
cmake --build --preset release --parallel 4
```

Stage only task-owned paths before Nix builds; flakes include tracked files.
Measure performance with `result/bin/gufo` or `build/release/gufo`. Preserve
compiler/dependency versions when comparing results.

## Focused tests

Formatting and Python repository checks may run on the editing host; they do
not need the remote GPU. Before committing C++ changes, run the shared CI check:

```sh
nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py
# Add --fix to apply formatting, then rerun the check.
```

Tests selected by `nix flake check` should stay fast and cache-friendly where
possible. Reuse expensive setup and compiler caches when valid, use the smallest
deterministic fixtures that preserve the assertions, and avoid redundant builds,
fixed sleeps and repeated whole-tree work. Measure added wall time; keep expensive
stress/model qualification explicitly selectable without weakening correctness.

Run the smallest check covering the change. `gpu-test` is RelWithDebInfo with
assertions enabled. Build only the affected target during iteration:

```sh
nix develop -c cmake --preset gpu-test
nix develop -c cmake --build --preset gpu-test --target <test-target>
nix develop -c ctest --preset gpu-full -R '^<test-name>$' --output-on-failure
```

The same CMake/CTest commands work outside Nix. `cmake --build --preset pr`
runs the focused contract subset after configuring `cpu-test`; `nix flake check`
runs the canonical CPU/repository gate. Full CPU checks,
sanitisers, GPU/operator/model checks and H3 quality tools remain available
locally; see [docs/internal/testing.md](docs/internal/testing.md). Do not run full model sweeps,
video generation or duplicate suites for routine edits. A missing-model skip
is not a quality pass. Broaden checks when shared behavior changes or failures
expose risk.

Text API functional/regression tests live in [tests/functional/](tests/functional/README.md).
Select suites explicitly. Model-specific changes need the affected model/modes;
use all four profiles only for shared behavior changes. Run correctness on the
candidate; use main for matched affected timing controls and suspected regressions,
not a duplicate full correctness run. Reuse baseline evidence only when revision,
toolchain, harness, model, settings and cache history match. Include `long-context`
and `cache` for continuation changes; verify loaded mode and actual drafts.
Correctness and expected prefill/cache work are strict; timings are
mandatory per request/phase at 5% and 10 ms, never averaged across requests.
Run once, investigate flags, then alternate main/PR only for affected histories;
use unchanged-main controls when needed and retain every result. Fix confirmed
regressions before publishing. Noisy evidence stays visibly inconclusive and
unqualified; do not widen margins or stop at reporting failures. Devlab keeps
inconclusive timing advisory on both ordinary and release PRs. Use the code
diff to select affected metrics; unrelated timing variance does not justify
another full matrix.
For reported coding-agent loops, also replay the affected workflow with
`tests/functional/pi_agent.py`; retain its sessions and inspect actual tool results.
Retain numerical quality tests and the standard speed benchmark. Keep these
tests in `tests/functional/`, outside hosted model CI; avoid full sweeps.

## Profiling and kernels

Apply [.agents/skills/optimize-kernel/SKILL.md](.agents/skills/optimize-kernel/SKILL.md).
Use `tools/bench/build.sh` for standalone HIP experiments,
`tools/bench/gfx1151_peak.hip` for measured hardware ceilings,
`tools/prof/prof.py` for pipeline/wall-time profiles, and
`tools/prof/isa_mix.py` for instruction analysis. Production paths must retain
quality; successful optimizations become the default, without extra switches.

## Development

- Keep model code, tests, tools and numerical contracts with their model.
- Follow [docs/AGENTS.md](docs/AGENTS.md) when writing or changing documentation.
- Use the shared Qwen control-token constants for framing; see
  [src/families/qwen/AGENTS.md](src/families/qwen/AGENTS.md). Inspect their definitions
  when reviewing or changing behavior, and keep reference fixtures independent.
- Put Windows-specific code in `src/core/platform/`: one leaf per OS facility
  (files, pages, sockets), whose sources may branch on `_WIN32` or split into
  `<leaf>_posix.cpp`/`<leaf>_windows.cpp` selected in CMake. Outside it and
  vendored code, call the leaf, with no `windows.h` or `_WIN32`/`_MSC_VER`
  checks. POSIX code moves into a leaf, verbatim, once it gains a Windows
  counterpart. Never emulate POSIX with macros or copy a file per OS; when the
  logic itself differs, add `<file>_windows.cpp` beside it.
- Use one canonical long option and backend name per behavior; avoid aliases.
- Use `gh` for GitHub operations after checking `gh auth status`.
- Follow Conventional Commits with a single-line message. Title pull requests
  according to [the release policy](docs/internal/releasing.md); squash merging retains
  that title on `main`. Mark breaking changes with `!` in the title, and leave
  version and changelog updates to the release workflow.
- Prefer `jj` when available (`jj version`); otherwise use Git.
- The source extension API is versioned independently from Gufo. Public headers
  under `include/gufo/ext/`, their transitive `SURFACE` closure (including
  forwarded core/cache headers), the metadata schema and public CMake contracts
  require an `ext-api` PR label and one new, uniquely named Markdown record under
  `include/gufo/ext/changes/<major>.<minor>/` when declarations/contracts change.
  Records are permanent and immutable. CMake derives the API revision from the
  active series in `include/gufo/ext/api-series.json` plus its revision-record
  count. The `baseline.md` record documents the initial contracts without advancing
  it. Do not manually bump versions for additive changes.
  Run `python3 tools/ci/check-extension-api.py --write` after header/include
  changes; the PR diff gate also checks forwarded declarations. Before 1.0, a minor bump
  may break extensions; revision bumps are additive. Fix in-tree users together.
  The guarantee is source compatibility after rebuilding, never binary ABI.
  Breaking changes explicitly advance the compatibility series in api-series.json
  and add a record in the new series, retaining all historical records. Extension
  API and the future experimental flake `extensions` argument are excluded from Gufo's
  release compatibility surface; CLI/HTTP/public existing flake breaks still
  require `!`. See [the release policy](docs/internal/releasing.md).
- Follow the user's remote workflow and preserve unrelated work.
