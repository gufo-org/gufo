# Development

Production targets Linux x86-64 on AMD Strix Halo (`gfx1151`). CMake owns the
compiler flags, dependencies and installation for every build; Nix supplies the
pinned toolchain. System dependencies and the qualified compiler/ROCm versions
are listed in [the README](../README.md#build-from-source). Repository rules for
agents and contributors are in [AGENTS.md](../AGENTS.md).

## Repository layout

| Path | Ownership |
| --- | --- |
| `src/cli` | Executable commands, terminal interfaces and HTTP adapters |
| `src/core` | Shared runtime: GGUF loading, sampling, quantization, HIP helpers, diagnostics |
| `src/models/<model>` | One model package: config, weights, engine, kernels, CPU reference |
| `src/eval` | HTTP capability evaluation |
| `src/testing` | Test-only comparison helpers |
| `tests` | Maintained fixtures and focused checks, mirroring `src` |
| `tools` | CI entrypoints, profiling, microbenchmarks, offline evaluation and reference runners |
| `cmake` | Shared CMake modules, including the hosted `check-pr` selection |
| `docs` | Cross-cutting guides; `docs/models/<model>` holds per-model documents |
| `.devops/nix` | Production package, reference packages and server configuration |
| `.github/workflows` | Hosted CI definition |
| `artifacts`, `models` | Ignored local outputs and weights; never committed |

Model code, kernels, tests, tools and numerical contracts stay with their model.
A model package builds as its own static library and is attached in the root
`CMakeLists.txt` with `add_subdirectory`; heavy CPU oracles are declared
`EXCLUDE_FROM_ALL` so routine builds stay cheap.

## Build

Nix builds the production package and the hosted checks. Stage new files in
Git first: flakes only see tracked files.

```sh
nix build                              # production package, no tests or tools
./result/bin/gufo diagnose
nix build .#checks.x86_64-linux.pr     # hosted CPU/repository checks
nix develop                            # GPU development, profilers, reference tools
```

The same CMake commands work without Nix once the documented dependencies are
installed; `nix develop -c <command>` runs them inside the pinned environment.

```sh
cmake --preset release
cmake --build --preset release --parallel 4
```

### Presets

| Configure preset | Build type | Contents |
| --- | --- | --- |
| `release` | RelWithDebInfo | Production inference with HIP; no tests, no tools |
| `gpu-test` | RelWithDebInfo + assertions | HIP, tests and development tools |
| `cpu-test` | RelWithDebInfo | Host-only tests, no HIP |
| `cpu-sanitizer` | Debug + ASan/UBSan | Host memory and undefined-behavior diagnostics |

Build preset `pr` configures against `cpu-test` and runs the hosted contract
target `check-pr`. Test presets: `cpu-test` (label `cpu`), `cpu-sanitizer`,
`gpu-fast` (excludes `slow` and `external-model`), `gpu-full` (complete GPU
tree), `deepseek-gpu` and `qwen-gpu-kernel-oracle`. Definitions live in
[CMakePresets.json](../CMakePresets.json).

### Options

| Option | Default | Effect |
| --- | --- | --- |
| `ENGINE_ENABLE_HIP` | `ON` | Build the gfx1151 HIP backend; only `gfx1151` is accepted |
| `BUILD_TESTING` | `OFF` | Build correctness tests |
| `GUFO_BUILD_TOOLS` | `OFF` | Build kernel benchmarks and tuning executables |
| `GUFO_ENABLE_WARNINGS` | `ON` | Compiler warnings |
| `GUFO_ENABLE_SANITIZERS` | `OFF` | AddressSanitizer and UndefinedBehaviorSanitizer |
| `GUFO_VERSION` | `development` | Reported build revision |
| `GUFO_FFMPEG_EXECUTABLE`, `GUFO_FFPROBE_EXECUTABLE` | `ffmpeg`, `ffprobe` | External media executables |

C++20 is required; HIP kernels compile from the same tree, and the adapted
llama.cpp quantized kernels keep their own C++17 dialect in isolation.

## Development loop

Run the smallest check that covers the change. Formatting and the Python
repository checks need no GPU and no Gufo build.

```sh
nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py
# Add --fix to apply formatting, then rerun the check.

nix develop -c cmake --preset gpu-test
nix develop -c cmake --build --preset gpu-test --target <test-target>
nix develop -c ctest --preset gpu-full -R '^<test-name>$' --output-on-failure
```

[Testing](TESTING.md) maps change categories to the checks that cover them.
Do not run full model sweeps, video generation or duplicate suites for routine
edits; a skip caused by a missing model or device is not a quality pass.

## Conventions

- Formatting follows [.clang-format](../.clang-format): Google style, C++20,
  two-space indent, 80 columns, regrouped and sorted includes.
- [.clang-tidy](../.clang-tidy) runs bugprone, cert, clang-analyzer,
  cppcoreguidelines, misc and performance checks over `src/` with
  `WarningsAsErrors: '*'`. Whole-tree static analysis is the explicit
  `static-analysis` check, not part of hosted CI.
- One canonical long option and backend name per behavior; no aliases.
- Successful optimizations become the default path, without extra switches, and
  production paths retain quality.

## Adding code

**A test.** Declare the executable and `add_test` in the root
`CMakeLists.txt` (or the model's `tests/models/<model>` subdirectory), and give
it labels — `cpu` for host-only checks, plus model, subsystem and cost labels
such as `slow` or `external-model`. Add cheap host contracts to `gufo_pr_targets`
in [cmake/Checks.cmake](../cmake/Checks.cmake) only when they belong in every PR;
model oracles stay out of the hosted set.

**A model.** Create `src/models/<model>` with its own `CMakeLists.txt` exporting
a `gufo_<model>` library, attach it with `add_subdirectory`, keep kernels under
`kernels/` and the CPU reference behind `EXCLUDE_FROM_ALL`. Add tests under
`tests/models/<model>`, a row in [the model index](models/README.md), and the
four required documents `README.md`, `BENCHMARKS.md`, `QUALITY.md` and
`EXPERIMENTS.md` under `docs/models/<model>/`.

**A dependency.** Record it in `THIRD_PARTY_NOTICES.md`; `tools/ci/check-dependencies.py`
validates that shipped and evaluation components carry license, version and
source records.

**Documentation.** `tools/ci/check-docs.py` enforces the presence of the
maintained documents, resolves every local link and heading anchor offline, and
parses fenced JSON blocks. Run `nix build .#checks.x86_64-linux.docs` for
documentation-only changes.

## Performance work

Measure with `result/bin/gufo` from `nix build` or a `release` build, keeping
compiler and dependency versions fixed across compared runs. Follow
[performance engineering](PERFORMANCE.md) and the
[kernel optimization skill](../.agents/skills/optimize-kernel/SKILL.md). Use
`tools/bench/build.sh` for standalone HIP experiments,
`tools/bench/gfx1151_peak.hip` for measured hardware ceilings,
`tools/prof/prof.py` for pipeline and wall-time profiles, and
`tools/prof/isa_mix.py` for instruction analysis. Retained numbers belong in
`docs/models/<model>/BENCHMARKS.md` under the rules in
[benchmarking](BENCHMARKS.md).

## Submitting changes

Hosted CI runs `nix build .#checks.x86_64-linux.pr` on Ubuntu with a 25-minute
limit: formatting, documentation, dependency inventory, server-command checks
and the CPU contract suite. It does not build ROCm, download weights or run
model oracles, so run the GPU checks that cover the change locally; the exact
hosted selection is described in [testing](TESTING.md).

- Prefer `jj` when available (`jj version`); otherwise use Git.
- Follow Conventional Commits with a single-line message.
- Use `gh` for GitHub operations after checking `gh auth status`.
- Preserve unrelated work and stage only task-owned paths before Nix builds.

## Related documents

[CLI](CLI.md), [server](SERVER.md), [testing](TESTING.md),
[performance](PERFORMANCE.md), [benchmarking](BENCHMARKS.md) and
[models](models/README.md).

The production package ships the HIP backend only. Experimental accelerator work
remains on `perf/qwen27b-dflash2-npu`; offline quantization research remains in
PR #234. Neither toolchain is required to install or run main.
