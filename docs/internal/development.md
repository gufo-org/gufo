# Development

How the repository is laid out, how to build and test it, and the conventions
for adding code. Production targets Linux x86-64 on AMD Strix Halo
(`gfx1151`); CMake owns compiler flags, dependencies and installation, and Nix
supplies the pinned toolchain. Dependencies and versions are in
[install](../install.md#build-from-source); repository rules for agents and
contributors are in [AGENTS.md](../../AGENTS.md).

## Repository layout

| Path | Ownership |
| --- | --- |
| `src/cli` | Executable commands, terminal interfaces and HTTP adapters |
| `src/core` | Shared runtime: GGUF loading, sampling, quantization, HIP helpers, diagnostics |
| `src/cache` | Continuation cache: tiers, ledger and the neutral capture/backing contract |
| `src/sched` | Text scheduling policy: work-unit decisions and prefill rate, without serving or model code |
| `src/ext` | Extension runtime: registry, options, neutral continuation and host-service contracts |
| `src/models/<model>` | One model extension: manifest, config, weights, engine, kernels, CPU reference |
| `src/families/<family>` | Libraries shared by several models of one family, such as Qwen |
| `src/eval` | HTTP capability evaluation |
| `src/testing` | Test-only comparison helpers |
| `tests` | Maintained fixtures and focused checks, mirroring `src` |
| `tools` | CI entrypoints, profiling, microbenchmarks, offline evaluation and reference runners |
| `cmake` | Shared CMake modules, including the hosted `check-pr` selection |
| `docs` | Public pages; `docs/internal` holds developer documents and per-model evidence (see [docs/AGENTS.md](../AGENTS.md)) |
| `.devops/nix` | Production package, reference packages and server configuration |
| `.github/workflows` | Hosted CI definition |
| `artifacts`, `models` | Ignored local outputs and weights; never committed |

Model code, kernels, tests, tools and numerical contracts stay with their model.
Every model is an extension: configure discovers its `extension.json`, builds
the libraries its `extension.cmake` declares, and links them into `gufo` through
the generated registry (see [extension build](extension-build.md)). Heavy CPU
oracles are `EXCLUDE_FROM_ALL` so routine builds stay cheap.

`gufo_serving` (`src/cli/serve`: scheduler, model runner, cache facade, model
selection) and `gufo_http` are the top of the graph. Nothing under
`gufo_ext_runtime` includes `src/cli/**` or links either library, which
`extension_serving_boundary_gate` and `extension_link_graph_gate` enforce; see
[testing](testing.md#boundary-and-link-graph-gates) and
[target ownership](target-ownership.md).

## Build

Nix builds the production package and the hosted checks. Stage new files in
Git first: flakes only see tracked files.

```sh
nix build                              # production package, no tests or tools
./result/bin/gufo diagnose
nix flake check     # hosted CPU/repository checks
nix develop                            # GPU development, profilers, reference tools
```

The same CMake commands work without Nix once the documented dependencies are
installed; `nix develop -c <command>` runs them inside the pinned environment.

Alongside the pinned compiler, `nix develop` carries the tools the loop needs:
`gdb` for the assertion-enabled test presets, `git` and `jj` for the repository,
`ripgrep` for searching, `radeontop` for live GPU load, and `ccache` for rebuilds.

```sh
cmake --preset release
cmake --build --preset release --parallel 4
```

CMake compiles C and C++ through `ccache` when it is on `PATH`, so reconfigured
builds and header changes reuse cached objects; the configure log reports either
`gufo: compiling C and CXX through ...` or `gufo: ccache not found`. HIP kernels
are not launched through the cache.

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
[CMakePresets.json](../../CMakePresets.json).

### Options

| Option | Default | Effect |
| --- | --- | --- |
| `ENGINE_ENABLE_HIP` | `ON` | Build the gfx1151 HIP backend; only `gfx1151` is accepted |
| `BUILD_TESTING` | `OFF` | Build correctness tests |
| `GUFO_BUILD_TOOLS` | `OFF` | Build kernel benchmarks and tuning executables |
| `GUFO_ENABLE_WARNINGS` | `ON` | Compiler warnings |
| `GUFO_ENABLE_SANITIZERS` | `OFF` | AddressSanitizer and UndefinedBehaviorSanitizer |
| `GUFO_RELEASE_VERSION` | `development` | Reported semantic release version |
| `GUFO_REVISION` | `unknown` | Reported source revision and provenance identity |
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

[Testing](testing.md) maps change categories to the checks that cover them.
Do not run full model sweeps, video generation or duplicate suites for routine
edits; a skip caused by a missing model or device is not a quality pass.

## Conventions

- Formatting follows [.clang-format](../../.clang-format): Google style, C++20,
  two-space indent, 80 columns, regrouped and sorted includes.
- [.clang-tidy](../../.clang-tidy) runs bugprone, cert, clang-analyzer,
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
in [cmake/Checks.cmake](../../cmake/Checks.cmake) only when they belong in every PR;
model oracles stay out of the hosted set.

**A model.** Follow [writing an extension](../writing-extensions.md): create
`src/models/<model>` with `extension.json`, `extension.cmake` and a
`library.json` ([target ownership](target-ownership.md)), keep kernels under
`kernels/` and the CPU reference behind `EXCLUDE_FROM_ALL`, and add tests under
`tests/models/<model>`. Set `docs` in the manifest, run
`python3 tools/ci/generate-infrastructure.py --write` to update the model index
and generated CI, then write the public page `docs/models/<model>.md` and
`BENCHMARKS.md`, `QUALITY.md` and `EXPERIMENTS.md` under
`docs/internal/models/<model>/`. Shipping it is a one-line change to
`extensions.lock`.

**A dependency.** Record it in `THIRD_PARTY_NOTICES.md`; `tools/ci/check-dependencies.py`
validates that shipped and evaluation components carry license, version and
source records.

**Documentation.** Follow [docs/AGENTS.md](../AGENTS.md).
`tools/ci/check-docs.py` checks required documents, every local link and
anchor, fenced JSON, and the public-page and figure rules. Run
`nix build .#checks.x86_64-linux.docs` for documentation-only changes.

## Performance work

Measure with `result/bin/gufo` from `nix build` or a `release` build, keeping
compiler and dependency versions fixed across compared runs. Follow
[performance engineering](performance.md) and the
[kernel optimization skill](../../.agents/skills/optimize-kernel/SKILL.md). Use
`tools/bench/build.sh` for standalone HIP experiments,
`tools/bench/gfx1151_peak.hip` for measured hardware ceilings,
`tools/prof/prof.py` for pipeline and wall-time profiles, and
`tools/prof/isa_mix.py` for instruction analysis. Retained numbers belong in
`docs/internal/models/<model>/BENCHMARKS.md` under the rules in
[benchmarking](benchmarks.md).

## Submitting changes

Hosted CI runs `nix flake check` on Ubuntu with a 25-minute
limit: formatting, documentation, dependency inventory, server-command checks
and the CPU contract suite. It does not build ROCm, download weights or run
model oracles, so run the GPU checks that cover the change locally; the exact
hosted selection is described in [testing](testing.md).

- Prefer `jj` when available (`jj version`); otherwise use Git.
- Follow Conventional Commits with a single-line message. Pull request titles
  must follow the types and version effects in the [release policy](releasing.md);
  the squash merge retains the title as the commit on `main`.
- Use `gh` for GitHub operations after checking `gh auth status`.
- Preserve unrelated work and stage only task-owned paths before Nix builds.

## Related documents

[Testing](testing.md), [CI](ci.md), [releasing](releasing.md),
[performance](performance.md), [benchmarking](benchmarks.md),
[server reference](server.md), [cache reference](cache.md) and
[extension build](extension-build.md).

The production package ships the HIP backend only.
