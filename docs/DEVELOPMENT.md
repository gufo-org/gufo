# Development

## Code structure

The codebase is organized in following way:

```sh
.
├── .devops # infra and nix derivations/helpers
├── benchmarks # raw docs to keep track of experiments and performances during development
│   ├── deepseek-v4-flash
│   ├── qwen3-asr
│   └── qwen3.8-27b
├── ...
├── docs # documentation for the final user
├── src
│   ├── cli # cli code
│   ├── core # core functionalities not specific to any model
│   ├── eval # gufo internal evaluation framework
│   ├── models # models collection, each model has its own subdir
│   │   ├── deepseek_v4_flash
│   │   ├── minimax_h3
│   │   ├── qwen
│   │   ├── qwen3_asr
│   │   └── qwen3_tts
│   └── testing # utils for testing
├── tests # unit tests and more
└── tools # tools, scripts, utils, for various tasks
```

Reproducible CMake presets are configured in `CMakePresets.json` and must be entered through the Nix development environment (`nix develop`). Direct host CMake is unsupported.

| Preset          | Purpose             | Description                                                                        |
| --------------- | ------------------- | ---------------------------------------------------------------------------------- |
| `development`   | Development         | CPU-only Debug build with warnings (`-Wall -Wextra -Wpedantic`)                    |
| `release`       | Release             | CPU-only optimized Release build                                                   |
| `cpu-sanitizer` | Diagnostics         | CPU-only build with AddressSanitizer (ASan) and UndefinedBehaviorSanitizer (UBSan) |
| `cpu-test`      | Unit Tests          | CPU-only test suite executed with CTest                                            |
| `gpu-test`      | GPU Tests           | ROCm/HIP enabled for `gfx1151` without XRT                                         |
| `hardware-test` | Full Hardware Build | Pinned ROCm/HIP (`gfx1151`) and XRT (`XDNA2`) stacks                               |

### Development Workflow

```sh
# Development build
nix develop -c cmake --preset development
nix develop -c cmake --build --preset development

# Release build
nix develop -c cmake --preset release
nix develop -c cmake --build --preset release

# Sanitizer build & test
nix develop -c cmake --preset cpu-sanitizer
nix develop -c cmake --build --preset cpu-sanitizer
nix develop -c ctest --preset cpu-sanitizer --output-on-failure

# CPU tests
nix develop -c cmake --preset cpu-test
nix develop -c cmake --build --preset cpu-test
nix develop -c ctest --preset cpu-test --output-on-failure

# HIP GPU tests (gfx1151)
nix develop -c cmake --preset gpu-test
nix develop -c cmake --build --preset gpu-test
nix develop -c ctest --preset gpu-full --output-on-failure

# Complete gfx1151 and XDNA2 tests
nix develop -c cmake --preset hardware-test
nix develop -c cmake --build --preset hardware-test
nix develop -c ctest --preset hardware-full --output-on-failure
```

Hardware presets label tests so unavailable devices skip normally during development. Strict presence validation can be enforced with:

- `GUFO_REQUIRE_HIP=1` (or `GUFO_REQUIRE_GPU=1`)
- `GUFO_REQUIRE_XDNA2=1` (or `GUFO_REQUIRE_NPU=1`)
