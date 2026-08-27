# gufo

gufo is a local inference runtime built specifically for AMD Strix
Halo systems with a `gfx1151` RDNA 3.5 GPU, an XDNA2 NPU, and up to 128 GiB of
unified memory.

The project is intentionally not a general-purpose inference framework. It
will support only the best available open-weights that can run comfortably on
the hardware.

Model-specific quantization, kernels, graph structure, scheduling policy, and
memory layout will be extremely tailored for Strix Halo.

## Current Status

The native C++ runtime supports model-owned ROCm inference paths for
Qwen3.8-27B BF16 and DeepSeek V4 Flash Q2-imatrix. Both models run through the
terminal `gufo prompt` command, `gufo bench`, and `gufo serve`.
DeepSeek uses its own graph, state, quantized layouts, and kernels under
`src/models/deepseek_v4_flash`; it does not call Qwen compute code.
`gufo eval` exercises any already running OpenAI-compatible text server with
the pinned Antirez DS4 capability questions and writes a sanitized regression
artifact.

Qwen3.5-0.8B remains the rapid-iteration quantization model. Qwen3.8-27B is the
primary dense text model, while DeepSeek V4 Flash is the first quantized MoE
model. GPU/NPU interoperability and NPU execution remain evidence-gated
parallel work.

See:

- [Project status and decisions](docs/PROJECT_STATUS.md)
- [Implementation roadmap](docs/ROADMAP.md)
- [Performance engineering and profiling](docs/PERFORMANCE.md)
- [Capability evaluation](docs/EVAL.md)
- [Offline tools](tools/README.md)
- [Qwen3.5-0.8B benchmark](benchmarks/qwen3.5-0.8b/README.md)
- [Qwen3.8-27B benchmark](benchmarks/qwen3.8-27b/README.md)
- [DeepSeek V4 Flash benchmark](benchmarks/deepseek-v4-flash/README.md)

## Supported Platform

Linux x86-64 on AMD Strix Halo (`gfx1151` GPU and XDNA2 NPU) is the only
planned production platform. Windows, macOS, and CUDA are out of scope.

Build with Nix only; direct host builds are unsupported:

```sh
nix build                          # build default package (gfx1151 + XRT)
./result/bin/gufo diagnose        # run hardware probe & diagnostics
./result/bin/gufo serve           # run server
nix build .#checks.x86_64-linux.pr # canonical PR test command (all gates)
```

## Development Presets

Reproducible CMake presets are configured in `CMakePresets.json` and must be entered through the Nix development environment (`nix develop`). Direct host CMake is unsupported.

| Preset | Purpose | Description |
| --- | --- | --- |
| `development` | Development | CPU-only Debug build with warnings (`-Wall -Wextra -Wpedantic`) |
| `release` | Release | CPU-only optimized Release build |
| `cpu-sanitizer` | Diagnostics | CPU-only build with AddressSanitizer (ASan) and UndefinedBehaviorSanitizer (UBSan) |
| `cpu-test` | Unit Tests | CPU-only test suite executed with CTest |
| `gpu-test` | GPU Tests | ROCm/HIP enabled for `gfx1151` without XRT |
| `hardware-test` | Full Hardware Build | Pinned ROCm/HIP (`gfx1151`) and XRT (`XDNA2`) stacks |

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

## Reference Projects

The initial design is informed by the following open source projects:

- `llama.cpp` for compact model serving, GGUF, and CPU/GPU correctness paths.
- `vLLM` for continuous batching and paged request scheduling.
- `hipEngine` for torch-free HIP execution, and native speculative-cycle work.
- `ROCmFPX` for activation-aware quantization and quality evaluation.
- `DS4` for DeepSeek V4 Flash, MoE scheduling, and DSpark.
- `ypapadop-amd/ggml` `hsa-backend` for XDNA2 HSA dispatch and MLIR-AIE
  integration patterns.
