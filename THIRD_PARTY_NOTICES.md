# Third-Party Notices and Inventory

This document records the complete inventory of third-party software, libraries, drivers,
and system components used, linked, or required by **gufo**.

For design policy details regarding licensing boundaries, see [docs/LICENSING.md](docs/LICENSING.md).

---

## Inventory Summary

| Component Name | Relationship | License (SPDX) | Pinned Revision / Version | Upstream Source / Location |
| --- | --- | --- | --- | --- |
| **XRT** | Linked | `Apache-2.0` | `8661761775a266b11992a3bd6eb08209d88aa845` | [Xilinx/XRT](https://github.com/Xilinx/XRT) |
| **xdna-driver** | Linked / Loaded | `Apache-2.0` | `4e5aed38f3b74a5a9a2c7a6222eaff1a8be54305` | [amd/xdna-driver](https://github.com/amd/xdna-driver) |
| **MLIR-AIE** | Build toolchain | `Apache-2.0 WITH LLVM-exception` | `1.4.1` | [Xilinx/mlir-aie](https://github.com/Xilinx/mlir-aie) |
| **LLVM-AIE** | Build toolchain | `Apache-2.0 WITH LLVM-exception` | `21.0.0.2026080301+c9c5ecb7` | [Xilinx/llvm-aie](https://github.com/Xilinx/llvm-aie) |
| **AIEBU** | Build toolchain | `MIT` | `27a302c5840773e79c79f0f2fc8a1832d6ab1774` | [Xilinx/aiebu](https://github.com/Xilinx/aiebu) |
| **ROCm / HIP** | Linked / Toolchain | `MIT OR Apache-2.0 WITH LLVM-exception` | Nixpkgs `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` | [ROCm/clr](https://github.com/ROCm/clr) |
| **hipBLAS** | Linked | `MIT` | Nixpkgs `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` | [ROCm/hipBLAS](https://github.com/ROCm/hipBLAS) |
| **hipBLASLt** | Linked | `MIT` | Nixpkgs `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` | [ROCm/hipBLASLt](https://github.com/ROCm/hipBLASLt) |
| **rocBLAS** | Linked | `MIT` | Nixpkgs `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` | [ROCm/rocBLAS](https://github.com/ROCm/rocBLAS) |
| **Composable Kernel** | Header / compiled kernels | `MIT` | Nixpkgs `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` | [ROCm/composable_kernel](https://github.com/ROCm/composable_kernel) |
| **ROCprofiler SDK / ROCTx** | Benchmark marker library / profiling tool | `MIT` | Nixpkgs `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` | [ROCm/rocprofiler-sdk](https://github.com/ROCm/rocprofiler-sdk) |
| **DS4** | Vendored model engine and ROCm kernels | `MIT` | `84cc882352757baf628a1776badf7cc54d584e28` | [antirez/ds4](https://github.com/antirez/ds4) |
| **h3.c** | Pinned implementation reference; selected code may be adapted model-privately | `MIT` | `8974cc055ea9c02fcd14cc27dfda3e1027c05153` | [antirez/h3.c](https://github.com/antirez/h3.c) |
| **ccv TensorOps matmul ancestry** | Algorithm/source ancestry identified by h3.c | `BSD-3-Clause` | Notice pinned through h3.c commit `8974cc055ea9c02fcd14cc27dfda3e1027c05153` | [libccv/ccv](https://github.com/liuliu/ccv) |
| **llama.cpp** | Source-derived algorithm | `MIT` | `e9fa0781f1c25fc4fe8c86be1edc6970661ad6f0` | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) |
| **libuuid** | Linked | `BSD-3-Clause` / `LGPL-2.1-or-later` | Nixpkgs `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` | [util-linux](https://git.kernel.org/pub/scm/utils/util-linux/util-linux.git) |
| **ICU** | Linked | `Unicode-3.0` | Nixpkgs `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` | [unicode-org/icu](https://github.com/unicode-org/icu) |
| **curl / libcurl** | Linked HTTP client | `curl` | `8.21.0`, Nixpkgs `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` | [curl/curl](https://github.com/curl/curl) |
| **FFmpeg** | Spawned runtime executable | `LGPL-2.1-or-later AND GPL-2.0-or-later` (enabled components may also be `LGPL-3.0-or-later` / `GPL-3.0-or-later`) | `8.1.2`, Nixpkgs `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` | [FFmpeg/FFmpeg](https://github.com/FFmpeg/FFmpeg) |
| **PyTorch ROCm** | Evaluation/offline-teacher tool only; not shipped | `BSD-3-Clause` | `2.12.0`, Nixpkgs `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` | [pytorch/pytorch](https://github.com/pytorch/pytorch) |
| **Torchvision** | Evaluation/LPIPS tool only; not shipped | `BSD-3-Clause` | `0.27.0`, Nixpkgs `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` | [pytorch/vision](https://github.com/pytorch/vision) |
| **LPIPS** | Perceptual-quality evaluation tool only; not shipped | `BSD-2-Clause` | `0.1.4`, Nixpkgs `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` | [richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity) |
| **Torchvision AlexNet weights** | Evaluation model data only; not shipped in the production package | `NOASSERTION` | SHA-256 `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02` | [PyTorch model distribution](https://download.pytorch.org/models/alexnet-owt-7be5be79.pth) |
| **`amdxdna` Kernel Driver** | System (Kernel) | `GPL-2.0-only` | System Kernel (`amdxdna.ko`) | [amd/xdna-driver](https://github.com/amd/xdna-driver) |
| **`amdxdna` UAPI Headers** | System / Header | `GPL-2.0 WITH Linux-syscall-note` | `4e5aed38f3b74a5a9a2c7a6222eaff1a8be54305` | [amd/xdna-driver](https://github.com/amd/xdna-driver) |
| **AMD NPU Firmware** | System (Firmware) | Proprietary Binary (`LICENSE.amdnpu`) | System Firmware (`linux-firmware`) | Host OS Distribution / AMD |
| **MiniMax H3 FL2VA checkpoint** | External operator-supplied model; not distributed | `LicenseRef-MiniMax-H3-Community-2026-08-02` or operator-specific authorization | `42ed227ee7df40d41602854ae760620d6eb651fe` | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) |
| **Qwen3-VL-32B encoder weights used by H3** | External operator-supplied model component; not distributed | `Apache-2.0` | Included by the pinned H3 FL2VA checkpoint | [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) |
| **Terminal-Bench 2.1 task definitions** | Redistributed (derived, modified) under `tools/eval/tasks/` | `Apache-2.0` | `5c8eadf1f393183288fa08b8f73ca9a469cc5e00` | [harbor-framework/terminal-bench-2-1](https://github.com/harbor-framework/terminal-bench-2-1) via [kyuz0/terminal-bench-mini](https://github.com/kyuz0/terminal-bench-mini) |

---

## 1. Distributed Userspace Dependencies

### 1.1 XRT (Xilinx Runtime)

- **Component Name**: XRT (Xilinx Runtime for AMD Ryzen AI NPU)
- **Upstream URL**: https://github.com/Xilinx/XRT
- **Pinned Revision**: Commit `8661761775a266b11992a3bd6eb08209d88aa845` (`unstable-2026-06-04`)
- **Component Used**: Userspace runtime libraries (`libxrt_core`, `libxrt_coreutil`, C/C++ headers)
- **SPDX License Identifier**: `Apache-2.0`
- **Copyright / Notice Source**:
  - Copyright (C) 2016-2022 Xilinx, Inc.
  - Copyright (C) 2022-2023 Advanced Micro Devices, Inc.
  - Additional upstream copyright holders preserved in project [NOTICE](NOTICE) file.
- **Relationship**: Linked (Dynamic runtime dependency in Nix derivation `.devops/nix/xrt.nix`)
- **Corresponding-Source Location**: https://github.com/Xilinx/XRT/tree/8661761775a266b11992a3bd6eb08209d88aa845

### 1.2 xdna-driver (AMD XDNA XRT Userspace Shim)

- **Component Name**: xdna-driver (AMD XDNA Driver Shim & Plugin for XRT)
- **Upstream URL**: https://github.com/amd/xdna-driver
- **Pinned Revision**: Commit `4e5aed38f3b74a5a9a2c7a6222eaff1a8be54305` (Branch `1.7`, plugin version `2.21.0`, `1.7-unstable-2026-07-22`)
- **Component Used**: Userspace driver plugin (`libxrt_driver_xdna.so`, shim headers under `src/shim`)
- **SPDX License Identifier**: `Apache-2.0`
- **Copyright / Notice Source**: Copyright (C) 2022-2025, Advanced Micro Devices, Inc. All rights reserved. (Upstream repository checked: no separate root NOTICE file exists at this revision).
- **Relationship**: Linked / Loaded (Dynamic XRT plugin loaded at runtime in `.devops/nix/xrt-plugin-amdxdna.nix`)
- **Corresponding-Source Location**: https://github.com/amd/xdna-driver/tree/4e5aed38f3b74a5a9a2c7a6222eaff1a8be54305

### 1.3 ROCm / HIP (AMD ROCm Compute Language Runtime)

- **Component Name**: ROCm / HIP (AMD ROCm Compute Language Runtime)
- **Upstream URL**: https://github.com/ROCm/clr
- **Pinned Revision**: Nixpkgs `nixos-unstable` lock revision `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` (`rocmPackages.clr` / `rocmPackages.llvm.clang`)
- **Component Used**: HIP runtime, headers, host/device headers, and Clang compiler for `gfx1151`
- **SPDX License Identifier**: `MIT OR Apache-2.0 WITH LLVM-exception`
- **Copyright / Notice Source**: Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
- **Relationship**: Linked / Toolchain (Build toolchain and runtime library dependency)
- **Corresponding-Source Location**: https://github.com/ROCm/clr (via Nix derivation `rocmPackages.clr`)

### 1.4 hipBLAS

- **Component Name**: hipBLAS (AMD ROCm BLAS Marshalling Library)
- **Upstream URL**: https://github.com/ROCm/hipBLAS
- **Pinned Revision**: Nixpkgs `nixos-unstable` lock revision `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` (`rocmPackages.hipblas`)
- **Component Used**: Dynamic BLAS abstraction library and interface headers
- **SPDX License Identifier**: `MIT`
- **Copyright / Notice Source**: Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
- **Relationship**: Linked (Dynamic library dependency)
- **Corresponding-Source Location**: https://github.com/ROCm/hipBLAS (via Nix derivation `rocmPackages.hipblas`)

### 1.5 hipBLASLt

- **Component Name**: hipBLASLt (AMD ROCm Tunable BLAS Library)
- **Upstream URL**: https://github.com/ROCm/hipBLASLt
- **Pinned Revision**: Nixpkgs `nixos-unstable` lock revision `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` (`rocmPackages.hipblaslt`)
- **Component Used**: Tuned BF16 matrix multiplication kernels and algorithm-selection interface for prompt processing
- **SPDX License Identifier**: `MIT`
- **Copyright / Notice Source**: Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
- **Relationship**: Linked (Dynamic library dependency)
- **Corresponding-Source Location**: https://github.com/ROCm/hipBLASLt (via Nix derivation `rocmPackages.hipblaslt`)

### 1.6 rocBLAS

- **Component Name**: rocBLAS (AMD ROCm Basic Linear Algebra Subprograms)
- **Upstream URL**: https://github.com/ROCm/rocBLAS
- **Pinned Revision**: Nixpkgs `nixos-unstable` lock revision `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` (`rocmPackages.rocblas`)
- **Component Used**: GPU BLAS execution kernels optimized for `gfx1151`
- **SPDX License Identifier**: `MIT`
- **Copyright / Notice Source**: Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
- **Relationship**: Linked (Dynamic library dependency)
- **Corresponding-Source Location**: https://github.com/ROCm/rocBLAS (via Nix derivation `rocmPackages.rocblas`)

### 1.7 Composable Kernel

- **Component Name**: Composable Kernel (ROCm GPU kernel library)
- **Upstream URL**: https://github.com/ROCm/composable_kernel
- **Pinned Revision**: Nixpkgs `nixos-unstable` lock revision `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` (`rocmPackages.composable_kernel`)
- **Component Used**: Header-instantiated causal grouped-query attention kernel compiled into the `gfx1151` HIP backend
- **SPDX License Identifier**: `MIT`
- **Copyright / Notice Source**: Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
- **Relationship**: Header / compiled kernels
- **Corresponding-Source Location**: https://github.com/ROCm/composable_kernel (via Nix derivation `rocmPackages.composable_kernel`)

### 1.8 ROCprofiler SDK / ROCTx

- **Component Name**: ROCprofiler SDK / ROCTx
- **Upstream URL**: https://github.com/ROCm/rocprofiler-sdk
- **Pinned Revision**: Nixpkgs `nixos-unstable` lock revision `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` (`rocmPackages.rocprofiler-sdk`)
- **Component Used**: The `rocprofv3` diagnostic profiler and the lightweight ROCTx marker library linked only by `gufo-kernel-bench`
- **SPDX License Identifier**: `MIT`
- **Copyright / Notice Source**: Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
- **Relationship**: Benchmark marker library / profiling tool; the inference server does not link the profiler SDK
- **Corresponding-Source Location**: https://github.com/ROCm/rocprofiler-sdk (via Nix derivation `rocmPackages.rocprofiler-sdk`)

### 1.9 ICU

- **Component Name**: ICU (International Components for Unicode)
- **Upstream URL**: https://github.com/unicode-org/icu
- **Pinned Revision**: Nixpkgs `nixos-unstable` lock revision
  `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` (`icu`)
- **Component Used**: Unicode NFC normalization and Unicode general-category /
  whitespace classification for the model-private MiniMax H3 tokenizer
- **SPDX License Identifier**: `Unicode-3.0`
- **Relationship**: Linked dynamic runtime dependency
- **Corresponding-Source Location**: https://github.com/unicode-org/icu
  (via Nix derivation `icu`)

### 1.10 FFmpeg

- **Component Name**: FFmpeg
- **Upstream URL**: https://github.com/FFmpeg/FFmpeg
- **Pinned Version**: `8.1.2` from Nixpkgs lock revision
  `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a`
- **Component Used**: Headless `ffmpeg` and `ffprobe` executables for
  concurrent RGB24/F32 PCM input, H.264/AAC MP4 encoding, and output metadata
  validation
- **SPDX License Identifiers**: `LGPL-2.1-or-later` and
  `GPL-2.0-or-later`; the pinned Nix derivation also declares
  `LGPL-3.0-or-later` and `GPL-3.0-or-later` for enabled optional components
- **Relationship**: Spawned as a separate runtime process through pipes.
  Gufo does not link FFmpeg libraries or copy FFmpeg source into the engine.
- **Corresponding-Source Location**: https://github.com/FFmpeg/FFmpeg
  (via Nix derivation `ffmpeg-headless`)

The Nix closure retains FFmpeg's complete license and corresponding-source
metadata. H.264/AAC availability and any patent obligations are deployment
considerations separate from the MiniMax model license and the engine's MIT
license.

### 1.11 llama.cpp

- **Component Name**: llama.cpp
- **Upstream URL**: https://github.com/ggml-org/llama.cpp
- **Pinned Revision**: Commit `e9fa0781f1c25fc4fe8c86be1edc6970661ad6f0`
- **Component Used**: Causal tiled-attention scheduling and online-softmax algorithm adapted into the native Qwen3.8 `gfx1151` HIP kernel
- **SPDX License Identifier**: `MIT`
- **Copyright / Notice Source**: Copyright (c) 2023-2026 The ggml authors
- **Relationship**: Source-derived algorithm; no llama.cpp runtime code or library is linked
- **Corresponding-Source Location**: https://github.com/ggml-org/llama.cpp/tree/e9fa0781f1c25fc4fe8c86be1edc6970661ad6f0/ggml/src/ggml-cuda

### 1.12 DS4

- **Component Name**: DS4
- **Upstream URL**: https://github.com/antirez/ds4
- **Pinned Revision**: Commit `84cc882352757baf628a1776badf7cc54d584e28`
- **Component Used**: DeepSeek V4 Flash GGUF loader, tokenizer, request-session
  graph, and ROCm numerical kernels
- **SPDX License Identifier**: `MIT`
- **Copyright / Notice Source**: Copyright (c) 2026 Salvatore Sanfilippo
  and DS4 contributors; the upstream license is retained under
  `src/models/deepseek_v4_flash/vendor/antirez/LICENSE`.
- **Relationship**: Vendored and adapted into a model-private ROCm backend.
  Gufo does not import the upstream command-line interface, HTTP server,
  agent, evaluator, or disk-cache frontend.
- **Corresponding-Source Location**:
  https://github.com/antirez/ds4/tree/84cc882352757baf628a1776badf7cc54d584e28

### 1.13 libuuid (util-linux)

- **Component Name**: libuuid (util-linux UUID library)
- **Upstream URL**: https://git.kernel.org/pub/scm/utils/util-linux/util-linux.git
- **Pinned Revision**: Nixpkgs `nixos-unstable` lock revision `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` (`libuuid`)
- **Component Used**: Dynamic library for Universally Unique Identifier (UUID) generation
- **SPDX License Identifier**: `BSD-3-Clause` / `LGPL-2.1-or-later`
- **Copyright / Notice Source**: Copyright (C) 1996, 1997, 1998 Theodore Ts'o.
- **Relationship**: Linked (Dynamic runtime library dependency required by XRT and gufo)
- **Corresponding-Source Location**: https://git.kernel.org/pub/scm/utils/util-linux/util-linux.git

### 1.14 MLIR-AIE / IRON

- **Pinned Version**: `1.4.1`
- **Component Used**: Ahead-of-time NPU2 program and DMA generation
- **SPDX License Identifier**: `Apache-2.0 WITH LLVM-exception`
- **Relationship**: Build-only toolchain; generated reviewed artifacts are packaged
- **Corresponding-Source Location**: https://github.com/Xilinx/mlir-aie/tree/v1.4.1

### 1.15 LLVM-AIE / Peano

- **Pinned Version**: `21.0.0.2026080301+c9c5ecb7`
- **Component Used**: AIE2P core compiler distributed as a pinned release wheel
- **SPDX License Identifier**: `Apache-2.0 WITH LLVM-exception`
- **Relationship**: Build-only toolchain
- **Corresponding-Source Location**: https://github.com/Xilinx/llvm-aie

### 1.16 AIEBU

- **Pinned Revision**: `27a302c5840773e79c79f0f2fc8a1832d6ab1774`
- **Component Used**: `aiebu-asm` control-code ELF assembler
- **SPDX License Identifier**: `MIT`
- **Copyright / Notice Source**: Copyright (C) 2022 Xilinx, Inc.; 2022-2024 Advanced Micro Devices, Inc.
- **Relationship**: Build-only toolchain
- **Corresponding-Source Location**: https://github.com/Xilinx/aiebu/tree/27a302c5840773e79c79f0f2fc8a1832d6ab1774

### 1.17 h3.c

- **Component Name**: h3.c
- **Upstream URL**: https://github.com/antirez/h3.c
- **Pinned Revision**: `8974cc055ea9c02fcd14cc27dfda3e1027c05153`
- **Component Used**: Native MiniMax H3 architecture, tensor naming, packed
  multimodal layout, scheduler, sampler, VAE, tokenizer, fixtures, and
  performance reference for the model-private ROCm/HIP port
- **SPDX License Identifier**: `MIT`
- **Copyright / Notice Source**: Copyright (c) 2026 Salvatore Sanfilippo
- **Relationship**: Pinned implementation reference. Selected compatible code
  may be copied or adapted into `src/models/minimax_h3/` with provenance and
  modifications recorded; Metal and Objective-C execution are not imported.
- **Corresponding-Source Location**:
  https://github.com/antirez/h3.c/tree/8974cc055ea9c02fcd14cc27dfda3e1027c05153

The upstream `THIRD_PARTY_NOTICES.md` identifies the rectangular Morton
decoder and dynamic INT8/TensorOps scheduling design in `h3_shaders.metal` as
adapted from ccv's `NAMatMulKernel` and `NAInt8MatMulKernel`, licensed
BSD-3-Clause with copyright (c) 2010, Liu Liu. Any adapted expression or design
retains that notice. See
[`src/models/minimax_h3/UPSTREAM.md`](src/models/minimax_h3/UPSTREAM.md).

### 1.18 curl / libcurl

- **Component Name**: curl / libcurl
- **Upstream URL**: https://github.com/curl/curl
- **Pinned Revision**: Version `8.21.0` from Nixpkgs lock revision
  `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a`
- **Component Used**: HTTPS-capable client library used only by `gufo eval`
  for OpenAI-compatible model discovery and chat-completion requests
- **SPDX License Identifier**: `curl`
- **Relationship**: Linked dynamic runtime dependency
- **Corresponding-Source Location**: https://github.com/curl/curl

---

## 2. System Boundary Dependencies (Non-Distributed)

The following components are required from the host environment or system kernel. They are **not** bundled, copied, or distributed by gufo.

### 2.1 Upstream Linux `amdxdna` Kernel Driver

- **Component Name**: Upstream Linux `amdxdna` Kernel Driver (`amdxdna.ko`)
- **Upstream URL**: https://github.com/amd/xdna-driver (kernel driver subdirectory) / Linux kernel mainline
- **Pinned Revision**: Host Linux kernel driver (`amdxdna`)
- **Component Used**: System device driver providing character device node `/dev/accel/accel*`
- **SPDX License Identifier**: `GPL-2.0-only`
- **Copyright / Notice Source**: Copyright (C) Advanced Micro Devices, Inc.
- **Relationship**: System (Non-distributed system kernel driver. gufo runs as an independent userspace process communicating via the standard kernel UAPI boundary without copying kernel implementation code).
- **Corresponding-Source Location**: Host OS Linux kernel package / distribution kernel source tree

### 2.2 `amdxdna` Kernel UAPI Headers

- **Component Name**: `amdxdna` Kernel User-Space API Headers (`amdxdna_accel.h`)
- **Upstream URL**: https://github.com/amd/xdna-driver
- **Pinned Revision**: Commit `4e5aed38f3b74a5a9a2c7a6222eaff1a8be54305` (`src/driver/amdxdna/amdxdna_accel.h`)
- **Component Used**: Header file defining ioctl numbers and kernel-userspace interface structs
- **SPDX License Identifier**: `GPL-2.0 WITH Linux-syscall-note`
- **Copyright / Notice Source**: Copyright (C) 2023-2025 Advanced Micro Devices, Inc.
- **Relationship**: System / Header (The Linux-syscall-note explicitly permits non-GPL userspace applications to include these UAPI header definitions without triggering GPL copyleft on the userspace application).
- **Corresponding-Source Location**: https://github.com/amd/xdna-driver/tree/4e5aed38f3b74a5a9a2c7a6222eaff1a8be54305

### 2.3 AMD NPU Firmware Binaries

- **Component Name**: AMD XDNA / NPU Firmware Artifacts
- **Upstream URL**: https://github.com/amd/xdna-driver / Host OS distribution firmware package (`linux-firmware`)
- **Pinned Revision**: Host system firmware files (e.g. `/lib/firmware/amdgpu/` or `/lib/firmware/amd/`)
- **Component Used**: Binary firmware blobs loaded into hardware NPU tiles by the kernel driver
- **SPDX License Identifier**: Proprietary Redistribution License (`LICENSE.amdnpu`)
- **Copyright / Notice Source**: Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All Rights Reserved.
- **Relationship**: System (Host system dependency. gufo does **not** bundle or redistribute NPU firmware binaries; firmware must be provided by the host Linux distribution).
- **Corresponding-Source Location**: Host OS `linux-firmware` package / AMD hardware driver packages

---

## 3. External Model Artifacts (Non-Distributed)

Model checkpoints are not part of the gufo source or binary
distribution. Operators obtain them directly from their publisher and remain
responsible for the terms governing their location and use.

### 3.1 MiniMax H3 FL2VA

- **Component Name**: MiniMax H3 Base FL2VA checkpoint
- **Upstream URL**: https://huggingface.co/MiniMaxAI/MiniMax-H3
- **Pinned Revision**: `42ed227ee7df40d41602854ae760620d6eb651fe`
- **Component Used**: Locally supplied BF16 text encoder, Omni Transformer,
  VisualVAE, AudioVAE, tokenizer, scheduler configuration, and metadata under
  `FL2VA/`
- **License**: MiniMax H3 Community License Agreement dated August 2, 2026, or
  separate operator-specific authorization where required
- **License File SHA-256 at Pinned Revision**:
  `59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44`
- **Copyright / Notice Source**: Copyright © 2026 MiniMax. All Rights Reserved.
- **Relationship**: External, access-controlled, operator-supplied runtime
  artifact. It is never committed, packaged, mirrored, automatically
  downloaded, or redistributed by gufo.
- **Operational Boundary**: Every operator must independently obtain access
  from MiniMax, accept or obtain the terms applicable to that operator, and
  configure a local checkpoint path. Possession of the gufo source
  does not grant model rights.
- **Serving Boundary**: The engine supplies numerical execution only. Anyone
  exposing H3 through an API is responsible for the publisher's user terms,
  acceptable-use, safeguards, disclosures, reporting, attribution, and
  territorial requirements.

The project records an operator attestation that the dedicated development
machine is authorized for this work. The repository does not contain private
license correspondence or credentials and does not independently make a legal
determination about a downstream operator.

### 3.2 Qwen3-VL-32B Encoder Component

- **Component Name**: Qwen3-VL-32B encoder weights used by MiniMax H3
- **Upstream URL**: https://github.com/QwenLM/Qwen3-VL
- **Pinned Revision**: Supplied as part of the pinned MiniMax H3 FL2VA package
- **Component Used**: H3 prompt encoder through layer 50
- **SPDX License Identifier**: `Apache-2.0`
- **Relationship**: External model component within the operator-supplied H3
  checkpoint; not distributed by gufo

See [docs/MINIMAX_H3.md](docs/MINIMAX_H3.md) for the acquisition, release, and
runtime boundary.

---

## 4. Evaluation-Only Quality Toolchain (Non-Shipped)

The following packages are present only in the pinned Nix development shell.
They are not linked into, copied into, or distributed with the production
`gufo` package.

### 4.1 PyTorch ROCm

- **Component Name**: PyTorch with ROCm support
- **Upstream URL**: https://github.com/pytorch/pytorch
- **Pinned Version**: `2.12.0` from Nixpkgs lock revision
  `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a`
- **Component Used**: Independent MiniMax H3 teacher execution and tensor
  inspection on the supported ROCm host
- **SPDX License Identifier**: `BSD-3-Clause`
- **Relationship**: Evaluation and offline tooling only; absent from the
  production package closure

### 4.2 Torchvision

- **Component Name**: Torchvision
- **Upstream URL**: https://github.com/pytorch/vision
- **Pinned Version**: `0.27.0` from Nixpkgs lock revision
  `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a`
- **Component Used**: AlexNet feature network used by the LPIPS evaluator
- **SPDX License Identifier**: `BSD-3-Clause`
- **Relationship**: Evaluation-only direct dependency of the pinned Python
  toolchain; absent from the production package closure

### 4.3 LPIPS

- **Component Name**: Learned Perceptual Image Patch Similarity (LPIPS)
- **Upstream URL**: https://github.com/richzhang/PerceptualSimilarity
- **Pinned Version**: `0.1.4` from Nixpkgs lock revision
  `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a`
- **Component Used**: AlexNet-based perceptual comparison of delivered
  MiniMax H3 video frames
- **SPDX License Identifier**: `BSD-2-Clause`
- **Relationship**: Evaluation-only tool; absent from the production package
  closure
- **Model Boundary**: Feature-network parameters are not committed or
  redistributed by gufo. Each promoted quality report records a
  canonical SHA-256 of the loaded module state and the exact
  LPIPS/PyTorch/Torchvision versions.

### 4.4 Torchvision AlexNet evaluation weights

- **Component Name**: Torchvision AlexNet ImageNet weights
- **Upstream URL**:
  https://download.pytorch.org/models/alexnet-owt-7be5be79.pth
- **Pinned Digest**:
  `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02`
- **Component Used**: Frozen AlexNet feature trunk for LPIPS evaluation
- **SPDX License Identifier**: `NOASSERTION`
- **Relationship**: Nix-fetched evaluation model data only; absent from the
  production package closure and not committed to the repository

The evaluator verifies this digest and the bundled LPIPS v0.1 AlexNet
calibration digest
`df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0`
before constructing the metric. It fails closed when either file is absent or
changed and therefore does not rely on an ambient Torch hub cache or a runtime
download.
