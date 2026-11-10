#!/usr/bin/env bash
# Configure HRX HEAD (Release, gfx1151) with the gufo flake's ROCm 10.1 toolchain.
# Store paths are the gufo flake ROCm 10.1 toolchain at main b62b1674; resolve yours with
# nix eval on inputs.nixpkgs-rocm10 rocmPackages_10.{rocm-device-libs,rocm-runtime,aqlprofile}.
set -euo pipefail
SP=${HRX_PROBE_DIR:?set HRX_PROBE_DIR to the probe work directory}
T=/nix/store/92bpbb2rmrkszhw83x63x7qzj9jsz324-rocm-toolchain
DL=/nix/store/gyzwfffwdwj8dpkrasiqd5hkr2adx844-rocm-device-libs-24.0.0-rocm/amdgcn/bitcode
RT=/nix/store/ym4x29qyz0pr1r4qd82y2836bdycfl2f-rocm-runtime-10.1
AQL=/nix/store/d8g8g4di4k5wf7r87bl2k8gf0rvcb2q1-aqlprofile-10.1
cmake -S "$SP/hrx-src" -B "$SP/hrx-build2" -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DIREE_ROCM_DEPENDENCY_MODE=pinned \
  -DCMAKE_PREFIX_PATH="$RT;$AQL" \
  -DIREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=llvm-tools \
  -DIREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLVM_TOOLS_DIR="$T/bin" \
  -DIREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_ROCM_DEVICE_LIB_PATH="$DL" \
  -DCMAKE_C_COMPILER="$T/bin/clang" \
  -DCMAKE_CXX_COMPILER="$T/bin/clang++" \
  -DCMAKE_ASM_COMPILER="$T/bin/clang" \
  -DCMAKE_AR="$T/bin/llvm-ar" \
  -DCMAKE_RANLIB="$T/bin/llvm-ranlib" \
  -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_MODULE_LINKER_FLAGS="-fuse-ld=lld" \
  -DIREE_HAL_AMDGPU_TARGETS=gfx1151 \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_ENABLE_WERROR_FLAG=OFF \
  "$@"
