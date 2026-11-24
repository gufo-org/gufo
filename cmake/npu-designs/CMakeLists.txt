# The in-tree NPU designs alone, without the rest of Gufo:
# .devops/nix/npu-designs.nix builds this as its own derivation, so CI caches
# the compiled designs until one of them changes, and the Gufo package
# installs them with GUFO_NPU_PREBUILT_DESIGNS.
cmake_minimum_required(VERSION 3.25)
project(gufo_npu_designs NONE)

set(GUFO_NPU_DESIGNS_REQUIRED ON CACHE BOOL "" FORCE)
include(${CMAKE_CURRENT_SOURCE_DIR}/../GufoNpuDesigns.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../GufoNpuDesignList.cmake)
