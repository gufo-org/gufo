// Wave64 192x128 dense-tier launchers for GLM-5.3-Flash. Separate from
// w8a8_dense.hip.cpp because the WMMA wave size is a per-TU codegen flag.
#pragma once

#include <cstddef>

#include <hip/hip_runtime.h>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::glm5_next::rocm {

void Glm5W8A8DenseWave64(core::GgmlType type, const void* w,
                         const void* q8_1_x, float* y, std::size_t batch,
                         std::size_t m, std::size_t k, hipStream_t stream);

}  // namespace gufo::models::glm5_next::rocm
