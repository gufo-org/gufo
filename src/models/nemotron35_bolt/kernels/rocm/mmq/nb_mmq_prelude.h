// SPDX-License-Identifier: MIT
// Every system header the vendored tier includes, pulled in at global scope
// before each translation unit wraps the tier in namespace nb_mmq. The tier's
// own includes then hit include guards, so its definitions land in the
// namespace while the runtime headers stay global. The wrap keeps this copy's
// ggml names (ggml_type, mul_mat_q_case, ...) apart from the DeepSeek and Qwen
// copies linked into the same binary.
#pragma once

// The HIP headers come first, as in the DeepSeek copy (ggml-common.h pulls in
// hip_fp16.h before vendors/hip.h): HIP's warp-sync builtins such as
// __syncwarp stay declared, and vendors/hip.h maps the *_sync shuffles onto
// the plain HIP shuffles afterwards.
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
