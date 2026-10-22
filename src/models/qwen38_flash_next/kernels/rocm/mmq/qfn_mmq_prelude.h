// Every system header the vendored tier includes, pulled in at global scope
// before the tier is wrapped in namespace qfn_mmq. The tier's own includes
// then hit include guards, so its definitions land in the namespace while
// the runtime headers stay global. The wrap keeps this copy from clashing
// with the identical tier linked for DeepSeek V4 Flash.
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stddef.h>
#include <stdint.h>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>
