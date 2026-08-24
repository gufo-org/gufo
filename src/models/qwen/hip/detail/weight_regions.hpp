#ifndef STRIX_MODELS_QWEN_HIP_DETAIL_WEIGHT_REGIONS_HPP_
#define STRIX_MODELS_QWEN_HIP_DETAIL_WEIGHT_REGIONS_HPP_

#include "src/models/qwen/hip/executor.hpp"

#if defined(ENGINE_ENABLE_HIP)
namespace strix::hip::detail {

void ReleaseWeightRegions(std::vector<QwenGpuWeightRegion>& regions) noexcept;

}  // namespace strix::hip::detail
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // STRIX_MODELS_QWEN_HIP_DETAIL_WEIGHT_REGIONS_HPP_
