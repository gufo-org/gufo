#ifndef GUFO_MODELS_QWEN_HIP_OPS_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_HPP_

// Compatibility umbrella. New call sites should include the narrow operation
// family they use.
#include "src/families/qwen/kernels/ops/attention.hpp"
#include "src/families/qwen/kernels/ops/gemm.hpp"
#include "src/families/qwen/kernels/ops/norm_residual.hpp"
#include "src/families/qwen/kernels/ops/ssm.hpp"
#include "src/families/qwen/kernels/ops/swiglu.hpp"
#include "src/families/qwen/kernels/ops/token.hpp"

#endif  // GUFO_MODELS_QWEN_HIP_OPS_HPP_
