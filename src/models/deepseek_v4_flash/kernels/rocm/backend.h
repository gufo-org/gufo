#pragma once

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hip/hip_fp16.h>
#include <hipcub/hipcub.hpp>
#include <rocwmma/rocwmma-version.hpp>
#include <rocwmma/rocwmma.hpp>

namespace cub = hipcub;

#include "detail/ds4_rocm_device.hip.hpp"

// Precise F32 router math for the established ROCm 7.2 path.
extern "C" __device__ __attribute__((pure))  float __ocml_exp_f32(float);
extern "C" __device__ __attribute__((pure))  float __ocml_log1p_f32(float);
extern "C" __device__ __attribute__((const)) float __ocml_sqrt_f32(float);

static __device__ __forceinline__ float ds4_precise_expf(float x)   { return __ocml_exp_f32(x); }
static __device__ __forceinline__ float ds4_precise_log1pf(float x) { return __ocml_log1p_f32(x); }
static __device__ __forceinline__ float ds4_precise_sqrtf(float x)  { return __ocml_sqrt_f32(x); }
