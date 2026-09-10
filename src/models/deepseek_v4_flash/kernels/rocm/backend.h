#pragma once

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hip/hip_fp16.h>
#include <hipcub/hipcub.hpp>
#include <rocwmma/rocwmma-version.hpp>
#include <rocwmma/rocwmma.hpp>

namespace cub = hipcub;

#include "detail/ds4_rocm_device.hip.hpp"

// Precise transcendentals for the MoE router top-k scores, immune to -fapprox-func.
// These functions are to be used on paths where small error can be translated to
// some macro effect - like expert selection kernels
extern "C" __device__ __attribute__((pure))  float __ocml_exp_f32(float);
extern "C" __device__ __attribute__((pure))  float __ocml_log1p_f32(float);
extern "C" __device__ __attribute__((const)) float __ocml_sqrt_f32(float);

static __device__ __forceinline__ float ds4_precise_expf(float x)   { return __ocml_exp_f32(x); }
static __device__ __forceinline__ float ds4_precise_log1pf(float x) { return __ocml_log1p_f32(x); }
static __device__ __forceinline__ float ds4_precise_sqrtf(float x)  { return __ocml_sqrt_f32(x); }
