#if defined(ENGINE_ENABLE_HIP)

// Euler arithmetic adapted from antirez/h3.c h3_shaders.metal at
// 8974cc055ea9c02fcd14cc27dfda3e1027c05153 (MIT). This is an independent
// gfx1151 ROCm/HIP kernel that keeps the sampler state in device F32.

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

#include "src/core/hip/arch.hpp"
#include "src/models/minimax_h3/sampling.hpp"

namespace gufo::minimax_h3 {
namespace {

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

bool ValidateTarget(std::string* error) {
  int device = 0;
  hipError_t status = hipGetDevice(&device);
  if (status != hipSuccess) {
    SetError(error, std::string("hipGetDevice: ") + hipGetErrorString(status));
    return false;
  }
  hipDeviceProp_t properties{};
  status = hipGetDeviceProperties(&properties, device);
  if (status != hipSuccess) {
    SetError(error, std::string("hipGetDeviceProperties: ") +
                        hipGetErrorString(status));
    return false;
  }
  if (!gufo::hip::IsRdna35(properties.gcnArchName) ||
      properties.integrated == 0) {
    SetError(error,
             "MiniMax H3 Euler sampling supports only integrated RDNA 3.5 "
             "APUs (gfx1151 Strix Halo, gfx1150 Strix Point); detected " +
                 std::string(properties.gcnArchName));
    return false;
  }
  return true;
}

__global__ void EulerUpdateKernel(float* sample, std::size_t sample_offset,
                                  const float* last, const float* previous,
                                  std::size_t elements,
                                  float sigma_from_timestep, float ratio,
                                  float extrapolation) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const float last_value = last[index];
  const float velocity_delta = last_value - previous[index];
  const float extrapolated_delta = extrapolation * velocity_delta;
  const float velocity = last_value + extrapolated_delta;
  const std::size_t sample_index = sample_offset + index;
  const float sample_value = sample[sample_index];
  // Preserve the eager Diffusers float32 operation boundaries:
  // denoised = sample + sigma_from_timestep * velocity
  // next = ratio * sample + (1 - ratio) * denoised
  const float denoised_delta = sigma_from_timestep * velocity;
  const float denoised = sample_value + denoised_delta;
  const float sample_term = ratio * sample_value;
  const float denoised_weight = 1.0F - ratio;
  const float denoised_term = denoised_weight * denoised;
  sample[sample_index] = sample_term + denoised_term;
}

}  // namespace

bool HipEulerUpdate(void* sample_f32, std::size_t sample_elements,
                    std::size_t sample_offset, const void* last_f32,
                    const void* previous_f32, std::size_t velocity_elements,
                    float sigma_from_timestep, float ratio,
                    float extrapolation, void* stream, std::string* error) {
  if (sample_f32 == nullptr || last_f32 == nullptr ||
      previous_f32 == nullptr || velocity_elements == 0 ||
      sample_offset > sample_elements ||
      velocity_elements > sample_elements - sample_offset ||
      velocity_elements >
          static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()) *
              256ULL ||
      !std::isfinite(sigma_from_timestep) || sigma_from_timestep <= 0.0F ||
      !std::isfinite(ratio) || ratio < 0.0F || ratio >= 1.0F ||
      !std::isfinite(extrapolation) || extrapolation < -2.0F ||
      extrapolation > 2.0F) {
    SetError(error, "invalid MiniMax H3 HIP Euler update");
    return false;
  }
  if (!ValidateTarget(error)) {
    return false;
  }
  constexpr unsigned int kThreads = 256;
  const unsigned int blocks =
      static_cast<unsigned int>((velocity_elements + kThreads - 1) / kThreads);
  hipLaunchKernelGGL(EulerUpdateKernel, dim3(blocks), dim3(kThreads), 0,
                     static_cast<hipStream_t>(stream),
                     static_cast<float*>(sample_f32), sample_offset,
                     static_cast<const float*>(last_f32),
                     static_cast<const float*>(previous_f32),
                     velocity_elements, sigma_from_timestep, ratio,
                     extrapolation);
  const hipError_t status = hipGetLastError();
  if (status != hipSuccess) {
    SetError(error, std::string("MiniMax H3 Euler kernel launch: ") +
                        hipGetErrorString(status));
    return false;
  }
  return true;
}

}  // namespace gufo::minimax_h3

#endif  // ENGINE_ENABLE_HIP
