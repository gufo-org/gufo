#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

// The model's MoE combine geometry: top-10 slots over a 2,560-wide hidden
// state, router row stride num_experts + 1.
constexpr std::uint32_t kTokens = 37;
constexpr std::uint32_t kSlots = 10;
constexpr std::uint32_t kHidden = 2560;
constexpr std::uint32_t kGateStride = 513;

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

template<typename T>
class HipBuffer {
public:
  explicit HipBuffer(std::size_t count) : count_(count) {
    void* allocation = nullptr;
    CheckHip(hipMalloc(&allocation, bytes()), "hipMalloc");
    data_ = static_cast<T*>(allocation);
  }
  ~HipBuffer() {
    if (data_ != nullptr) {
      (void)hipFree(data_);
    }
  }

  HipBuffer(const HipBuffer&) = delete;
  HipBuffer& operator=(const HipBuffer&) = delete;
  HipBuffer(HipBuffer&&) = delete;
  HipBuffer& operator=(HipBuffer&&) = delete;

  [[nodiscard]] T* get() noexcept { return data_; }
  [[nodiscard]] std::size_t bytes() const noexcept {
    return count_ * sizeof(T);
  }

private:
  T* data_{nullptr};
  std::size_t count_{0};
};

std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

std::vector<float> MakeValues(std::size_t count, std::uint32_t seed,
                              float scale, float offset = 0.0F) {
  std::vector<float> values(count);
  for (float& value : values) {
    value = offset +
            scale *
                static_cast<float>(
                    static_cast<int>(NextRandom(&seed) & 0xFFFFU) - 32768) /
                32768.0F;
  }
  return values;
}

void Upload(HipBuffer<float>* destination, const std::vector<float>& source) {
  CheckHip(hipMemcpy(destination->get(), source.data(), destination->bytes(),
                     hipMemcpyHostToDevice),
           "upload");
}

template<typename T>
std::vector<T> Download(HipBuffer<T>* source, std::size_t count) {
  std::vector<T> values(count);
  CheckHip(hipMemcpy(values.data(), source->get(), source->bytes(),
                     hipMemcpyDeviceToHost),
           "download");
  return values;
}

// Independent host evaluation of the inject contract: one rounded product,
// the remaining stream/component FMAs, then descending wave sums and the
// ordered eight-wave total. The ordinary mixed epilogue reassociates its
// first products under fast-math and is not an exact F32 reference.
std::vector<float> InjectReference(const std::vector<__half>& norm,
                                   const std::vector<float>& weights,
                                   std::uint32_t parts) {
  constexpr std::size_t kThreads = 256;
  constexpr std::size_t kStreams = 4;
  constexpr std::size_t kDim = kStreams * kHidden;
  std::vector<float> reference(kTokens * kStreams * parts);
  for (std::size_t t = 0; t < kTokens; ++t) {
    for (std::size_t o = 0; o < kStreams; ++o) {
      for (std::size_t part = 0; part < parts; ++part) {
        std::array<float, kThreads> dots{};
        for (std::size_t lane = 0; lane < kThreads; ++lane) {
          const std::size_t i = (part * kThreads + lane) * 4;
          if (i >= kHidden)
            continue;
          const std::size_t first = o * kDim + i;
          volatile float first_product =
              weights[first] * __half2float(norm[t * kDim + i]);
          float dot = first_product;
          for (std::size_t s = 0; s < kStreams; ++s) {
            for (std::size_t c = s == 0 ? 1 : 0; c < 4; ++c) {
              dot = std::fma(weights[o * kDim + s * kHidden + i + c],
                             __half2float(norm[t * kDim + s * kHidden + i + c]),
                             dot);
            }
          }
          dots[lane] = dot;
        }
        for (std::size_t off = 16; off > 0; off /= 2) {
          const auto previous = dots;
          for (std::size_t lane = 0; lane < kThreads; ++lane) {
            volatile float sum = previous[lane] + previous[lane ^ off];
            dots[lane] = sum;
          }
        }
        float total = 0.0F;
        for (std::size_t wave = 0; wave < kThreads / 32; ++wave) {
          volatile float sum = total + dots[wave * 32];
          total = sum;
        }
        reference[(t * kStreams + o) * parts + part] = total;
      }
    }
  }
  return reference;
}

}  // namespace

int main() {
  try {
    constexpr std::size_t kExpert =
        static_cast<std::size_t>(kTokens) * kSlots * kHidden;
    constexpr std::size_t kOut = static_cast<std::size_t>(kTokens) * kHidden;
    const auto expert_out = MakeValues(kExpert, 0x1234ABCDU, 1.0F);
    const auto weights = MakeValues(static_cast<std::size_t>(kTokens) * kSlots,
                                    0xBADC0FFEU, 0.5F, 0.5F);
    const auto shared = MakeValues(kOut, 0xDEADBEEFU, 1.0F);
    const auto gate = MakeValues(
        static_cast<std::size_t>(kTokens) * kGateStride, 0xC0FFEE11U, 3.0F);
    HipBuffer<float> d_expert(kExpert);
    HipBuffer<float> d_weights(weights.size());
    HipBuffer<float> d_shared(kOut);
    HipBuffer<float> d_gate(gate.size());
    HipBuffer<float> d_ref(kOut);
    HipBuffer<float> d_vec(kOut);
    Upload(&d_expert, expert_out);
    Upload(&d_weights, weights);
    Upload(&d_shared, shared);
    Upload(&d_gate, gate);
    q::MoeEpilogue(d_expert.get(), d_weights.get(), d_shared.get(),
                   d_gate.get(), kGateStride, d_ref.get(), kTokens, kSlots,
                   kHidden, nullptr);
    q::MoeEpilogueVec4(d_expert.get(), d_weights.get(), d_shared.get(),
                       d_gate.get(), kGateStride, d_vec.get(), kTokens, kSlots,
                       kHidden, nullptr);
    CheckHip(hipDeviceSynchronize(), "MoE epilogue synchronization");
    const auto ref = Download(&d_ref, kOut);
    const auto vec = Download(&d_vec, kOut);
    double worst = 0.0;
    for (std::size_t i = 0; i < kOut; ++i) {
      worst = std::max(worst, std::abs(static_cast<double>(ref[i] - vec[i])));
    }
    std::cout << "MoE epilogue vec4 worst absolute error " << worst << '\n';
    // Omitting an unused norm must preserve the fused residual exactly.
    // This also covers ragged token counts and all ten routed experts.
    constexpr std::size_t kResidual = kOut * 4;
    const auto residual = MakeValues(kResidual, 0x13579U, 1.0F);
    const auto parts = q::HcInjectPartsVec4(kHidden);
    HipBuffer<float> d_res_norm(kResidual);
    HipBuffer<float> d_res_only(kResidual);
    HipBuffer<float> d_res_separate(kResidual);
    HipBuffer<float> d_inject(kTokens * 4 * parts);
    HipBuffer<float> d_gamma(4 * kHidden);
    HipBuffer<__half> d_half(kExpert);
    HipBuffer<__half> d_norm(kResidual);
    HipBuffer<std::uint8_t> d_q8(q::Q8TiledBytes(kTokens, 4 * kHidden));
    Upload(&d_res_norm, residual);
    Upload(&d_res_only, residual);
    Upload(&d_res_separate, residual);
    Upload(&d_inject, MakeValues(kTokens * 4 * parts, 0x24680U, 0.5F));
    Upload(&d_gamma, MakeValues(4 * kHidden, 0x98765U, 0.5F, 1.0F));
    q::NarrowActivations(d_expert.get(), d_half.get(), false, kExpert, nullptr);
    for (bool normalize : {true, false}) {
      if (!q::HcCombineMoeF16(normalize ? d_res_norm.get() : d_res_only.get(),
                              d_half.get(), d_weights.get(), d_shared.get(),
                              d_gate.get(), kGateStride, kSlots, d_inject.get(),
                              parts, normalize ? d_gamma.get() : nullptr,
                              normalize ? d_norm.get() : nullptr,
                              normalize ? d_q8.get() : nullptr, kTokens,
                              kHidden, 4, 1e-6F, nullptr)) {
        throw std::runtime_error("fused MoE residual rejected");
      }
    }
    if (Download(&d_res_norm, kResidual) != Download(&d_res_only, kResidual)) {
      throw std::runtime_error("skipping unused norm changed the residual");
    }
    q::MoeEpilogueVec4F16(d_half.get(), d_weights.get(), d_shared.get(),
                          d_gate.get(), kGateStride, d_vec.get(), kTokens,
                          kSlots, kHidden, nullptr);
    q::HcCombine(d_res_separate.get(), d_vec.get(), d_inject.get(), parts,
                 nullptr, nullptr, kTokens, kHidden, 4, 1e-6F, nullptr);
    if (Download(&d_res_separate, kResidual) !=
        Download(&d_res_only, kResidual)) {
      throw std::runtime_error("fusing the residual-only combine changed it");
    }
    // A deferred update must keep the separate F32 store's rounding before
    // the next combine and preserve the downstream F16/Q8 normalization.
    HipBuffer<float> d_prior(kOut);
    HipBuffer<float> d_prior_w(kTokens * 4);
    HipBuffer<__half> d_norm_separate(kResidual);
    HipBuffer<std::uint8_t> d_q8_separate(d_q8.bytes());
    Upload(&d_prior, MakeValues(kOut, 0xFEED123U, 0.23F));
    Upload(&d_prior_w, MakeValues(kTokens * 4, 0xABC789U, 0.75F, 1.0F));
    HipBuffer<float> d_next_inject(kTokens * 4 * parts);
    HipBuffer<float> d_next_inject_w(kHidden * 4 * 4);
    const auto next_inject_weights =
        MakeValues(kHidden * 4 * 4, 0xDED123U, 0.1F);
    Upload(&d_next_inject_w, next_inject_weights);
    for (unsigned mode = 0; mode < 3; ++mode) {
      const bool normalize = mode != 0;
      const bool inject_next = mode == 2;
      Upload(&d_res_norm, residual);
      Upload(&d_res_separate, residual);
      CheckHip(hipMemset(d_q8.get(), 0, d_q8.bytes()), "clear fused Q8");
      CheckHip(hipMemset(d_q8_separate.get(), 0, d_q8_separate.bytes()),
               "clear reference Q8");
      q::HcApplyDeferred(d_res_separate.get(), d_prior.get(), d_prior_w.get(),
                         kTokens, kHidden, nullptr);
      for (bool deferred : {false, true}) {
        if (!q::HcCombineMoeF16(
                deferred ? d_res_norm.get() : d_res_separate.get(),
                d_half.get(), d_weights.get(), d_shared.get(), d_gate.get(),
                kGateStride, kSlots, d_inject.get(), parts,
                normalize ? d_gamma.get() : nullptr,
                normalize ? (deferred ? d_norm.get() : d_norm_separate.get())
                          : nullptr,
                normalize ? (deferred ? d_q8.get() : d_q8_separate.get())
                          : nullptr,
                kTokens, kHidden, 4, 1e-6F, nullptr,
                deferred ? d_prior.get() : nullptr,
                deferred ? d_prior_w.get() : nullptr,
                deferred && inject_next ? d_next_inject.get() : nullptr,
                deferred && inject_next ? d_next_inject_w.get() : nullptr)) {
          throw std::runtime_error("deferred fused combine rejected");
        }
      }
      const auto fused = Download(&d_res_norm, kResidual);
      const auto separate = Download(&d_res_separate, kResidual);
      if (std::memcmp(fused.data(), separate.data(), d_res_norm.bytes()) != 0) {
        throw std::runtime_error("deferred combine changed residual bytes");
      }
      if (normalize) {
        const auto norm = Download(&d_norm, kResidual);
        const auto reference_norm = Download(&d_norm_separate, kResidual);
        if (std::memcmp(norm.data(), reference_norm.data(), d_norm.bytes()) !=
            0) {
          std::size_t changed = 0;
          for (std::size_t i = 0; i < norm.size(); ++i) {
            if (std::memcmp(&norm[i], &reference_norm[i], sizeof(__half)) !=
                0) {
              if (changed < 8)
                std::cerr << "norm mismatch " << i << ": "
                          << __half2float(norm[i]) << " vs "
                          << __half2float(reference_norm[i]) << '\n';
              ++changed;
            }
          }
          throw std::runtime_error("deferred combine changed " +
                                   std::to_string(changed) +
                                   " F16 norm values");
        }
        if (Download(&d_q8, d_q8.bytes()) !=
            Download(&d_q8_separate, d_q8_separate.bytes())) {
          throw std::runtime_error("deferred combine changed Q8 norm bytes");
        }
        if (inject_next) {
          const auto actual_inject =
              Download(&d_next_inject, kTokens * 4 * parts);
          const auto reference_inject =
              InjectReference(reference_norm, next_inject_weights, parts);
          if (std::memcmp(actual_inject.data(), reference_inject.data(),
                          d_next_inject.bytes()) != 0) {
            throw std::runtime_error("deferred combine changed inject bytes");
          }
        }
      }
    }
    return worst == 0.0 ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
