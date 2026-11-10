#ifndef GUFO_TESTS_MODELS_GLM5_NEXT_HIP_TEST_HPP_
#define GUFO_TESTS_MODELS_GLM5_NEXT_HIP_TEST_HPP_

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// Shared scaffolding for the glm5_next GPU operator checks: a RAII device
// buffer, a deterministic xorshift generator, host/device transfer helpers
// and error reductions. Each check computes its own CPU reference (the
// canonical gufo_core dequantizers plus double-precision dots) and compares
// it against the kernel output.
namespace gufo::tests::glm5_next {

inline void CheckHip(hipError_t error, const char* operation) {
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

inline std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

inline std::vector<float> MakeValues(std::size_t count, std::uint32_t seed,
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

inline std::vector<std::uint8_t> MakeBytes(std::size_t count,
                                           std::uint32_t seed) {
  std::vector<std::uint8_t> bytes(count);
  for (std::uint8_t& byte : bytes) {
    byte = static_cast<std::uint8_t>(NextRandom(&seed) & 0xFFU);
  }
  return bytes;
}

// A finite binary16 payload: sign and mantissa random, exponent restricted to
// [6, 25] so no dequant path ever observes an inf/NaN scale word.
inline std::uint16_t FiniteHalf(std::uint32_t* seed) {
  const std::uint32_t r = NextRandom(seed);
  return static_cast<std::uint16_t>(((r >> 16) & 0x8000U) |
                                    ((6U + (r & 0x13U)) << 10) |
                                    ((r >> 5) & 0x03FFU));
}

inline void PutLittle16(std::uint8_t* p, std::uint16_t v) {
  p[0] = static_cast<std::uint8_t>(v & 0xFFU);
  p[1] = static_cast<std::uint8_t>(v >> 8);
}

// Overwrite every fp16 scale field of one super-block (offsets per the GGML
// block layouts; see ggml_dequant.hpp) with a finite half, leaving the code
// bytes fully random. IQ1_M stores its d in the top nibbles of the four
// scale words at byte 48; clearing bit 14 of the last word keeps the
// reassembled exponent < 0x1F.
inline void PatchScaleFields(std::uint8_t* blk, const std::uint16_t* offsets,
                             std::uint32_t* seed) {
  for (const std::uint16_t* off = offsets; *off != 0xFFFFU; ++off) {
    PutLittle16(blk + *off, FiniteHalf(seed));
  }
}

inline void Upload(HipBuffer<float>* destination,
                   const std::vector<float>& source) {
  CheckHip(hipMemcpy(destination->get(), source.data(), destination->bytes(),
                     hipMemcpyHostToDevice),
           "upload");
}

inline std::vector<float> Download(HipBuffer<float>* source,
                                   std::size_t count) {
  std::vector<float> values(count);
  CheckHip(hipMemcpy(values.data(), source->get(), count * sizeof(float),
                     hipMemcpyDeviceToHost),
           "download");
  return values;
}

// Worst relative error over the elementwise scale floor, so near-zero
// reference values do not inflate the metric.
inline double WorstRelative(const std::vector<float>& reference,
                            const std::vector<float>& candidate,
                            double floor = 1e-3) {
  double worst = 0.0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const double scale =
        std::max(floor, std::abs(static_cast<double>(reference[i])));
    worst = std::max(
        worst,
        std::abs(static_cast<double>(reference[i]) - candidate[i]) / scale);
  }
  return worst;
}

}  // namespace gufo::tests::glm5_next

#endif  // GUFO_TESTS_MODELS_GLM5_NEXT_HIP_TEST_HPP_
