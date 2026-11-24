#ifndef GUFO_MODELS_QWEN3_TTS_NPU_NPU_COMMON_HPP_
#define GUFO_MODELS_QWEN3_TTS_NPU_NPU_COMMON_HPP_

// Helpers shared by the Qwen3-TTS NPU runtimes: BF16 rounding as the GPU
// path applies it, Q8_0 quantization of BF16 checkpoint matrices for
// CPU decode products and BfpGemm, and the residual-add RMS norm.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

#include "gufo/ext/experimental/npu.hpp"
#include "src/models/qwen3_tts/npu/cpu_gemv.hpp"
#include "src/models/qwen3_tts/tensor.hpp"

namespace gufo::models::qwen3_tts::npu {

/// Rounds to the nearest BF16 value (ties to even).
inline float Bf16(float x) {
  std::uint32_t u;
  std::memcpy(&u, &x, 4);
  if ((u & 0x7f800000u) != 0x7f800000u) {
    u += 0x7fffu + ((u >> 16) & 1u);
    u &= 0xffff0000u;
  } else {
    u &= 0xffff0000u;
  }
  float r;
  std::memcpy(&r, &u, 4);
  return r;
}

/// A BF16 or F32 tensor as floats.
inline std::vector<float> FloatVector(const Tensor& t) {
  const std::size_t n = *t.NumElements();
  std::vector<float> out(n);
  if (t.dtype == DType::kBF16)
    gufo::npu::host::Bf16ToFloat(reinterpret_cast<const std::uint16_t*>(t.data),
                                 out.data(), n);
  else if (t.dtype == DType::kF32)
    std::memcpy(out.data(), t.data, n * sizeof(float));
  else
    throw std::runtime_error("Qwen3-TTS NPU: unsupported dtype of " + t.name);
  return out;
}
inline std::vector<float> Bf16Vector(const Tensor& t) {
  return FloatVector(t);
}

/// The row-wise concatenation of `parts` ([n_i][k] each, BF16 or F32).
inline std::vector<float> ConcatRows(std::span<const Tensor* const> parts,
                                     std::size_t* n, std::size_t* k) {
  *n = 0;
  *k = 0;
  std::vector<float> rows;
  for (const Tensor* t : parts) {
    if (t->shape.size() != 2)
      throw std::runtime_error("Qwen3-TTS NPU: matrix expected: " + t->name);
    if (*k != 0 && *k != t->shape[1])
      throw std::runtime_error("Qwen3-TTS NPU: mismatched matrix widths");
    *n += t->shape[0];
    *k = t->shape[1];
    const std::vector<float> w = FloatVector(*t);
    rows.insert(rows.end(), w.begin(), w.end());
  }
  return rows;
}

/// Q8_0 quantization (per 32 values: scale = max |x| / 127, codes rounded to
/// nearest) of rows [n][k].
inline CpuQ8 QuantizeQ8(const std::vector<float>& w, std::size_t n,
                        std::size_t k) {
  if (k % 32 != 0)
    throw std::runtime_error("Qwen3-TTS NPU: width not a multiple of 32");
  CpuQ8 q;
  q.n = n;
  q.k = k;
  q.codes.resize(n * k);
  q.scales.resize(n * (k / 32));
  gufo::npu::host::ParallelFor(n, 64, [&](std::size_t b, std::size_t e) {
    for (std::size_t r = b; r < e; ++r)
      for (std::size_t blk = 0; blk < k / 32; ++blk) {
        const float* x = w.data() + r * k + blk * 32;
        float top = 0;
        for (int i = 0; i < 32; ++i)
          top = std::max(top, std::fabs(x[i]));
        const float scale = top / 127.0F;
        const float inv = scale > 0 ? 1.0F / scale : 0.0F;
        std::int8_t* c = q.codes.data() + r * k + blk * 32;
        for (int i = 0; i < 32; ++i)
          c[i] = static_cast<std::int8_t>(std::lround(x[i] * inv));
        q.scales[r * (k / 32) + blk] = scale;
      }
  });
  return q;
}

/// The Q8_0 rows of the concatenation of `parts`, for decode products.
inline CpuQ8 MakeCpuQ8(std::span<const Tensor* const> parts) {
  std::size_t n = 0, k = 0;
  const std::vector<float> w = ConcatRows(parts, &n, &k);
  return QuantizeQ8(w, n, k);
}

/// Uploads the concatenation of `parts` for many-row products.
inline int UploadBfp(gufo::npu::BfpGemm& gemm,
                     std::span<const Tensor* const> parts) {
  std::size_t n = 0, k = 0;
  const std::vector<float> w = ConcatRows(parts, &n, &k);
  return gemm.Upload(int(n), int(k), w);
}

/// x = bf16(bf16(x) + bf16(update)) (update may be null), then
/// norm = bf16(bf16(x / rms(x)) * weight), per row: the GPU path's
/// Bfloat16ResidualAddRMSNormKernel.
inline void ResidualNorm(float* x, const float* update, float* norm,
                         std::size_t rows, std::size_t dim, const float* weight,
                         float eps) {
  for (std::size_t r = 0; r < rows; ++r) {
    float* row = x + r * dim;
    float sum = 0;
    for (std::size_t i = 0; i < dim; ++i) {
      const float left = Bf16(row[i]);
      const float v =
          update == nullptr ? left : Bf16(left + Bf16(update[r * dim + i]));
      row[i] = v;
      sum += v * v;
    }
    const float inv = 1.0F / std::sqrt(sum / float(dim) + eps);
    for (std::size_t i = 0; i < dim; ++i)
      norm[r * dim + i] = Bf16(Bf16(row[i] * inv) * weight[i]);
  }
}

}  // namespace gufo::models::qwen3_tts::npu

#endif  // GUFO_MODELS_QWEN3_TTS_NPU_NPU_COMMON_HPP_
