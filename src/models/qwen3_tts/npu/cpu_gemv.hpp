#ifndef GUFO_MODELS_QWEN3_TTS_NPU_CPU_GEMV_HPP_
#define GUFO_MODELS_QWEN3_TTS_NPU_CPU_GEMV_HPP_

// Q8_0 matrix-vector products on the CPU for the talker's and the code
// predictor's decode steps. With one or two rows per product, a Q8Gemm
// dispatch on the NPU costs about 90 us before it streams weights at about
// 36 GB/s, while the CPU threads stream Q8_0 weights at 50-85 GB/s (the
// predictor's partly from L3), so decode products run here and the NPU takes
// the many-row work (prompt prefill, text projection, codec decoding).

#include <immintrin.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "gufo/ext/experimental/npu.hpp"

namespace gufo::models::qwen3_tts::npu {

/// Q8_0 rows on the CPU: codes [n][k], one scale per 32 codes.
struct CpuQ8 {
  std::size_t n{}, k{};
  std::vector<std::int8_t> codes;
  std::vector<float> scales;

  /// y[r] = sum_k w[r][k] x[k] for rows [begin, end).
  void Rows(const float* x, float* y, std::size_t begin,
            std::size_t end) const {
    const std::size_t blocks = k / 32;
    for (std::size_t r = begin; r < end; ++r) {
      const std::int8_t* w = codes.data() + r * k;
      const float* s = scales.data() + r * blocks;
#if defined(__AVX2__) && defined(__FMA__)
      __m256 total = _mm256_setzero_ps();
      for (std::size_t b = 0; b < blocks; ++b) {
        __m256 acc = _mm256_setzero_ps();
        for (int q = 0; q < 4; ++q) {
          const __m256i v = _mm256_cvtepi8_epi32(_mm_loadl_epi64(
              reinterpret_cast<const __m128i*>(w + b * 32 + q * 8)));
          acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(v),
                                _mm256_loadu_ps(x + b * 32 + q * 8), acc);
        }
        total = _mm256_fmadd_ps(acc, _mm256_set1_ps(s[b]), total);
      }
      __m128 h = _mm_add_ps(_mm256_castps256_ps128(total),
                            _mm256_extractf128_ps(total, 1));
      h = _mm_add_ps(h, _mm_movehl_ps(h, h));
      h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
      y[r] = _mm_cvtss_f32(h);
#else
      float total = 0;
      for (std::size_t b = 0; b < blocks; ++b) {
        float acc = 0;
        for (int i = 0; i < 32; ++i)
          acc += float(w[b * 32 + i]) * x[b * 32 + i];
        total += acc * s[b];
      }
      y[r] = total;
#endif
    }
  }
};

/// y [rows][n] = W x for a few rows of x, on the CPU threads.
inline void Gemv(const CpuQ8& m, const float* x, std::size_t rows, float* y) {
  const std::size_t n = m.n, k = m.k;
  gufo::npu::host::ParallelFor(rows * n, 64, [&](std::size_t b, std::size_t e) {
    while (b < e) {
      const std::size_t r = b / n, o = b % n;
      const std::size_t stop = std::min(e - b, n - o);
      m.Rows(x + r * k, y + r * n, o, o + stop);
      b += stop;
    }
  });
}

}  // namespace gufo::models::qwen3_tts::npu

#endif  // GUFO_MODELS_QWEN3_TTS_NPU_CPU_GEMV_HPP_
