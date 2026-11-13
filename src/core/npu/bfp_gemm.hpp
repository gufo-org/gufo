#pragma once

// BF16 x BFP16 matrix products on the NPU for many rows (prefill, encoders,
// embeddings): y = W x, with the weights resident in one arena buffer as
// bfp16ebs8 (blocks of 8 values sharing an exponent, 8-bit mantissas). The
// program is the IRON design in src/core/npu/kernels/bfpgemm.
//
// x is rounded to bf16 and converted to bfp16 on the cores; outputs
// accumulate in bf16. Each product is within about 1% relative of f32
// arithmetic. Rows go in bands of 512 and outputs in groups of 1024, so for a
// few rows Q8Gemm is faster.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

#include "src/core/npu/npu.hpp"

namespace gufo::npu {

class BfpGemm {
public:
  /// One product: y [rows][n] = x [rows][k] times matrix `id` transposed.
  struct Call {
    int id;
    const float* x;
    int rows;
    float* y;
  };

  /// Loads `dir`/bfpgemm.xclbin and `dir`/bfpgemm_dispatch.so. The weight
  /// arena holds `arena_bytes` of packed matrices, at most 1.875 GiB (the
  /// design's weight tensor).
  static BfpGemm Open(Device& device, const std::filesystem::path& dir,
                      std::size_t arena_bytes = std::size_t(1) << 30);

  BfpGemm();
  ~BfpGemm();
  BfpGemm(BfpGemm&&) noexcept;
  BfpGemm& operator=(BfpGemm&&) noexcept;

  [[nodiscard]] explicit operator bool() const noexcept {
    return impl_ != nullptr;
  }

  /// A matrix of n outputs x k inputs, row-major [n][k] f32. Returns its id.
  /// Throws Error when the arena is full.
  int Upload(int n, int k, std::span<const float> weights);
  /// As Upload, element (o, i) at weights[o * row_stride + i * col_stride]
  /// (a transposed or strided view, such as attention keys or values).
  int Upload(int n, int k, const float* weights, std::size_t row_stride,
             std::size_t col_stride);
  /// As Upload, from bf16 bit patterns.
  int UploadBf16(int n, int k, std::span<const std::uint16_t> weights);
  /// The arena state; Release(mark) drops the matrices uploaded after it,
  /// for per-request operands (attention keys and values).
  [[nodiscard]] std::size_t Mark() const;
  void Release(std::size_t mark);
  /// Runs independent products in as few dispatches as the staging buffers
  /// allow.
  void Run(std::span<const Call> calls);
  void Run(int id, const float* x, int rows, float* y) {
    const Call call{id, x, rows, y};
    Run(std::span<const Call>(&call, 1));
  }

  /// bfp16ebs8 encoding of 8 values: [exponent byte][8 int8 mantissas], value
  /// = mantissa * 2^(exponent - 127) / 64, rounded to nearest. For tests.
  static void EncodeBlock(const float* values, std::uint8_t* out);

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::npu
