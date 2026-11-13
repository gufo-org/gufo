#pragma once

// Q8_0 matrix products on the NPU: y = W x for any row count, with the
// weights resident in one arena buffer. The program is the IRON design in
// src/core/npu/kernels/q8gemm (q8gemm.xclbin and its dispatch library).
//
// x is quantized per row to int16. Each 32-code block's integer dot is
// scaled by the block's weight scale in integer arithmetic (int16 fractions
// of the output's largest block scale, accumulated in 64 bits), so results
// match f32 activations to about 5e-5 relative. The weights' Q8_0 codes are
// used as they are.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

#include "src/core/npu/npu.hpp"

namespace gufo::npu {

class Q8Gemm {
public:
  /// One product: y [rows][n] = x [rows][k] times matrix `id` transposed.
  struct Call {
    int id;
    const float* x;
    int rows;
    float* y;
  };

  /// Loads `dir`/q8gemm.xclbin and `dir`/q8gemm_dispatch.so. The weight
  /// arena holds `arena_bytes` of packed matrices.
  static Q8Gemm Open(Device& device, const std::filesystem::path& dir,
                     std::size_t arena_bytes = std::size_t(1) << 30);

  Q8Gemm();
  ~Q8Gemm();
  Q8Gemm(Q8Gemm&&) noexcept;
  Q8Gemm& operator=(Q8Gemm&&) noexcept;

  [[nodiscard]] explicit operator bool() const noexcept {
    return impl_ != nullptr;
  }

  /// A matrix of n outputs x k inputs (k a multiple of 32): Q8_0 codes
  /// [n][k] and block scales [n][k / 32]. Returns its id. Throws Error when
  /// the arena is full.
  int Upload(int n, int k, std::span<const std::int8_t> codes,
             std::span<const float> scales);
  /// Runs independent products in one dispatch (cached per shape).
  void Run(std::span<const Call> calls);
  void Run(int id, const float* x, int rows, float* y) {
    const Call call{id, x, rows, y};
    Run(std::span<const Call>(&call, 1));
  }

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::npu
