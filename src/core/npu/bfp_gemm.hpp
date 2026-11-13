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
//
// Run is thread-safe: up to four threads stage their products at once, so one
// thread's host work (packing, unpacking, its own CPU operators) overlaps the
// other's NPU work.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "src/core/npu/npu.hpp"

namespace gufo::npu {

class BfpGemm {
public:
  /// One product: y [rows][n] = x [rows][k] times matrix `id` transposed.
  /// Consumes one output tile instead of writing y: rows [r0, r0 + rows)
  /// of the call, outputs [n0, n0 + w), as bf16 rows of 128 (tile[i * 128 +
  /// j] is output n0 + j of row r0 + i). Runs on pool threads, concurrently
  /// for distinct tiles.
  using Epilogue = std::function<void(int r0, int rows, int n0, int w,
                                      const std::uint16_t* tile)>;

  /// One product: y [rows][n] = x [rows][k] times matrix `id` transposed.
  /// With x16 set, the rows come as bf16 (x is unused); with epilogue set,
  /// it receives the output tiles (y is unused).
  struct Call {
    int id;
    const float* x;
    int rows;
    float* y;
    const std::uint16_t* x16 = nullptr;
    const Epilogue* epilogue = nullptr;
  };

  /// Loads `dir`/bfpgemm.xclbin and `dir`/bfpgemm_dispatch.so. The weight
  /// arena holds `arena_bytes` of packed matrices, at most 2 GiB (the
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
  /// As Upload, from bf16 bit patterns.
  int UploadBf16(int n, int k, std::span<const std::uint16_t> weights);

  /// A region of the arena for per-request operands (attention keys and
  /// values), reserved once. Uploads into it are valid until Reset; ids of
  /// different scratches never collide. One thread uses a scratch at a time.
  class Scratch {
  public:
    Scratch() = default;
    /// Element (o, i) at weights[o * row_stride + i * col_stride]: a
    /// transposed or strided view. Returns an id for Call. Throws Error when
    /// the region is full.
    int Upload(int n, int k, const float* weights, std::size_t row_stride,
               std::size_t col_stride);
    /// One matrix of UploadMany, as for Upload.
    struct Spec {
      int n, k;
      const float* weights;
      std::size_t row_stride, col_stride;
      /// When >= 0: a copy of specs[same_as] (earlier in the list), not
      /// encoded again.
      int same_as = -1;
    };
    /// Uploads several matrices in one pass, consecutively; returns their ids.
    std::vector<int> UploadMany(std::span<const Spec> specs);
    /// A copy of matrix `id` (of any table) in this region, without encoding
    /// it again: for calls that must use distinct, evenly spaced matrices.
    int Duplicate(int id);
    /// Drops every matrix uploaded since the last Reset.
    void Reset();

  private:
    friend class BfpGemm;
    BfpGemm* owner_{};
    int index_{-1};
  };
  [[nodiscard]] Scratch NewScratch(std::size_t bytes);

  /// Runs independent products in as few dispatches as the staging buffers
  /// allow.
  void Run(std::span<const Call> calls);
  void Run(const Call& call) { Run(std::span<const Call>(&call, 1)); }
  void Run(int id, const float* x, int rows, float* y) {
    const Call call{id, x, rows, y};
    Run(std::span<const Call>(&call, 1));
  }

  /// With GUFO_NPU_PROFILE set: the NPU's busy time (union of in-flight
  /// dispatches) since the last call, in ms; 0 otherwise.
  double TakeBusyMs();

  /// bfp16ebs8 encoding of 8 values: [exponent byte][8 int8 mantissas], value
  /// = mantissa * 2^(exponent - 127) / 64, rounded to nearest. For tests.
  static void EncodeBlock(const float* values, std::uint8_t* out);

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::npu
