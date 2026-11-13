#ifndef GUFO_MODELS_QWEN3_ASR_NPU_SPLIT_Q8_HPP_
#define GUFO_MODELS_QWEN3_ASR_NPU_SPLIT_Q8_HPP_

// Decode-step products y = W x of one row, Q8_0 weights, with the outputs
// split between the NPU (gufo::npu::Q8Gemm) and the CPU, which run at the
// same time: one decode token is memory-bound, and the NPU's DMA (about
// 57 GB/s) and the CPU cores (about 100 GB/s) read DRAM independently.
// Both sides quantize x per row to int16 and scale each 32-code block's
// integer dot by its weight scale, as Q8Gemm does.

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "gufo/ext/experimental/npu.hpp"

namespace gufo::models::qwen3_asr::npu {

/// A thread that runs one posted task at a time, so the caller can work on
/// the CPU while it waits for the NPU.
class Lane {
public:
  Lane();
  ~Lane();
  Lane(const Lane&) = delete;
  Lane& operator=(const Lane&) = delete;
  void Post(std::function<void()> task);
  /// Waits for the posted task; rethrows its exception.
  void Wait();

private:
  void Loop();
  std::mutex mutex_;
  std::condition_variable wake_, done_;
  std::function<void()> task_;
  std::exception_ptr error_;
  bool busy_{false}, stop_{false};
  std::thread thread_;
};

class SplitQ8 {
public:
  /// Quantizes BF16 rows [n][k] to Q8_0; the first `npu_rows` outputs go to
  /// `gemm`'s arena, the rest stay on the host.
  SplitQ8(gufo::npu::Q8Gemm& gemm, const std::uint16_t* bf16, int n, int k,
          int npu_rows);
  /// y [n] = W x [k].
  void Run(Lane& lane, const float* x, float* y) const;
  [[nodiscard]] int n() const { return n_; }

private:
  gufo::npu::Q8Gemm* gemm_;
  int n_, k_, npu_rows_, npu_id_{-1};
  std::vector<std::int8_t> codes_;  // CPU rows [n - npu_rows][k]
  std::vector<float> scales_;       // [n - npu_rows][k / 32]
};

/// The NPU's share of a matrix's rows that finishes both sides together
/// (rounded to 8 rows), from measured per-dispatch and per-byte costs. Used
/// only when the fused decode program cannot hold the context.
int NpuRows(int n, int k);

}  // namespace gufo::models::qwen3_asr::npu

#endif  // GUFO_MODELS_QWEN3_ASR_NPU_SPLIT_Q8_HPP_
