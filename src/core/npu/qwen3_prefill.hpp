#pragma once

// A whole Qwen3 dense prefill (28 layers x 2048, 16 query / 8 KV heads of
// 128, SwiGLU 6144) in one NPU dispatch: the IRON design in
// src/core/npu/kernels/qwen3pre runs every GEMM, residual + RMSNorm, q/k norm
// + RoPE and causal attention on the NPU cores, for up to M rows per program
// (qwen3pre_<M>.xclbin, one per M). The host stages the input rows and reads
// back the K/V cache entries, for handing the sequence over to Qwen3Decoder.
//
// GEMMs run as bf16 x bfp16 (about 1% relative per product); the residual
// stream and activations are bf16.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>

#include "src/core/npu/npu.hpp"

namespace gufo::npu {

class Qwen3Prefill {
public:
  static constexpr int kLayers = 28, kHidden = 2048, kHeads = 16, kKvHeads = 8,
                       kHeadDim = 128, kFfn = 6144;

  /// One layer's weights, row-major [out][in] bf16 bit patterns, and its norm
  /// weights in f32.
  struct Layer {
    const std::uint16_t *q, *k, *v, *o, *gate, *up, *down;
    const float *input_norm, *post_norm, *q_norm, *k_norm;
  };

  /// Loads every `dir`/qwen3pre_<M>.xclbin with its instruction stream and
  /// plan. Throws Error when there is none.
  static Qwen3Prefill Open(Device& device, const std::filesystem::path& dir);

  Qwen3Prefill();
  ~Qwen3Prefill();
  Qwen3Prefill(Qwen3Prefill&&) noexcept;
  Qwen3Prefill& operator=(Qwen3Prefill&&) noexcept;

  [[nodiscard]] explicit operator bool() const noexcept {
    return impl_ != nullptr;
  }

  /// Packs layer `l` (bfp16 of the bf16 weights, norm weights folded).
  void SetLayer(int l, const Layer& layer);
  /// RoPE frequencies theta^(-2i / 128).
  void SetRopeTheta(double theta);
  /// The most rows one Run takes.
  [[nodiscard]] int max_rows() const noexcept;

  /// Runs rows x [n][2048] (f32, the embeddings) at positions [0, n).
  void Run(const float* x, int n);
  /// After Run: layer `l`'s cache entries of positions [0, n), K after its
  /// norm and rotation and V, each [n][8][128] f32.
  void Kv(int l, int n, float* k, float* v) const;

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::npu
