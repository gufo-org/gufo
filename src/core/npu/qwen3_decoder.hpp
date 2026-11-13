#pragma once

// One Qwen3 decode token per NPU dispatch: the 28-layer dense stack (hidden
// 2048, 16 query and 8 KV heads of 128, SwiGLU 6144, per-head q/k RMSNorm,
// NeoX RoPE) of Qwen3-ASR's text decoder and Qwen3-TTS's talker, plus an
// output head of up to 152064 rows. The program is the IRON design in
// src/core/npu/kernels/qwen3dec (a phase interpreter on all 32 cores);
// weights are Q8_0, activations are quantized to Q8_0 on the cores, the KV
// cache is bf16 and holds up to 4096 positions.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

#include "src/core/npu/npu.hpp"

namespace gufo::npu {

class Qwen3Decoder {
public:
  static constexpr int kLayers = 28, kHidden = 2048, kHeads = 16, kKvHeads = 8,
                       kHeadDim = 128, kFfn = 6144;
  static constexpr int kMaxHeadRows = 152064;

  /// One layer's weights, row-major [out][in] bf16 bit patterns, and its norm
  /// weights in f32.
  struct Layer {
    const std::uint16_t *q, *k, *v, *o, *gate, *up, *down;
    const float *input_norm, *post_attention_norm, *q_norm, *k_norm;
  };

  /// Loads `dir`/qwen3dec.xclbin, its dispatch library and layout.
  static Qwen3Decoder Open(Device& device, const std::filesystem::path& dir);

  Qwen3Decoder();
  ~Qwen3Decoder();
  Qwen3Decoder(Qwen3Decoder&&) noexcept;
  Qwen3Decoder& operator=(Qwen3Decoder&&) noexcept;

  [[nodiscard]] explicit operator bool() const noexcept {
    return impl_ != nullptr;
  }

  /// Packs layer `l` (Q8_0 of the bf16 weights).
  void SetLayer(int l, const Layer& layer);
  /// The output head ([rows][2048] bf16, rows <= kMaxHeadRows) and the final
  /// norm.
  void SetHead(const std::uint16_t* weights, int rows, const float* norm);
  /// RoPE frequencies theta^(-2i / 128).
  void SetRopeTheta(double theta);
  /// Positions the KV cache holds (4096).
  [[nodiscard]] int capacity() const noexcept;

  /// The cache entries of `layer` at positions [pos, pos + count), as prefill
  /// computed them: k after its norm and rotation, and v, each row
  /// [8 heads][128] at the given row strides (floats).
  void WriteKv(int layer, int pos, int count, const float* k,
               std::size_t k_stride, const float* v, std::size_t v_stride);

  /// Starts the token whose input row (2048 f32, the embedding) sits at
  /// position `pos`; the cache must hold positions [0, pos).
  void Start(const float* row, int pos);
  /// Waits for the started token. Its logits (head rows) and its final
  /// hidden state before the final norm are then readable.
  void Wait();
  [[nodiscard]] std::span<const float> logits();
  void Hidden(float* out);

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::npu
