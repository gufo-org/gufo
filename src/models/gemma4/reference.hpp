#ifndef GUFO_MODELS_GEMMA4_REFERENCE_HPP_
#define GUFO_MODELS_GEMMA4_REFERENCE_HPP_

#include <cstdint>
#include <span>
#include <vector>

#include "src/models/gemma4/tokenizer.hpp"
#include "src/models/gemma4/weights.hpp"

namespace gufo::models::gemma4 {

/// Scalar reference of the Gemma 4 graph (llama.cpp 391fac16
/// `src/models/gemma4.cpp`). Weights are dequantized on the fly, every
/// reduction accumulates in double, and every layer keeps full K/V for all
/// positions, so it only suits short prefixes. It pins operator semantics:
/// scaled embeddings, weighted q/k norms, the unweighted V norm and K=V on
/// global layers, NEOX rope with per-layer bases and frequency divisors,
/// window-masked attention with scale 1, sandwich norms, GeGLU, routed
/// experts (26B-A4B), the layer output scale, and the final logit softcap.
class Reference {
public:
  /// kFloat32 keeps K/V exact; kHalfKv rounds stored K/V through binary16,
  /// as the GPU cache does. kQ8Activations additionally rounds every input of
  /// a quantized projection to Q8_1 blocks (32 values, binary16 scale), the
  /// activation format of llama.cpp's and Gufo's integer GEMM kernels; it
  /// separates operator semantics from activation rounding.
  enum class Storage { kFloat32, kHalfKv, kQ8Activations };

  Reference(const ModelWeights& weights, Storage storage);

  /// Rows [row, row + count) of one Forward call hold an image: `embedding`
  /// ([count][hidden], the encoder output) replaces their token embeddings
  /// unscaled, and in sliding layers each of them sees every key of its
  /// image (the window still bounds older keys); global layers stay causal.
  struct Image {
    std::uint32_t row{0};
    std::uint32_t count{0};
    const float* embedding{nullptr};
  };

  /// Appends `tokens` after the current position. `logits` receives
  /// [tokens.size()][vocab] softcapped rows and `hidden` the post-norm
  /// [tokens.size()][hidden] rows fed to the LM head and to MTP; either may
  /// be null. Processing a prefix in several calls gives identical results.
  void Forward(std::span<const TokenId> tokens, std::vector<float>* logits,
               std::vector<float>* hidden, std::span<const Image> images = {});

  /// When enabled, Forward records the residual stream after every layer:
  /// [layer][row][hidden] for the rows of the latest call.
  void SetTrace(bool enabled) noexcept { trace_ = enabled; }
  [[nodiscard]] const std::vector<float>& LayerTrace() const noexcept {
    return layer_trace_;
  }

  [[nodiscard]] std::uint32_t Position() const noexcept { return position_; }
  void Reset();

private:
  /// out[r][o] = sum_c W[o][c] * x[r][c] for `rows` input rows.
  void MatMul(const TensorRef& weight, const float* x, std::size_t rows,
              float* out) const;
  /// `ends` (sliding layers, may be empty): per row, the exclusive key end
  /// when it exceeds the row's own position + 1.
  void Attention(std::uint32_t layer, const float* q, std::size_t rows,
                 std::span<const std::uint32_t> ends, float* out) const;
  /// Routed expert mixture of `rows` attention residual rows (26B-A4B):
  /// router, top-k softmax weights, per-expert scale, GeGLU experts.
  void Experts(const LayerWeights& w, const float* x, std::size_t rows,
               float* out) const;

  const ModelWeights& weights_;
  Storage storage_;
  std::uint32_t position_{0};
  /// Per layer, [position][kv_heads * head_dim].
  std::vector<std::vector<float>> keys_;
  std::vector<std::vector<float>> values_;
  bool trace_{false};
  std::vector<float> layer_trace_;
};

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_REFERENCE_HPP_
