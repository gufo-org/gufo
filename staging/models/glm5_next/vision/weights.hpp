#ifndef GUFO_MODELS_GLM5_NEXT_VISION_WEIGHTS_HPP_
#define GUFO_MODELS_GLM5_NEXT_VISION_WEIGHTS_HPP_

#include <cstdint>
#include <optional>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "staging/models/glm5_next/vision/config.hpp"
#include "staging/models/glm5_next/weights.hpp"

namespace gufo::models::glm5_next::vision {

/// One vision transformer block. Projections are F16 (the mmproj ships
/// F16/F32); norms and biases are F32. Q/K/V live in one fused `qkv` row
/// block laid out as the reference `qkv` linear: rows [0, emb) are Q (head
/// h owns rows h*hd..), [emb, 2*emb) K and [2*emb, 3*emb) V. `q_norm`/
/// `k_norm` are per-head RMSNorm weights of length head_dim (64); V has no
/// norm. Every projection carries its F32 bias.
struct BlockWeights {
  TensorRef ln1;        ///< [1024] pre-attention RMSNorm.
  TensorRef qkv;        ///< [1024 -> 3072].
  TensorRef qkv_bias;   ///< [3072].
  TensorRef q_norm;     ///< [64] per-head RMSNorm.
  TensorRef k_norm;     ///< [64] per-head RMSNorm.
  TensorRef attn_out;   ///< [1024 -> 1024].
  TensorRef attn_bias;  ///< [1024].
  TensorRef ln2;        ///< [1024] pre-FFN RMSNorm.
  TensorRef ffn_up;     ///< [1024 -> 4096].
  TensorRef ffn_up_bias;
  TensorRef ffn_gate;   ///< [1024 -> 4096].
  TensorRef ffn_gate_bias;
  TensorRef ffn_down;       ///< [4096 -> 1024].
  TensorRef ffn_down_bias;  ///< [1024].
};

/// The `glm5v` vision tower bound from the mmproj sidecar. The reference
/// patch embedding is a temporal-2 Conv3d; the GGUF splits it into
/// `v.patch_embd.weight` and `.weight.1` with identical [kx=14, ky=14,
/// cin=3, cout=1024] layout, so the two kernels sum into one stride-14
/// conv2d over the single decoded frame. `Bind` folds that sum host-side
/// into `patch_weight` as row-major [cout][kx + 14*ky + 196*cin] F32.
/// `merger` is the 2x2 stride-2 conv folded the same way into row-major
/// [cout=4096][kx + 2*ky + 4*cin]; everything else binds as-is.
struct VisionWeights {
  Config config;
  std::vector<float> patch_weight;  ///< 1024 rows x 588 cols, F32 (summed).
  TensorRef patch_bias;             ///< [1024] F32.
  TensorRef post_ln;                ///< [1024] F32 (tower tail).
  std::vector<BlockWeights> blocks;
  TensorRef merger;      ///< [4096 -> 4096] folded, F16.
  TensorRef merger_bias; ///< [4096] F32.
  TensorRef fc;          ///< [4096 -> 4096], no bias.
  TensorRef ln_weight;   ///< [4096] projector LayerNorm.
  TensorRef ln_bias;     ///< [4096].
  TensorRef up;          ///< [4096 -> 10240], no bias.
  TensorRef gate;        ///< [4096 -> 10240], no bias.
  TensorRef down;        ///< [10240 -> 4096], no bias.

  [[nodiscard]] static std::optional<VisionWeights> Bind(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

}  // namespace gufo::models::glm5_next::vision

#endif  // GUFO_MODELS_GLM5_NEXT_VISION_WEIGHTS_HPP_
